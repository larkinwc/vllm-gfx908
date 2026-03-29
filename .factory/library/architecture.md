# Architecture: TurboQuant Custom Attention Backend for vLLM

## System Overview

TurboQuant KV cache compression is integrated into vLLM as a custom attention backend via the `register_backend(AttentionBackendEnum.CUSTOM)` API. The backend extends `RocmAttentionBackend` and is instantiated inside each GPU worker process naturally, solving the multi-process architecture problem.

## Key Components

### TurboQuant Backend (`/opt/turboquant/turboquant/backends/vllm_rocm.py`)

- **TurboQuantRocmBackend**: Extends `RocmAttentionBackend`. Registered as `AttentionBackendEnum.CUSTOM`. Provides `TurboQuantRocmImpl` as the implementation class.
- **TurboQuantRocmImpl**: Extends `RocmAttentionImpl`. Each instance owns per-layer TQ state (CompressedKVStore, KVCaptureEngine, quantizer). Overrides `do_kv_cache_update()` to capture KV into compressed store, and `forward()` to optionally use TQ hybrid decode.

### Per-Layer State (lives in worker process)

Each `TurboQuantRocmImpl` instance owns:
- `CompressedKVStore` -- chunked compressed KV history (3-bit keys via MSE+QJL, 2-bit values via group quantization)
- `KVCaptureEngine` -- ring buffer (128 recent tokens in full precision) + bulk capture for prefill
- `TurboQuantProd` quantizer -- rotation matrix Pi (DxD), QJL matrix S (DxD), codebook (8 centroids)

### Qwen3.5-9B Hybrid Architecture

- 32 total layers: 8 full-attention + 24 linear-attention (GDN)
- TQ only applies to the 8 full-attention layers
- Linear-attention layers use a separate backend (GDNAttentionBackend) -- TQ does not touch them
- vLLM routes layers to their respective backends automatically based on layer type

### Data Flow

**Capture-only mode (Phase 1):**
```
Request → vLLM Scheduler → Worker Process → Model Forward
  → TurboQuantRocmImpl.do_kv_cache_update():
      1. Write to paged KV cache (standard path via super())
      2. Capture K,V into CompressedKVStore (quantize and store)
  → TurboQuantRocmImpl.forward():
      1. Always delegate to super().forward() (standard flash/paged attention)
      2. TQ compressed store is populated but not used for decode
```

**Hybrid mode (Phase 2):**
```
Prefill:
  → do_kv_cache_update(): write to paged cache + capture into TQ store
  → forward(): use standard flash attention for prefill (super())

Decode (single token):
  → do_kv_cache_update(): append to ring buffer, flush oldest to compressed store if full
  → forward():
      IF compressed store has >= 16 tokens:
        Use TQ hybrid decode (Triton fused kernel over compressed history + ring buffer)
      ELSE:
        Fall back to standard paged attention (super())
```

### Registration Flow

```python
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
register_backend(AttentionBackendEnum.CUSTOM, "turboquant.backends.vllm_rocm.TurboQuantRocmBackend")
```

This must be called before vLLM engine initialization. The launch script handles this.

### vLLM Backend Selection Path

```
vllm/v1/attention/selector.py::get_attn_backend()
  → _cached_get_attn_backend()
    → current_platform.get_attn_backend_cls() [rocm.py]
      → If selected_backend is not None: validate and return its path
      → Else: auto-select from priority list
    → resolve_obj_by_qualname(class_path) → actual backend class
```

When `--attention-backend CUSTOM` is passed (or equivalent config), vLLM selects `AttentionBackendEnum.CUSTOM` which resolves to whatever was registered.

### Memory Budget

Per full-attention layer overhead:
- Pi rotation matrix: 256x256 float32 = 256 KB
- S QJL matrix: 256x256 float32 = 256 KB  
- Codebook: 8 float32 = 32 B
- Ring buffer: 128 tokens x 256 dim x 2 (K+V) x 2 bytes = 128 KB per KV head
- Total per layer: ~600 KB (negligible vs 32 GB VRAM)

### HIP Graph Compatibility

FULL_DECODE_ONLY captures decode forward passes into HIP graphs. If TQ Triton kernels are graph-capturable, hybrid decode benefits from graph mode. If not, hybrid decode must run in eager mode while capture_only can still use graph mode (since it delegates to standard forward).

## Existing Infrastructure

- Baseline benchmarks: `/root/benchmark-results/` (from previous mission)
- Benchmark scripts: `/root/benchmark-scripts/` (reusable)
- Production launch: `/root/launch-vllm-optimized.sh` (FULL_DECODE_ONLY + prefix caching)
- TQ standalone tests: Already verified 3 Triton kernels compile/run on gfx908
