# Architecture: TurboQuant Custom Attention Backend for vLLM

## System Overview

TurboQuant KV cache compression is integrated into vLLM as a custom attention backend via the `register_backend(AttentionBackendEnum.TRITON_ATTN, ...)` override API. The backend extends `TritonAttentionBackend` (not RocmAttentionBackend) and is instantiated inside each GPU worker process naturally, solving the multi-process architecture problem.

## Key Components

### TurboQuant Backend (`/opt/turboquant/turboquant/backends/vllm_rocm.py`)

- **TurboQuantTritonBackend** (alias: TurboQuantRocmBackend): Extends `TritonAttentionBackend`. Registered as TRITON_ATTN override. Provides `TurboQuantTritonImpl` as the implementation class.
- **TurboQuantTritonImpl** (alias: TurboQuantRocmImpl): Extends `TritonAttentionImpl`. Each instance owns per-layer TQ state (CompressedKVStore, KVCaptureEngine) **initialized eagerly in **init**** (required for HIP graph compatibility). Overrides `do_kv_cache_update()` to capture KV into compressed store, and `forward()` to optionally use TQ hybrid decode.

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

```text
Request → vLLM Scheduler → Worker Process → Model Forward
  → TurboQuantRocmImpl.do_kv_cache_update():
      1. Write to paged KV cache (standard path via super())
      2. Capture K,V into CompressedKVStore (quantize and store)
  → TurboQuantRocmImpl.forward():
      1. Always delegate to super().forward() (standard flash/paged attention)
      2. TQ compressed store is populated but not used for decode
```

**Hybrid mode (Phase 2):**

```text
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

**CRITICAL:** Override TRITON_ATTN, NOT ROCM_ATTN or CUSTOM:

```python
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
register_backend(AttentionBackendEnum.TRITON_ATTN, "turboquant.backends.vllm_rocm.TurboQuantTritonBackend")
```

This must execute in every worker process (via sitecustomize.py in PYTHONPATH). No `--attention-backend` flag needed.

**Why TRITON_ATTN, not ROCM_ATTN or CUSTOM:**

- On ROCm, vLLM selects TRITON_ATTN (not ROCM_ATTN) for standard attention layers in `_get_backend_priorities()`
- ROCM_ATTN is only selected if `use_prefill_decode_attention=True`, which is not the default
- CUSTOM + `--attention-backend CUSTOM` forces ALL layers through system 1, crashing GDN layers
- TRITON_ATTN is the actual default for standard layers on ROCm → overriding it correctly captures only full-attention layers

### vLLM Dual Backend Routing

vLLM has TWO separate backend routing systems:

1. **AttentionBackendEnum** (standard attention) → `get_attn_backend()` → TRITON_ATTN on ROCm
2. **MambaAttentionBackendEnum** (mamba/GDN/linear) → `get_mamba_attn_backend()` → GDN_ATTN for GDN layers

Qwen3.5-9B's 8 full-attention layers use system 1 (TRITON_ATTN → now TQ).
Qwen3.5-9B's 24 GDN layers use system 2 (GDN_ATTN → unchanged).

### Memory Budget

Per full-attention layer overhead:

- Pi rotation matrix: 256x256 float32 = 256 KB
- S QJL matrix: 256x256 float32 = 256 KB  
- Codebook: 8 float32 = 32 B
- Ring buffer: 128 tokens x 256 dim x 2 (K+V) x 2 bytes = 128 KB per KV head
- Total per layer: ~600 KB (negligible vs 32 GB VRAM)

### HIP Graph Compatibility

FULL_DECODE_ONLY captures decode forward passes into HIP graphs. Both TQ capture_only and TQ hybrid modes ARE compatible with FULL_DECODE_ONLY graphs.

**VALIDATED:**

- TQ capture_only + FULL_DECODE_ONLY: 35 graphs captured, 10/10 requests coherent (VAL-GRAPH-001)
- TQ hybrid + FULL_DECODE_ONLY: 35 graphs captured, 10/10 requests coherent (VAL-GRAPH-002)

**Requirements for HIP graph compatibility:**

1. All random tensor generation must specify `device='cpu'` explicitly (e.g., `torch.randn(..., device='cpu')`)
2. Tensors used in forward() operations must be pre-registered as module buffers via `register_buffer()`, not created dynamically during forward
3. TQ state (CompressedKVStore, KVCaptureEngine) must be initialized eagerly in `__init__`, not lazily during first forward pass
4. The hybrid decode path uses PyTorch matmuls (`_matmul_attend` in score.py) which are graph-compatible

**Recommended production config for Qwen3.5-9B on MI100:**

**DO NOT use TurboQuant in production.** Benchmarking (2026-03-31) showed TQ hybrid causes 5-11% synthetic throughput regression and 42-49% coding throughput regression, with TPOT doubling. VRAM savings of 10.8% do not compensate. Use the optimized baseline instead:

```bash
/root/launch-vllm-optimized.sh  # FULL_DECODE_ONLY + prefix caching, no TQ
```

See `.factory/library/benchmarking-results.md` and `docs/turboquant-production-recommendation.md` for full data.

## Existing Infrastructure

- Baseline benchmarks: `/root/benchmark-results/` (from previous mission)
- Benchmark scripts: `/root/benchmark-scripts/` (reusable)
- Production launch: `/root/launch-vllm-optimized.sh` (FULL_DECODE_ONLY + prefix caching)
- TQ standalone tests: Already verified 3 Triton kernels compile/run on gfx908
