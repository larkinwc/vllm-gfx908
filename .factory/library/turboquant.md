# TurboQuant Integration Notes

## Installation

TurboQuant is installed from https://github.com/0xSero/turboquant at `/opt/turboquant`.

```bash
# Install (after ensuring PyTorch ROCm is present)
/opt/vllm-env/bin/pip install -e /opt/turboquant
```

**CRITICAL**: TurboQuant's setup.py specifies `torch>=2.1` which causes pip to replace the ROCm PyTorch with the CUDA version! Always reinstall ROCm PyTorch after installing TurboQuant:

```bash
/opt/vllm-env/bin/pip uninstall torch triton -y
/opt/vllm-env/bin/pip install --pre torch==2.11.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2
```

## Triton Kernels on ROCm/gfx908

**All three Triton kernels compile and execute successfully on ROCm 7.12/gfx908:**

1. `_turboquant_mse_score_kernel` - MSE attention score computation
2. `_turboquant_qjl_score_kernel` - QJL residual score computation
3. `_turboquant_fused_decode_kernel` - Full fused decode attention (online softmax)

No ROCm-specific Triton issues observed (no `tl.atomic_*` failures, no `cluster_dims` errors).

## Quantization Quality

| Quantizer | Bits | Cosine Similarity | Notes |
|-----------|------|-------------------|-------|
| MSE (keys) | 3-bit | 0.983 | Near-lossless |
| Prod (MSE+QJL) | 3-bit | 0.919 | Acceptable |
| Values | 2-bit | ~0.94 | Quality bottleneck per paper |

## Compression Ratio

- Original bf16 K+V per token: 1024 bytes (D=256)
- Compressed: 196 bytes per token
- Compression ratio: **5.22x** (exceeds paper's ~4.4x claim)

## vLLM Integration

TurboQuant provides vLLM integration via monkey-patching:

```python
from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE

# After LLM engine initialization
hooks = install_turboquant_hooks(
    model_runner,
    key_bits=3,
    value_bits=2,
    buffer_size=128,
    mode=MODE_ACTIVE
)
```

Integration files:
- `turboquant/vllm_attn_backend.py` - Legacy API shim
- `turboquant/integration/vllm.py` - New modular integration

## Qwen3.5-9B Specifics

- 8 full-attention layers out of 32 (25% compressible)
- 24 linear-attention layers (not compressible by TQ)
- Expected KV savings: ~30% on full-attention layers
- head_dim=256 (supported by TurboQuant)

## Modes

- `MODE_OFF`: No TQ, passthrough
- `MODE_CAPTURE_ONLY`: Capture KV to compressed store, use flash output
- `MODE_HYBRID`: Use compressed history + exact recent for decode
- `MODE_FULL_TQ`: (future) Full TQ path including prefill

## Fallback

If Triton kernels fail, TurboQuant has a pure-PyTorch fallback path in `score.py` (`compute_hybrid_attention`).

## Validation Contract Status

- VAL-TQ-001: PASS (turboquant import succeeds)
- VAL-TQ-002: PASS (Triton kernels compile on ROCm)
- VAL-TQ-007: PASS (fallback path available)
