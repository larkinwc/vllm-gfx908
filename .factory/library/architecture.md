# Architecture: vLLM MI100 Throughput Optimization

## System Overview

4x AMD MI100 (gfx908, CDNA1) GPUs connected via XGMI Infinity Bridge (full mesh, 1-hop) on an AMD EPYC 7742 host with 64GB RAM. ROCm 7.12, PyTorch 2.11+rocm7.2, vLLM 0.18.1.dev4 (mi100-fixes fork).

## vLLM Architecture on MI100

### Attention Backends

- **TRITON_ATTN**: Pure Triton-based attention, works on all ROCm GPUs including gfx908. Supports ALWAYS cudagraph compatibility. This is the primary backend for MI100.
- **ROCM_ATTN**: Legacy 2-path backend (Triton prefill + HIP paged attention decode). Supports custom paged attention on gfx9 family including gfx908.
- **ROCM_AITER_FA / ROCM_AITER_MLA**: AITER-based backends for MI300X+ (gfx942/gfx950) only. NOT available on MI100.

### MI100 Constraints

- No native FP8 hardware (CDNA1 limitation)
- `VLLM_ROCM_USE_SKINNY_GEMM=0` required (wvSplitK kernels are MI300X-only)
- `VLLM_ROCM_USE_AITER=1` enables Triton-based AITER ops that DO work on gfx908
- torch.compile/inductor has `KernelMetadata.cluster_dims` error on gfx908
- Originally ran with `--enforce-eager` and `TORCH_COMPILE_DISABLE=1`; now FULL_DECODE_ONLY graph mode is used

### CUDA/HIP Graph Modes (vLLM v1)

- NONE: No graphs (via --enforce-eager)
- PIECEWISE: Attention stays eager, everything else in graph (requires piecewise compilation) — NOT tested on gfx908
- FULL_DECODE_ONLY: Full graph for decode only, no graph for prefill — **VERIFIED WORKING on gfx908** (MI100 milestone 2)
- FULL_AND_PIECEWISE: Full for decode, piecewise for prefill (most performant but most memory) — NOT tested on gfx908
- TORCH_COMPILE_DISABLE=1 is still required with FULL_DECODE_ONLY on gfx908 (graph capture does not use inductor)
- Piecewise compatibility on gfx908: still unknown (skipped in favor of FULL_DECODE_ONLY which worked immediately)

### Model Architecture: Qwen3.5-9B

- Hybrid attention: 8 full-attention + 24 linear-attention layers (32 total)
- full_attention_interval: 4 (every 4th layer is full attention)
- head_dim: 256, num_attention_heads: 16, num_key_value_heads: 4 (GQA 4:1)
- Linear attention layers: linear_key_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32
- Native MTP: mtp_num_hidden_layers=1
- Max position embeddings: 262144
- Vision encoder present but we use --language-model-only

### Key Data Flows

1. **Request** → Scheduler → Model Runner → QKV Projection → Attention Backend → Sampling → Response
2. **KV Cache**: Paged block allocation, stored per-layer. Full-attention layers use standard KV cache. Linear-attention layers use their own cache format.
3. **TP=4**: Model sharded across 4 GPUs via XGMI. All-reduce for attention heads, expert parallelism for MoE (N/A for 9B).

### TurboQuant Integration Points

- Monkey-patches vLLM attention layers after initialization
- Captures KV entries during prefill, quantizes them (3-bit keys via MSE+QJL, 2-bit values via group quant)
- Frees original KV cache after quantization
- Hybrid decode: dequantizes compressed cache for decode attention
- Only compresses full-attention layers (8/32 for Qwen3.5-9B = 25%)
- Uses Triton kernels for fused decode attention (should work on ROCm but untested)

### MTP (Multi-Token Prediction)

- Native to Qwen3.5 models (mtp_num_hidden_layers=1)
- Predicts 1 extra token per step, verified against actual generation
- Reduces effective TPOT by ~30-50% when acceptance rate is high
- Adds one extra MTP head layer to VRAM but minimal overhead for 9B model
- Compatible with enforce-eager; compatibility with graph modes TBD

## File Locations

- vLLM source: `/root/vllm-gfx908-src` (also at worktree path)
- vLLM env: `/opt/vllm-env/`
- Models: `/models/`
- ROCm: `/opt/rocm/core-7.12/`
- Launch script: `/root/launch-vllm.sh`
- MI100-specific code: `vllm/platforms/rocm.py` (on_mi100(), on_gfx9())
- Attention backends: `vllm/v1/attention/backends/`
- Benchmark tools: `benchmarks/` dir + `vllm bench` CLI
