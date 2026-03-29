# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Required Environment Variables for MI100

```bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export VLLM_ROCM_USE_AITER=1
export PATH=/opt/rocm/core-7.12/bin:$PATH
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export TORCH_COMPILE_DISABLE=1          # Must keep even in graph mode (avoids gfx908 cluster_dims error)
export VLLM_ROCM_USE_SKINNY_GEMM=0     # wvSplitK is MI300X-only
```

## Python Environment

- Virtual env: `/opt/vllm-env/`
- Python: 3.12
- PyTorch: 2.11.0+rocm7.2
- Triton: pytorch-triton-rocm 3.5.1
- vLLM: 0.18.1.dev4 installed editable from `/root/vllm-gfx908-src`

## Hardware

- 4x MI100 (gfx908), 32GB HBM2 each, 120 CUs, 1.23 TB/s bandwidth
- XGMI Infinity Bridge full mesh (1-hop)
- AMD EPYC 7742 64C/64T
- 64GB DDR4 system RAM
- 1.2TB free disk (ZFS on NVMe)

## Known Issues

- `TORCH_COMPILE_DISABLE=1` needed to avoid `KernelMetadata.cluster_dims` error
- `VLLM_ROCM_USE_SKINNY_GEMM=0` needed to avoid MI300X-only kernel crash
- flashinfer is CUDA-only, must be uninstalled
- PyTorch ROCm version must match host ROCm major version (use rocm7.2 wheels with ROCm 7.12)
- Upstream triton (CUDA-only) conflicts with pytorch-triton-rocm

## Models

- `/models/Qwen3.5-27B-AWQ-BF16-INT4` (27GB, currently deployed)
- `/models/Qwen3.5-9B` (FP16 ~19GB, downloaded and verified; use `--max-model-len 32768 --language-model-only`)
- `/models/Llama-2-7b-hf` (FP16 ~13GB, NousResearch mirror; **max_position_embeddings=4096**, use `--max-model-len 4096`; requires `--chat-template /root/vllm-gfx908-src/vllm/transformers_utils/chat_templates/template_llama2.jinja`)
- INT4 models (GPTQ, AWQ compressed-tensors) are NOT supported on MI100/ROCm — see `model-verification.json` quantization_feasibility section for details

## Benchmark Infrastructure

- Scripts: `/root/benchmark-scripts/` (run_synthetic_bench.sh, coding_agent_bench.py, run_all_baselines.sh, compare_results.py)
- Results: `/root/benchmark-results/` (JSON benchmark files + GPU stats .txt files)
- Requires `aiohttp` for coding_agent_bench.py (`/opt/vllm-env/bin/pip install aiohttp`)
- `aggregate_decode_tok_per_s` in coding_agent_bench.py correctly uses wall clock time (`total_decode_tokens / wall_clock_seconds`). Qwen3.5-9B baselines: c1=21.4 tok/s, c2=41.7 tok/s, c4=82.7 tok/s (scales ~4x with concurrency as expected).
- **Note**: Llama-2-7b-hf coding agent benchmarks in baseline-report.json use pre-fix data (wall_clock_seconds=0, aggregate=avg at all concurrency levels). Re-run required for reliable Llama-2-7b-hf coding agent baselines.
- GPU VRAM utilization: ~93% per GPU for both Qwen3.5-9B (max-model-len=32768) and Llama-2-7b-hf (max-model-len=4096), TP=4

## HIP Graph Mode Notes

- Graph mode server startup: ~100 seconds (vs ~60s for enforce-eager). Health check / readiness probes must use timeout >= 120s.
- FULL_DECODE_ONLY graph memory overhead: 0.16 GiB (35 graph sizes captured)
- TORCH_COMPILE_DISABLE=1 is still required with graph mode on gfx908; FULL_DECODE_ONLY does NOT use torch.compile/inductor
- Piecewise graph compilation on gfx908: not yet tested; only FULL_DECODE_ONLY has been validated
- Optimized launch script: `/root/benchmark-scripts/launch-config-tuned.sh` (graph mode + prefix caching + max-model-len 32768)
