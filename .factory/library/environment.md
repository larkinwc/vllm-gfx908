# Environment

**What belongs here:** Required env vars, external dependencies, setup notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

## System

- 4x AMD MI100 (gfx908), 32GB VRAM each
- 64 CPU cores (AMD EPYC), 62GB RAM
- ROCm 7.12, Ubuntu 24.04
- Python 3.12.3 at `/opt/vllm-env/bin/python3`

## Key Paths

- vLLM env: `/opt/vllm-env/`
- vLLM source (read-only): current worktree
- TurboQuant: `/opt/turboquant/` (editable install)
- Models: `/models/Qwen3.5-9B`
- Benchmark scripts: `/root/benchmark-scripts/`
- Benchmark results: `/root/benchmark-results/`
- Production launch: `/root/launch-vllm-optimized.sh`

## Environment Variables

- `HF_TOKEN` -- set via environment variable (required for gated models)
- `TORCH_COMPILE_DISABLE=1` (required for gfx908 to avoid torch.compile issues)
- `VLLM_TORCH_COMPILE_CONFIG` -- set to config path for graph mode
- `VLLM_USE_V1=1` -- v1 engine (default in 0.18.1)

## Known Issues

- TurboQuant's `setup.py` specifies `torch>=2.1` which causes pip to replace ROCm PyTorch with CUDA. Always verify after install.
- rocm-smi `--all --json` doesn't include temperature; use `--showtemp --json`
- `tests/distributed/` fails due to pre-existing RANK env var issue
- Only one vLLM instance at a time (93% VRAM per GPU)

## Previous Mission Results (Reference)

- Baseline Qwen3.5-9B: ~228 tok/s at c=1, ~412 tok/s at c=2
- Optimized (FULL_DECODE_ONLY + prefix cache): ~248 tok/s at c=1, ~478 tok/s at c=2
- TPOT: 50ms baseline → 14ms optimized (FULL_DECODE_ONLY)
- TQ Triton kernels: All 3 compile and run on gfx908, 5.22x compression, cos_sim=0.983
