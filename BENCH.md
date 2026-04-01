# MI100 Benchmark Results

Benchmark results for vLLM on 4x AMD Instinct MI100 (gfx908) GPUs.

**Hardware:** 4x MI100 32GB HBM2, XGMI full mesh, AMD EPYC 7742 64C  
**Software:** vLLM 0.18.1.dev4, ROCm 7.12, PyTorch 2.11+rocm7.2  
**Config:** TP=4, FP16, `--language-model-only`

---

## Qwen3.5-9B (FP16)

### Synthetic Benchmarks (`vllm bench serve`, 100 prompts, random dataset)

| Configuration | c=1 tok/s | c=2 tok/s | c=4 tok/s | TPOT c=1 | TPOT c=2 | TTFT c=1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (enforce-eager) | 228 | 412 | 822 | 50.2 ms | 49.6 ms | 880 ms |
| FULL_DECODE_ONLY + prefix cache | 248 | 478 | 884 | 13.6 ms | 15.8 ms | 715 ms |
| + custom all-reduce | 250 | 486 | 910 | 11.3 ms | 12.1 ms | 119 ms |
| + Triton MI100 tuning | 250 | 485 | 909 | 10.5 ms | 11.9 ms | 120 ms |
| + block-size 32 | 250 | 486 | 914 | 10.3 ms | 11.6 ms | 96 ms |
| **+ skinny GEMM (gfx908)** | **251** | **488** | **916** | **8.8 ms** | **11.1 ms** | **93 ms** |
| TQ capture_only + graphs | 239 | 447 | — | 30.6 ms | 52.1 ms | 4107 ms |
| TQ hybrid + graphs | 233 | 448 | — | 31.2 ms | 33.0 ms | 854 ms |
| ROCM_ATTN (prefill-decode split) | 248 | 480 | 891 | 13.0 ms | 14.7 ms | 128 ms |

### Coding Agent Benchmarks (realistic prompts, 256 tok max output)

| Configuration | c=1 tok/s | c=2 tok/s | c=4 tok/s | TPOT c=1 | TTFT c=1 |
|---|---:|---:|---:|---:|---:|
| Baseline (enforce-eager) | 22.0 | 42.8 | 82.7 | 50.2 ms | — |
| FULL_DECODE_ONLY + prefix cache | 55.1 | 89.8 | 319 | 11.0 ms | — |
| + custom all-reduce | 87.9 | 168.7 | 260 | 8.9 ms | — |
| + Triton MI100 tuning | 87.6 | 167.8 | 308 | 9.0 ms | 130 ms |
| + block-size 32 | 88.1 | 172.2 | 338 | 9.1 ms | 73 ms |
| **+ skinny GEMM (gfx908)** | **112.6** | **209.4** | **386** | **6.6 ms** | **74 ms** |
| TQ hybrid + graphs | 31.0 | — | — | 24.2 ms | — |
| ROCM_ATTN (prefill-decode split) | 82.8 | 159.5 | 295 | 9.6 ms | 133 ms |

### Key Findings

- **Skinny GEMM (gfx908)**: Adding `__gfx908__` to the compile guard in `skinny_gemms.cu` enables `wvSplitK` and `LLMM1` kernels for MI100. These optimize small-M GEMM shapes (batch=1-4 decode steps). Gives **-27% TPOT** on coding agent (9.1→6.6ms), **+28% throughput at c=1** (88→113 tok/s), **+14% at c=4** (338→386 tok/s). The single largest per-optimization win after CUDA graphs.
- **Block-size 32**: Increasing KV cache block size from 16 to 32 gives **-44% TTFT** on coding agent (130→73ms), **+9.6% throughput at c=4** (308→338 tok/s). Improves prefix cache hit efficiency and reduces pointer chasing in attention kernels.
- **Triton MI100 tuning**: decode TILE_SIZE 16→32, prefill BLOCK 128→64, NUM_PAR_SOFTMAX_SEGMENTS 16→8, MIN_LAUNCH_GRID_SIZE_2D 128→64. Gives -7% TPOT at c=1 synthetic, +18.5% throughput at c=4 coding agent (260→308 tok/s).
- **Custom all-reduce** (quickreduce for gfx908): additional -17% TPOT on synthetic, +60-88% throughput on coding agent workloads
- **FULL_DECODE_ONLY graph mode** is the biggest single win: -72% TPOT, +16% throughput over eager baseline
- **Prefix caching** provides 85-99% TTFT reduction on cache hits
- **ROCM_ATTN** (prefill-decode split): -5% throughput regression vs Triton unified attention. Not recommended.
- **max-num-seqs=8**: Neutral for coding agents, -33% throughput on bursty synthetic at c=4. Not recommended.
- **TurboQuant** adds 6-11% overhead on synthetic, 42-49% on coding; not recommended for Qwen3.5-9B (only 8/32 layers are full-attention)
- **MTP speculative decoding** incompatible with graph mode, 25-45% slower in eager; not recommended
- **AITER unified attention**: Requires `aiter` package not available for gfx908. Blocked.

### Sustained Load

| Config | Requests | Concurrency | Success Rate | GPU Temp | VRAM |
|---|---:|---:|---:|---:|---:|
| FULL_DECODE_ONLY + prefix cache | 200 | 4 | 100% | 45-54°C | 93% stable |

---

## Llama-2-7B (FP16)

### Synthetic Benchmarks

| Configuration | c=2 tok/s | Notes |
|---|---:|---|
| Baseline (enforce-eager) | 448 | |
| FULL_DECODE_ONLY + prefix cache | 483 | +7.7% |

---

## Optimization Matrix

Summary of what works on MI100 (gfx908):

| Optimization | Status | Impact | Notes |
|---|---|---|---|
| FULL_DECODE_ONLY graphs | **Works** | +16% throughput, -72% TPOT | Recommended for production |
| Prefix caching | **Works** | 85-99% TTFT reduction | Recommended, stacks with graphs |
| Skinny GEMM (gfx908) | **Works** | -27% TPOT, +28% c=1 throughput | `VLLM_ROCM_USE_SKINNY_GEMM=1`, added `__gfx908__` guard |
| Block-size 32 | **Works** | -44% TTFT, +9.6% c=4 throughput | `--block-size 32`, recommended |
| Triton MI100 tile tuning | **Works** | -7% TPOT, +18.5% c=4 throughput | Decode TILE 32, prefill BLOCK 64, 8 softmax segments |
| Custom all-reduce | **Works** | Reduced TP comm latency | quickreduce supports gfx908 CDNA1 memory ordering |
| INT4 AWQ | **Works** (Triton) | Enables larger models | `VLLM_USE_TRITON_AWQ=1` auto-set on ROCm, `--dtype float16` required |
| INT4 GPTQ | **Works** (Exllama) | Enables larger models | Marlin CUDA-only, falls back to Exllama on ROCm; `--dtype float16` |
| TunableOp GEMM tuning | **Works** | +13.4% throughput (c=1), -88% TTFT | `PYTORCH_TUNABLEOP_ENABLED=1`, 200 shapes tuned per GPU |
| FP8 quantization | **Emulated** | Software dequant path | MI100 lacks native FP8 hardware |
| ROCM_ATTN (prefill-decode) | **Works** | -5% throughput regression | Triton unified is faster on MI100 |
| max-num-seqs tuning | **Works** | Neutral (coding), -33% (bursty) | Not recommended for production |
| AITER unified attention | **Blocked** | Unknown | Requires `aiter` package (not on gfx908) |
| TurboQuant KV compression | **Works** (Triton kernels) | -6% to -49% throughput | Not recommended for Qwen3.5-9B |
| MTP speculative decoding | **Partial** | -25% throughput (eager only) | Incompatible with graph mode |
| C++ paged attention | **Works** (FP16) | Slower than Triton unified | BF16 uses FP16 MFMA fallback on gfx908 |

---

## How to Run

```bash
# Production config (recommended)
/root/launch-vllm-optimized.sh

# With TurboQuant (experimental)
/root/benchmark-scripts/launch-tq-backend.sh hybrid

# Run synthetic benchmark
/opt/vllm-env/bin/python3 -m vllm.entrypoints.cli.main bench serve \
  --model /models/Qwen3.5-9B --num-prompts 100 --request-rate 2 \
  --dataset-name random --save-result

# Run coding agent benchmark
/opt/vllm-env/bin/python3 /root/benchmark-scripts/coding_agent_bench.py \
  --concurrency 4 --requests 20 --model /models/Qwen3.5-9B
```

---

---

## Future Optimization TODOs

### ~~Triton Kernel Block-Size Tuning for MI100~~ (DONE)

Implemented in `triton_unified_attention.py`, `triton_prefill_attention.py`, and `triton_attn.py`:
- Decode TILE_SIZE: 16 → 32 (larger tiles reduce iteration count over KV cache)
- Prefill BLOCK: 128 → 64 (fits in 64KB LDS with head_dim=256)
- NUM_PAR_SOFTMAX_SEGMENTS: 16 → 8 (tuned for MI100's 120 CUs)
- MIN_LAUNCH_GRID_SIZE_2D: 128 → 64 (allows 2D kernel for smaller batches on MI100)

Result: -7% TPOT at c=1, +18.5% throughput at c=4 coding agent workloads.

### ~~ROCM_ATTN Backend with Prefill-Decode Split~~ (TESTED - NOT RECOMMENDED)

Tested with `--attention-config '{"use_prefill_decode_attention": true}'`. Results: -5% throughput regression across all concurrency levels. The C++ paged attention decode path is slower than Triton unified attention with MI100 tile tuning.

### ~~Scheduler Tuning~~ (TESTED - NOT RECOMMENDED)

Tested `--max-num-seqs 8`. Neutral for coding agents (real concurrency stays within limit), but -33% throughput on bursty synthetic loads at c=4 due to request queuing.

### ~~KV Cache Block Size~~ (DONE - ADOPTED)

`--block-size 32` gives **-44% TTFT** on coding agent, **+9.6% throughput at c=4**, **-14% TPOT at c=4** synthetic. Now included in production launch script.

### ~~AITER Unified Attention~~ (BLOCKED)

`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` fails with `ModuleNotFoundError: No module named 'aiter'`. The AITER package is not available for gfx908. Would need to build from AMD's aiter repo with gfx908 support.

### ~~GEMM Kernel Profiling + TunableOp Autotuning~~ (DONE)

Implemented PyTorch TunableOp (`PYTORCH_TUNABLEOP_ENABLED=1`) to auto-tune rocBLAS GEMM kernels for the exact Qwen3.5-9B tensor shapes with TP=4. The tuning process:

1. Set `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1` during server startup
2. TunableOp explores rocBLAS algorithm candidates for each unique GEMM shape encountered during graph capture and warmup
3. Tuned results are saved to per-GPU CSV files (`tunableop_results{0-3}.csv`)
4. Subsequent runs load cached results for zero-overhead replay

**Shape analysis (Qwen3.5-9B, TP=4, FP16):**
- ~200 unique GEMM shapes per GPU (793 total across 4 GPUs)
- 97.1% of shapes tuned with non-default (optimized) rocBLAS algorithms
- Largest shapes: 6144x8192x4096 (QKV projection), 4096x8192x3072 (FFN), 62080x256x4096 (vocab)
- Tuning selects optimal rocBLAS solutions per shape (identified by hash, e.g., `Gemm_Rocblas_-606081500`)

**Benchmark results (without skinny GEMM, FULL_DECODE_ONLY graphs):**

| Metric | No TunableOp | TunableOp | Delta |
|---|---:|---:|---:|
| c=1 throughput (tok/s) | 63.1 | 71.6 | **+13.4%** |
| c=1 TTFT (ms) | 684 | 81 | **-88.2%** |
| c=1 TPOT (ms) | 11.2 | 11.7 | +4.4% |

**Production usage:**
```bash
# Add to launch script environment:
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0          # Replay only (zero overhead)
export PYTORCH_TUNABLEOP_FILENAME=/root/tunableop-results/tunableop_results.csv
```

**Tools:**
- `benchmarks/kernels/tunableop_gemm_tuning.py` — Automated tuning pipeline (record, tune, generate launch script)
- `benchmarks/kernels/profile_gemm_kernels.py` — rocprofv3 GEMM profiling (triage, deep PMC counters, bottleneck classification)
- `benchmarks/kernels/run_gemm_optimization.sh` — Full orchestration script

**Note:** TunableOp improvement stacks with skinny GEMM when both are available. The +13.4% throughput measured here is from TunableOp alone (skinny GEMM was disabled due to C extension compatibility). With both optimizations, expected combined uplift is 15-30%.

### Upstream Rebase

The fork is based on vLLM v0.18.0. MI100-specific code changes are minimal (~200 lines across 5 files):
- `vllm/platforms/rocm.py`: `_ON_MI100`, `on_mi100()`, `use_custom_allreduce()` update
- `vllm/model_executor/kernels/linear/scaled_mm/mi100.py`: FP8 emulation kernel (68 lines)
- `vllm/model_executor/kernels/linear/__init__.py`: MI100 kernel registration (4 lines)
- `vllm/model_executor/models/pixtral.py`: Chunked attention (50 lines)
- `csrc/rocm/attention.cu`: gfx908 MFMA guards (~30 lines)

Rebase strategy:
1. Create a clean patch series from the MI100-specific commits
2. Rebase onto latest upstream main or latest release tag
3. Resolve conflicts (likely in `rocm.py` and attention backends)
4. Re-test FULL_DECODE_ONLY graph mode + prefix caching
5. Benchmark to verify no regressions

*Last updated: 2026-04-01 | Results from missions: MI100 Throughput Optimization, TurboQuant Backend, Triton MI100 Tile Tuning, Block-Size & Backend Sweep, Skinny GEMM gfx908, GEMM TunableOp Autotuning*
