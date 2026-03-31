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
| **FULL_DECODE_ONLY + prefix cache** | **248** | **478** | **884** | **13.6 ms** | **15.8 ms** | **715 ms** |
| TQ capture_only + graphs | 239 | 447 | — | 30.6 ms | 52.1 ms | 4107 ms |
| TQ hybrid + graphs | 233 | 448 | — | 31.2 ms | 33.0 ms | 854 ms |

### Coding Agent Benchmarks (realistic prompts, 1000 tok max output)

| Configuration | c=1 tok/s | c=2 tok/s | c=4 tok/s | TPOT c=1 |
|---|---:|---:|---:|---:|
| Baseline (enforce-eager) | 22.0 | 42.8 | 82.7 | 50.2 ms |
| **FULL_DECODE_ONLY + prefix cache** | **55.1** | **89.8** | **319** | **11.0 ms** |
| TQ hybrid + graphs | 31.0 | — | — | 24.2 ms |

### Key Findings

- **FULL_DECODE_ONLY graph mode** is the biggest win: -72% TPOT, +16% throughput
- **Prefix caching** provides 85-99% TTFT reduction on cache hits
- **TurboQuant** adds 6-11% overhead on synthetic, 42-49% on coding; not recommended for Qwen3.5-9B (only 8/32 layers are full-attention)
- **MTP speculative decoding** incompatible with graph mode, 25-45% slower in eager; not recommended

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
| TurboQuant KV compression | **Works** (Triton kernels) | -6% to -49% throughput | Not recommended for Qwen3.5-9B |
| MTP speculative decoding | **Partial** | -25% throughput (eager only) | Incompatible with graph mode |
| INT4 (GPTQ/AWQ) | **Blocked** | N/A | GPTQ marlin CUDA-only, AWQ unsupported |
| FP8 quantization | **Emulated** | Software dequant path | MI100 lacks native FP8 hardware |

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

*Last updated: 2026-03-31 | Results from missions: MI100 Throughput Optimization, TurboQuant Backend*
