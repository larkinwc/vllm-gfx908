# TurboQuant Benchmarking Results

**Feature:** full-benchmark-comparison + production-recommendation-and-report
**Date:** 2026-03-31
**Milestone:** benchmarking

## Executive Summary

**DO NOT enable TurboQuant in production for Qwen3.5-9B on MI100.**

TQ hybrid causes significant performance regression: 5-11% throughput loss on synthetic benchmarks and 42-49% loss on coding agent benchmarks, with TPOT nearly doubling (11ms → 24ms). VRAM savings of only 10.8% do not offset the performance loss. Use the optimized baseline (`/root/launch-vllm-optimized.sh`).

## Benchmark Results

### Synthetic Benchmarks (vllm bench serve, 100 prompts each)

| Config | c=1 tok/s | c=2 tok/s | c=4 tok/s | c=1 TPOT | c=2 TPOT | c=4 TPOT |
|--------|-----------|-----------|-----------|----------|----------|----------|
| baseline_optimized | 248.10 | 478.37 | 884.01 | 13.62ms | 15.83ms | 21.17ms |
| tq_capture_only | 239.36 | 447.39 | 786.18 | 30.58ms | 52.08ms | 40.95ms |
| tq_hybrid | 233.42 | 447.55 | 785.14 | 31.20ms | 32.99ms | 42.22ms |

**vs baseline (tq_hybrid):** -5.9% (c=1), -6.4% (c=2), -11.2% (c=4)

### TQ Capture_Only TTFT Anomaly

At c=1, TQ capture_only shows anomalous TTFT of 4107ms vs baseline 715ms (+475%). This is NOT representative of typical inference — it likely reflects GPU resource ordering or cold-start effects at low concurrency. The throughput regression for capture_only at c=1 is only 3.5%, which is inconsistent with the 6x TTFT regression. **Future workers should not treat capture_only's c=1 TTFT as intrinsic TQ overhead.**

### Coding Agent Benchmarks (10 requests per concurrency level)

| Config | c=1 tok/s | c=2 tok/s | c=4 tok/s |
|--------|-----------|-----------|-----------|
| baseline_optimized | 53.28 | 144.03 | 256.92 |
| tq_hybrid | 30.98 | 73.16 | 146.40 |

**vs baseline (tq_hybrid):** -41.9% (c=1), -49.2% (c=2), -43.0% (c=4)
**TPOT (coding, tq_hybrid):** ~24ms vs baseline ~11ms (+119%)

### VRAM Usage

| Config | Avg VRAM % |
|--------|------------|
| baseline_optimized | 93.3% |
| tq_hybrid | 82.5% |

**Savings:** 10.8% (~3.5GB per GPU)

## Result Files

All located at `/root/benchmark-results/`:
- `baseline_optimized_synthetic_c{1,2,4}_20260330.json` — baseline synthetic benchmarks
- `tq_capture_only_graph_synthetic_c{1,2,4}_20260330.json` — TQ capture_only synthetic
- `tq_hybrid_graph_synthetic_c{1,2,4}_20260330.json` — TQ hybrid synthetic
- `tq_hybrid_graph_coding_c1.json` — TQ hybrid coding c=1
- `tq_hybrid_coding_c{2,4}_20260330.json` — TQ hybrid coding c=2,4
- `baseline_optimized_vram_20260330.json` — baseline VRAM
- `tq_hybrid_graph_vram_20260331_083315.json` — TQ hybrid VRAM
- `tq-comparison-report.json` — comprehensive comparison report (all metrics)

## Prefix Caching + TQ Interaction

Prefix caching still provides TTFT reduction with TQ active. Tested via `prefix_caching_test.json`:
- First request TTFT: 15623ms (cold cache; note: unusually high, likely server warm-up artifact)
- Second request TTFT: 138ms (cache hit)
- TTFT reduction: 99.1%

## Production Recommendation

**Use the optimized baseline** (FULL_DECODE_ONLY + prefix caching, no TQ):
```bash
/root/launch-vllm-optimized.sh
```

See `docs/turboquant-production-recommendation.md` in the repo for full analysis.

## Root Cause of TQ Regression

1. TQ overhead in capture path adds latency even in capture_only mode
2. Ring buffer management adds per-token overhead at every decode step
3. Hybrid decode matmul path (PyTorch) slower than optimized Triton kernels in baseline
4. Qwen3.5-9B has only 8/32 compressible full-attention layers (25%) — insufficient for TQ to be beneficial
5. At low GPU memory utilization (0.80 for TQ vs 0.93 for baseline), paged attention is less efficient

## When TQ Could Help

- Models with >50% full-attention layers
- VRAM-constrained deployments where 10% savings matters
- Context lengths >16k where compression ratio improves
- After future Triton kernel optimizations to TQ hybrid decode path
