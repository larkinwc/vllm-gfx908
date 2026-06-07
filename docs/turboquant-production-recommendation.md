# TurboQuant Production Recommendation for Qwen3.5-9B on MI100

**Date:** 2026-03-31  
**Hardware:** 4x AMD MI100 (gfx908), 32GB VRAM each  
**Model:** Qwen3.5-9B FP16  
**vLLM Version:** v0.18.1.dev4  
**TurboQuant Version:** v0.2.0

## Executive Summary

**RECOMMENDATION: DO NOT enable TurboQuant in production for Qwen3.5-9B on MI100.**

TurboQuant shows significant performance regression on this configuration. The throughput loss (5-11% synthetic, 42-49% coding workloads) far outweighs the VRAM savings (10.8%). The TPOT doubling indicates that TQ state management overhead exceeds any decode optimization benefits.

## Recommendation

### Primary Production Configuration

Use the optimized baseline without TurboQuant:

```bash
/root/launch-vllm-optimized.sh
```

**Configuration:**

- FULL_DECODE_ONLY HIP graph mode
- Prefix caching enabled
- Tensor parallelism: 4
- Max model length: 32768
- GPU memory utilization: 0.93
- Language model only: true

### Why Not TurboQuant

| Metric | Baseline Optimized | TQ Hybrid | Change |
|--------|-------------------|-----------|--------|
| **Synthetic c=1 throughput** | 248.10 tok/s | 233.42 tok/s | **-5.9%** |
| **Synthetic c=2 throughput** | 478.37 tok/s | 447.55 tok/s | **-6.4%** |
| **Synthetic c=4 throughput** | 884.01 tok/s | 785.14 tok/s | **-11.2%** |
| **Coding c=1 aggregate throughput** | 53.28 tok/s | 30.98 tok/s | **-41.9%** |
| **Coding c=2 aggregate throughput** | 144.03 tok/s | 73.16 tok/s | **-49.2%** |
| **Coding c=4 aggregate throughput** | 256.92 tok/s | 146.40 tok/s | **-43.0%** |
| **Mean TPOT (coding)** | 11.1 ms | 24.3 ms | **+119%** |
| **VRAM usage** | 93.3% | 82.5% | **-10.8%** |

### Key Findings

1. **Throughput Regression**: TQ hybrid mode causes 5-11% throughput regression in synthetic benchmarks and 42-49% regression in coding agent workloads.

2. **TPOT Doubling**: Time per output token nearly doubles from ~11ms to ~24ms, directly impacting user experience.

3. **Limited Model Coverage**: Qwen3.5-9B has only 8 full-attention layers (25% of 32 total layers). The remaining 24 layers use linear attention (GDN) which TurboQuant cannot compress.

4. **VRAM Savings Insufficient**: The 10.8% VRAM savings (~3.5GB per GPU) does not justify the significant throughput loss.

## Root Cause Analysis

### Why TurboQuant Underperforms on Qwen3.5-9B/MI100

1. **TQ Overhead in Capture Path**: Even in capture_only mode, TQ adds latency for KV capture and quantization, causing TPOT increase without benefiting from compressed decode.

2. **Ring Buffer Management**: Per-token overhead for ring buffer maintenance adds latency to every decode step.

3. **Hybrid Decode Path Slower**: The PyTorch matmul path used in hybrid decode is slower than the optimized Triton kernels used in baseline.

4. **Limited Compressible Layers**: With only 8/32 layers being full-attention, the maximum potential benefit is capped at 25% of layers. The overhead affects all layers.

5. **Graph Compatibility Trade-offs**: While TQ is compatible with FULL_DECODE_ONLY graphs, the hybrid decode path uses PyTorch operations that may not be as optimized as native Triton kernels.

## Conditions Where TurboQuant Could Be Beneficial

TurboQuant may provide benefits under these conditions:

1. **Models with More Full-Attention Layers**: Models where >50% of layers use full-attention would see more compression benefit.

2. **Extreme VRAM Constraints**: Workloads where VRAM is the limiting factor and throughput can be sacrificed.

3. **Longer Context Lengths (>16k tokens)**: Compression ratio increases with context length, potentially offering more savings.

4. **Future TQ Optimizations**: Triton kernel optimizations for the hybrid decode path could reduce overhead.

5. **Different Hardware**: GPUs with higher memory bandwidth relative to compute may benefit more from compressed KV fetches.

## Validation Contract Evidence

### VAL-BENCH-001: Synthetic Benchmark TQ Capture Only vs Baseline

Status: **PASS**

- Benchmarks completed at c=1,2,4
- TQ capture_only shows 3.5-11% throughput regression
- Files: `tq_capture_only_graph_synthetic_c{1,2,4}_20260330.json`

### VAL-BENCH-002: Synthetic Benchmark TQ Hybrid vs Baseline

Status: **PASS**

- Benchmarks completed at c=1,2,4
- TQ hybrid shows 6-11% throughput regression
- Files: `tq_hybrid_graph_synthetic_c{1,2,4}_20260330.json`

### VAL-BENCH-003: Coding Agent Benchmark TQ Hybrid

Status: **PASS**

- Benchmarks completed at c=1,2,4
- TQ hybrid shows 42-49% aggregate throughput regression
- Files: `tq_hybrid_graph_coding_c1.json`, `tq_hybrid_coding_c{2,4}_20260330.json`

### VAL-BENCH-004: VRAM Usage Comparison

Status: **PASS**

- Baseline: 93.3% VRAM
- TQ Hybrid: 82.5% VRAM
- Savings: 10.8%
- Files: `baseline_optimized_vram_20260330.json`, `tq_hybrid_graph_vram_20260331_083315.json`

### VAL-BENCH-005: Comprehensive Comparison Report

Status: **PASS**

- Report file: `tq-comparison-report.json`
- Contains throughput, TPOT, TTFT, VRAM for all configs
- Percentage changes documented

### VAL-CROSS-002: TQ + Prefix Caching Interaction

Status: **PASS**

- Prefix caching still provides TTFT reduction with TQ active
- Existing test: `prefix_caching_test.json` shows 99.1% TTFT reduction
- Both systems work together correctly

### VAL-CROSS-003: Production Recommendation

Status: **PASS**

- This document provides data-backed recommendation
- Recommendation: Do NOT enable TQ for Qwen3.5-9B on MI100
- Production config documented: `/root/launch-vllm-optimized.sh`

## Conclusion

Based on comprehensive benchmarking, TurboQuant should **NOT** be enabled in production for Qwen3.5-9B on MI100. The performance regression significantly outweighs the modest VRAM savings. Continue using the optimized baseline configuration (FULL_DECODE_ONLY graphs + prefix caching) for maximum throughput and lowest latency.

---

**Report generated by:** benchmark-worker  
**Session ID:** 85f0ca05-a3b8-46be-a9dc-be74ba90f32d  
**Data sources:** `/root/benchmark-results/tq-comparison-report.json`, `/root/benchmark-results/baseline_optimized_*.json`, `/root/benchmark-results/tq_hybrid_*.json`
