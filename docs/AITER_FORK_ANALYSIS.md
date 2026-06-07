# AITER Fork Analysis for MI100 (gfx908)

Evaluation of forking [ROCm/aiter](https://github.com/ROCm/aiter) to provide fused operator kernels for MI100 (gfx908) GPUs.

**Date:** 2026-03-31
**Context:** vLLM MI100 fork, Qwen3.5-9B FP16, TP=4, 4x MI100 32GB

---

## What is AITER?

AITER (AI Tensor Engine for ROCm) is AMD's centralized high-performance AI operator library. It provides fused kernels for attention, GEMM, MoE, RMSNorm, RoPE, quantization, and communication primitives. vLLM has deep integration with AITER via the `VLLM_ROCM_USE_AITER=1` environment variable and the `rocm_aiter_ops` abstraction layer.

**Repository:** <https://github.com/ROCm/aiter> (MIT license, 397 stars, 229 contributors, ~1590 commits)

## Current Status

AITER is **not available on MI100 (gfx908)**. The package is developed and tested exclusively on gfx942 (MI300X) and gfx950 (MI350). Even gfx90a (MI250) has build failures due to FP8 kernel dependencies ([issue #179](https://github.com/ROCm/aiter/issues/179)).

When `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` is set in our fork, vLLM crashes with:

```text
ModuleNotFoundError: No module named 'aiter'
```

## AITER Ops Relevant to Our Workload

Qwen3.5-9B is a **dense FP16 model** (not MoE, not FP8). With TP=4, each decode step executes 32 layers, each with ~12 kernel launches. Only a subset of AITER's fused ops are relevant:

| AITER Fused Op | Replaces | Kernel Type | gfx908 Portability | Relevance |
|---|---|---|---|---|
| Fused RoPE + KV cache write | 2 separate Triton kernels | **Triton** | Easy | **High** |
| Fused all-reduce + RMSNorm | quickreduce + RMSNorm kernel | C++/HIP/ASM | Hard (gfx942 assembly) | **High** |
| Triton unified attention | vLLM's Triton attention | **Triton** | Easy | Low (already MI100-tuned) |
| Fused RMSNorm + quantization | Separate RMSNorm + quant | C++/HIP | N/A | None (FP16, not FP8) |
| Fused MoE | Separate gate + experts | C++/CK | N/A | None (dense model) |
| FP8 GEMM (a8w8) | N/A | C++/CK | N/A | None (FP16) |
| Triton RoPE | vLLM's RoPE | **Triton** | Easy | Low (small kernel) |
| FP8 batched GEMM | N/A | C++/CK | N/A | None (FP16) |

## Performance Impact Estimate

### Per-layer analysis (32 layers, ~10.3ms TPOT at batch=1)

Each layer currently executes ~12 kernel launches. MI100 kernel launch overhead is ~5-10us per launch. Fusing operations eliminates launches and intermediate memory traffic:

| Optimization | Launches Saved | Memory Passes Saved | Per-Layer Saving | Total (32 layers) |
|---|---|---|---|---|
| Fused RoPE + KV cache | 1 | 1 | ~10us | ~0.3ms |
| Fused all-reduce + RMSNorm | 1-2 | 1 | ~15-25us | ~0.5-0.8ms |
| **Total** | | | | **~0.8-1.1ms** |

### Projected results

| Metric | Current | With AITER fused ops | Change |
|---|---|---|---|
| TPOT (c=1) | 10.3 ms | ~9.2-9.8 ms | **-5% to -12%** |
| Throughput (c=1) | 250 tok/s | ~255-263 tok/s | +2-5% |
| Throughput (c=4) | 338 tok/s | ~355-370 tok/s | +5-10% |

At higher concurrency, GEMMs become compute-bound and kernel launch overhead becomes a smaller fraction, so gains diminish to 3-8%.

## Fork Strategy Options

### Option A: Full AITER fork (NOT RECOMMENDED)

Fork `ROCm/aiter`, add gfx908 as a build target, disable FP8/CK-dependent ops.

**Effort:** 2-4 weeks (build system changes, gfx908 ISA for assembly kernels, testing)
**Maintenance:** High -- AITER has multiple commits per day; keeping fork in sync is a constant burden
**Risk:** High -- assembly kernels in `hsa/` directory are gfx942-specific and would need full rewrites for gfx908 CDNA1 memory ordering

### Option B: Triton-only AITER fork (MODERATE)

Fork AITER, build with `ENABLE_CK=0`, include only the `aiter/ops/triton/` directory (pure Triton kernels).

**Portable Triton ops:** unified attention, RoPE, reshape-and-cache
**Effort:** 3-5 days
**Gain:** ~3-5% TPOT reduction (RoPE+cache fusion, Triton attention)
**Maintenance:** Lower -- Triton ops change less frequently than CK/ASM ops
**Risk:** The Triton unified attention in AITER may not be faster than vLLM's already MI100-tuned version

### Option C: Cherry-pick individual Triton kernels (RECOMMENDED for near-term)

Don't fork AITER at all. Instead, write standalone fused Triton kernels directly in this vLLM fork:

1. **Fused RoPE + KV cache write** -- combine the existing `triton_reshape_and_cache_flash` and RoPE into a single kernel. Save 1 launch + 1 memory pass per layer.
2. **Fused RMSNorm + residual add** -- already partially done in vLLM's compilation passes. Could be enhanced with MI100-specific tuning.

**Effort:** 1-3 days per kernel
**Gain:** ~3-5% TPOT reduction
**Maintenance:** Zero (self-contained in our fork)
**Risk:** Low

### Option D: Full AITER fork targeting MoE models (FUTURE)

If/when the workload shifts to MoE models (e.g., Qwen3.5-27B-MoE, Mixtral, DeepSeek), AITER's fused MoE kernels become the killer feature. At that point, a fork with CK-based MoE ops would be justified.

**Trigger:** Switching primary model to MoE architecture
**Gain:** 15-30% on MoE inference (fused gate+expert+reduce)

## Decision Matrix

| Factor | Full Fork (A) | Triton-only Fork (B) | Cherry-pick (C) |
|---|---|---|---|
| Expected gain | 5-12% | 3-5% | 3-5% |
| Effort | 2-4 weeks | 3-5 days | 1-3 days |
| Maintenance burden | High | Medium | None |
| Risk | High | Low | Low |
| MoE model support | Yes | No | No |
| **Recommendation** | No | Maybe later | **Yes** |

## Conclusion

For the current workload (dense FP16 Qwen3.5-9B), forking AITER provides a maximum **5-12% TPOT improvement** at significant engineering and maintenance cost. The highest-value fused op (all-reduce + RMSNorm) requires gfx942-specific assembly rewrites that account for most of the effort.

**Recommended path:** Option C (cherry-pick). Write 1-2 standalone fused Triton kernels directly in this fork. Revisit full AITER fork if workload shifts to MoE models.

---

*See also: [BENCH.md](../BENCH.md) for current optimization results, [MI100_SETUP.md](../MI100_SETUP.md) for hardware setup.*
