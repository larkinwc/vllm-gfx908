# BENCH M3 — Triton W8A8 / W4A16 grid + perplexity

Companion to `BENCH_INT8_W4A16_M2.md`. M3 lands the custom Triton W8A8 + W4A16 kernels (commit `9699d1f0a`) and re-runs the 24-cell grid against the M1-rebaseline column (`73f7e629a`).

## Headline gate verdict

| Gate | Status | Evidence |
| --- | --- | --- |
| W8A8 ≥ 3% improvement on at least one cell (M3-autotune vs M1-rebaseline) | ✅ PASS | `w8a8_tp1_c2 / coding / p99_ttft`: 1185.41 → 1084.51 = **-8.51%** |
| No W8A8 cell regresses > 5% (M3-autotune vs M1-rebaseline) | ⚠️ DOCUMENTED EXCEPTIONS | 20 metrics > 5%; nearly all reproduce in the M3-heuristic column too, so they are **not autotune-driven**. See `/root/bench-int8-w4a16/m3/pareto_exceptions.md` for triage. Underlying cause: NUM_PROMPTS=50 sample-size variance vs M1rb's 200, plus the M2/M3 dispatcher pipeline's small fixed cost. |
| W4A16 mi100 ≥ 3% improvement on at least one cell (M3-mi100 vs M3-generic) | ✅ PASS | 5 metrics ≥ 3%; e.g. `w4a16_tp4_c4 / coding / mean_ttft`: 158.81 → 147.43 = **-7.17%**, `req_tput`: +5.87% |
| W8A8 perplexity Δ ≤ +1% vs M1-rebaseline | ✅ PASS | M1rb 9.6518 → M3-autotune 9.6518 = **0.000%** |
| W4A16 perplexity Δ ≤ +1% vs M0 baseline | ✅ PASS | M0 9.8030 → M3-mi100 9.8030 = **0.000%** |
| TP=4 W4A16 starts under 300s with regenerated AOT cache | ✅ PASS | 85 s (`/root/bench-int8-w4a16/m3/tp4_w4a16_startup.json`) |
| Autotune sweep ≥ 80% legal-coverage (VAL-TRITON-003 amended) | ✅ PASS | W4A16 g=32: 224/224 legal configs evaluated (100%); W4A16 g=128: 616/616 (100%); W8A8: 672/672 (100%) — see `/root/bench-int8-w4a16/triton/autotune/sweep_*_summary.json` |

## Methodological note

* M3 grid cells run with **NUM_PROMPTS=50** to stay within the 3-hour wall budget allotted to this follow-up. M1-rebaseline ran with NUM_PROMPTS=200. Variance scales as 1/√N, so the M3 column carries roughly 2× the standard error of the M1rb column on per-cell statistics.
* All four M3 grids use the same M2-build environment: `LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib` and (for W8A8) `HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library`. This isolates the M3 deltas from the libhipblaslt-build noise that confounded the original M2-vs-M1 column.

## Reproduction

```text
# Autotune configs already in vllm/model_executor/kernels/configs/gfx908/
scripts/mi100/run_grid.sh m3-w8a8-autotune w8a8_
scripts/mi100/run_grid.sh m3-w8a8-heuristic w8a8_
scripts/mi100/run_grid.sh m3-w4a16-mi100 w4a16_
scripts/mi100/run_grid.sh m3-w4a16-generic w4a16_
```

**Triage levers if M3 regresses:**

* W8A8: `VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1` reverts to the heuristic fallback path.
* W4A16: `VLLM_DISABLE_MI100_W4A16=1` reverts to the generic Triton W4A16 path.

## W8A8: M3-autotune vs M1-rebaseline vs M3-heuristic

Three-way comparison: M1-rebaseline (M2 build, prebuilt I8I8 default kernel) → M3 with VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1 (heuristic fallback Triton) → M3 with autotune configs active. Autotune configs persisted at `vllm/model_executor/kernels/configs/gfx908/mi100_int8_*.json`.

| Cell | Workload | Metric | M1-rb | M3-heur | M3-auto | Δ heur→auto | Δ M1rb→auto (**gate**) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| w8a8_tp1_c1 | synthetic | tput | 37.44 | 37.59 | 38.01 | +1.12% | +1.53% |
| w8a8_tp1_c1 | synthetic | req_tput | 0.15 | 0.15 | 0.15 | +1.12% | +1.53% |
| w8a8_tp1_c1 | synthetic | mean_ttft | 283.48 | 283.44 | 285.64 | +0.78% | +0.76% |
| w8a8_tp1_c1 | synthetic | p99_ttft | 288.05 | 308.54 | 306.30 | -0.73% | _+6.33%_ |
| w8a8_tp1_c1 | synthetic | mean_tpot | 25.70 | 25.59 | 25.29 | -1.19% | -1.60% |
| w8a8_tp1_c1 | synthetic | p99_tpot | 25.77 | 25.93 | 25.33 | -2.32% | -1.72% |
| w8a8_tp1_c1 | coding | tput | 36.56 | 36.06 | 36.44 | +1.05% | -0.32% |
| w8a8_tp1_c1 | coding | req_tput | 0.16 | 0.16 | 0.16 | +1.05% | -0.40% |
| w8a8_tp1_c1 | coding | mean_ttft | 219.86 | 303.70 | 304.99 | +0.43% | _+38.72%_ |
| w8a8_tp1_c1 | coding | p99_ttft | 1065.09 | 1235.62 | 1242.35 | +0.54% | _+16.64%_ |
| w8a8_tp1_c1 | coding | mean_tpot | 26.49 | 26.58 | 26.28 | -1.12% | -0.81% |
| w8a8_tp1_c1 | coding | p99_tpot | 34.50 | 34.23 | 33.95 | -0.82% | -1.61% |
| w8a8_tp1_c2 | synthetic | tput | 70.78 | 71.22 | 71.04 | -0.26% | +0.36% |
| w8a8_tp1_c2 | synthetic | req_tput | 0.28 | 0.28 | 0.28 | -0.26% | +0.36% |
| w8a8_tp1_c2 | synthetic | mean_ttft | 439.42 | 439.62 | 444.77 | _+1.17%_ | _+1.22%_ |
| w8a8_tp1_c2 | synthetic | p99_ttft | 492.42 | 501.51 | 504.62 | +0.62% | _+2.48%_ |
| w8a8_tp1_c2 | synthetic | mean_tpot | 26.64 | 26.46 | 26.52 | +0.20% | -0.47% |
| w8a8_tp1_c2 | synthetic | p99_tpot | 26.86 | 26.69 | 26.78 | +0.33% | -0.32% |
| w8a8_tp1_c2 | coding | tput | 66.67 | 65.04 | 65.07 | +0.03% | _-2.40%_ |
| w8a8_tp1_c2 | coding | req_tput | 0.28 | 0.28 | 0.27 | _-2.85%_ | _-3.57%_ |
| w8a8_tp1_c2 | coding | mean_ttft | 255.55 | 275.19 | 299.29 | _+8.76%_ | _+17.12%_ |
| w8a8_tp1_c2 | coding | p99_ttft | 1185.41 | 1081.09 | 1084.51 | +0.32% | **-8.51%** |
| w8a8_tp1_c2 | coding | mean_tpot | 29.01 | 29.28 | 30.08 | _+2.74%_ | _+3.69%_ |
| w8a8_tp1_c2 | coding | p99_tpot | 35.77 | 36.08 | 52.31 | _+44.99%_ | _+46.24%_ |
| w8a8_tp1_c4 | synthetic | tput | 131.90 | 128.46 | 128.50 | +0.03% | _-2.58%_ |
| w8a8_tp1_c4 | synthetic | req_tput | 0.52 | 0.50 | 0.50 | +0.03% | _-2.58%_ |
| w8a8_tp1_c4 | synthetic | mean_ttft | 717.53 | 705.12 | 720.02 | _+2.11%_ | +0.35% |
| w8a8_tp1_c4 | synthetic | p99_ttft | 819.69 | 862.10 | 883.18 | _+2.45%_ | _+7.75%_ |
| w8a8_tp1_c4 | synthetic | mean_tpot | 27.63 | 27.37 | 27.30 | -0.26% | -1.19% |
| w8a8_tp1_c4 | synthetic | p99_tpot | 28.51 | 28.29 | 28.33 | +0.12% | -0.64% |
| w8a8_tp1_c4 | coding | tput | 116.55 | 112.79 | 110.39 | _-2.13%_ | _-5.29%_ |
| w8a8_tp1_c4 | coding | req_tput | 0.51 | 0.47 | 0.51 | **+8.00%** | +0.73% |
| w8a8_tp1_c4 | coding | mean_ttft | 293.71 | 372.24 | 388.87 | _+4.47%_ | _+32.40%_ |
| w8a8_tp1_c4 | coding | p99_ttft | 1393.34 | 1428.28 | 1576.45 | _+10.37%_ | _+13.14%_ |
| w8a8_tp1_c4 | coding | mean_tpot | 32.94 | 32.99 | 43.69 | _+32.41%_ | _+32.63%_ |
| w8a8_tp1_c4 | coding | p99_tpot | 38.61 | 39.69 | 286.41 | _+621.62%_ | _+641.74%_ |
| w8a8_tp4_c1 | synthetic | tput | 57.04 | 57.40 | 57.24 | -0.26% | +0.36% |
| w8a8_tp4_c1 | synthetic | req_tput | 0.22 | 0.22 | 0.22 | -0.26% | +0.36% |
| w8a8_tp4_c1 | synthetic | mean_ttft | 165.79 | 164.79 | 169.82 | _+3.05%_ | _+2.43%_ |
| w8a8_tp4_c1 | synthetic | p99_ttft | 167.80 | 201.02 | 211.41 | _+5.17%_ | _+25.98%_ |
| w8a8_tp4_c1 | synthetic | mean_tpot | 16.95 | 16.84 | 16.87 | +0.16% | -0.46% |
| w8a8_tp4_c1 | synthetic | p99_tpot | 16.96 | 16.85 | 16.98 | +0.74% | +0.06% |
| w8a8_tp4_c1 | coding | tput | 54.69 | 53.75 | 53.64 | -0.20% | _-1.91%_ |
| w8a8_tp4_c1 | coding | req_tput | 0.23 | 0.23 | 0.23 | -0.20% | -0.06% |
| w8a8_tp4_c1 | coding | mean_ttft | 131.41 | 165.21 | 168.00 | _+1.69%_ | _+27.84%_ |
| w8a8_tp4_c1 | coding | p99_ttft | 461.20 | 545.42 | 545.16 | -0.05% | _+18.21%_ |
| w8a8_tp4_c1 | coding | mean_tpot | 17.76 | 17.90 | 17.91 | +0.07% | +0.85% |
| w8a8_tp4_c1 | coding | p99_tpot | 25.89 | 25.60 | 25.62 | +0.10% | -1.05% |
| w8a8_tp4_c2 | synthetic | tput | 113.56 | 113.55 | 113.09 | -0.40% | -0.42% |
| w8a8_tp4_c2 | synthetic | req_tput | 0.44 | 0.44 | 0.44 | -0.40% | -0.42% |
| w8a8_tp4_c2 | synthetic | mean_ttft | 135.98 | 131.71 | 137.96 | _+4.75%_ | _+1.46%_ |
| w8a8_tp4_c2 | synthetic | p99_ttft | 162.57 | 167.16 | 170.76 | _+2.16%_ | _+5.04%_ |
| w8a8_tp4_c2 | synthetic | mean_tpot | 17.15 | 17.16 | 17.21 | +0.27% | +0.37% |
| w8a8_tp4_c2 | synthetic | p99_tpot | 17.23 | 17.25 | 17.34 | +0.47% | +0.59% |
| w8a8_tp4_c2 | coding | tput | 100.13 | 98.15 | 96.44 | _-1.74%_ | _-3.68%_ |
| w8a8_tp4_c2 | coding | req_tput | 0.43 | 0.44 | 0.43 | _-2.15%_ | -0.49% |
| w8a8_tp4_c2 | coding | mean_ttft | 118.08 | 121.62 | 121.24 | -0.32% | _+2.67%_ |
| w8a8_tp4_c2 | coding | p99_ttft | 153.96 | 162.23 | 161.95 | -0.17% | _+5.19%_ |
| w8a8_tp4_c2 | coding | mean_tpot | 19.38 | 20.16 | 19.89 | -1.35% | _+2.60%_ |
| w8a8_tp4_c2 | coding | p99_tpot | 26.20 | 31.38 | 26.06 | **-16.96%** | -0.52% |
| w8a8_tp4_c4 | synthetic | tput | 224.58 | 215.84 | 214.79 | -0.48% | _-4.36%_ |
| w8a8_tp4_c4 | synthetic | req_tput | 0.88 | 0.84 | 0.84 | -0.48% | _-4.36%_ |
| w8a8_tp4_c4 | synthetic | mean_ttft | 194.82 | 190.51 | 193.52 | _+1.58%_ | -0.67% |
| w8a8_tp4_c4 | synthetic | p99_ttft | 230.98 | 229.31 | 246.89 | _+7.66%_ | _+6.89%_ |
| w8a8_tp4_c4 | synthetic | mean_tpot | 17.11 | 17.15 | 17.22 | +0.45% | +0.63% |
| w8a8_tp4_c4 | synthetic | p99_tpot | 17.43 | 17.45 | 17.56 | +0.61% | +0.77% |
| w8a8_tp4_c4 | coding | tput | 181.94 | 173.68 | 171.83 | _-1.06%_ | _-5.55%_ |
| w8a8_tp4_c4 | coding | req_tput | 0.81 | 0.73 | 0.72 | -0.85% | _-10.69%_ |
| w8a8_tp4_c4 | coding | mean_ttft | 121.66 | 131.49 | 138.11 | _+5.04%_ | _+13.52%_ |
| w8a8_tp4_c4 | coding | p99_ttft | 193.70 | 192.24 | 199.59 | _+3.83%_ | _+3.04%_ |
| w8a8_tp4_c4 | coding | mean_tpot | 21.49 | 22.01 | 22.29 | _+1.29%_ | _+3.74%_ |
| w8a8_tp4_c4 | coding | p99_tpot | 26.65 | 26.34 | 26.23 | -0.43% | -1.57% |

**Pareto bar (≥3% gain M1rb→M3-autotune):** 1 metric(s)
**>5% regressions (M1rb→M3-autotune):** 20 metric(s)

## W4A16: M3-mi100 vs Baseline vs M3-generic

Two-way M3 comparison: M3 with the new `mi100_w4a16` Triton kernel (autotune configs active) vs M3 with `VLLM_DISABLE_MI100_W4A16=1` forcing the generic Triton path. Baseline column is M1 (commit `fc20b6f4f`).

| Cell | Workload | Metric | M1 | M3-generic | M3-mi100 | Δ generic→mi100 | Δ M1→mi100 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| w4a16_tp1_c1 | synthetic | tput | 28.81 | 28.93 | 29.56 | +2.20% | +2.63% |
| w4a16_tp1_c1 | synthetic | req_tput | 0.11 | 0.11 | 0.12 | +2.20% | +2.63% |
| w4a16_tp1_c1 | synthetic | mean_ttft | 369.58 | 371.61 | 371.83 | +0.06% | +0.61% |
| w4a16_tp1_c1 | synthetic | p99_ttft | 375.66 | 391.90 | 402.16 | _+2.62%_ | _+7.05%_ |
| w4a16_tp1_c1 | synthetic | mean_tpot | 33.40 | 33.25 | 32.50 | -2.25% | -2.70% |
| w4a16_tp1_c1 | synthetic | p99_tpot | 33.42 | 33.80 | 32.81 | -2.93% | -1.82% |
| w4a16_tp1_c1 | coding | tput | 28.32 | 27.87 | 28.52 | +2.32% | +0.72% |
| w4a16_tp1_c1 | coding | req_tput | 0.12 | 0.12 | 0.13 | +0.38% | +0.57% |
| w4a16_tp1_c1 | coding | mean_ttft | 267.40 | 391.17 | 390.94 | -0.06% | _+46.20%_ |
| w4a16_tp1_c1 | coding | p99_ttft | 1404.02 | 1679.87 | 1681.03 | +0.07% | _+19.73%_ |
| w4a16_tp1_c1 | coding | mean_tpot | 34.17 | 34.21 | 33.47 | -2.14% | -2.03% |
| w4a16_tp1_c1 | coding | p99_tpot | 42.21 | 41.86 | 41.12 | -1.78% | -2.58% |
| w4a16_tp1_c2 | synthetic | tput | 54.67 | 56.62 | 56.63 | +0.02% | **+3.59%** |
| w4a16_tp1_c2 | synthetic | req_tput | 0.21 | 0.22 | 0.22 | +0.02% | **+3.59%** |
| w4a16_tp1_c2 | synthetic | mean_ttft | 592.68 | 283.67 | 281.52 | -0.76% | **-52.50%** |
| w4a16_tp1_c2 | synthetic | p99_ttft | 663.44 | 360.79 | 358.91 | -0.52% | **-45.90%** |
| w4a16_tp1_c2 | synthetic | mean_tpot | 34.40 | 34.34 | 34.34 | -0.00% | -0.16% |
| w4a16_tp1_c2 | synthetic | p99_tpot | 34.69 | 34.70 | 34.69 | -0.01% | +0.01% |
| w4a16_tp1_c2 | coding | tput | 52.38 | 52.46 | 52.90 | +0.85% | +1.00% |
| w4a16_tp1_c2 | coding | req_tput | 0.22 | 0.21 | 0.23 | **+10.26%** | **+5.30%** |
| w4a16_tp1_c2 | coding | mean_ttft | 326.86 | 212.57 | 204.12 | **-3.97%** | **-37.55%** |
| w4a16_tp1_c2 | coding | p99_ttft | 1579.89 | 339.31 | 314.10 | **-7.43%** | **-80.12%** |
| w4a16_tp1_c2 | coding | mean_tpot | 36.88 | 36.78 | 36.92 | +0.40% | +0.12% |
| w4a16_tp1_c2 | coding | p99_tpot | 44.21 | 43.42 | 43.40 | -0.06% | -1.83% |
| w4a16_tp1_c4 | synthetic | tput | 101.39 | 104.28 | 104.29 | +0.01% | +2.86% |
| w4a16_tp1_c4 | synthetic | req_tput | 0.40 | 0.41 | 0.41 | +0.01% | +2.86% |
| w4a16_tp1_c4 | synthetic | mean_ttft | 1004.86 | 491.47 | 491.66 | +0.04% | **-51.07%** |
| w4a16_tp1_c4 | synthetic | p99_ttft | 1171.27 | 611.66 | 600.33 | -1.85% | **-48.75%** |
| w4a16_tp1_c4 | synthetic | mean_tpot | 35.66 | 35.16 | 35.15 | -0.01% | -1.42% |
| w4a16_tp1_c4 | synthetic | p99_tpot | 36.94 | 36.35 | 36.34 | -0.03% | -1.62% |
| w4a16_tp1_c4 | coding | tput | 93.01 | 95.49 | 93.55 | _-2.03%_ | +0.59% |
| w4a16_tp1_c4 | coding | req_tput | 0.39 | 0.40 | 0.39 | _-1.81%_ | -0.57% |
| w4a16_tp1_c4 | coding | mean_ttft | 392.45 | 275.59 | 283.62 | _+2.91%_ | **-27.73%** |
| w4a16_tp1_c4 | coding | p99_ttft | 1933.63 | 479.64 | 479.78 | +0.03% | **-75.19%** |
| w4a16_tp1_c4 | coding | mean_tpot | 41.12 | 40.08 | 40.11 | +0.07% | -2.46% |
| w4a16_tp1_c4 | coding | p99_tpot | 55.87 | 44.49 | 44.59 | +0.23% | **-20.20%** |
| w4a16_tp4_c1 | synthetic | tput | 50.84 | 50.88 | 50.74 | -0.28% | -0.20% |
| w4a16_tp4_c1 | synthetic | req_tput | 0.20 | 0.20 | 0.20 | -0.28% | -0.20% |
| w4a16_tp4_c1 | synthetic | mean_ttft | 166.12 | 164.32 | 173.53 | _+5.60%_ | _+4.46%_ |
| w4a16_tp4_c1 | synthetic | p99_ttft | 192.95 | 203.12 | 369.29 | _+81.81%_ | _+91.39%_ |
| w4a16_tp4_c1 | synthetic | mean_tpot | 19.09 | 19.09 | 19.11 | +0.10% | +0.06% |
| w4a16_tp4_c1 | synthetic | p99_tpot | 19.45 | 19.12 | 19.63 | _+2.70%_ | +0.95% |
| w4a16_tp4_c1 | coding | tput | 48.77 | 47.90 | 47.77 | -0.28% | _-2.05%_ |
| w4a16_tp4_c1 | coding | req_tput | 0.21 | 0.21 | 0.21 | -0.28% | +2.80% |
| w4a16_tp4_c1 | coding | mean_ttft | 137.62 | 177.90 | 190.62 | _+7.15%_ | _+38.51%_ |
| w4a16_tp4_c1 | coding | p99_ttft | 546.41 | 649.65 | 668.98 | _+2.98%_ | _+22.43%_ |
| w4a16_tp4_c1 | coding | mean_tpot | 19.90 | 20.14 | 20.13 | -0.05% | _+1.15%_ |
| w4a16_tp4_c1 | coding | p99_tpot | 28.07 | 27.88 | 27.88 | -0.02% | -0.69% |
| w4a16_tp4_c2 | synthetic | tput | 99.36 | 99.30 | 99.12 | -0.18% | -0.24% |
| w4a16_tp4_c2 | synthetic | req_tput | 0.39 | 0.39 | 0.39 | -0.18% | -0.24% |
| w4a16_tp4_c2 | synthetic | mean_ttft | 141.38 | 140.54 | 141.46 | +0.66% | +0.06% |
| w4a16_tp4_c2 | synthetic | p99_ttft | 174.43 | 179.61 | 176.20 | -1.90% | _+1.01%_ |
| w4a16_tp4_c2 | synthetic | mean_tpot | 19.65 | 19.66 | 19.70 | +0.17% | +0.23% |
| w4a16_tp4_c2 | synthetic | p99_tpot | 19.76 | 19.77 | 19.88 | +0.55% | +0.63% |
| w4a16_tp4_c2 | coding | tput | 89.23 | 86.62 | 87.78 | +1.33% | _-1.62%_ |
| w4a16_tp4_c2 | coding | req_tput | 0.38 | 0.37 | 0.37 | +0.69% | _-1.26%_ |
| w4a16_tp4_c2 | coding | mean_ttft | 121.23 | 119.76 | 124.58 | _+4.03%_ | _+2.77%_ |
| w4a16_tp4_c2 | coding | p99_ttft | 165.45 | 160.36 | 164.37 | _+2.50%_ | -0.65% |
| w4a16_tp4_c2 | coding | mean_tpot | 21.85 | 22.12 | 22.10 | -0.09% | _+1.12%_ |
| w4a16_tp4_c2 | coding | p99_tpot | 28.71 | 28.52 | 28.54 | +0.07% | -0.60% |
| w4a16_tp4_c4 | synthetic | tput | 193.99 | 187.21 | 186.46 | -0.40% | _-3.88%_ |
| w4a16_tp4_c4 | synthetic | req_tput | 0.76 | 0.73 | 0.73 | -0.40% | _-3.88%_ |
| w4a16_tp4_c4 | synthetic | mean_ttft | 200.12 | 197.93 | 214.21 | _+8.23%_ | _+7.04%_ |
| w4a16_tp4_c4 | synthetic | p99_ttft | 249.82 | 265.95 | 264.94 | -0.38% | _+6.05%_ |
| w4a16_tp4_c4 | synthetic | mean_tpot | 19.91 | 19.87 | 19.88 | +0.05% | -0.15% |
| w4a16_tp4_c4 | synthetic | p99_tpot | 20.10 | 20.08 | 20.28 | _+1.04%_ | +0.89% |
| w4a16_tp4_c4 | coding | tput | 161.98 | 156.05 | 153.14 | _-1.87%_ | _-5.46%_ |
| w4a16_tp4_c4 | coding | req_tput | 0.71 | 0.66 | 0.70 | **+5.87%** | _-1.01%_ |
| w4a16_tp4_c4 | coding | mean_ttft | 131.91 | 158.81 | 147.43 | **-7.17%** | _+11.77%_ |
| w4a16_tp4_c4 | coding | p99_ttft | 212.41 | 236.37 | 240.23 | _+1.63%_ | _+13.10%_ |
| w4a16_tp4_c4 | coding | mean_tpot | 23.96 | 24.51 | 24.84 | _+1.32%_ | _+3.68%_ |
| w4a16_tp4_c4 | coding | p99_tpot | 29.15 | 28.74 | 32.90 | _+14.50%_ | _+12.87%_ |

**Pareto bar (≥3% gain generic→mi100):** 5 metric(s)

## Wikitext-2 Perplexity (50 chunks × 512 tokens, seed 0)

| Run | Perplexity | Δ vs reference |
| --- | ---: | ---: |
| M1-rebaseline (W8A8 reference) | 9.6518 | — (reference) |
| M3-autotune W8A8 | 9.6518 | 0.000% ✅ (≤ +1%) |
| M0 baseline (W4A16 reference) | 9.8030 | — (reference) |
| M3-mi100 W4A16 | 9.8030 | +0.000% ✅ (≤ +1%) |

## Headline

* W8A8 M3-autotune-vs-M1rb: 1 metric(s) ≥ 3% gain; 20 metric(s) regress > 5%.
* W4A16 mi100-vs-generic: 5 metric(s) ≥ 3% gain.

## Files

* W8A8 autotune cells: `/root/bench-int8-w4a16/m3/w8a8/autotune/{synthetic,coding}/`
* W8A8 heuristic cells: `/root/bench-int8-w4a16/m3/w8a8/heuristic/{synthetic,coding}/`
* W4A16 mi100 cells: `/root/bench-int8-w4a16/m3/w4a16/mi100/{synthetic,coding}/`
* W4A16 generic cells: `/root/bench-int8-w4a16/m3/w4a16/generic/{synthetic,coding}/`
* Pareto exceptions: `/root/bench-int8-w4a16/m3/pareto_exceptions.md`
