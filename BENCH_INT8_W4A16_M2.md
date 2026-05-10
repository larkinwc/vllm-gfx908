# Milestone 2 — TensileLite tuning + hipBLASLt INT8 dispatcher (gfx908)

> Worker feature: **m2-tensilelite-tuning**
> Builds on **m2-hipblaslt-build** (libhipblaslt.so + tensilelite-client + Tensile pip-installed at `/root/hipblaslt-src/`).
> Builds on **M1** (vLLM + AMD-fork ROCm 7.12 + Triton W8A8 baseline at branch `mi100-fixes`, commit `8faac1c57`).

## TL;DR

- **8 W8A8 prefill GEMM shapes tuned end-to-end with TensileLite**
  (gfx908-compatible MFMA grid; INT8 / INT32 acc / **INT32 dest**
  matching hipBLASLt's prebuilt I8I8_II8 contraction at runtime).
- **8 tuned logic YAMLs merged into a self-contained library**
  (`/root/bench-int8-w4a16/tensilelite/merged_library/`) via
  `scripts/mi100/merge_tensile_logic.sh`.
- **Runtime selection-proof confirmed via rocprofv3 kernel-trace**:
  the merged library swaps the runtime tile from
  `MT64x64x64_MI32x32x1` (prebuilt) to one of our tuned
  `MT128x64x32_MI32x32x1` / `MT128x32x64_MI16x16x1` tiles.
- **Dispatcher routes tuned shapes to `torch._int_mm` (hipBLASLt
  INT8) and falls back to Triton for everything else.**
  `VLLM_DISABLE_HIPBLASLT=1` forces 100% Triton.
- **17/17 pytests pass**: 8 dispatch tests + 9 numerical-correctness
  tests (`atol=1e-2`, `rtol=5e-2` vs fp32 reference) for every tuned
  (M, N, K).
- **Full 12-cell W8A8 grid measured three ways** (M1 May-7 / M1
  rebaselined on M2 build / M2 with merged library) by
  `m2-rebaseline-and-fill-grid`. Result: **the apparent M2 win vs
  May-7 M1 is entirely a `libhipblaslt.so` build/system-swap delta;
  TensileLite tuning by itself produces a *flat* result** — no cell
  improves ≥ 3% in the M2-vs-M1-rebaselined comparison. **The
  mission's tuning-gate FAILS.**
- **Wikitext-2 perplexity gate PASSES**: M2 ppl 9.6518 vs M1rb ppl
  9.6518 (Δ = 0.000%); vs May-7 M1 ppl 9.6561 (Δ = -0.045%).
- **Selection-proof + numerical correctness still hold**: the
  infrastructure (TensileLite tuning client, merged-library build,
  hipBLASLt dispatcher) is intact and is required for any future
  per-shape kernel substitution. The negative result is _on uplift
  for these shapes_, not on the build/dispatch chain.
- **Reproducibility canary differs across runs** — Tensile's "best"
  kernel selection is sensitive to measurement noise on this MI100
  host. With the new finding that tuning provides ~0% uplift, the
  canary failure is no longer blocking — there is no tuning win to
  lose. Tracked under the separate `m2-repro-pin` feature.

## Files added/modified (this resume)

```
scripts/mi100/
├── merge_tensile_logic.sh           # NEW — yaml staging + TensileCreateLibrary
├── verify_tuned_kernel_selected.sh  # NEW — rocprofv3-driven selection proof
├── run_grid.sh                      # NEW — M2 wrapper around M1 bench harness
├── run_triton_fallback_smoke.sh     # NEW — VAL-TENSILE-008 fallback evidence
└── gen_tensilelite_configs.py       # MODIFIED — DestDataType I (INT32) fix
```

(All other files from the prior commit `b69172e07` remain in place.)

## Configuration fix

The pre-resume tuning campaign emitted YAMLs with `DestDataType: I8`. At
runtime, `torch._int_mm` requests `cType=Int32, dType=Int32` (the
`I8I8_II8_Type_I8I_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908`
prebuilt contraction), and the prior tuned YAMLs **could not be matched
to that runtime descriptor** because of the dest-type mismatch
(`HIPBLAS_STATUS_NOT_SUPPORTED` was returned when we tried).

The resume:

1. Patched `scripts/mi100/gen_tensilelite_configs.py` to set
   `DestDataType: I` (INT32) and `TransposeB: False` (matching the
   prebuilt `Ailk_Bljk` operation identifier).
2. Re-ran `scripts/mi100/run_tensilelite.sh` for all 8 hot shapes
   (succeeded 8/8, ~5 min wall on this host).
3. Built the merged library via the new `merge_tensile_logic.sh`.
4. Verified runtime kernel selection via rocprofv3.

## Tuned tile shapes (8 / 8 succeeded after dest-type fix)

| Logic YAML | M | N | K | Tile (winner) | Reported eff |
| --- | --- | --- | --- | --- | --- |
| `tune_M512_N4096_K4096`  | 512 | 4096  | 4096  | `MT128x32x64_MI16x16x1` | 55.16 |
| `tune_M512_N10240_K4096` | 512 | 10240 | 4096  | `MT128x32x64_MI16x16x1` | 65.27 |
| `tune_M512_N24576_K4096` | 512 | 24576 | 4096  | `MT128x32x64_MI16x16x1` | 66.96 |
| `tune_M512_N4096_K12288` | 512 | 4096  | 12288 | `MT128x32x64_MI16x16x1` | 63.51 |
| `tune_M513_N4096_K4096`  | 513 | 4096  | 4096  | `MT128x64x32_MI32x32x1` | 47.12 |
| `tune_M513_N10240_K4096` | 513 | 10240 | 4096  | `MT128x32x64_MI16x16x1` | 47.25 |
| `tune_M513_N24576_K4096` | 513 | 24576 | 4096  | `MT128x64x32_MI32x32x1` | 49.79 |
| `tune_M513_N4096_K12288` | 513 | 4096  | 12288 | `MT128x64x32_MI32x32x1` | 53.07 |

These 8 shapes cover **9,720 / 17,760 = 54.7%** of W8A8 GEMM calls
(rest are decode-time small-M, bandwidth-bound on the existing Triton
kernel — they remain on the Triton path).

## Lazy-merge pipeline

```
1. Stage 8 tuned YAMLs into merge_workspace/merged_logic/
     scripts/mi100/merge_tensile_logic.sh (Step 1)

2. TensileCreateLibrary against the merged logic dir
     scripts/mi100/merge_tensile_logic.sh (Step 2)
     -> /root/bench-int8-w4a16/tensilelite/merged_library/
            library/
                TensileLibrary_lazy_gfx908.dat                 (NEW lazy index, our 8 tuned solutions)
                TensileLibrary_I8I8_II8_UA_Type_I8I_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.{dat,co}  (NEW, our merged tile choices)
                Kernels.so-000-gfx908.hsaco                    (NEW assembled MFMA kernels)
                TensileLiteLibrary_lazy_Mapping.dat            (NEW lazy mapping)
        Build wall time: ~9 s (only 2 unique tile shapes across 8 logic files).

3. Stage prebuilt fallback contractions alongside (Step 3)
     scripts/mi100/merge_tensile_logic.sh (manual cp loop after Step 2)
     -> non-I8 contractions and the prebuilt I8I8_II8_Type_I8I (default)
        copied into merged_library/library/, NOT overwriting our
        TensileLibrary_lazy_gfx908.dat.

Note: this is a Path-1 alternative to the canonical "TensileMergeLibrary"
flow described in the AMD docs. The canonical flow merges incremental
logic YAMLs into the upstream arcturus_*.yaml then re-runs
TensileCreateLibrary on the merged file. That fails on this version
because the upstream ROCm-7.2.0 arcturus repo only ships the
*Cijk_Ailk_Bljk_I8II_BH* (I8 -> FP16 dest) logic YAML, not the
*Cijk_Ailk_Bljk_I8II_BH_UserArgs* (I8 -> INT32 dest) variant we tune.
The merge tool checks ProblemType equality and fails with
"DestDataType: 6 != 8" / "TransposeB: false != true" because the only
upstream YAML has incompatible types.

The "stage tuned YAMLs as the logic input directly" approach we use here
produces exactly the same result for the contraction we care about
(I8 input, INT32 output, TransposeA=False, TransposeB=False) — every
tuned (M,N,K) gets its tile choice baked into the merged
TensileLibrary_lazy_gfx908.dat. The merge_tensile_logic.sh comment
documents this trade-off.
```

## Selection Proof (VAL-TENSILE-003 evidence)

```
$ scripts/mi100/verify_tuned_kernel_selected.sh
===== M2 selection-proof summary (2026-05-09T23:34:35Z) =====

PROBE 1 -- prebuilt-only library at /root/hipblaslt-src/build/release/hipblaslt-install/lib/hipblaslt/library
  rocprofv3 kernel name(s):
    Cijk_Ailk_Bljk_I8II_BH_MT64x64x64_MI32x32x1

PROBE 2 -- merged library at /root/bench-int8-w4a16/tensilelite/merged_library/library
  rocprofv3 kernel name(s):
    Cijk_Ailk_Bljk_I8II_BH_UserArgs_MT128x64x32_MI32x32x1
  matched tuned-tile pattern? YES

===== verdict =====
PASS: merged library selected a tuned MT*_MI* tile,
      and the prebuilt-only library selected a DIFFERENT tile.
```

`HIPBLASLT_LOG_LEVEL` up to 6 does **not** print the selected kernel
name on this ROCm 7.12 build, so the verify script captures the
selection via `rocprofv3 --kernel-trace` (already used by M1
profiling). The trace CSV contains the `KERNEL_DISPATCH` records with
the assembly kernel name embedded, and we grep for the
`MT*_MI*` tile fragment.

Output traces:
- `/root/bench-int8-w4a16/tensilelite/select_proof_prebuilt_kernel_trace.csv`
- `/root/bench-int8-w4a16/tensilelite/select_proof_merged_kernel_trace.csv`
- `/root/bench-int8-w4a16/tensilelite/select_proof_summary.txt`

## Bench grid (M2 vs M1, full W8A8 grid — 12 cells)

This section was re-generated by feature **m2-rebaseline-and-fill-grid**
on 2026-05-09/10. We now have **three** comparable columns for every
W8A8 cell (12 cells = TP∈{1,4} × c∈{1,2,4} × workload∈{synthetic,coding}):

| Column | Build | hipBLASLt library | TensileLite logic |
| --- | --- | --- | --- |
| **M1** (May-7)        | system `libhipblaslt.so` from ROCm 7.12 (`/opt/rocm/core-7.12/lib`) | system | system prebuilt |
| **M1-rebaselined**    | M2-build `libhipblaslt.so` (`/root/hipblaslt-src/build/release/library`) | M2 build | M2 build's prebuilt I8I8 default kernel |
| **M2** (M2 env + tuned) | M2-build `libhipblaslt.so` (same as M1rb) | M2 build | merged TensileLite library at `/root/bench-int8-w4a16/tensilelite/merged_library/library` (`HIPBLASLT_TENSILE_LIBPATH`) |

The point of the M1-rebaseline column is to **isolate the libhipblaslt
build/system-swap delta from the actual TensileLite tuning gain**.
M1 vs M1-rebaseline tells us "what changed when we swapped in the M2
build of `libhipblaslt.so`"; M1-rebaseline vs M2 tells us "what the
TensileLite tuning by itself bought us, controlling for the library
swap".

Files:
- M1 baseline: `/root/bench-int8-w4a16/baseline/{synthetic,coding}/w8a8_*.json` (May-7)
- M1-rebaselined: `/root/bench-int8-w4a16/m1-rebaseline/{synthetic,coding}/w8a8_*.json` (May-9/10)
- M2 (tuned): `/root/bench-int8-w4a16/m2/{synthetic,coding}/w8a8_*.json` (May-9/10)
- Aggregator: `scripts/mi100/aggregate_m2_threeway.py`

### Three-way comparison table (full 12-cell W8A8 grid)

Markup: **bold** = improvement ≥ 3% on this metric (Pareto-bar);
_italic_ = regression > 1%.

| Cell | Workload | Metric | M1 (May-7) | M1-rebaseline | M2 | Δ M1→M2 | Δ M1→M1rb (lib-swap) | Δ M1rb→M2 (**tuning**) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w8a8_tp1_c1 | synthetic | tput | 38.97 | 37.44 | 37.17 | _-4.61%_ | _-3.91%_ | -0.73% |
| w8a8_tp1_c1 | synthetic | mean_ttft | 314.33 | 283.48 | 283.76 | **-9.73%** | **-9.81%** | +0.10% |
| w8a8_tp1_c1 | synthetic | p99_ttft | 318.00 | 288.05 | 295.71 | **-7.01%** | **-9.42%** | _+2.66%_ |
| w8a8_tp1_c1 | synthetic | mean_tpot | 24.53 | 25.70 | 25.90 | _+5.57%_ | _+4.77%_ | +0.76% |
| w8a8_tp1_c1 | synthetic | p99_tpot | 24.55 | 25.77 | 25.91 | _+5.55%_ | _+4.98%_ | +0.54% |
| w8a8_tp1_c1 | coding    | tput | 38.06 | 36.56 | 36.31 | _-4.58%_ | _-3.92%_ | -0.68% |
| w8a8_tp1_c1 | coding    | mean_ttft | 231.00 | 219.86 | 218.16 | **-5.56%** | **-4.82%** | -0.78% |
| w8a8_tp1_c1 | coding    | p99_ttft | 1183.77 | 1065.09 | 1063.76 | **-10.14%** | **-10.03%** | -0.13% |
| w8a8_tp1_c1 | coding    | mean_tpot | 25.34 | 26.49 | 26.69 | _+5.35%_ | _+4.58%_ | +0.74% |
| w8a8_tp1_c1 | coding    | p99_tpot | 33.36 | 34.50 | 34.72 | _+4.07%_ | _+3.42%_ | +0.62% |
| w8a8_tp1_c2 | synthetic | tput | 73.81 | 70.78 | 70.67 | _-4.26%_ | _-4.11%_ | -0.16% |
| w8a8_tp1_c2 | synthetic | mean_ttft | 488.31 | 439.42 | 439.20 | **-10.06%** | **-10.01%** | -0.05% |
| w8a8_tp1_c2 | synthetic | p99_ttft | 556.20 | 492.42 | 492.33 | **-11.48%** | **-11.47%** | -0.02% |
| w8a8_tp1_c2 | synthetic | mean_tpot | 25.29 | 26.64 | 26.69 | _+5.55%_ | _+5.37%_ | +0.18% |
| w8a8_tp1_c2 | synthetic | p99_tpot | 25.56 | 26.86 | 26.91 | _+5.25%_ | _+5.09%_ | +0.15% |
| w8a8_tp1_c2 | coding    | tput | 68.61 | 66.67 | 66.07 | _-3.71%_ | _-2.83%_ | -0.91% |
| w8a8_tp1_c2 | coding    | mean_ttft | 277.97 | 255.55 | 254.33 | **-8.51%** | **-8.07%** | -0.48% |
| w8a8_tp1_c2 | coding    | p99_ttft | 1341.32 | 1185.41 | 1214.56 | **-9.45%** | **-11.62%** | _+2.46%_ |
| w8a8_tp1_c2 | coding    | mean_tpot | 27.88 | 29.01 | 29.32 | _+5.15%_ | _+4.05%_ | _+1.06%_ |
| w8a8_tp1_c2 | coding    | p99_tpot | 37.91 | 35.77 | 37.54 | -0.96% | **-5.64%** | _+4.96%_ |
| w8a8_tp1_c4 | synthetic | tput | 135.90 | 131.90 | 132.01 | _-2.86%_ | _-2.94%_ | +0.08% |
| w8a8_tp1_c4 | synthetic | mean_ttft | 814.96 | 717.53 | 717.53 | **-11.96%** | **-11.96%** | +0.00% |
| w8a8_tp1_c4 | synthetic | p99_ttft | 956.65 | 819.69 | 820.69 | **-14.21%** | **-14.32%** | +0.12% |
| w8a8_tp1_c4 | synthetic | mean_tpot | 26.35 | 27.63 | 27.60 | _+4.75%_ | _+4.85%_ | -0.09% |
| w8a8_tp1_c4 | synthetic | p99_tpot | 27.41 | 28.51 | 28.48 | _+3.93%_ | _+4.03%_ | -0.10% |
| w8a8_tp1_c4 | coding    | tput | 120.36 | 116.55 | 117.12 | _-2.69%_ | _-3.16%_ | +0.49% |
| w8a8_tp1_c4 | coding    | mean_ttft | 322.69 | 293.71 | 293.36 | **-9.09%** | **-8.98%** | -0.12% |
| w8a8_tp1_c4 | coding    | p99_ttft | 1436.67 | 1393.34 | 1436.77 | +0.01% | **-3.02%** | _+3.12%_ |
| w8a8_tp1_c4 | coding    | mean_tpot | 31.76 | 32.94 | 33.01 | _+3.95%_ | _+3.73%_ | +0.22% |
| w8a8_tp1_c4 | coding    | p99_tpot | 42.66 | 38.61 | 41.43 | -2.87% | **-9.48%** | _+7.30%_ |
| w8a8_tp4_c1 | synthetic | tput | 63.95 | 57.04 | 57.18 | _-10.58%_ | _-10.80%_ | +0.25% |
| w8a8_tp4_c1 | synthetic | mean_ttft | 165.10 | 165.79 | 164.19 | -0.55% | +0.41% | -0.96% |
| w8a8_tp4_c1 | synthetic | p99_ttft | 166.57 | 167.80 | 167.07 | +0.30% | +0.74% | -0.43% |
| w8a8_tp4_c1 | synthetic | mean_tpot | 15.05 | 16.95 | 16.91 | _+12.37%_ | _+12.61%_ | -0.22% |
| w8a8_tp4_c1 | synthetic | p99_tpot | 15.08 | 16.96 | 16.92 | _+12.23%_ | _+12.53%_ | -0.26% |
| w8a8_tp4_c1 | coding    | tput | 60.98 | 54.69 | 54.82 | _-10.11%_ | _-10.32%_ | +0.24% |
| w8a8_tp4_c1 | coding    | mean_ttft | 132.13 | 131.41 | 130.46 | -1.27% | -0.54% | -0.73% |
| w8a8_tp4_c1 | coding    | p99_ttft | 494.09 | 461.20 | 460.94 | **-6.71%** | **-6.66%** | -0.05% |
| w8a8_tp4_c1 | coding    | mean_tpot | 15.86 | 17.76 | 17.72 | _+11.72%_ | _+11.97%_ | -0.22% |
| w8a8_tp4_c1 | coding    | p99_tpot | 24.00 | 25.89 | 25.85 | _+7.73%_ | _+7.89%_ | -0.15% |
| w8a8_tp4_c2 | synthetic | tput | 125.30 | 113.56 | 113.08 | _-9.75%_ | _-9.37%_ | -0.43% |
| w8a8_tp4_c2 | synthetic | mean_ttft | 137.05 | 135.98 | 135.99 | -0.77% | -0.78% | +0.01% |
| w8a8_tp4_c2 | synthetic | p99_ttft | 165.91 | 162.57 | 163.75 | -1.30% | -2.02% | +0.73% |
| w8a8_tp4_c2 | synthetic | mean_tpot | 15.49 | 17.15 | 17.22 | _+11.21%_ | _+10.72%_ | +0.44% |
| w8a8_tp4_c2 | synthetic | p99_tpot | 15.59 | 17.23 | 17.32 | _+11.10%_ | _+10.57%_ | +0.48% |
| w8a8_tp4_c2 | coding    | tput | 110.08 | 100.13 | 101.14 | _-8.13%_ | _-9.04%_ | +1.01% |
| w8a8_tp4_c2 | coding    | mean_ttft | 116.15 | 118.08 | 119.55 | _+2.93%_ | _+1.66%_ | _+1.25%_ |
| w8a8_tp4_c2 | coding    | p99_ttft | 156.93 | 153.96 | 159.80 | _+1.83%_ | -1.89% | _+3.80%_ |
| w8a8_tp4_c2 | coding    | mean_tpot | 17.70 | 19.38 | 19.38 | _+9.52%_ | _+9.51%_ | +0.01% |
| w8a8_tp4_c2 | coding    | p99_tpot | 24.27 | 26.20 | 26.26 | _+8.20%_ | _+7.92%_ | +0.25% |
| w8a8_tp4_c4 | synthetic | tput | 247.74 | 224.58 | 223.82 | _-9.66%_ | _-9.35%_ | -0.34% |
| w8a8_tp4_c4 | synthetic | mean_ttft | 199.47 | 194.82 | 194.65 | -2.41% | -2.33% | -0.09% |
| w8a8_tp4_c4 | synthetic | p99_ttft | 249.35 | 230.98 | 228.98 | **-8.17%** | **-7.37%** | -0.86% |
| w8a8_tp4_c4 | synthetic | mean_tpot | 15.43 | 17.11 | 17.18 | _+11.35%_ | _+10.95%_ | +0.36% |
| w8a8_tp4_c4 | synthetic | p99_tpot | 15.76 | 17.43 | 17.49 | _+10.94%_ | _+10.56%_ | +0.34% |
| w8a8_tp4_c4 | coding    | tput | 196.29 | 181.94 | 181.29 | _-7.64%_ | _-7.31%_ | -0.36% |
| w8a8_tp4_c4 | coding    | mean_ttft | 124.39 | 121.66 | 124.22 | -0.13% | -2.19% | _+2.10%_ |
| w8a8_tp4_c4 | coding    | p99_ttft | 208.17 | 193.70 | 195.01 | **-6.32%** | **-6.95%** | +0.68% |
| w8a8_tp4_c4 | coding    | mean_tpot | 19.70 | 21.49 | 21.42 | _+8.74%_ | _+9.09%_ | -0.32% |
| w8a8_tp4_c4 | coding    | p99_tpot | 24.87 | 26.65 | 26.55 | _+6.75%_ | _+7.15%_ | -0.37% |

(Aggregator: `scripts/mi100/aggregate_m2_threeway.py`; raw cells under
`/root/bench-int8-w4a16/{baseline,m1-rebaseline,m2}/`. The `request_throughput`
metric is omitted from this table for compactness — it tracks `tput` 1:1.)

### Headline reading

1. **Δ M1→M2 (the headline column)** still shows large TTFT
   improvements on TP=1 cells (-9% to -14% mean and p99 TTFT) and
   regressions on TPOT/throughput (+4% to +12%). This was the
   "improvement" we previously claimed for M2.
2. **Δ M1→M1-rebaseline (the lib-swap column)** is _virtually
   identical_ to Δ M1→M2 across every metric. **The change in `libhipblaslt.so`
   alone (system → M2 build) accounts for nearly all of the apparent
   M2 win.** TP=4 in particular shows a uniform -7% to -10% throughput
   regression and +7% to +12% TPOT regression that has nothing to do
   with TensileLite tuning.
3. **Δ M1-rebaseline→M2 (the tuning column)** is **flat**. Every cell
   sits inside the ±1% noise envelope on every metric except for a
   handful of high-variance p99 cells (`tp1_c1_synth p99_ttft +2.66%`,
   `tp1_c2_coding p99_tpot +4.96%`, `tp1_c4_coding p99_tpot +7.30%`)
   which all swing in the *negative* direction in absolute terms but
   are still inside the noise band of repeated p99-of-200-prompts
   measurements. **No cell shows a ≥ 3% improvement** from
   TensileLite tuning over the M2-build prebuilt I8I8 default kernel.

### Pareto outcome (vs VAL-TENSILE-006)

The bench gate is `≥ +3% throughput OR p99 latency improvement per
cell, no cell regressing > 1% on any metric without an exception
entry`. The TPOT/throughput regressions vs the M1 May-7 baseline
exceed the 1% bar **but the lib-swap-isolation row makes clear that
this is environmental, not caused by tuning**.

A Triton-fallback smoke at `w8a8_tp1_c1_synthetic` with
`VLLM_DISABLE_HIPBLASLT=1` (100% of W8A8 calls go through Triton)
showed numbers virtually identical to the M2 run, and the M1-rebaseline
column above now confirms this across the **entire 12-cell W8A8 grid**:
the M2-build `libhipblaslt.so` itself produces the regression, not
our merged TensileLite logic.

The full per-cell exception inventory is documented in
`/root/bench-int8-w4a16/tensilelite/pareto_exceptions.md` (updated
2026-05-09/10 with the three-way comparison).

### Mission-gate outcome (m2-rebaseline-and-fill-grid) — **GATE FAIL**

The follow-up feature `m2-rebaseline-and-fill-grid` raises a stricter
gate than VAL-TENSILE-006: **at least one prefill-dominated cell must
improve ≥ 3% in the M2-vs-M1-rebaselined column**, isolating the
TensileLite tuning win from the libhipblaslt build/system swap.

**Result: GATE FAIL.** No prefill-dominated cell improves ≥ 3% on any
metric (throughput, TTFT, or TPOT) in the M2-vs-M1-rebaselined
comparison. The largest "tuning-only" deltas across all 60 cell ×
metric combinations are:

| Cell | Workload | Metric | Δ M1rb→M2 | Verdict |
| --- | --- | --- | ---: | --- |
| w8a8_tp4_c2 | coding    | p99_ttft  | +3.80% (worse) | regression |
| w8a8_tp1_c4 | coding    | p99_tpot  | +7.30% (worse) | regression |
| w8a8_tp1_c2 | coding    | p99_tpot  | +4.96% (worse) | regression |
| w8a8_tp4_c4 | coding    | mean_ttft | +2.10% (worse) | regression |
| w8a8_tp1_c1 | synthetic | p99_ttft  | +2.66% (worse) | regression |

All other (cell × metric) deltas are inside the ±1% noise envelope.
**The TensileLite tuning provides no measurable speedup over the
M2-build prebuilt I8I8 default kernel** on Qwen3.5-9B W8A8 at the
batch sizes vLLM actually issues. The selection-proof
(`scripts/mi100/verify_tuned_kernel_selected.sh`) confirms the
runtime *did* pick our tuned `MT128x64x32_MI32x32x1` /
`MT128x32x64_MI16x16x1` tiles when the merged library was loaded —
they simply do not run faster than the prebuilt
`MT64x64x64_MI32x32x1` kernel for these shapes.

This is consistent with the M1 strategic-implication note: the
hot W8A8 kernel is **memory-bandwidth-bound** (~21% of HBM peak),
not compute-bound, so different MFMA tile shapes do not move the
needle. Tile tuning helps when the kernel is compute-bound; here
the bottleneck is HBM weight bandwidth and the cure (per the M1
report) is fused dequant+GEMM in M3 / packed-INT4 register unpack
for W4A16 in M3.

**Recommendation to orchestrator:** treat M2 as a documented
negative-result-on-tuning-uplift but accept it as a pass on the
infrastructure VAL-TENSILE-001..-005, -008, -009 (numerical
correctness, dispatcher routing, runtime selection-proof, Triton
fallback intact). Proceed to **M3 (Triton custom kernels with fused
dequant+GEMM)** — the highest-ROI path on memory-bound hot kernels.

### Wikitext-2 quality gate (VAL-TENSILE-007)

The W8A8 + merged-library path produces **bit-identical perplexity**
to the M1-rebaseline baseline across the full Wikitext-2 50×512 chunk
suite (seed 0):

| Run | Wikitext-2 ppl (50 × 512, seed 0) | Δ vs M1 (May-7) | Δ vs M1-rebaseline |
| --- | ---: | ---: | ---: |
| M1 (May-7, system libhipblaslt)   | 9.6561 | —        | —        |
| M1-rebaseline (M2 libhipblaslt, no `HIPBLASLT_TENSILE_LIBPATH`) | 9.6518 | -0.045% | — |
| M2 (M2 libhipblaslt + merged TensileLite logic) | 9.6518 | -0.045% | **0.000%** |

`(ppl_M2 - ppl_M1rb) / ppl_M1rb = 0.000%` — the runtime kernel
selection differs (selection-proof above), but the INT8 GEMM result is
numerically equivalent in fp16-output precision, so perplexity is
unaffected. **Δ ≤ +1% gate PASSES.** Evidence:

- M1 baseline ppl: `/root/bench-int8-w4a16/baseline/ppl_w8a8.json`
- M1-rebaseline ppl: `/root/bench-int8-w4a16/m1-rebaseline/ppl_w8a8_m1rb.json`
- M2 ppl: `/root/bench-int8-w4a16/m2/ppl_w8a8_m2.json`

## Triton fallback (VAL-TENSILE-008 evidence)

```
$ scripts/mi100/run_triton_fallback_smoke.sh
[fallback] VLLM_DISABLE_HIPBLASLT=1
[fallback] LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
[fallback] CELLS_FILTER=w8a8_tp1_c1_synthetic
...
wrote cell w8a8_tp1_c1_synthetic: tput=37.37 tok/s ttft_p50=274.7 ms tpot_p50=25.75 ms
```

Result file at
`/root/bench-int8-w4a16/tensilelite/triton_fallback_smoke.json`.
Within the same run-to-run noise envelope as the M2 number, confirming
the Triton fallback is intact.

## Numerical correctness (VAL-TENSILE-005)

```
$ pytest tests/kernels/quantization/test_hipblaslt_int8_correctness.py -v
  test_hipblaslt_matches_reference[512-4096-12288]      PASSED
  test_hipblaslt_matches_reference[512-24576-4096]      PASSED
  test_hipblaslt_matches_reference[513-4096-12288]      PASSED
  test_hipblaslt_matches_reference[513-24576-4096]      PASSED
  test_hipblaslt_matches_reference[512-4096-4096]       PASSED
  test_hipblaslt_matches_reference[512-10240-4096]      PASSED
  test_hipblaslt_matches_reference[513-4096-4096]       PASSED
  test_hipblaslt_matches_reference[513-10240-4096]      PASSED
  test_bias_is_applied                                  PASSED
9 passed in 4.75s
```

Reference is fp32 GEMM (operands cast int8→fp32, scales applied in
fp32, then cast). On ROCm `torch.matmul(int32, int32)` is unimplemented;
for K ≤ 12288 with operands in [-32, 31] fp32 retains full
INT32-equivalent precision (max product ~12.5M < 2^24).

## Dispatch routing (VAL-TENSILE-004)

```
$ pytest tests/kernels/quantization/test_mi100_w8a8_dispatch.py -v
test_supports_returns_true_for_listed_shape           PASSED
test_supports_returns_false_for_unknown_shape         PASSED
test_disable_env_forces_triton_path                   PASSED
test_dispatcher_prefers_hipblaslt_for_tuned_shape     PASSED
test_dispatcher_falls_back_to_triton_for_untuned      PASSED
test_disable_env_forces_triton_even_for_tuned_shape   PASSED
test_disable_env_output_matches_default_triton        PASSED
test_committed_manifest_is_loadable                   PASSED
8 passed in 5.84s
```

The off-list shape produces bit-identical output with and without
`VLLM_DISABLE_HIPBLASLT=1`, proving the fallback path is unchanged
from M1.

## Reproducibility canary (VAL-TENSILE-009)

The canary still fails: Tensile picks a different "best" kernel for
`M=512,N=4096,K=4096` on the second run. Both kernels are valid I8
MFMA solutions; the runtime measurement noise is large enough relative
to their measured gflops to swap winners. This is tracked under the
separate `m2-repro-pin` feature; it is **not blocking** for this
resume because the *committed* logic YAMLs are reproducible (the
*selection* is not).

Mitigation paths (deferred to `m2-repro-pin`):
- Pin GPU clocks via `rocm-smi --setperflevel high` + manual SCLK lock.
- Larger Tensile `BenchmarkRepeats` count to reduce variance.
- Monte-Carlo agreement instead of single-winner — pick the kernel
  that wins ≥ N/M trials.

## How to enable hipBLASLt at serve time

```bash
# Use our build's libhipblaslt.so + merged TensileLite library
export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
export HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library

# The dispatcher is enabled by default; to force Triton-only:
export VLLM_DISABLE_HIPBLASLT=1
```

## Deferred follow-ups (out of M2 scope)

- [x] Run remaining {TP=1 c=2}, {TP=4} cells of the W8A8 grid — done in
      `m2-rebaseline-and-fill-grid` (this report). Conclusion holds:
      lib-swap accounts for the TTFT improvement; tuning is flat.
- [x] Wikitext-2 perplexity Δ ≤ +1% vs M1 — done in
      `m2-rebaseline-and-fill-grid`. Δ vs M1 = -0.045%; Δ vs M1rb =
      0.000%. PASS.
- [x] M1 rebaseline grid for like-for-like M3 comparison — done at
      `/root/bench-int8-w4a16/m1-rebaseline/{synthetic,coding}/w8a8_*.json`.
      M3 should compare against this rebaselined column, not the
      May-7 M1 numbers.
- [ ] `m2-repro-pin` feature: pin GPU clocks, repeat-count tune
      tuning client; rerun `verify_tensile_repro.sh` until canary
      passes. (Lower priority now that the tuning gain is shown to
      be ~0% — the canary failing is no longer blocking M2 because
      there is no tuning win to lose.)
- [ ] M3 (custom Triton W8A8 with fused dequant+GEMM) — the highest
      ROI path per M1 strategic-implication note (memory-bandwidth-
      bound hot kernel). Now elevated from "next milestone" to
      "primary remaining throughput path", since M2 TensileLite
      tuning by itself does not Pareto-improve the like-for-like
      baseline.
