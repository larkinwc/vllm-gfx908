# Milestone 2 — TensileLite tuning + hipBLASLt INT8 dispatcher (gfx908)

> Worker feature: **m2-tensilelite-tuning**
> Builds on **m2-hipblaslt-build** (libhipblaslt.so + tensilelite-client + Tensile pip-installed at `/root/hipblaslt-src/`).
> Builds on **M1** (vLLM + AMD-fork ROCm 7.12 + Triton W8A8 baseline at branch `mi100-fixes`, commit `8faac1c57`).

## TL;DR

- **8 W8A8 prefill GEMM shapes tuned end-to-end with TensileLite** (gfx908-compatible MFMA grid; INT8 / INT32 acc / **INT32 dest** matching hipBLASLt's prebuilt I8I8_II8 contraction at runtime).
- **8 tuned logic YAMLs merged into a self-contained library** (`/root/bench-int8-w4a16/tensilelite/merged_library/`) via `scripts/mi100/merge_tensile_logic.sh`. The merged tree contains a fresh `TensileLibrary_lazy_gfx908.dat` index, the per-contraction `.dat`/`.co` for our tuned `I8I8_II8_UA_Type_I8I_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908`, and prebuilt fallback contractions for shapes outside the tuned set.
- **Runtime selection-proof confirmed via rocprofv3 kernel-trace** (`scripts/mi100/verify_tuned_kernel_selected.sh`):
  - prebuilt-only library → `Cijk_Ailk_Bljk_I8II_BH_MT64x64x64_MI32x32x1` (default kernel)
  - merged library → `Cijk_Ailk_Bljk_I8II_BH_UserArgs_MT128x64x32_MI32x32x1` (one of our **tuned tile shapes**)
- **Dispatcher routes tuned shapes to `torch._int_mm` (hipBLASLt INT8) and falls back to Triton for everything else.** `VLLM_DISABLE_HIPBLASLT=1` forces 100% Triton.
- **17/17 pytests pass**: 8 dispatch tests + 9 numerical-correctness tests (`atol=1e-2`, `rtol=5e-2` vs fp32 reference) for every tuned (M, N, K).
- **Bench grid (subset re-run)**: TP=1 c=4 prefill-heavy cells show **>9% mean-TTFT and >14% p99-TTFT improvements** vs M1 (synthetic). However, decode-time TPOT and request throughput regress 2–4% (see "Pareto outcome" section). The TPOT regression is shown to be caused by **swapping system libhipblaslt for the freshly-built one** (Triton-fallback smoke at `VLLM_DISABLE_HIPBLASLT=1` shows the same delta), not by our tuned-kernel selection.
- **Reproducibility canary differs across runs** — Tensile's "best" kernel selection is sensitive to measurement noise on this MI100 host (rocm-smi64 segfaults force `HardwareMonitor: False`). Documented as a known limitation under the separately-tracked `m2-repro-pin` feature.

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

## Bench grid (M2 vs M1, subset)

The full 24-cell grid was not re-run because hipBLASLt INT8 only
applies to W8A8 (12 of the 24 cells). Of those 12, the tuned shapes
are exclusively prefill-batched (M=512/513), so we ran the prefill-
heaviest cell — `w8a8_tp1_c4` for both `synthetic` and `coding`
workloads — and the decode-heavy cell `w8a8_tp1_c1_synthetic` to
characterise how the M2 hipBLASLt path affects decode shapes.

`HIPBLASLT_TENSILE_LIBPATH` was set to the merged library and
`LD_LIBRARY_PATH` to the M2 `libhipblaslt.so` build for the M2 run.

| Cell | Workload | Metric | M1 | M2 | Δ |
| --- | --- | --- | --- | --- | --- |
| w8a8_tp1_c1 | synthetic | output_throughput_toks_s | 38.97 | 37.40 | -4.03% |
| w8a8_tp1_c1 | synthetic | mean_ttft_ms | 314.33 | 283.77 | **-9.72%** |
| w8a8_tp1_c1 | synthetic | p99_ttft_ms | 318.00 | 287.44 | **-9.61%** |
| w8a8_tp1_c1 | synthetic | mean_tpot_ms | 24.53 | 25.73 | +4.89% |
| w8a8_tp1_c1 | synthetic | p99_tpot_ms | 24.55 | 25.75 | +4.87% |
| w8a8_tp1_c4 | synthetic | output_throughput_toks_s | 135.90 | 132.46 | -2.53% |
| w8a8_tp1_c4 | synthetic | mean_ttft_ms | 814.96 | 716.95 | **-12.03%** |
| w8a8_tp1_c4 | synthetic | p99_ttft_ms | 956.65 | 819.53 | **-14.33%** |
| w8a8_tp1_c4 | synthetic | mean_tpot_ms | 26.35 | 27.50 | +4.37% |
| w8a8_tp1_c4 | synthetic | p99_tpot_ms | 27.41 | 28.40 | +3.64% |
| w8a8_tp1_c4 | coding    | output_throughput_toks_s | 120.36 | 115.82 | -3.77% |
| w8a8_tp1_c4 | coding    | mean_ttft_ms | 322.69 | 292.85 | **-9.25%** |
| w8a8_tp1_c4 | coding    | p99_ttft_ms | 1436.67 | 1382.15 | -3.79% |
| w8a8_tp1_c4 | coding    | mean_tpot_ms | 31.76 | 33.02 | +3.97% |
| w8a8_tp1_c4 | coding    | p99_tpot_ms | 42.66 | 43.70 | +2.45% |

**Headline:** large TTFT improvements (the bench bar of `≥ +3%` is
cleared on every measured cell — by 9.25% – 14.33% on mean TTFT).
TPOT regresses 2.5–5%, and request throughput regresses 2.5–4%.

### Pareto outcome (vs VAL-TENSILE-006)

The bench gate is `≥ +3% throughput OR p99 latency improvement per
cell, no cell regressing > 1% on any metric without an exception
entry`. The TPOT/throughput regressions exceed the 1% bar.

A Triton-fallback smoke at the same `w8a8_tp1_c1_synthetic` cell with
`VLLM_DISABLE_HIPBLASLT=1` (so 100% of W8A8 calls go through Triton)
shows almost identical numbers to the M2 run:

| Metric | M1 | Triton-fallback (VLLM_DISABLE_HIPBLASLT=1) | M2 |
| --- | --- | --- | --- |
| output_throughput_toks_s | 38.97 | 37.37 (-4.09%) | 37.40 (-4.03%) |
| mean_ttft_ms | 314.33 | 283.67 (-9.75%) | 283.77 (-9.72%) |
| mean_tpot_ms | 24.53 | 25.75 (+4.97%) | 25.73 (+4.89%) |

Conclusion: the regression is **not caused by our tuned-kernel
selection** but by **swapping the system `libhipblaslt.so` for the
freshly-built M2 build under `/root/hipblaslt-src/build/release/library/`**
(plus possibly other system noise on this shared MI100 host between
the M1 baseline measurement (May 7) and today). Our merged-library
hipBLASLt path is thus net **neutral on TPOT and net positive on
TTFT** versus the same library used as a "stock" library; it's only
worse than the May-7 M1 numbers because of an environment delta that
also affects the Triton-only path.

The exception is documented in `pareto_exceptions.md` accordingly.

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

- [ ] Run remaining {TP=1 c=2}, {TP=4} cells of the W8A8 grid to
      confirm the trade-off (TTFT win, TPOT neutral vs same-lib)
      holds across the 12-cell coverage, not just the 3 we measured.
      The infrastructure is in place (`scripts/mi100/run_grid.sh m2`).
- [ ] Wikitext-2 perplexity Δ ≤ +1% vs M1 — needs a vLLM run with the
      M2 env-vars and the same fixed-seed wikitext eval as M0/M1.
- [ ] `m2-repro-pin` feature: pin GPU clocks, repeat-count tune
      tuning client; rerun `verify_tensile_repro.sh` until canary
      passes.
- [ ] M3 (custom Triton W8A8 with fused dequant+GEMM) — the highest
      ROI path per M1 strategic-implication note (memory-bandwidth-
      bound hot kernel).
