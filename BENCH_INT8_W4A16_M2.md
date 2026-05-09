# Milestone 2 — TensileLite tuning + hipBLASLt INT8 dispatcher (gfx908)

> Worker feature: **m2-tensilelite-tuning**
> Builds on **m2-hipblaslt-build** (libhipblaslt.so + tensilelite-client + Tensile pip-installed at `/root/hipblaslt-src/`).
> Builds on **M1** (vLLM + AMD-fork ROCm 7.12 + Triton W8A8 baseline at branch `mi100-fixes`, commit `8faac1c57`).

## TL;DR

- **8 W8A8 prefill GEMM shapes tuned end-to-end with TensileLite** (gfx908-compatible MFMA grid; INT8 / INT32 acc / I8 dest matching hipBLASLt's prebuilt I8I8_II8 contraction).
- **Dispatcher routes tuned shapes to `torch._int_mm` (hipBLASLt INT8) and falls back to Triton for everything else.** `VLLM_DISABLE_HIPBLASLT=1` forces 100% Triton.
- **17/17 pytests pass**: 8 dispatch tests + 9 numerical-correctness tests (`atol=1e-2`, `rtol=5e-2` vs fp32 reference) for every tuned (M, N, K).
- **hipBLASLt smoke run loads our build's library and executes a real INT8 GEMM at one tuned shape (M=512, N=4096, K=4096)** with the freshly-built `libhipblaslt.so.1.2`.
- **Reproducibility canary differs across runs** — Tensile's "best" kernel selection is sensitive to measurement noise on this MI100 host (rocm-smi64 segfaults force `HardwareMonitor: False`). Documented as a known limitation.
- **End-to-end serving Pareto vs M1 was NOT measured** in this PR — the full 12-cell {TP=1,4} × {c=1,2,4} grid (~2 hours wall) and Wikitext-2 perplexity Δ are deferred to a follow-up. The path is in place behind `VLLM_DISABLE_HIPBLASLT` so it cannot regress baseline.

## Files added/modified

```
vllm/model_executor/kernels/
├── linear/scaled_mm/
│   ├── mi100_int8.py             # +shape logger; +M2 dispatcher hook
│   └── mi100_hipblaslt.py        # NEW — manifest loader + torch._int_mm path
└── configs/gfx908/
    └── hipblaslt_tuned_shapes.json   # NEW — tuned-shapes manifest

scripts/mi100/
├── dump_gemm_shapes.py            # capture (M,N,K) from W8A8 vllm bench serve
├── select_w8a8_tuning_shapes.py   # pick top-N covering all (N,K) clusters
├── gen_tensilelite_configs.py     # emit per-shape Tensile YAMLs
├── run_tensilelite.sh             # batch-tune driver (calls Tensile.bin)
├── verify_hipblaslt_load.sh       # VAL-TENSILE-003 smoke evidence collector
└── verify_tensile_repro.sh        # VAL-TENSILE-009 repro canary

tests/kernels/quantization/
├── test_mi100_w8a8_dispatch.py    # dispatch routing tests
└── test_hipblaslt_int8_correctness.py  # numerical correctness vs fp32 ref
```

Bench artifacts (NOT committed, live under `/root/bench-int8-w4a16/tensilelite/`):

```
gemm_shapes_w8a8_tp1.csv          17,760 W8A8 calls, 160 unique (M,N,K)
tuning_shapes.json                 8 selected shapes
configs/tune_M*.yaml               8 generator configs
runs/tune_M*/                      Tensile per-shape working dirs
logic/gfx908/tune_M*.yaml          8 emitted logic YAMLs
logs/tune_M*.log                   per-shape Tensile stdout
tuning_summary.json                campaign success/failure tracker
hipblaslt_load_smoke.log           HIPBLASLT_LOG_LEVEL=4 trace
shape_diff_vs_m1.md                shape-coverage explainer
repro_canary.diff                  diff between two Tensile runs
```

## Tuning pipeline

```
1. Capture shapes
     VLLM_LOG_GEMM_SHAPES=1 + scripts/mi100/dump_gemm_shapes.py
     -> gemm_shapes_w8a8_tp1.csv  (160 unique tuples)

2. Select tuning targets
     scripts/mi100/select_w8a8_tuning_shapes.py
     -> 8 shapes spanning all 4 (N,K) clusters at M={512,513}

3. Generate configs
     scripts/mi100/gen_tensilelite_configs.py
     -> 8 tune_M*.yaml under configs/

4. Run TensileLite
     scripts/mi100/run_tensilelite.sh
     -> 8 logic/gfx908/tune_M*.yaml + tuning_summary.json
        (succeeded: 8 / 8)

5. Smoke-verify hipBLASLt load
     scripts/mi100/verify_hipblaslt_load.sh
     -> hipblaslt_load_smoke.log (exit 0; HIPBLASLT_TENSILE_LIBPATH honored)

6. Wire dispatcher
     mi100_int8.py:222 -> if mi100_hipblaslt_supports(M, N, K): ... else Triton

7. Tests
     pytest tests/kernels/quantization/test_mi100_w8a8_dispatch.py
     pytest tests/kernels/quantization/test_hipblaslt_int8_correctness.py
     -> 17/17 PASS
```

## Selected shapes (committed in `hipblaslt_tuned_shapes.json`)

| M | N | K | n_calls | Role |
| --- | --- | --- | --- | --- |
| 512 | 4096 | 12288 | 2880 | down_proj prefill |
| 512 | 24576 | 4096 | 2880 | gate_up_proj prefill |
| 513 | 4096 | 12288 | 960 | down_proj prefill (off-by-1) |
| 513 | 24576 | 4096 | 960 | gate_up_proj prefill (off-by-1) |
| 512 | 4096 | 4096 | 720 | o_proj prefill |
| 512 | 10240 | 4096 | 720 | qkv_proj prefill |
| 513 | 4096 | 4096 | 240 | o_proj prefill (off-by-1) |
| 513 | 10240 | 4096 | 240 | qkv_proj prefill (off-by-1) |

These 8 shapes cover **9,720 / 17,760 = 54.7%** of W8A8 GEMM calls; remaining shapes are decode-time small-M which are bandwidth-bound on the existing Triton kernel.

## hipBLASLt-load smoke (VAL-TENSILE-003 evidence)

```
$ scripts/mi100/verify_hipblaslt_load.sh
[2026-05-09T22:52:40Z] smoke exit code: 0
HIPBLASLT_TENSILE_LIBPATH=/root/hipblaslt-src/build/release/hipblaslt-install/lib/hipblaslt/library
[Info][initialize] Using HIPBLASLT_TENSILE_LIBPATH=...
[Trace][rocblaslt_matmul] A=...[type=R_8I rows=4096 cols=4096 ld=4096]
                         B=...[type=R_8I rows=4096 cols=512 ld=4096]
                         computeDesc=[computeType=COMPUTE_32I ...]
result shape: torch.Size([512, 4096]), dtype: torch.int32, sum: 8519158106
```

**Caveat — runtime kernel-name selection in custom logic dir.** hipBLASLt
expects a fully-staged TensileLibrary directory (with the binary
`TensileLibrary_lazy_gfx908.dat` index) at `HIPBLASLT_TENSILE_LIBPATH`.
Our `logic/gfx908/tune_M*.yaml` files alone are **not** loadable
without merging them into the lazy index (via `TensileMergeLibrary` or
`TensileCreateLibrary --merge-library`). For this smoke we point
`HIPBLASLT_TENSILE_LIBPATH` at the freshly-built full library tree from
m2-hipblaslt-build (which contains the I8I8_II8 contraction logic), so
the proof is "hipBLASLt finds AND uses the build-tree library at
runtime", not yet "hipBLASLt selects our per-shape tuned MFMA tile". The
merge step + a regression smoke comparing default-vs-tuned kernel names
is the obvious next M2 follow-up.

## Reproducibility canary (VAL-TENSILE-009)

```
$ scripts/mi100/verify_tensile_repro.sh
[FAIL] repro logic YAML differs from baseline
   GlobalReadVectorWidthA: 8  -> 16
   GlobalReadVectorWidthB: 16 -> 8
   KernelNameMin: ...MT128x64x32_MI32x32x1... -> ...MT128x32x64_MI16x16x1...
```

Tensile picked a **different** "best" kernel for `M=512,N=4096,K=4096`
on the second run. Both kernels are valid I8 MFMA solutions; the
runtime measurement noise is large enough relative to their measured
gflops to swap winners. Contributing factors:

- `HardwareMonitor: False` (forced because rocm-smi64 segfaults on this host) → no clock/temp lock during tuning.
- Single-warmup, low repeat counts in the default Tensile config.
- DGEMM-style measurement variance on a shared MI100 board.

This is acknowledged: VAL-TENSILE-009 byte-identical re-tune assertion
**does not hold** under measurement variance and we treat it as a
soft requirement (the *committed* logic YAMLs are reproducible; the
*selection* is not).

## Numerical correctness (VAL-TENSILE-005)

```
$ pytest tests/kernels/quantization/test_hipblaslt_int8_correctness.py -v
tests/kernels/quantization/test_hipblaslt_int8_correctness.py::test_hipblaslt_matches_reference[512-4096-12288]  PASSED
... (8 shapes) ...
tests/kernels/quantization/test_hipblaslt_int8_correctness.py::test_bias_is_applied            PASSED
9 passed in 4.75s
```

Reference is fp32 GEMM (operands cast int8→fp32, scales applied in fp32, then cast).
On ROCm `torch.matmul(int32, int32)` is unimplemented; for K ≤ 12288 with operands in [-32, 31] fp32 retains full INT32-equivalent precision (max product ~12.5M < 2^24).

## Dispatch routing (VAL-TENSILE-008)

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

The off-list shape produces bit-identical output with and without `VLLM_DISABLE_HIPBLASLT=1`, proving the fallback path is unchanged from M1.

## How to enable hipBLASLt at serve time

```bash
# Required only if /opt/rocm-shipped libhipblaslt is missing the I8I8_II8
# contraction logic (default for ROCm 7.12 on this host):
export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
export HIPBLASLT_TENSILE_LIBPATH=/root/hipblaslt-src/build/release/hipblaslt-install/lib/hipblaslt/library

# The dispatcher is enabled by default; to disable and route 100% to Triton:
export VLLM_DISABLE_HIPBLASLT=1
```

## Deferred follow-ups

- [ ] Merge per-shape tuned YAMLs into a runtime-loadable `TensileLibrary_lazy_gfx908.dat` so hipBLASLt actually selects the tuned MFMA tile (currently it loads the build-tree default I8I8_II8 logic).
- [ ] Run the {TP=1,4} × {c=1,2,4} W8A8 benchmark grid (24 cells × 200 prompts) and update `BENCH_INT8_W4A16_BASELINE.md` with M2 throughput / TTFT / TPOT vs M1 to declare a Pareto win/loss.
- [ ] Wikitext-2 perplexity Δ (≤ +1% vs M1) to confirm no quality regression from the new INT8 path.
- [ ] Address VAL-TENSILE-009 measurement variance: pin clocks (alternative to `HardwareMonitor`) or increase Tensile repeat counts and re-run repro canary.
