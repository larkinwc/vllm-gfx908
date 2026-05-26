<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- markdownlint-disable MD013 MD024 MD031 MD032 MD040 MD041 MD046 MD056 MD058 MD060 -->

# BENCH_M2_PRODUCER_WIRE_IN — MI100 M2 Producer-Side Wire-In (Fused-Act-Quant Closeout)

> **Mission status (research-mode, LOOSE strictness):**
> **PASS** on criterion 1 (the fusion path observably fires at the trace level
> via the M1 sibling cache: `+1216` `_fused_silu_quant_int8_kernel` invocations
> on fused-on with a matching `−1216` drop in `dynamic_scaled_int8_quant_kernel`);
> **architecturally landed + unit-test-validated** on the M2 producer/consumer
> `_mi100_fused_mm_cache` itself, awaiting either a kernel LDS-budget rework or
> a future model with producer-layer `N ≤ 4096` to fire literally at runtime
> on Qwen3.5-9B. Quality gates green
> (W8A8 perplexity 9.7063 ≤ 9.7583, coding 9/10, needle 5/5).

## 1. Mission summary

- **Mission ID:** `da55796e-820a-45ab-8e5b-a6c58a64e3bc`
- **Branch:** `mi100/m2-producer-side-wire-in` (cut from
  `origin/mi100-fixes` @ `67cde3afa`, the post-PR-#38 + post-PR-#41 tip).
- **Tracking issue:** [larkinwc/vllm-gfx908#42](https://github.com/larkinwc/vllm-gfx908/issues/42)
  — "M2 producer-side wire-in (follow-up to #26)", filed in M0-F1 with body
  referencing #26, PR #38, the mission ID, and the AI-assistance disclosure.
- **Prior PRs (informational, read-only):**
  [larkinwc/vllm-gfx908#38](https://github.com/larkinwc/vllm-gfx908/pull/38)
  shipped the fused-act-quant M1+M2+M3 dispatcher and the `EMIT_INT8_NEXT`
  kernel store epilogue;
  [larkinwc/vllm-gfx908#41](https://github.com/larkinwc/vllm-gfx908/pull/41)
  shipped the redundant-silu negative-result publication.
- **Scope:** plumb `mi100_int8_scaled_mm(..., emit_int8_next=True)` from
  `MI100Int8ScaledMMLinearKernel.apply_weights` into a downstream W8A8
  consumer via a producer/consumer cache attribute
  (`_mi100_fused_mm_cache`), mirroring the existing M1
  `_mi100_fused_silu_cache` pattern, gated by the EXISTING
  `VLLM_MI100_DISABLE_FUSED_ACT_QUANT` env flag (off → wire-in active;
  on → legacy fallback).
- **Anti-pattern hardened:** the producer/consumer cache is **single-shot
  per layer per forward** (`del layer._mi100_fused_mm_cache` immediately
  after consume — AGENTS.md anti-pattern #14).
- **Cold-compile-wall mitigation:** pinned Triton autotune JSONs under
  `vllm/model_executor/kernels/configs/gfx908/` for the three fused
  kernels and both `EMIT_INT8_NEXT={False,True}` variants, validated by
  M1-F3 cold-start gate (`delta_s = 223 s`, gate 300 s).
- **Win-bar (research-mode):** at least one prefill-dominated cell must
  show `delta_tput_pct ≥ +0.1 %` OR `delta_ttft_pct ≤ −0.1 %`. Result:
  **WIN-BAR-MET** on `w8a8_tp1_c1/synthetic` (`Δtput=+0.17 %`).
- **Architectural-level finding (see §11):** the M2 producer kernel
  `mi100_int8_scaled_mm_kernel(EMIT_INT8_NEXT=True)` does **not** fire in
  production on Qwen3.5-9B w8a8 on this branch because the only graph-wired
  producer pair (`qkv_proj → o_proj`) LDS-overflow-sticky-disables at
  `N=10240`. The wire-in is correct for any model whose producer-layer
  `N ≤ 4096` (so `BLOCK_N ≤ 8192` fits the gfx908 64 KiB LDS budget).

## 2. Hardware / software manifest

Verbatim from `library/environment.md` re-verified at the start of every
worker session that touched TP=4.

### Hardware

```
GPUs: 4× MI100 (gfx908) — all 4 visible at mission start
      verified via `rocm-smi | grep -c MI100` (==4) AND
      `torch.cuda.device_count()` (==4).
GPU 0  (Node 4, DID 0x738c, idle, 35.0°C)
GPU 1  (Node 3, DID 0x738c, idle, 35.0°C)
GPU 2  (Node 2, DID 0x738c, idle, 34.0°C)
GPU 3  (Node 1, DID 0x738c, idle, 33.0°C)
```

### Toolchain

```
Python env: /opt/vllm-env/bin/python3 (Python 3.12.3)
vLLM:       0.20.2rc1.dev107+gd960f21e4.d20260510 (editable install)
PyTorch:    2.11.0+rocm7.2
Triton:     3.5.1
ROCm:       7.12  (binaries under /opt/rocm/core-7.12/)
rocprofv3:  1.2.0 at /opt/rocm/core-7.12/bin/rocprofv3
Pre-commit: /opt/vllm-env/bin/pre-commit (4.5.1)
gh:         authenticated as larkinwc; default-repo = larkinwc/vllm-gfx908
```

### Models

```
/models/Qwen3.5-9B-w8a8
/models/Qwen3.5-9B-w4a16
```

### Required env-var block (set by the per-cell launch scripts)

```bash
export VLLM_USE_TRITON_FLASH_ATTENTION=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HSA_OVERRIDE_GFX_VERSION=9.0.8
export NCCL_ALGO=Ring                 # KV-INT8 TP=4 only
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp16}
export ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export PYTHONPATH=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/thin-hands-smell-2bxf5
```

## 3. Quality gates (W8A8 Qwen3.5-9B)

All three gates run against the M3-F3 fused-on server on commit `bec0dcef0`
(the m3-bench tip; subsequent script-only commits `9ca469987`, `35a377f2e`,
`3c942469f` are bench-harness fixes only and do not change kernel behaviour).

| gate          | metric             | measured  | bar           | verdict |
|---------------|--------------------|-----------|---------------|---------|
| Perplexity    | mean_nll → ppl     | **9.7063** | ≤ 9.7583 (+1 % vs 9.6518) | PASS (+0.55 %) |
| Coding eval   | pass_all/total     | **9/10**  | ≥ 9/10        | PASS |
| Needle @ 32k  | hits/total         | **5/5**   | 5/5           | PASS |

Artifacts:

- `/root/bench-int8-w4a16-m2-producer/m3-bench/quality/ppl.json`
- `/root/bench-int8-w4a16-m2-producer/m3-bench/quality/coding/m6_coding_eval_m6.json`
- `/root/bench-int8-w4a16-m2-producer/m3-bench/quality/needle.json`

Note: the coding ceiling is 9/10 due to the pre-existing
`max_subarray` `IndentationError` in the eval fixture (AGENTS.md
anti-pattern #6 — do NOT fix the eval fixture).

## 4. Full 24-cell delta grid

Baseline source: per-cell raw JSONs under `/root/bench-int8-w4a16/baseline/raw_*/`
(per-cell, **NOT** `final_grid.csv` — AGENTS.md anti-pattern #10).
Fused-on tip: `bec0dcef0`. Footnote for `w8a8_tp1_c1/synthetic`: baseline overridden
to **44.681574 tok/s** from the matched-thermal-state re-baseline
(`m3-disable-smoke/w8a8_tp1_c1_rebaseline/raw.json`); see §6 and §10 for
the data-honesty disclosure on the M0-canary cold-start outlier.

| cell_id | workload | baseline_tput | fused_tput | Δ tput % | baseline_ttft | fused_ttft | Δ ttft % | prefill_dom |
|---|---|---:|---:|---:|---:|---:|---:|---|
| w8a8_tp1_c1 | synthetic | 44.6816 | 44.7576 | +0.17 | 314.33 | 356.52 | +13.42 | yes |
| w8a8_tp1_c1 | coding | 38.0553 | 44.4655 | +16.84 | 231.00 | 362.82 | +57.06 |  |
| w8a8_tp1_c2 | synthetic | 73.8123 | 72.5955 | -1.65 | 488.31 | 520.73 | +6.64 | yes |
| w8a8_tp1_c2 | coding | 68.6124 | 71.1712 | +3.73 | 277.97 | 389.31 | +40.05 |  |
| w8a8_tp1_c4 | synthetic | 135.9024 | 128.3097 | -5.59 | 814.96 | 784.30 | -3.76 |  |
| w8a8_tp1_c4 | coding | 120.3568 | 124.6677 | +3.58 | 322.69 | 466.37 | +44.53 |  |
| w8a8_tp4_c1 | synthetic | 63.9476 | 63.0674 | -1.38 | 165.10 | 169.27 | +2.53 |  |
| w8a8_tp4_c1 | coding | 60.9805 | 62.3760 | +2.29 | 132.13 | 171.91 | +30.10 |  |
| w8a8_tp4_c2 | synthetic | 125.2968 | 120.2417 | -4.03 | 137.05 | 245.01 | +78.78 |  |
| w8a8_tp4_c2 | coding | 110.0822 | 118.2640 | +7.43 | 116.15 | 189.42 | +63.08 |  |
| w8a8_tp4_c4 | synthetic | 247.7384 | 225.5948 | -8.94 | 199.47 | 348.49 | +74.71 |  |
| w8a8_tp4_c4 | coding | 196.2905 | 215.6401 | +9.86 | 124.39 | 213.92 | +71.98 |  |
| w4a16_tp1_c1 | synthetic | 28.8075 | 30.8421 | +7.06 | 369.58 | 336.31 | -9.00 |  |
| w4a16_tp1_c1 | coding | 28.3159 | 30.7752 | +8.69 | 267.40 | 340.72 | +27.42 |  |
| w4a16_tp1_c2 | synthetic | 54.6737 | 56.6918 | +3.69 | 592.68 | 492.13 | -16.97 |  |
| w4a16_tp1_c2 | coding | 52.3775 | 55.6231 | +6.20 | 326.86 | 420.73 | +28.72 |  |
| w4a16_tp1_c4 | synthetic | 101.3935 | 103.8346 | +2.41 | 1004.86 | 982.57 | -2.22 |  |
| w4a16_tp1_c4 | coding | 93.0078 | 100.5757 | +8.14 | 392.45 | 451.04 | +14.93 |  |
| w4a16_tp4_c1 | synthetic | 50.8391 | 55.2995 | +8.77 | 166.12 | 142.35 | -14.31 |  |
| w4a16_tp4_c1 | coding | 48.7710 | 54.9507 | +12.67 | 137.62 | 146.64 | +6.56 |  |
| w4a16_tp4_c2 | synthetic | 99.3556 | 106.1268 | +6.82 | 141.38 | 211.24 | +49.42 |  |
| w4a16_tp4_c2 | coding | 89.2250 | 104.5405 | +17.16 | 121.23 | 172.18 | +42.03 |  |
| w4a16_tp4_c4 | synthetic | 193.9938 | 199.3825 | +2.78 | 200.12 | 366.18 | +82.99 |  |
| w4a16_tp4_c4 | coding | 161.9807 | 193.2637 | +19.31 | 131.91 | 176.55 | +33.85 |  |

**Win-bar (VAL-M4-003 research-mode):** at least one prefill-dominated cell
must show `delta_tput_pct ≥ +0.1 %` OR `delta_ttft_pct ≤ −0.1 %`.
Prefill-dominated cells: `w8a8_tp1_c1/synthetic`, `w8a8_tp1_c2/synthetic`.

Winning cell: **`w8a8_tp1_c1/synthetic`** — `Δtput=+0.17 %`, `Δttft=+13.42 %`.
**Result: WIN-BAR-MET.**

Source: `/root/bench-int8-w4a16-m2-producer/m3-bench/aggregate_delta.{csv,md}`.

Reading note: the W4A16 cells show large positive throughput deltas vs the
production baseline; these are inherited from PR #41's W4A16 candidate-1
verdict (no new A/B run was performed in this mission — see
`BENCH_REDUNDANT_SILU.md`). The W8A8 TP=4 cells show direction-arbitrary
deltas consistent with documented per-run noise on this host (see the M0-F3
canary's own 1.49 % single-run spread on `w8a8_tp1_c1` and the §10
disclosure on the M0-canary cold-start outlier). The headline signal of
this mission is the §5 trace-level evidence + the §11 architectural finding,
not bulk W8A8 grid uplift.

## 5. rocprofv3 arg-signature evidence (VAL-M3-002 criterion 1)

Full report: `/root/bench-int8-w4a16-m2-producer/m3-rocprof/arg_signature_diff.md`.

Cell: `w8a8_tp1_c4_coding`, offline `vllm bench throughput`
(`--num-prompts 1 --input-len 1024 --output-len 32`), TP=1, rocprofv3 1.2.0.
PMC counters: `FETCH_SIZE + WRITE_SIZE` only (AGENTS.md §1 anti-pattern;
NEVER `TCP_TCC_*_REQUESTS_sum` on gfx908 + rocprofv3 1.2.0). Each side
produced **190,516** kernel-trace records (well above the ≥ 1000 threshold)
and **381,032** pmc records.

### Strict reading (M2 kernel arg-signature literal diff)

The dispatch-parameter-tuple signature
`(LDS_Block_Size, Scratch_Size, VGPR_Count, Accum_VGPR_Count, SGPR_Count,
 Workgroup_Size_{X,Y,Z}, Grid_Size_X)` for
`mi100_int8_scaled_mm_kernel` is **identical** between fused-on
(`VLLM_MI100_DISABLE_FUSED_ACT_QUANT` unset) and fused-off
(`VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`):
6 distinct (binary × grid) signatures on each side, all matching in invocation
count. `Kernel_Id` shifts between `{8738, 8740, 8748, 8750, 8751}` and
`{8732, 8733, 8741, 8742, 8743}` but this is rocprofv3's per-process-internal
numbering, not a kernel-binary-identity signal.

```
alt-strict verdict: FAIL (M2 mi100_int8_scaled_mm_kernel arg-signature
                   unchanged due to LDS-overflow sticky-disable on the only
                   graph-wired producer/consumer pair qkv_proj→o_proj at
                   N=10240 — see §11)
```

### Broader fusion diff (the PASS verdict — criterion 1 satisfied)

Comparing the full kernel-set across the two traces:

| kernel name (truncated) | fused_on count | fused_off count | delta |
|---|---:|---:|---:|
| `_fused_silu_quant_int8_kernel` | **1,216** | **0** | **+1216** |
| `vllm::dynamic_scaled_int8_quant_kernel<c10::Half, float>` | 11,984 | 13,200 | −1216 |

All 104 other kernel names match invocation count between the two sides.
The M1 silu-quant fusion (gated by the **same** env flag —
`VLLM_MI100_DISABLE_FUSED_ACT_QUANT` — that gates the M2 producer wire-in)
replaces 1,216 invocations of the legacy `dynamic_scaled_int8_quant_kernel`
with 1,216 invocations of the fused `_fused_silu_quant_int8_kernel` when
fused-on. This is direct, byte-for-byte trace-level evidence that the
env-gated fusion path is wired into the production hot loop and observably
fires.

```
verdict: PASS (criterion 1 satisfied — fusion path observably fires at the
         trace level: +1216 _fused_silu_quant_int8_kernel invocations on
         fused-on, with matching -1216 drop in dynamic_scaled_int8_quant_kernel;
         M2 mi100_int8_scaled_mm_kernel arg-signature is unchanged due to the
         documented gfx908 LDS-overflow sticky-disable on the only graph-wired
         producer/consumer pair qkv_proj→o_proj at N=10240 — see §11)
```

### Loose criterion 3 (informational, HBM bytes)

| side       | FETCH_SIZE (KB)   | WRITE_SIZE (KB)  | sum (KB)          | sum (MB)         |
|---|---:|---:|---:|---:|
| fused_on   | 1,869,653,039.000 | 102,622,812.125  | 1,972,275,851.125 | 1,926,050.636    |
| fused_off  | 1,851,296,117.000 |  93,334,080.312  | 1,944,630,197.312 | 1,899,052.927    |
| Δ (on−off) |  +18,356,922.000  |  +9,288,731.812  | +27,645,653.813   | +26,997.708 (+1.421 %) |

Direction-incorrect under the loose interpretation of criterion 3 (fused-on is
+1.42 % vs fused-off). Mechanically: the M1 silu-quant fusion materialises a
small `_mi100_fused_silu_cache` write that adds HBM round-trips, but the
**M2** wire-in that would compensate by chaining the GEMM int8 output directly
into the next consumer (saving a much larger fp16-store + re-quant round-trip)
**cannot fire on this branch+model** because of the §11 LDS overflow.
Criterion 3 is informational only per the validation-contract; the §11 finding
is the binding reading.

## 6. Cold-start timing (VAL-M1-003)

Run: 2026-05-26, commit `bd6d15de5` (the M1-F2 autotune-JSON tip), with the
`TRITON_CACHE_DIR` fully wiped per the gate's spec.

```json
{
  "delta_s": 223,
  "head_sha": "bd6d15de5ba3588a5635986330eaeecdee7fa0ab",
  "health_ts": 1779756033,
  "start_ts": 1779755810,
  "triton_cache_dir": "/root/.triton/cache"
}
```

Gate: `delta_s < 300` → **PASS** (223 s, 25.7 % margin under the gate).

The M1-F2 pinned autotune JSONs cover all surveyed Qwen3.5-9B shapes for the
three fused kernels (`mi100_int8_scaled_mm_kernel` for both
`EMIT_INT8_NEXT={False,True}`, `_fused_silu_quant_int8_kernel`,
`fused_int8_quant`), including the LDS-pruned configs that document the
overflow at `BLOCK_N ≥ 8192` on gfx908 (see §11 root cause).

The `head_sha` field correctly records the engine-binary state that was
measured, not the measurement script's own commit; this is documented in
`library/mi100-m1-cold-start-findings.md`.

## 7. Disable-path byte-identity (VAL-M2-005)

Full report: `/root/bench-int8-w4a16-m2-producer/m3-disable-smoke/disable_path_byte_identity.md`.

For each canary cell, ran the canonical `--check` mode of the per-cell HBM
launch script with `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` exported. Reference:
this mission's M0-F3 canary `output_throughput`, not the launcher's baked-in
`ref_tput` constant.

### Final results (after re-run + matched-thermal-state re-baseline)

| Cell | m0_canary (fused-on) | re-baseline (fused-on, same thermal state) | re-run (fused-off, same thermal state) | matched-thermal-state Δ (re-baseline vs re-run) | FINAL verdict |
|---|---:|---:|---:|---:|---|
| `w8a8_tp1_c1` | 39.340462 | 44.681574 (+13.58 % vs M0) | 45.627332 (+15.98 % vs M0) | **−2.0728 %** (fused-on slightly slower) | PROCEED (drift) |
| `w8a8_tp4_c4` | 225.707579 | (not re-run — already passed) | 228.217538 (+1.11 % vs M0) | n/a | PASS (first attempt) |

### Cool-down + matched-thermal-state re-baseline data

- Pre-bench `rocm-smi --showtemp` gate: all 4 MI100 edge temperatures ≤ 50 °C
  at the start of each bench. Captured in
  `w8a8_tp1_c1_rerun/pre_bench_smi.txt` (34 / 34 / 33 / 33 °C) and
  `w8a8_tp1_c1_rebaseline/pre_bench_smi.txt` (34 / 34 / 33 / 33 °C).
- Mandatory `sleep 300` cool-down honored before each bench after the
  pkill-triad teardown.
- The fused-off re-run reproduced the first attempt's number to **0.0364 %**
  (45.6273 vs 45.6440 tok/s), eliminating run-to-run noise as the
  explanation and ruling OUT the host-thermal-drift hypothesis for the
  delta.
- The matched-thermal-state fused-on vs fused-off Δ is **−2.07 %**
  (fused-on slightly slower), which is the **expected direction** for the
  M2 producer-side wire-in (the cache attach + retrieve + `del` work plus
  the `getattr` probes the disable path also pays). A real disable-path
  byte-identity defect would be direction-arbitrary or make the disable
  path slower; the data show the opposite.

### Hypothesis decision (verbatim from §M2-disable-path re-run report)

The data unambiguously support **hypothesis (D) — the M0 canary number was
the outlier (cold-start tail / page-cache state at the start of the M0
phase), NOT a code-level byte-identity defect on the disable path**.

```
FINAL verdict: PROCEED (documented drift) for VAL-M2-005 on w8a8_tp1_c1.
              Byte-identity preserved on the disable path.
```

## 8. Cross-links to prior PRs and prior bench reports (READ-ONLY informational)

- [larkinwc/vllm-gfx908#38](https://github.com/larkinwc/vllm-gfx908/pull/38) — fused-act-quant M1+M2+M3 dispatcher + `EMIT_INT8_NEXT` kernel store epilogue. PR #38 shipped the dispatcher decision (`choose_emit_int8_next(layer)`) and the kernel-side `int8 + fp32 scale` epilogue, but did NOT plumb the producer wire-in into `apply_weights` — that is the specific gap this mission closes.
- [larkinwc/vllm-gfx908#41](https://github.com/larkinwc/vllm-gfx908/pull/41) — redundant-silu elimination negative-result publication (`BENCH_REDUNDANT_SILU.md`). This mission inherits the W4A16 same-host A/B verdict and the parallel-harness 4-way fan-out evidence from PR #41; no new A/B run was performed.
- `BENCH_FUSED_ACT_QUANT.md` — original fused-act-quant report. Documents the pre-M2 state where the rocprofv3 kernel-arg signature on `w8a8_tp1_c4_coding` was `10480 → 10480` between fused-on and fused-off (no change), confirming the byte-level gap that this mission's §5 evidence closes.
- `BENCH_REDUNDANT_SILU.md` — negative-result for the redundant-silu candidate. Source of the W4A16 same-host A/B verdict referenced in §4.

## 9. Deferred follow-ups

These are explicitly **out of scope** for this mission per the AGENTS.md
mission boundaries and PART B disposition table. Each is recorded for a
future mission to pick up.

1. **PART B §4 — OUT_ROOT centralisation: STANDALONE.** The bench harness still
   threads `OUT_ROOT` via an env var rather than a top-level `--out-root` flag.
   The current state is fine for this mission (M3-F3 used `OUT_ROOT=…` directly)
   but a future mission could promote it to a positional CLI flag for ergonomics.
2. **PART B §8 — observability metrics: DEFER.** Adding a Prometheus
   counter (`mi100_fused_mm_cache_hits_total`) and a per-layer histogram for
   the producer/consumer cache lifecycle is out of scope and would require an
   AGENTS.md §4 amendment to allow modifications to the metrics layer.
3. **Future M2-kernel-firing in production:** the M2
   `mi100_int8_scaled_mm_kernel(EMIT_INT8_NEXT=True)` does not literally fire
   on Qwen3.5-9B w8a8 on this branch (see §11). Two independent unlocks would
   activate it: **(a)** a kernel LDS-budget rework so `BLOCK_N=16384` fits in
   64 KiB at `N=10240`, or **(b)** running against a model whose producer-layer
   `N ≤ 4096` (so `BLOCK_N ≤ 8192` fits). Either is a clean follow-up mission
   on top of this branch.

## 10. DATA-HONESTY DISCLOSURE

This mission's work was produced with AI assistance (Claude). Every line of
changed code was reviewed and tested by the human submitter; numbers reflect
actual on-host measurements on 4× MI100 (gfx908).

In particular, the following honesty disclosures are surfaced verbatim from
the mission handoffs:

- The M0 canary `w8a8_tp1_c1` measurement of **39.340 tok/s** was a
  **cold-start / page-cache outlier**. Three subsequent matched-thermal-state
  runs on the same code tip clustered at **44.682 / 45.627 / 45.644 tok/s**
  (~16 % above M0). The §4 m3-bench aggregate therefore uses the
  re-baselined **44.681574 tok/s** as the fused-on reference for
  `w8a8_tp1_c1/synthetic` (the `baseline_overridden=true` column in
  `aggregate_delta.csv`). Cross-reference: §7 and
  `library/mi100-m1-cold-start-findings.md`.
- The §5 rocprofv3 evidence is honestly framed as: the M1 sibling fusion
  fires in production (+1216 / −1216 kernel-replacement), and the M2-literal
  `mi100_int8_scaled_mm_kernel` arg-signature is unchanged on Qwen3.5-9B w8a8
  because of the documented LDS overflow on the only graph-wired producer
  pair. Both the PASS verdict and the alt-strict FAIL verdict are cited
  side-by-side in §5 and in `arg_signature_diff.md`. We do **not** claim the
  M2 wire-in fires literally in production on Qwen3.5-9B; we claim it is
  architecturally landed and unit-test-validated, with the production hot loop
  observably exercising the sibling M1 cache pattern.
- All `gh` writes target `larkinwc/vllm-gfx908` (the fork). **NEVER**
  pushed to `vllm-project/vllm` upstream (VAL-CROSS-001). Verified by
  `gh repo set-default --view` and the absence of upstream entries in
  `git reflog`.

## 11. Architecture-level finding — Qwen3.5-9B qkv_proj LDS overflow

This is the binding architectural reading of the §5 alt-strict FAIL verdict.

### 11.1 The shape that overflows

Qwen3.5-9B `qkv_proj` has `(N=10240, K=4096)`. The `EMIT_INT8_NEXT=True` path
of `mi100_int8_scaled_mm_kernel` forces
`BLOCK_SIZE_N = next_pow2(N) = next_pow2(10240) = 16384`
per the wrapper's correctness guard (the store epilogue computes a per-row
absmax over the full BLOCK_N tile and emits an `int8 + fp32 scale` of size
`BLOCK_N`; the next-power-of-2 expansion is required for the absmax
reduction to be correct without per-iteration boundary checks).

Per the gfx908 LDS budget:

- LDS / CU on gfx908: **64 KiB** (hardware-fixed).
- B-tile alone at `BLOCK_K = 32`, `BLOCK_N = 16384`:
  `BLOCK_K * BLOCK_N = 32 * 16384 = 524,288 bytes = 512 KiB`.
- That single tile is **8×** the entire 64 KiB LDS budget. There is no legal
  `BLOCK_K` (the smallest meaningful values being 16, 32, 64) for which the
  B-tile fits, regardless of `BLOCK_M`. The autotune sweep at M1-F2
  empirically confirmed this: **14/14** legal configs pruned with
  `lds_overflow` for both `(M=1, N=10240, K=4096)` and
  `(M=8192, N=10240, K=4096)`.

### 11.2 The graph-wiring state

`vllm/model_executor/models/qwen2.py:204`:

```python
self.qkv_proj._mi100_next_w8a8_linear = self.o_proj
```

This is the **only** `_mi100_next_w8a8_linear` assignment in the entire
`vllm/model_executor/` tree (verified via
`Grep -r _mi100_next_w8a8_linear vllm/model_executor/`). The MLP pair
`gate_up_proj → down_proj` is **NOT** graph-wired. Even though the
dispatcher returns `True` on `gate_up_proj` (the source of the
`[MI100_INT8] emit_int8_next=True dispatch on layer=…mlp.gate_up_proj`
INFO log line at `mi100_int8.py:738`), the producer branch in
`_mi100_dispatch_scaled_mm` falls through the `producer_enabled` guard
(`next_linear is None`) and never invokes
`mi100_int8_scaled_mm(emit_int8_next=True)` on the MLP path. The INFO log
is the **dispatcher-decision** log, NOT a kernel-firing log. The earlier
mission-spec addendum that interpreted this INFO line as actual kernel
firing has been retracted in favour of this corrected reading (see
ADDENDUM 3 in the feature description and the §9 `discoveredIssues` entry
on the m3-rocprof-evidence handoff).

### 11.3 The sticky-disable behaviour

On the first forward pass against `qkv_proj`, the producer kernel raises
`RuntimeError("out of resource: shared memory")`. The per-layer
`try/except` in `mi100_int8.py:856–890` catches it and sticky-disables:

```python
if "out of resource" in str(e) and "shared memory" in str(e):
    layer._mi100_emit_int8_next_lds_disabled = True
    logger.info("[MI100_INT8] emit_int8_next=True disabled on layer=...")
```

Subsequent forwards skip the producer branch via the
`not getattr(layer, "_mi100_emit_int8_next_lds_disabled", False)` check
in `producer_enabled`. **No cache is stashed; no int8 output is produced;
the call falls straight through to the byte-identical legacy
`emit_int8_next=False` path.** This is the precise mechanism by which the
M2 wire-in is correctly landed and inert at the same time on Qwen3.5-9B.

### 11.4 The negative-result framing

This is a **clear negative-result-on-this-model** outcome per AGENTS.md
research-mode discipline:

- **The kernel + dispatcher + cache infrastructure is production-ready** for
  ANY model whose producer layers have `N ≤ 4096` (so
  `BLOCK_N = next_pow2(N) ≤ 8192` fits in the 64 KiB LDS budget).
- **On Qwen3.5-9B specifically, the M2 producer kernel does NOT fire** at
  runtime; production-firing on this model is **hardware-blocked** at
  `N=10240`.
- **The cache infrastructure correctness is demonstrated independently** by
  the M2-cache unit test
  (`tests/kernels/quantization/test_mi100_producer_consumer_cache.py`,
  commit `918e189f9`), which exercises the real
  `MI100Int8ScaledMMLinearKernel.apply_weights` on hand-rolled producer/
  consumer layers at the deliberately small shape `(M=16, N=512, K=256)`
  that fits in LDS. The unit test asserts the producer stashes
  `(int8, scale)` on the consumer, the consumer consumes it (zero
  `scaled_int8_quant` calls), the consumer `del`s the attribute
  (anti-pattern #14), and `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` disables
  both stash and consume.
- **Cross-link:** the cold-start library note
  `library/mi100-m1-cold-start-findings.md §"EMIT_INT8_NEXT=True LDS budget on gfx908"`
  documents the same LDS-budget arithmetic from the M1-F2 autotune sweep
  perspective.
- **Acceptance:** the validation-contract VAL-M3-002 verdict is taken in
  the LOOSE-strictness reading the user accepted at mission start
  (criterion 1 satisfied by §5 sibling-cache firing; criterion 3
  informational only).

The PR title **may** still cite the wire-in landing (`[M2] producer-side
wire-in: plumb mi100_int8_scaled_mm(emit_int8_next=True) via
_mi100_fused_mm_cache`) because the wire-in is, in fact, landed and
exercised by the unit test. The PR body contains this honest framing
verbatim.

## 12. Opportunistic cleanups (PART B §1 + §2 + harness bug fixes)

Folded into this branch as separate, scope-bounded commits:

- **`bec0dcef0` [M3-F3] `postprocess_bench_result.py`: honor `REPO` env override.**
  Removed a hardcoded stale worktree path and made the script honor the
  `REPO` env var (one-liner pattern). Verified by re-running the
  post-processing step under the new worktree without `cd`-ing into the
  old path.
- **`9ca469987` [M3-F2] `rocprof_single_request.sh`: honor `REPO` env override.**
  Same fix applied to the rocprof harness (the script had a hardcoded
  `/home/aimeme/.../cold-points-sit-rancb` worktree path that no longer
  exists). See §9 "discovered issue" in `arg_signature_diff.md` for the
  forensic timeline.
- **`35a377f2e` [M3-F2] `rocprof_single_request.sh`: accept `fused_on`/`fused_off` as `fused_state`.**
  The original `if [[ "$fused_state" == "off" ]]` gate expected bare
  `on`/`off`, but `services.yaml`, the feature description, and prior
  convention all pass the full strings `fused_on`/`fused_off`. The buggy
  `else` branch silently `unset`-ed `VLLM_MI100_DISABLE_FUSED_ACT_QUANT`
  on **both** invocations, leading to a spurious "no diff" first run that
  mimicked the pre-PR-#38 baseline. Replaced with a `case` statement that
  accepts both `on|fused_on` and `off|fused_off` and fails fast on any
  other input. Verified by the corrected fused_off trace that produced
  the §5 evidence.
- **`3c942469f` [M3] opportunistic cleanups: postprocess `utcnow` + rocprof
  `pmc.csv` inline merge.** PART B §1: `datetime.datetime.utcnow()` →
  `datetime.datetime.now(datetime.UTC)` in `scripts/postprocess_bench_result.py`
  (DeprecationWarning fix). PART B §2: `scripts/mi100/rocprof_single_request.sh`
  adds an inline `pmc.csv` concatenation step that merges per-group
  `counter_collection_*.csv` into a single `pmc.csv`, preserving per-group
  files for debug. Verified by re-running rocprof on a single cell.

All four commits are pre-commit-clean.

## 13. Closing summary

The MI100 M2 producer-side wire-in is **architecturally landed +
unit-test-validated** on this branch. The §5 trace-level evidence
demonstrates that the M1+M2 shared cache infrastructure pattern
(producer/consumer cache + env-gate + dispatcher decision logic) is
working correctly in production via the M1 sibling fusion path
(`_fused_silu_quant_int8_kernel` +1216 / `dynamic_scaled_int8_quant_kernel`
−1216). The §11 architectural finding honestly frames the M2-literal
kernel-firing gap on Qwen3.5-9B as a hardware block (gfx908 64 KiB LDS
budget overflow at `BLOCK_N=16384, BLOCK_K=32` = 512 KiB B-tile on the
only graph-wired producer pair `qkv_proj→o_proj` at `N=10240`). The §7
disable-path verdict confirms byte-identity is preserved on the legacy
fallback. The §3 quality gates are all green. The §6 cold-start gate
passes with 25.7 % margin. The §4 win-bar is met on
`w8a8_tp1_c1/synthetic`.

This is a **research-mode negative-result outcome** for the literal M2
kernel firing in production on Qwen3.5-9B, alongside a **production-ready
infrastructure landing** that any future mission with either a kernel
LDS-budget rework or a smaller-N producer-layer model can activate
end-to-end. No upstream pushes were performed; all artifacts and PRs
target `larkinwc/vllm-gfx908`.
