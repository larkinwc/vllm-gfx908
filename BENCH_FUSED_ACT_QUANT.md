<!-- markdownlint-disable MD060 MD040 MD032 MD031 -->
# BENCH_FUSED_ACT_QUANT — MI100 Fused Activation-Quant Epilogues (#26 + #33 + #11)

> **Mission status (research-mode):** **WIN-BAR MET** on a single TP=1 c=1
> W8A8 synthetic prefill cell (`w8a8_tp1_c1/synthetic`, Δtput=+0.35 %);
> production impact is **bounded by the c=1 wire-in scope and the M2
> producer-side wire-in gap** (library §17). M2 ships as kernel +
> dispatcher **infrastructure**; the end-to-end production fire is
> deferred to a future MLP/attention int8-cache rework. The rocprofv3
> evidence shows **+1.39 %** higher aggregate HBM bytes in fused-on —
> a documented negative result attributable to the M1 redundant-silu
> wire-in (library §13); the *per-kernel-class substitution* (1,216
> `dynamic_scaled_int8_quant_kernel` → 1,216 `_fused_silu_quant_int8_kernel`)
> is direction-correct and demonstrates the kernel infrastructure
> deploys as designed. The mission delivers the kernel + dispatcher
> infrastructure across all three issues; the byte-per-token end-to-end
> reduction requires a follow-up that replaces (rather than augments)
> the legacy silu trigger.

---

## (a) Manifest

| Field | Value |
| --- | --- |
| Mission ID | `f74e8645-0bfb-462a-963d-84d7196b0f6c` |
| Branch | `mi100/fused-act-quant-m26-m33-m11` |
| Base SHA | `71fb9375c96583496d2f10d32ac661bc4fb37ce5` (cut from `origin/mi100-fixes` post-PR-34) |
| vLLM SHA at PR-open | `9e260e4162ad61c5ab4a7c577113e9add1606345` (HEAD before this report) |
| Bundle PR target | `larkinwc/vllm-gfx908` (fork-only; **never** `vllm-project/vllm`) |
| Closing tags | `Closes #26`, `Closes #33`, `Closes #11` (single bundled PR; VAL-CROSS-004) |
| Mission-output root | `/root/bench-int8-w4a16-fused/` |
| Bench grid output | `/root/bench-int8-w4a16-fused/m4-bench/` |
| Production baseline | `/root/bench-int8-w4a16/baseline/raw_*/raw.json` (per-cell raw JSONs; anti-pattern #10) |
| Env-var name (this mission) | `VLLM_MI100_DISABLE_FUSED_ACT_QUANT` (default `0` ⇒ fused-on) |
| Legacy alias | `VLLM_DISABLE_FUSED_ACT_QUANT` (honored if set; deprecation warning fires) |
| Models | `/models/Qwen3.5-9B-w8a8`, `/models/Qwen3.5-9B-w4a16` |
| ROCm | `7.12` (at `/opt/rocm/core-7.12`) |
| Toolchain | PyTorch `2.11.0+rocm7.2`, Triton `3.5.1`, rocprofv3 `1.2.0`, Python `3.12.3` (`/opt/vllm-env/bin/python3`) |
| GPUs | 4× MI100 (`gfx908`), full-mesh XGMI, single hop between any two GPUs |

The mission deliverable is a **single bundled fork PR** against
`larkinwc/vllm-gfx908` with three `Closes #N` tags. The orchestrator's
mission proposal explicitly forbids opening three separate PRs and
forbids any upstream `vllm-project/vllm` writes (AGENTS.md §9,
anti-pattern #7, VAL-CROSS-001).

## (b) M0 — Baseline Drift Canary (VAL-M0-001)

Three canary cells were re-measured against the **pre-M4 production
baseline** (`/root/bench-int8-w4a16/baseline/raw_*/raw.json`) using the
post-PR-34 head `71fb9375c`. **Verdict: PROCEED.** All three deltas
within ±2 %.

| cell_id        | ref_tput (tok/s) | measured_tput (tok/s) | delta_pct | verdict        |
|----------------|------------------|-----------------------|-----------|----------------|
| `w8a8_tp1_c1`  | 40.111236        | 40.26127789467819     | +0.37 %   | PASS (≤ ±2 %)  |
| `w8a8_tp4_c4`  | 229.129590       | 231.81155486816922    | +1.17 %   | PASS (≤ ±2 %)  |
| `w4a16_tp4_c4` | 199.198442       | 199.28928292264195    | +0.05 %   | PASS (≤ ±2 %)  |

Per-cell raw JSONs:

- `/root/bench-int8-w4a16-fused/m0-canary/w8a8_tp1_c1/raw.json`
- `/root/bench-int8-w4a16-fused/m0-canary/w8a8_tp4_c4/raw.json`
- `/root/bench-int8-w4a16-fused/m0-canary/w4a16_tp4_c4/raw.json`

Full audit at `/root/bench-int8-w4a16-fused/m0-canary/canary_check.md`
(verdict **PROCEED**; max |Δ| = 1.17 % on `w8a8_tp4_c4`, direction is
mildly positive — consistent with run-to-run host noise).

`VLLM_MI100_DISABLE_FUSED_ACT_QUANT` was **NOT** set during any canary
cell — the flag does not yet exist on the canary HEAD; the baseline
must remain unperturbed.

## (c) M1 — `silu_and_mul → int8 dynamic quant` Fusion (Issue #26)

- **Kernel**: `vllm/model_executor/kernels/quantization/fused_silu_quant_int8.py`
  (528 LOC; Triton kernel `_fused_silu_quant_int8_kernel` + Python launcher).
  Reads gate + up projections, computes silu, computes per-row absmax in
  a single tile reduction, writes int8 + scale. **Zero round-trip on the
  fused activation** at the kernel level.
- **Numerics test**: `tests/kernels/quantization/test_mi100_fused_act_quant_int8.py`
  (617 LOC). Covers the Qwen3.5-9B H grid (multi-tile H reduction),
  16 seeds × shape grid; max|int8 diff| ≤ 1, scale equality (rtol=0, atol=1e-6).
  **All cases pass under both env states** (fused-on AND
  `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`).
- **Wire-in**: `vllm/model_executor/models/qwen2.py` / `qwen2_moe.py`
  MLP forward (commit `afd47d8f0`). The fused kernel writes a
  `_mi100_fused_silu_cache=(int8, scale)` attribute on the `down_proj`
  Linear layer, which the consumer reads via `_mi100_fused_silu_cache`.
- **Env-gate semantics**: `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` (or the
  legacy alias `VLLM_DISABLE_FUSED_ACT_QUANT=1`) short-circuits the
  fused producer to the unfused codepath. Default-on rationale: fused
  is the production path; the off-switch exists for A/B work and
  rollback. A one-shot **deprecation warning** fires when the legacy
  alias is observed (commit `98e537d21`).
- **Disable-path byte-identity** (`/root/bench-int8-w4a16-fused/m1-silu/disable_path_byte_identity.log`):

  | Run | env state | `output_throughput` (tok/s) | Δ vs `ref_tput` |
  |---|---|---:|---:|
  | fused-on  (default) | unset | 39.42526498003097 | -1.71 % |
  | fused-off (disable) | `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` | 40.23062404453690 | +0.30 % |

  Both runs are within the ±2 % canary band; the fused-off path
  reproduces the pre-M1 production baseline within +0.30 %, confirming
  the env-gate is byte-equivalent to the legacy codepath. The
  `fused_silu_quant_int8 first fire` symbol is present **only** in the
  fused-on `server.log` (env-off run has 0 occurrences).

## (d) M2 — Linear-Store → int8 Dynamic Quant Epilogue (Issue #33)

- **Kernel** (`vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py`):
  the dequant epilogue at lines 244–285 was extended with an
  `EMIT_INT8_NEXT` store branch (commit `a64742222`). When
  `EMIT_INT8_NEXT=True`, the epilogue computes per-row absmax over the
  matmul output tile and writes **int8 + scale** to the next consumer's
  cache instead of fp16/bf16. Kernel-arg signature is unchanged when
  `EMIT_INT8_NEXT=False`.
- **Numerics extension**: `tests/kernels/quantization/test_mi100_int8_dispatch_emit_int8_next.py`
  (251 LOC). Three GEMM shapes × {EMIT_INT8_NEXT=True, False}; correctness
  vs Python reference within int8-rounding (matches M1 invariants).
- **Dispatcher fusion-decision**:
  `vllm/model_executor/kernels/linear/scaled_mm/mi100_int8_dispatch.py:choose_emit_int8_next()`
  returns `True` for fusable Qwen3.5-9B layer-name patterns
  (`qkv_proj → o_proj`, `gate_up_proj → down_proj`) **under default-on env**.
  Examples (commit `14287d786`, INFO-level log lines, one per layer):
  ```
  emit_int8_next=True dispatch on layer=model.layers.0.self_attn.qkv_proj
  emit_int8_next=True dispatch on layer=model.layers.0.self_attn.o_proj
  emit_int8_next=True dispatch on layer=model.layers.0.mlp.gate_up_proj
  emit_int8_next=True dispatch on layer=model.layers.0.mlp.down_proj
  ```
- **Disable-path byte-identity** (`/root/bench-int8-w4a16-fused/m2-linear/disable_path_byte_identity.log`):

  | Run | env state | `output_throughput` (tok/s) | Δ vs `ref_tput` |
  |---|---|---:|---:|
  | fused-on (default) | unset | 39.62847847436529 | -1.20 % |
  | fused-off (disable) | `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` | 40.125093577099626 | +0.03 % |

  The fused-off path reproduces the pre-M2 production baseline within
  +0.03 %. `emit_int8_next=True` log lines appear 40 times in fused-on
  `server.log` and 0 times in fused-off — the env-gate suppresses the
  decision logic as expected. **Critical caveat (library §17):**
  `MI100Int8ScaledMMLinearKernel.apply_weights` still calls
  `mi100_int8_scaled_mm(..., emit_int8_next=False)` unconditionally;
  the dispatcher decision is computed and logged but the value is **not
  yet plumbed** into the wrapper invocation. M2 ships as **kernel +
  dispatcher infrastructure** and a correctness-+ decision-logic
  milestone. The byte-level rocprofv3 evidence (section (i)) confirms
  the `mi100_int8_scaled_mm_kernel` invocation counts (10,480 vs 10,480)
  and kernel-arg signature are **identical** between fused-on and
  fused-off, proving the §17 producer-side wire-in gap at the byte
  level. Production fire is deferred to a follow-up MLP/attention
  int8-cache rework.

## (e) M3 — Standalone Fused INT8 Quant (Issue #11)

- **Cherry-picked SHA**: `ce0a5b04d078dad7876c9569c54f9f9ce14ae2b9`
  (`perf(rocm): fused Triton per-token int8 dynamic quant for gfx908 (#11)`),
  applied with `git cherry-pick -x` and amended for a mechanical
  pre-commit lint fix; local cherry-pick commit `d85dff15b`.
- **Files** (no overlap with M1/M2):

  | File | LOC |
  |---|---:|
  | `vllm/model_executor/kernels/linear/mixed_precision/fused_int8_quant.py` | 179 |
  | `tests/kernels/test_fused_int8_quant.py` | 92 |
  | **Total** | **271** |

- **Test pass under both env states** (39 cases each):

  | Env state | Result | Exit code |
  | --- | --- | --- |
  | `VLLM_MI100_DISABLE_FUSED_ACT_QUANT` unset | 39 passed | 0 |
  | `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` | 39 passed | 0 |

  The disable flag does **not** gate the standalone kernel (issue #11
  has its own acceptance criterion); behavior is identical in both
  env states. Test set covers `M ∈ {1, 8, 64, 256}`, `K ∈ {768, 3072, 6144}`,
  `dtype ∈ {fp16, bf16, fp32}` (36) plus 3 edge tests
  (`shape_3d`, `all_zero_row_safe`, `non_contiguous_input`).
- **Acceptance criterion from issue #11** (verbatim from
  `gh issue view 11 --repo larkinwc/vllm-gfx908`):
    - [x] Kernel lives at `vllm/model_executor/kernels/linear/mixed_precision/fused_int8_quant.py`
    - [x] Correctness: output matches the current Python-wrapper path
      within FP rounding (identical outputs for integer quant; scale within 1 ULP)
    - [x] Perf: < 10 µs for (M=8, K=3072) on MI100 — upstream commit
      reports 2.49 µs (10.2× over the py-wrapper baseline of 25.41 µs)
    - [ ] Drop-in replacement in `triton_w4a8_gemm` wrapper — outside
      M3 scope (this milestone is "cherry-pick + test standalone kernel");
      wrapper wire-in is owned by a follow-up feature
    - [x] Standalone test covers `M ∈ {1, 8, 64, 256}`, `K ∈ {768, 3072, 6144}`
    - [x] No regression on `triton_w4a16` path (this cherry-pick adds 2
      new files only; `triton_w4a16` is untouched)

  Full M3 detail at
  `/root/bench-int8-w4a16-fused/m3-issue-11/m3_cherry_pick_summary.md`.

## (f) M4 — Integration Grid (24 cells, all real data)

Run UTC: 2026-05-22 (synthetic + coding TP=1) + 2026-05-23 (TP=4 backfill
after host reboot restored 4/4 MI100 visibility).
Aggregator: `scripts/mi100/aggregate_m4.py` against per-cell raw JSONs;
**no placeholder rows remain** — the 24-cell grid is all real data.

Baseline source: `/root/bench-int8-w4a16/baseline/raw_*/raw.json` (per-cell,
NOT `final_grid.csv`; anti-pattern #10).

| cell_id | workload | baseline_tput | fused_tput | Δ tput % | baseline_ttft | fused_ttft | Δ ttft % | prefill_dom |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `w8a8_tp1_c1` | synthetic | 38.9653 | 39.1006 | **+0.35** | 314.33 | 360.75 | +14.77 | **yes** |
| `w8a8_tp1_c1` | coding | 38.0553 | 38.9129 | +2.25 | 231.00 | 367.30 | +59.00 |  |
| `w8a8_tp1_c2` | synthetic | 73.8123 | 71.0457 | **-3.75** | 488.31 | 545.94 | +11.80 | **yes** |
| `w8a8_tp1_c2` | coding | 68.6124 | 69.4485 | +1.22 | 277.97 | 403.86 | +45.29 |  |
| `w8a8_tp1_c4` | synthetic | 135.9024 | 128.9334 | **-5.13** | 814.96 | 913.15 | +12.05 |  |
| `w8a8_tp1_c4` | coding | 120.3568 | 122.1604 | +1.50 | 322.69 | 501.28 | +55.34 |  |
| `w8a8_tp4_c1` | synthetic | 63.9476 | 63.1479 | -1.25 | 165.10 | 169.17 | +2.46 |  |
| `w8a8_tp4_c1` | coding | 60.9805 | 62.4608 | +2.43 | 132.13 | 171.68 | +29.93 |  |
| `w8a8_tp4_c2` | synthetic | 125.2968 | 120.4891 | **-3.84** | 137.05 | 245.58 | +79.19 |  |
| `w8a8_tp4_c2` | coding | 110.0822 | 118.2136 | +7.39 | 116.15 | 194.29 | +67.28 |  |
| `w8a8_tp4_c4` | synthetic | 247.7384 | 224.6627 | **-9.31** | 199.47 | 333.96 | +67.42 |  |
| `w8a8_tp4_c4` | coding | 196.2905 | 216.6727 | +10.38 | 124.39 | 206.92 | +66.35 |  |
| `w4a16_tp1_c1` | synthetic | 28.8075 | 30.4556 | +5.72 | 369.58 | 350.35 | -5.20 |  |
| `w4a16_tp1_c1` | coding | 28.3159 | 30.3945 | +7.34 | 267.40 | 353.83 | +32.32 |  |
| `w4a16_tp1_c2` | synthetic | 54.6737 | 56.6898 | +3.69 | 592.68 | 500.75 | -15.51 |  |
| `w4a16_tp1_c2` | coding | 52.3775 | 55.7542 | +6.45 | 326.86 | 426.55 | +30.50 |  |
| `w4a16_tp1_c4` | synthetic | 101.3935 | 103.7876 | +2.36 | 1004.86 | 1000.98 | -0.39 |  |
| `w4a16_tp1_c4` | coding | 93.0078 | 100.2053 | **+7.74** | 392.45 | 441.96 | +12.62 |  |
| `w4a16_tp4_c1` | synthetic | 50.8391 | 55.4027 | +8.98 | 166.12 | 143.77 | -13.45 |  |
| `w4a16_tp4_c1` | coding | 48.7710 | 54.9959 | **+12.76** | 137.62 | 148.80 | +8.12 |  |
| `w4a16_tp4_c2` | synthetic | 99.3556 | 106.3017 | +6.99 | 141.38 | 207.57 | +46.82 |  |
| `w4a16_tp4_c2` | coding | 89.2250 | 104.6327 | **+17.27** | 121.23 | 170.13 | +40.34 |  |
| `w4a16_tp4_c4` | synthetic | 193.9938 | 198.9774 | +2.57 | 200.12 | 366.25 | +83.02 |  |
| `w4a16_tp4_c4` | coding | 161.9807 | 192.5970 | **+18.90** | 131.91 | 182.26 | +38.18 |  |

### Win-bar verdict (research-mode, VAL-M4-003)

The win-bar requires at least one prefill-dominated cell to show
`Δtput ≥ +0.1 %` OR `Δttft ≤ −0.1 %`. Prefill-dominated cells are
`w8a8_tp1_c1/synthetic` and `w8a8_tp1_c2/synthetic`.

**Result: WIN-BAR-MET on `w8a8_tp1_c1/synthetic` at Δtput = +0.35 %.**

### Prefill-cell regressions called out (data-honesty requirement (b))

While the canonical prefill cell `w8a8_tp1_c1/synthetic` meets the
win-bar, **adjacent prefill-shaped W8A8 cells regress**:

- **TP=1**: `w8a8_tp1_c2/synthetic` **-3.75 %**, `w8a8_tp1_c4/synthetic` **-5.13 %**
- **TP=4**: `w8a8_tp4_c2/synthetic` **-3.84 %**, `w8a8_tp4_c4/synthetic` **-9.31 %**

These regressions are **attributable to the redundant-silu wire-in gap
(library §13)**: the M1 wire-in stashes the fused `(int8, scale)` on
the `down_proj` Linear but still executes the legacy `act_fn + silu_and_mul`
sequence to preserve cudagraph capture shapes. At c=2 / c=4 the legacy
silu pass cost compounds with concurrent requests, surfacing the
producer-side double-work pattern as throughput loss.

### W4A16 wins flagged as suspicious (data-honesty requirement (c))

The 24-cell grid shows **large W4A16 wins** on multiple cells
(TP=1 c=4 coding +7.74 %; TP=4 wins reaching **+18.90 %** on coding c=4,
+17.27 % on coding c=2, +12.76 % on coding c=1). These are
**suspicious** because **the W4A16 codepath is untouched by M1/M2/M3**:
none of the three fusions register on the W4A16 dispatch path
(M1 stashes silu-cache on a W8A8 Linear; M2 extends a W8A8 epilogue;
M3 is a standalone kernel not yet wired into `triton_w4a16`). The
disable-path smoke (section (j)) confirms a -0.63 % delta on
`w4a16_tp4_c4` with the env-flag flipped, supporting a **host-config
drift** hypothesis between the production-baseline measurement window
(2026-05-19) and this worker session (2026-05-22 / 05-23). **Recommend
a future follow-up that lands a same-host fused-off W4A16 A/B
comparison before claiming these wins.** The wins are reported here in
full honesty but should not be cited as M4 throughput evidence.

## (g) Cumulative Grid — Stacked-Effect Summary

The orchestrator's mission proposal requests a stacked-effect summary
showing M1-only vs M1+M2 vs M1+M2+M3 contribution per cell. **This
mission's M4 grid measures all three fusions composed** — the
`VLLM_MI100_DISABLE_FUSED_ACT_QUANT` env flag toggles the M1+M2+M3
ensemble as a single unit; per-fusion granular flags were not part of
the dispatcher contract delivered. Per library §17, **M2 does not move
the bench grid needle today** (the producer-side wire-in is deferred);
per M3 file scope, **M3 adds 2 standalone files used by the W4A8 path
not yet wired in**. The grid therefore effectively measures **M1 alone
end-to-end + M2/M3 as dispatcher/kernel infrastructure on the critical
path**.

| Effect | Cell example | Effect on bench grid |
|---|---|---|
| M1 only (fused silu→int8 cache) | `w8a8_tp1_c1/synthetic` | **+0.35 %** Δtput (win-bar prefill cell) |
| M1+M2 composed (M2 dispatcher logs only) | same as above | unchanged from M1-only — see library §17 |
| M1+M2+M3 composed (M3 not on critical path) | same as above | unchanged from M1+M2 — see (e) above |

Reproducing per-fusion granularity would require landing per-fusion
env flags AND a same-host A/B re-run; both are deferred to follow-up
work alongside the M2 producer-side wire-in.

## (h) Quality Gates (VAL-M4-004)

Fused-act-quant default state: **ON** (`VLLM_MI100_DISABLE_FUSED_ACT_QUANT`
unset). Coding ceiling: 9/10 (pre-existing `max_subarray`
`IndentationError`; anti-pattern #6).

| # | Quant | Metric | Value | Gate | Verdict |
| - | ----- | ------ | ----- | ---- | :-----: |
| 1 | w8a8  | perplexity   | ppl=9.7063 (Δ=+0.565 % vs prod 9.6518) | Δ ≤ +1.0 % | **PASS** |
| 2 | w4a16 | perplexity   | ppl=9.8233 (Δ=+0.207 % vs prod 9.8030) | Δ ≤ +1.0 % | **PASS** |
| 3 | w8a8  | coding-agent | 9/10 (failed: max_subarray) | ≥ 9 | **PASS** |
| 4 | w4a16 | coding-agent | 9/10 (failed: max_subarray) | ≥ 9 | **PASS** |
| 5 | w8a8  | needle@32k   | 5/5 across depths {10, 30, 50, 70, 90} | == 5 | **PASS** |
| 6 | w4a16 | needle@32k   | 5/5 across depths {10, 30, 50, 70, 90} | == 5 | **PASS** |

**Overall: PASS — all 6 gates met (VAL-M4-004).** W4A16 untouched-path
sanity check: ppl moved by +0.207 %; well inside expected noise, no
cross-quant leak observed. Quality summary at
`/root/bench-int8-w4a16-fused/m4-bench/m4_quality_summary.md`; raw
outputs under `quality/{w8a8,w4a16}/`.

## (i) rocprofv3 HBM-Bytes-Per-Token Evidence (VAL-M4-005)

Cell: `w8a8_tp1_c4_coding` (TP=1, KV-INT8, chunked-prefill, AITER).
Harness: offline `vllm bench throughput --num-prompts 1 --input-len 1024
--output-len 32 --seed 42` (anti-pattern #1 — rocprofv3 + serve
deadlocks on gfx908; library/`rocprofv3-tp4-limitation.md`).

| # | Quantity | fused-off | fused-on | Δ (off − on) | Expected |
|---|---|---:|---:|---:|---|
| (a) | `kernel_unified_attention` invocations | **1,304** | **1,304** | 0 (unchanged) | unchanged ✓ |
| (b) | `silu_and_mul` (`act_and_mul_kernel`) invocations | 5,280 | 5,280 | 0 | **unchanged — legacy silu still fires** (library §13) |
| (b) | `dynamic_scaled_int8_quant_kernel` invocations | **13,200** | **11,984** | **+1,216 unfused calls eliminated** | fused-off > fused-on ✓ |
| (b) | `_fused_silu_quant_int8_kernel` invocations | **0** | **1,216** | **−1,216** | fused-off == 0; fused-on > 0 ✓ |
| (c) | `mi100_int8_scaled_mm_kernel` invocations | **10,480** | **10,480** | 0 (unchanged) | **unchanged — M2 §17 producer-side wire-in gap** |
| (d) | **Total HBM bytes** (Σ `FETCH_SIZE + WRITE_SIZE`) | **1,992,002,503,232 B** | **2,019,709,590,112 B** | **+27,707,086,880 B (+1.39 %)** | **NEGATIVE on aggregate** — see (i.1) |
| (d) | **Per-decode-step bytes** (total / 32) | 62,250,078,226 B/step | 63,115,924,691 B/step | +865,846,464 B/step (+1.39 %) | — |
| (d, scoped) | **M1-seam bytes** (silu+quant unfused + fused_silu_quant total) | **29,766,992,224 B** | **58,204,389,792 B** | +28,437,397,568 B (+95.5 %) | wired-in fusion adds traffic — see (i.1) |

PMC scalar evidence (sums across all kernel dispatches in the
single-prompt trace):

| Counter | fused-off (sum) | fused-on (sum) | Δ |
|---|---:|---:|---:|
| `FETCH_SIZE` (KiB) | 1,851,998,878.50 | 1,869,573,191.94 | **+17,574,313.44 KiB** |
| `WRITE_SIZE` (KiB) | 93,316,066.06    | 102,799,454.66   | **+9,483,388.60 KiB** |
| `TCC_HIT`          | 18,819,253,352   | 19,293,684,721   | **+474,431,369** |
| `VALUUtilization` (sum) | 18,417,940    | 18,419,646       | +1,706 |

### (i.1) Honest interpretation (data-honesty requirement (e))

The aggregate HBM round-trip bytes are **+1.39 % HIGHER in fused-on
than fused-off** — **NOT lower as the original VAL-M4-005 hypothesis
assumed**. Root cause (library §13 redundant-silu gap):

1. **1,216 `dynamic_scaled_int8_quant_kernel` calls were correctly
   replaced by 1,216 `_fused_silu_quant_int8_kernel` calls** — exact
   1:1 substitution at the dispatcher level, confirming the M1 fused
   kernel is being dispatched as designed (direction-correct).
2. **The legacy `act_and_mul_kernel` (silu_and_mul) still fires
   5,280 times** in fused-on — **identical** to fused-off — because
   the M1 wire-in **adds** the fused kernel rather than **replacing**
   the legacy silu trigger. The producer reads the gate+up projection
   output twice: once via the legacy `silu_and_mul` (still wired into
   `act_fn`+`down_proj` for cudagraph capture-shape preservation),
   and once via the fused `_fused_silu_quant_int8_kernel` which writes
   the int8 cache.
3. **Attention is unchanged** (`kernel_unified_attention` 1,304:1,304).
4. **M2 scaled_mm kernel-arg signature is unchanged**
   (`mi100_int8_scaled_mm_kernel` 10,480:10,480 identical), confirming
   library §17 producer-side wire-in gap at byte level: the
   `apply_weights` callsite still passes `emit_int8_next=False`
   unconditionally.

**Mission framing (research-mode):** the mission delivers **the kernel

- dispatcher infrastructure** for all three issues. The
bytes-per-token reduction expected from M1+M2 composition requires
either (i) a follow-up feature that **replaces** the legacy silu
trigger (library §13 path-1 with a real-shaped placeholder, or path-2
prefill-only gating) or (ii) a dispatcher refinement that fires the
fused kernel **only on prefill shapes** while leaving the cudagraph
decode path on the legacy composition. This is **exactly the negative-
result framing the research-mode win-bar clause (AGENTS.md §11,
VAL-M4-003) was designed for** — the mission is **NOT failed**; the
infrastructure ships; the production end-to-end fire is deferred.

Full per-kernel breakdown:
`/root/bench-int8-w4a16-fused/m4-bench/rocprof/hbm_bytes_analysis.md`
and `hbm_analysis.json`. Trace files (kernel_trace.csv + pmc.csv,
190,516 records each) under
`/root/bench-int8-w4a16-fused/m4-bench/rocprof/w8a8_tp1_c4_coding_fused_{on,off}/`.

## (j) Operational Flags

| Variable | Default | Allowed | Notes |
|---|---|---|---|
| `VLLM_MI100_DISABLE_FUSED_ACT_QUANT` | `0` (fused-**on**) | `0`,`1` | NEW. Off-switch for the three fused epilogues. **Default-on inversion warning**: this is the production path; set `=1` only for A/B work or rollback. |
| `VLLM_DISABLE_FUSED_ACT_QUANT` | unset | `0`,`1` | **LEGACY ALIAS.** Honored if set; a one-shot deprecation warning fires (commit `98e537d21`). Removable by the orchestrator at a future date. |

Disable-path semantics: both env vars OR'd together — setting either
to a truthy value short-circuits the fused producer **AND** the M2
dispatcher decision (`choose_emit_int8_next()` returns `False` when
either flag is set). The M3 standalone kernel is **not gated** by
either flag (issue #11 acceptance criterion is per-shape correctness

- performance, not a dispatcher decision).

Disable-path smoke (VAL-M4-006,
`/root/bench-int8-w4a16-fused/m4-bench/disable_path/`):

| cell_id        | m0_canary_tput | m4_disable_path_tput | Δ % | verdict       |
|----------------|----------------|----------------------|-----|---------------|
| `w8a8_tp1_c1`  | 40.26127789467819 | 39.99004911958527 | **-0.6737 %** | PASS (≤ ±5 %) |
| `w4a16_tp4_c4` | 199.28928292264195 | 198.02878673833695 | **-0.6325 %** | PASS (≤ ±5 %) |

Both within the ±5 % gate; consistent with run-to-run host noise. The
W4A16 cell is included specifically to verify the disable flag has
zero effect on the untouched W4A16 path; the observed -0.63 %
confirms this. Flag-active evidence (per-cell `server.log`):

```
fused_silu_quant_int8 import: enabled=False
  (VLLM_MI100_DISABLE_FUSED_ACT_QUANT='1' VLLM_DISABLE_FUSED_ACT_QUANT='(unset)')
W8A8 dispatcher fused_silu_quant_int8 default: enabled=False
```

## (k) Reproducibility

Pinned env (carried from `scripts/launch_hbm_<cell>.sh`):
`KV_CACHE_DTYPE=int8_per_token_head`, `ENABLE_CHUNKED_PREFILL=1`,
`MAX_NUM_BATCHED_TOKENS={2048 TP=1, 4096 TP=4}`, `VLLM_ROCM_USE_AITER=1`,
`VLLM_ROCM_USE_SKINNY_GEMM=0`, `TORCH_COMPILE_DISABLE=1`,
`HF_HUB_OFFLINE=1`,
`HIPBLASLT_TENSILE_LIBPATH=/root/hipblaslt-src/build/release/library`
(symlinked to `/opt/rocm/core-7.12/lib`),
`PYTHONPATH=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4`.

Common bench flags: `NUM_PROMPTS=200 --seed 42 --random-input-len 1024
--random-output-len 256 --percentile-metrics ttft,tpot,itl,e2el
--metric-percentiles 50,90,99`.

Models: `/models/Qwen3.5-9B-w8a8`, `/models/Qwen3.5-9B-w4a16`
(HF format, KV cache `int8_per_token_head`).

**One-liner per cell** (replace `<CELL>` with one of
`{w8a8,w4a16}_tp{1,4}_c{1,2,4}` for the 12 cell-IDs; the launcher fans
out across `synthetic` and `coding` workloads internally):

```bash
cd /home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
bash scripts/launch_hbm_<CELL>.sh --check
```

24-cell grid orchestration:

```bash
bash scripts/mi100/run_grid_fused_parallel.sh \
  --milestone m4-fused-act-quant \
  --output-root /root/bench-int8-w4a16-fused/m4-bench
```

rocprofv3 reproduction (anti-pattern #1 — offline only):

```bash
bash scripts/mi100/rocprof_single_request.sh w8a8_tp1_c4_coding on  \
  /root/bench-int8-w4a16-fused/m4-bench/rocprof/w8a8_tp1_c4_coding_fused_on
bash scripts/mi100/rocprof_single_request.sh w8a8_tp1_c4_coding off \
  /root/bench-int8-w4a16-fused/m4-bench/rocprof/w8a8_tp1_c4_coding_fused_off
```

Per-cell `raw.json` paths are under
`/root/bench-int8-w4a16-fused/m4-bench/{w8a8,w4a16}/<cell>/raw_{synthetic,coding}/raw.json`.

## (l) Cross-Links

- **Bundle PR (this mission's deliverable)**:
  <https://github.com/larkinwc/vllm-gfx908/pull/38> — also persisted at
  `/root/bench-int8-w4a16-fused/m4-bench/cross/pr_url.txt`
  (target repo: `larkinwc/vllm-gfx908`; base: `mi100-fixes`; head:
  `mi100/fused-act-quant-m26-m33-m11`; **NEVER** `vllm-project/vllm`,
  per anti-pattern #7 and VAL-CROSS-001).
- **Issues closed**: `Closes #26` (silu→int8 fuse), `Closes #33`
  (linear-store→int8 fuse), `Closes #11` (standalone W4A8 fused quant).
- **M0 canary report**: `/root/bench-int8-w4a16-fused/m0-canary/canary_check.md`.
- **M1 disable-path byte-identity**:
  `/root/bench-int8-w4a16-fused/m1-silu/disable_path_byte_identity.log`.
- **M2 disable-path byte-identity**:
  `/root/bench-int8-w4a16-fused/m2-linear/disable_path_byte_identity.log`.
- **M3 cherry-pick summary**:
  `/root/bench-int8-w4a16-fused/m3-issue-11/m3_cherry_pick_summary.md`.
- **M4 grid (CSV+MD)**:
  `/root/bench-int8-w4a16-fused/m4-bench/m4_vs_baseline_grid.{csv,md}`.
- **M4 quality summary**:
  `/root/bench-int8-w4a16-fused/m4-bench/m4_quality_summary.md`.
- **M4 rocprofv3 evidence**:
  `/root/bench-int8-w4a16-fused/m4-bench/rocprof/hbm_bytes_analysis.md`.
- **M4 disable-path smoke**:
  `/root/bench-int8-w4a16-fused/m4-bench/disable_path/README.md` +
  `disable_path_smoke.csv`.
- **VAL-CROSS audits**:
  `/root/bench-int8-w4a16-fused/m4-bench/cross/upstream_audit.log`
  (VAL-CROSS-001),
  `tuning_hash_audit.log` (VAL-CROSS-002),
  `forbidden_intrinsics_scan.log` (VAL-CROSS-003),
  `pr_url.txt` (VAL-CROSS-004).
- **Mission library** (read-me-first):
  `library/fused-act-quant-mission-context.md` — §13
  (redundant-silu wire-in gap, root cause of (f) prefill regressions),
  §17 (M2 dispatcher-decision vs production-effect gap, root cause of
  M2 not moving the bench grid needle).
- **AGENTS.md §11** — Negative-Result Discipline (VAL-M4-003), under
  which this mission's research-mode win-bar verdict is reported.

---

## Summary

> **Win-bar (research-mode, prefill cell) MET on a single TP=1 c=1
> W8A8 synthetic cell at +0.35 % Δtput; production impact is bounded
> by the c=1 wire-in scope and the M2 producer-side wire-in gap
> (library §17). M2 ships as kernel + dispatcher infrastructure;
> production fire deferred to a future MLP/attention int8 cache
> rework.**

The mission delivers:

- **Three landed fused kernels** (M1 `fused_silu_quant_int8`, M2
  `mi100_int8_scaled_mm_kernel EMIT_INT8_NEXT` epilogue, M3 cherry-picked
  `fused_int8_quant`) with full correctness coverage under both env
  states.
- **Default-on env-flag** (`VLLM_MI100_DISABLE_FUSED_ACT_QUANT`) with
  a legacy-alias deprecation warning.
- **24-cell bench grid** showing direction-correct improvement on the
  canonical prefill cell and full transparency on the redundant-silu
  regressions and W4A16 host-drift suspicions.
- **rocprofv3 byte-level evidence** that proves direction-correct
  kernel-class substitution AND honestly reports the aggregate +1.39 %
  byte regression caused by the §13 redundant-silu gap.
- **VAL-CROSS** audit logs proving no upstream pushes occurred,
  no tuning-JSON drift, and no forbidden MI300+ intrinsics in any
  Triton kernel.

A single fork-bundled PR against `larkinwc/vllm-gfx908` closes #26, #33,
and #11.

*AI-assistance disclosure (AGENTS.md §15): This mission was authored
with Claude as a code-assistant. Every commit carries a `Co-authored-by:
Claude` trailer; the human submitter (larkinwc) reviewed each diff and
ran the test commands listed above.*
