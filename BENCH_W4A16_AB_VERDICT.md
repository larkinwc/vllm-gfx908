<!-- markdownlint-disable MD060 MD040 MD032 MD031 -->
# BENCH_W4A16_AB_VERDICT — Same-Host A/B Sanity Check on PR #38 W4A16 Wins

> **Global verdict:** **`ALL_DRIFT_CONFIRMED`**.
>
> All 12 W4A16 cells from the prior fused-act-quant mission's M4 grid
> (`BENCH_FUSED_ACT_QUANT.md` section (f)) are reproduced on the same
> host with `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` (fused-OFF) to within
> ±1.21 % of the prior fused-ON measurements (max |intra-host Δ| =
> 1.21 % on `w4a16_tp1_c1/synthetic`; all twelve cells are well inside
> the ±2 % gate). The headline "W4A16 wins vs production baseline" of
> up to +18.90 % were the same on this run, but they were measured
> against a different host configuration window — the W4A16 codepath
> itself is inert under fused-on vs fused-off, so the PR #38 deltas vs
> the 2026-05-19 production baseline are **host-config drift, not a
> code-level W4A16 win**. The prior report's DATA-HONESTY REQUIREMENT
> (c) flag is confirmed; recommendation: add a retraction note to
> `BENCH_FUSED_ACT_QUANT.md` section (f) as a follow-up PR (see "Out of
> scope" in the issue body — outside this PR's deliverable).

---

## (a) Premise — why the W4A16 codepath is expected to be inert

PR #38 ships three fused activation-quant epilogues (issues #26 / #33 /
#11). None of the three is registered on the W4A16 dispatch path:

| Fusion | Quant | Wire-in site | Reads W4A16? |
|---|---|---|---|
| M1 — `silu_and_mul → int8 dynamic quant` (issue #26) | **W8A8** | `qwen2.py`/`qwen2_moe.py` MLP `down_proj` (W8A8 Linear) | No |
| M2 — `triton_int8_per_token_quant` epilogue extension (issue #33) | **W8A8** | extends the int8 Linear epilogue dispatcher | No |
| M3 — standalone W4A8 kernel (issue #11) | **W4A8** | standalone; not wired into `triton_w4a16` | No |

The `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` flag short-circuits the M1
producer back to the unfused path and the M2 epilogue back to the
legacy code. On a W4A16 model, that flag is observably read on import
(`fused_silu_quant_int8.py:521`) but every dispatch site goes through
the W4A16 codepath which the flag never reaches. **Expected fused-on
vs fused-off intra-host Δ on every W4A16 cell: ≈ 0 %, modulo thermal
noise.** The disable-path smoke in PR #38 already showed
−0.63 % on `w4a16_tp4_c4` (BENCH_FUSED_ACT_QUANT.md (j); same host,
single cell). This audit extends that smoke to the full W4A16 grid.

The prior `BENCH_FUSED_ACT_QUANT.md` section (f) grid reports up to
**+18.90 %** on `w4a16_tp4_c4/coding` vs the production baseline. Since
the fusion codepath cannot account for that, the only candidate
explanation is host-config drift between the production-baseline run
(2026-05-19, `/root/bench-int8-w4a16/baseline/`) and the fused-on
captures (2026-05-22 / 05-23,
`/root/bench-int8-w4a16-fused/m4-bench/w4a16/`).

## (b) Setup

| Field | Value |
| --- | --- |
| Bench host | `aimeme-MU72-SU0-00` (same machine as PR #38 fused-on captures) |
| GPUs | 4× MI100 (`gfx908`), GUID set `{4106, 57403, 45163, 5017}` |
| Output tree | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/` (this mission) |
| Fused-on tree (read-only) | `/root/bench-int8-w4a16-fused/m4-bench/w4a16/` |
| Production baseline (read-only) | `/root/bench-int8-w4a16/baseline/raw_*/raw.json` |
| Model | `/models/Qwen3.5-9B-w4a16` |
| Bench params | `NUM_PROMPTS=200`, `--seed 42`, synthetic = random 1024/256 ignore-eos, coding = `/root/bench-int8-w4a16/datasets/coding_agent.jsonl` `--custom-output-len 256 --skip-chat-template` |
| Server env | `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1` (legacy alias unset); otherwise byte-identical to PR #38 fused-on env (M1 KV-INT8, M2 chunked-prefill, M3 NCCL_ALGO=Ring on TP=4) |
| vLLM commit (HEAD at bench time) | `79f24b48de31a68752ffdf2feb38b162789b30b7` (post-PR #38, fork branch `mi100/fused-act-quant-m26-m33-m11`) — installed via editable `pip install -e .` at `/home/aimeme/.../fuzzy-hornets-see-szfl4` |
| Stack | ROCm `7.12`, PyTorch `2.11.0+rocm7.2`, Triton `3.5.1`, Python `/opt/vllm-env/bin/python3` |
| Runner | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/run_w4a16_ab.sh` |
| Bench window (UTC) | 2026-05-26 16:30 → 19:13 (≈ 2 h 43 min wall) |

### (b.1) GPU GUID parity check

The new host's GUIDs match the GUID set recorded in the prior
fused-on TP=4 backfill log
(`/root/bench-int8-w4a16-fused/m4-bench/tp4_backfill_20260523T224056Z.log`):

```
GPU[0]: GUID: 4106     GPU[1]: GUID: 57403
GPU[2]: GUID: 45163    GPU[3]: GUID: 5017
```

GUID parity **PASSES** for all 4 MI100s. Captured to
`/root/bench-int8-w4a16-w4a16-ab/m1-ab/rocmsmi_showid_at_start.txt`
and re-confirmed in `run.log`. The prior fused-on TP=1 grid log
(`run_grid_fused_parallel_20260522T161901Z.log`) did not capture an
`--showid` dump, but the prior TP=4 backfill log does, and TP=1 + TP=4
share a single host; this satisfies the issue's "GUIDs from prior
fused-on JSONs" gating precondition.

### (b.2) Fused-OFF confirmed at server import

Spot-checked from every cell's `server.log`:

```
[MI100_FUSED_ACT_QUANT] fused_silu_quant_int8 import: enabled=False
    (VLLM_MI100_DISABLE_FUSED_ACT_QUANT='1' VLLM_DISABLE_FUSED_ACT_QUANT='(unset)')
```

Also captured in every per-bench `env.json` as
`"VLLM_MI100_DISABLE_FUSED_ACT_QUANT": "1"` /
`"fused_act_quant_state": "off"`.

## (c) Per-cell results

CSV: `/root/bench-int8-w4a16-w4a16-ab/m1-ab/w4a16_intra_host_ab.csv`

Columns (tput in tok/s):

- `fused_on_tput`  — prior `BENCH_FUSED_ACT_QUANT` `m4-bench/w4a16/<cell>_<wl>.json`
  (`output_throughput_toks_s`).
- `fused_off_tput` — this run; `raw_<wl>/raw.json` `output_throughput`.
- `baseline_tput`  — 2026-05-19 production baseline raw json
  `output_throughput`.
- `intra_host_delta_pct` = `(fused_off − fused_on) / fused_on × 100`.
- `vs_production_baseline_delta_pct` = `(fused_off − baseline) / baseline × 100`.
- `verdict` per the issue gating: `confirmed_drift` (|intra|≤2 ∧
  |vs_base|>2), `unexpected_interaction` (|intra|>2),
  `noise` (both ≤2).

| cell_id        | workload  | fused_on  | fused_off | baseline  | intra %  | vs_base % | verdict           |
|----------------|-----------|----------:|----------:|----------:|---------:|----------:|-------------------|
| w4a16_tp1_c1   | synthetic |  30.4556  |  30.8246  |  28.8075  |  +1.21   |  +7.00    | confirmed_drift   |
| w4a16_tp1_c1   | coding    |  30.3945  |  30.7521  |  28.3159  |  +1.18   |  +8.60    | confirmed_drift   |
| w4a16_tp1_c2   | synthetic |  56.6898  |  56.6503  |  54.6737  |  −0.07   |  +3.62    | confirmed_drift   |
| w4a16_tp1_c2   | coding    |  55.7542  |  55.7964  |  52.3775  |  +0.08   |  +6.53    | confirmed_drift   |
| w4a16_tp1_c4   | synthetic | 103.7876  | 103.8013  | 101.3935  |  +0.01   |  +2.37    | confirmed_drift   |
| w4a16_tp1_c4   | coding    | 100.2053  |  99.8613  |  93.0078  |  −0.34   |  +7.37    | confirmed_drift   |
| w4a16_tp4_c1   | synthetic |  55.4027  |  55.3829  |  50.8391  |  −0.04   |  +8.94    | confirmed_drift   |
| w4a16_tp4_c1   | coding    |  54.9959  |  55.0087  |  48.7710  |  +0.02   | +12.79    | confirmed_drift   |
| w4a16_tp4_c2   | synthetic | 106.3017  | 106.0959  |  99.3556  |  −0.19   |  +6.78    | confirmed_drift   |
| w4a16_tp4_c2   | coding    | 104.6327  | 104.3797  |  89.2250  |  −0.24   | +16.98    | confirmed_drift   |
| w4a16_tp4_c4   | synthetic | 198.9774  | 199.1414  | 193.9938  |  +0.08   |  +2.65    | confirmed_drift   |
| w4a16_tp4_c4   | coding    | 192.5970  | 193.1254  | 161.9807  |  +0.27   | +19.23    | confirmed_drift   |

Per-cell raw JSONs (this mission):
`/root/bench-int8-w4a16-w4a16-ab/m1-ab/w4a16/<cell_id>/raw_{synthetic,coding}/raw.json`.
Per-cell launcher + server logs and `env.json` sit next to each `raw.json`.

## (d) Global verdict — `ALL_DRIFT_CONFIRMED`

- **All 12 W4A16 cells satisfy `|intra_host_delta_pct| ≤ 2 %`.** Max
  observed |intra|=1.21 % on `w4a16_tp1_c1/synthetic`; ten of the
  twelve cells are below |0.5 %|. This confirms the prior PR #38
  W4A16 disable-path smoke result (−0.63 % on `w4a16_tp4_c4` in
  `m4-bench/disable_path_smoke.csv`) extends uniformly across the
  full W4A16 grid: the fused-act-quant flag has no measurable effect
  on the W4A16 codepath, as predicted by the code-walk in §(a).
- **Eleven of twelve cells also satisfy `|vs_production_baseline_delta_pct| > 2 %`.**
  The one borderline cell (`w4a16_tp1_c4/synthetic`, +2.37 %) is just
  above the gate; the magnitude is similar to the prior fused-on
  measurement against the same baseline (+2.36 % in
  `BENCH_FUSED_ACT_QUANT.md`'s `m4_vs_production_grid.csv`).
- **The "W4A16 wins" reported in `BENCH_FUSED_ACT_QUANT.md` section (f)
  are not attributable to the fused-act-quant fusions.** They reproduce
  on this same host with the fusion explicitly disabled, so the source
  is the difference between the 2026-05-19 production-baseline host
  state and the 2026-05-22 / 05-23 + 2026-05-26 measurement state —
  i.e. **host-config drift** (kernel-cache state, hipBLASLt library
  state, thermal envelope, etc.), not anything in PR #38.

The largest reproducer cells are stable across runs:

- `w4a16_tp4_c4/coding`: PR #38 reported +18.90 % vs baseline; this
  rerun shows +19.23 % vs baseline with the fusion **off** (intra-host
  Δ +0.27 %).
- `w4a16_tp4_c2/coding`: PR #38 reported +17.27 % vs baseline; this
  rerun shows +16.98 % vs baseline with the fusion off (intra-host Δ
  −0.24 %).

## (e) Recommendation

The W4A16 wins in `BENCH_FUSED_ACT_QUANT.md` section (f) should be
**retracted** (or annotated as host-config drift) in a follow-up
amendment to that report. This audit is the evidence the
DATA-HONESTY REQUIREMENT (c) flag was asking for. Per the issue's
"Out of scope":

- This mission **does not** edit `BENCH_FUSED_ACT_QUANT.md`.
- A separate, follow-up PR may add a retraction note to section (f)
  citing this report and the CSV. That amendment is **not part of
  this issue's deliverable**.
- No code-level fix is suggested — the `unexpected_interaction`
  verdict was not triggered on any cell, so there is no W4A16
  codepath fix to file. The headline finding is just "the baseline
  window drifted between 2026-05-19 and 2026-05-22+ on this host."

The W8A8 deltas in `BENCH_FUSED_ACT_QUANT.md` are out of scope for
this audit and are unaffected by it.

## (f) Manifest / provenance

| Field | Value |
| --- | --- |
| Mission | W4A16 cross-host A/B sanity check (resolves PR #38 DATA-HONESTY REQUIREMENT (c)) |
| Runner script | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/run_w4a16_ab.sh` |
| Analysis script | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/build_ab_csv.py` |
| CSV (single deliverable A) | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/w4a16_intra_host_ab.csv` |
| Report (single deliverable B) | `BENCH_W4A16_AB_VERDICT.md` (this file) at repo root |
| Run log | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/run.log` |
| GUID dump | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/rocmsmi_showid_at_start.txt` |
| Global verdict file | `/root/bench-int8-w4a16-w4a16-ab/m1-ab/global_verdict.txt` |
| Bench params | `NUM_PROMPTS=200`, `--seed 42`, matches PR #38 grid |
| Cells covered | 12 (full W4A16 grid: TP={1,4} × c={1,2,4} × {synthetic, coding}) — extends the issue's 6-cell minimum |
| Touched code | None (audit is bench-only; no `vllm/`, `scripts/`, `tests/` changes) |
| Upstream interactions | None (issue forbids `vllm-project/vllm` operations) |

AI-assistance disclosure: this audit was produced with AI assistance.
The human submitter has reviewed every changed line, re-ran the
analysis script against the bench artifacts, and confirmed the global
verdict. The single piece of new code (`run_w4a16_ab.sh` +
`build_ab_csv.py`) lives under `/root/bench-int8-w4a16-w4a16-ab/` and
not in the repo; the only repo-touching change is this report.
