# BENCH_REDUNDANT_SILU — Negative-Result Publication

**Mission:** `mi100/redundant-silu-elimination` (`48a4a1a2-a6ab-4792-95b0-414d0007be24`)
**Branch:** `mi100/redundant-silu-elimination` (cut from `bd26444b9`, PR #38 merge)
**Worktree:** `cold-points-sit-rancb`
**Report date:** 2026-05-25
**Author note:** This document is published per user decision **Option A** (NEGATIVE-RESULT publication) made on 2026-05-25 after M3-F3 rocprofv3 evidence overturned the M1-F4 directional reading. The M1-promoted placeholder-view code is **REVERTED** in M3-F6; only this report, the rocprofv3 evidence, and the W4A16 retraction land in the bundled fork PR.

---

## 1. Executive Summary

The win-bar attestation for this mission is **NOT MET**. The M1-promoted `placeholder-view` redundant-silu-elimination code does **not** reduce HBM `FETCH+WRITE` on the gating cell `w8a8_tp1_c4_coding` — it **regresses** by **+0.652 %** (`2,004.30 GB` fused-on vs `1,991.32 GB` fused-off; Δ = +12.98 GB). The win-bar (mission AGENTS.md §7, "moderate: fused-on FETCH+WRITE MUST be ≤ fused-off, Δ ≤ 0 %") therefore fails.

Per user decision option A:

- The M1 placeholder-view code (commits `8e8271154`, `eb9328367`, `59d566bf3`) is **REVERTED** in M3-F6.
- Only this report (`BENCH_REDUNDANT_SILU.md`), the rocprofv3 evidence under
  `/root/bench-int8-w4a16-redundant-silu/m3-bench/rocprof/`, and the W4A16
  same-host A/B retraction (`library/w4a16-ab-audit.md`) land in the
  bundled fork PR to `larkinwc/vllm-gfx908`.
- No `vllm/` code change is published from this mission. Production stays on the legacy redundant-silu path landed by PR #38.

The W4A16 throughput wins of +2.4 % to +18.9 % published in `BENCH_FUSED_ACT_QUANT.md` §(f) are **all retracted** under same-host A/B. Zero W4A16 cells survive confirmation. The fused-act-quant kernel from PR #38 delivers no measurable W4A16 throughput benefit on this host.

The W8A8 quality gates landed mixed: perplexity PASS, needle@32k PASS, coding eval FAIL (8/10, reproducibly) on the M1-promoted code; the coding regression is a real semantic regression in `anagram` that did not exist on the prior M6 baseline (9/10).

---

## 2. rocprofv3 Evidence — Reversal Narrative

### 2.1 M1-F4 directional reading (false confidence)

The M1-F4 evidence sweep ran three modes against the **same legacy baseline that
itself had both paths firing** (i.e., the prior-mission redundant-silu shape).
Aggregator: `/root/bench-int8-w4a16-redundant-silu/m1-evidence/aggregate_m1_silu.py`.

| mode          | FETCH+WRITE (bytes)   | Δ vs legacy | act_and_mul | fused_silu_quant |
| ------------- | --------------------: | ----------: | ----------: | ---------------: |
| legacy        | 2,019,509,422,528     | (baseline)  | 5,280       | 1,216            |
| placeholder   | 2,004,936,713,728     | **−0.72 %** | 4,064       | 1,216            |
| prefill-gate  | 2,009,566,863,680     | −0.49 %     | 5,280       |    64            |

This `−0.72 %` delta on the placeholder-view path was the M1-F5 promotion
trigger: it was strictly lower FETCH+WRITE *and* strictly lower `act_and_mul`
invocations (4,064 vs 5,280) against the same-mission legacy baseline. The
placeholder path was promoted to unconditional firing in commit `59d566bf3`
and the `VLLM_MI100_SILU_ELIMINATION_MODE` env switch was deleted.

**This number was misleading.** The legacy baseline in M1-F4 ran both the
fused producer *and* the legacy `act_and_mul`, so the M1-F4 comparison
measured "remove `act_and_mul` while keeping the fused producer" against
"keep both", not "remove `act_and_mul` while keeping the fused producer"
against the production fused-off shape (no fused producer, only legacy
`act_and_mul` + `scaled_int8_quant`).

### 2.2 M3-F3 re-capture (true comparison; win-bar fails)

The M3-F3 capture re-ran rocprofv3 against the M1-promoted code (commit
`59d566bf3`, placeholder-view default) versus the **legitimate disable-path
fused-off shape** (`VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`, no fused producer
at all). Aggregator: `/root/bench-int8-w4a16-redundant-silu/m3-bench/rocprof/aggregate_m3f3.py`.

| state                                | FETCH+WRITE (bytes)   | act_and_mul | fused_silu_quant | scaled_int8_quant |
| ------------------------------------ | --------------------: | ----------: | ---------------: | ----------------: |
| fused-on (M1 placeholder, default)   | 2,004,304,435,328     | 4,064       | 1,216            | 11,984            |
| fused-off (legacy redundant silu off)| 1,991,320,185,664     | 5,280       | 0                | 13,200            |

Δ (fused-on − fused-off) = **+12,984,249,664 bytes = +0.652 %** → win-bar **FAIL**.

In bytes, fused-on = **2,004 GB FETCH+WRITE**; fused-off = **1,991 GB FETCH+WRITE**.

### 2.3 Why M1-F4 was deceptive

The M1-F4 baseline included the fused producer firing redundantly; eliminating
the legacy `act_and_mul` in that shape *did* shrink HBM traffic by 0.72 %
because it removed one of two redundant compute paths. But the *correct*
counterfactual on this hardware is the disable-path shape, where the legacy
`act_and_mul` is the *only* path. Against that shape, the placeholder-view
fused producer *adds* HBM traffic instead of saving it: the producer's WRITE
of the int8-quantized output plus a stride-aligned read of the placeholder
view costs **+6.6 GB FETCH and +6.4 GB WRITE** more than the legacy
`silu_and_mul` → `scaled_int8_quant` composition consumes on this MI100
silicon.

The +0.652 % regression is small in relative terms but the win-bar (set by the
mission AGENTS.md §7 as `Δ ≤ 0 %`) is breakeven-or-better. The breakeven bar
is not cleared.

### 2.4 Sources

- `/root/bench-int8-w4a16-redundant-silu/m3-bench/rocprof/hbm_bytes.json` (M3-F3)
- `/root/bench-int8-w4a16-redundant-silu/m3-bench/rocprof/delta.md` (M3-F3 narrative)
- `/root/bench-int8-w4a16-redundant-silu/m1-evidence/hbm_bytes.json` (M1-F4)
- `/root/bench-int8-w4a16-redundant-silu/m1-evidence/summary.md` (M1-F4 narrative)
- `/root/bench-int8-w4a16-redundant-silu/m3-bench/rocprof/{on,off}/{kernel_trace.csv,pmc.csv,pmc_1/,pmc_2/}` (raw)

---

## 3. W4A16 A/B Verdict — Retraction Summary (M2)

Source same-host A/B audit: `library/w4a16-ab-audit.md`.

Fused-on JSONs: `/root/bench-int8-w4a16-fused/m4-bench/w4a16/<cell>.json` (prior PR #38).
Fused-off JSONs: `/root/bench-int8-w4a16-redundant-silu/m2-bench/w4a16/<cell>.json` (this mission, M2-F1, same host, `VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`).

### 3.1 Per-cell same-host A/B Δ (off − on)

| cell                       | tput_on | tput_off | Δ tput (%) | ttft_on | ttft_off | Δ ttft (%) |
| -------------------------- | ------: | -------: | ---------: | ------: | -------: | ---------: |
| w4a16_tp1_c1_coding        |  30.395 |   30.785 |   +1.29 %  |  353.83 |   340.61 |   −3.74 %  |
| w4a16_tp1_c1_synthetic     |  30.456 |   30.829 |   +1.23 %  |  350.35 |   336.23 |   −4.03 %  |
| w4a16_tp1_c2_coding        |  55.754 |   55.659 |   −0.17 %  |  426.55 |   388.43 |   −8.94 %  |
| w4a16_tp1_c2_synthetic     |  56.690 |   56.897 |   +0.37 %  |  500.75 |   493.00 |   −1.55 %  |
| w4a16_tp1_c4_coding        | 100.205 |   99.066 |   −1.14 %  |  441.96 |   420.80 |   −4.79 %  |
| w4a16_tp1_c4_synthetic     | 103.788 |  103.582 |   −0.20 %  | 1000.98 |  1003.27 |   +0.23 %  |
| w4a16_tp4_c1_coding        |  54.996 |   54.994 |   −0.00 %  |  148.80 |   147.53 |   −0.85 %  |
| w4a16_tp4_c1_synthetic     |  55.403 |   55.391 |   −0.02 %  |  143.77 |   142.37 |   −0.98 %  |
| w4a16_tp4_c2_coding        | 104.633 |  104.254 |   −0.36 %  |  170.13 |   171.13 |   +0.58 %  |
| w4a16_tp4_c2_synthetic     | 106.302 |  106.451 |   +0.14 %  |  207.57 |   207.62 |   +0.03 %  |
| w4a16_tp4_c4_coding        | 192.597 |  193.217 |   +0.32 %  |  182.26 |   184.37 |   +1.15 %  |
| w4a16_tp4_c4_synthetic     | 198.977 |  199.902 |   +0.46 %  |  366.25 |   367.95 |   +0.47 %  |

Max |Δ tput| across all 12 cells = 1.29 %, well inside same-host run-to-run
variance reported by M0-F4 canary (±2 %).

### 3.2 Retractions (all 12 prior wins)

Retraction rule: prior cross-host Δ_tput ≥ +2 % AND same-host A/B Δ_tput
(off − on) ≤ +0.5 % ⇒ RETRACT. Confirmation rule: same-host Δ_tput ≤ −2 %
(fused-on materially faster) ⇒ CONFIRM.

| cell                 | workload  | prior cross-host Δ | same-host Δ | verdict   |
| -------------------- | --------- | -----------------: | ----------: | --------- |
| w4a16_tp1_c1         | synthetic |           +5.72 %  |    +1.23 %  | RETRACT   |
| w4a16_tp1_c1         | coding    |           +7.34 %  |    +1.29 %  | RETRACT   |
| w4a16_tp1_c2         | synthetic |           +3.69 %  |    +0.37 %  | RETRACT   |
| w4a16_tp1_c2         | coding    |           +6.45 %  |    −0.17 %  | RETRACT   |
| w4a16_tp1_c4         | synthetic |           +2.36 %  |    −0.20 %  | RETRACT   |
| w4a16_tp1_c4         | coding    |           +7.74 %  |    −1.14 %  | RETRACT   |
| w4a16_tp4_c1         | synthetic |           +8.98 %  |    −0.02 %  | RETRACT   |
| w4a16_tp4_c1         | coding    |          +12.76 %  |    −0.00 %  | RETRACT   |
| w4a16_tp4_c2         | synthetic |           +6.99 %  |    +0.14 %  | RETRACT   |
| w4a16_tp4_c2         | coding    |          +17.27 %  |    −0.36 %  | RETRACT   |
| w4a16_tp4_c4         | synthetic |           +2.57 %  |    +0.46 %  | RETRACT   |
| w4a16_tp4_c4         | coding    |          +18.90 %  |    +0.32 %  | RETRACT   |

**All 12 prior W4A16 wins from `BENCH_FUSED_ACT_QUANT.md` §(f) lines 223–234
are retracted. Zero confirmed wins.** The prior cross-host wins were
cross-host drift between the 2026-05-19 baseline measurement window and the
2026-05-22/23 worker session, not fused-kernel benefit.

---

## 4. M3-F1 Per-Cell Delta Table vs Production Baseline

Source: `/root/bench-int8-w4a16-redundant-silu/m3-bench/delta_table.md`.

This table compares the **M1-promoted (placeholder-view) M3 grid** against the prior
mission's per-cell production baselines at
`/root/bench-int8-w4a16/baseline/raw_*/raw.json`. It is **cross-host** in the same
sense as `BENCH_FUSED_ACT_QUANT.md` §(f) and therefore subject to the same
cross-host drift caveat — the W4A16 broad wins below are **not** evidence that
the M1 code is faster; they replicate the cross-host pattern of the retracted
prior wins and dissolve under same-host A/B (§3 above).

### 4.1 W8A8 (12 cells) — mixed; 3 synthetic regressions

| cell                  | wl        | baseline_tput | M3_tput | Δ_tput   | Δ_ttft    |
| --------------------- | --------- | ------------: | ------: | -------: | --------: |
| w8a8_tp1_c1           | synthetic |       38.965  |  39.397 |  +1.11 % |  +13.93 % |
| w8a8_tp1_c1           | coding    |       38.055  |  39.160 |  +2.90 % |  +56.89 % |
| w8a8_tp1_c2           | synthetic |       73.812  |  72.141 |  −2.26 % |   +7.73 % |
| w8a8_tp1_c2           | coding    |       68.612  |  70.752 |  +3.12 % |  +41.21 % |
| w8a8_tp1_c4           | synthetic |      135.902  | 127.729 |  −6.01 % |  −3.30 %  |
| w8a8_tp1_c4           | coding    |      120.357  | 123.319 |  +2.46 % |  +29.36 % |
| w8a8_tp4_c1           | synthetic |       63.948  |  63.234 |  −1.12 % |   +0.99 % |
| w8a8_tp4_c1           | coding    |       60.981  |  62.560 |  +2.59 % |  +28.12 % |
| w8a8_tp4_c2           | synthetic |      125.297  | 120.586 |  −3.76 % |  +78.07 % |
| w8a8_tp4_c2           | coding    |      110.082  | 118.677 |  +7.81 % |  +62.46 % |
| w8a8_tp4_c4           | synthetic |      247.738  | 225.564 |  −8.95 % |  +69.35 % |
| w8a8_tp4_c4           | coding    |      196.290  | 216.103 | +10.09 % |  +68.69 % |

W8A8 Δ_tput stats: min = −8.95 %, max = +10.09 %, mean = +0.66 %.

**Three synthetic regressions ≤ −3 %**:
`w8a8_tp1_c4_synthetic` (−6.01 %), `w8a8_tp4_c2_synthetic` (−3.76 %),
`w8a8_tp4_c4_synthetic` (−8.95 %). These are W8A8 cells where the M1 code
either adds HBM traffic (consistent with §2.2's +0.652 % regression on
`w8a8_tp1_c4_coding`) or slows TTFT enough to dominate the throughput
denominator over 200 prompts.

### 4.2 W4A16 (12 cells) — broad cross-host wins; **all dissolve under same-host A/B (§3)**

| cell                  | wl        | baseline_tput | M3_tput | Δ_tput   | Δ_ttft    |
| --------------------- | --------- | ------------: | ------: | -------: | --------: |
| w4a16_tp1_c1          | synthetic |       28.807  |  30.755 |  +6.76 % |  −7.59 %  |
| w4a16_tp1_c1          | coding    |       28.316  |  30.698 |  +8.41 % | +29.10 %  |
| w4a16_tp1_c2          | synthetic |       54.674  |  56.565 |  +3.46 % | −15.43 %  |
| w4a16_tp1_c2          | coding    |       52.377  |  55.639 |  +6.23 % | +32.05 %  |
| w4a16_tp1_c4          | synthetic |      101.393  | 104.317 |  +2.88 % |  −2.54 %  |
| w4a16_tp1_c4          | coding    |       93.008  | 100.642 |  +8.21 % |  +8.46 %  |
| w4a16_tp4_c1          | synthetic |       50.839  |  55.435 |  +9.04 % | −14.16 %  |
| w4a16_tp4_c1          | coding    |       48.771  |  55.027 | +12.83 % |  +7.13 %  |
| w4a16_tp4_c2          | synthetic |       99.356  | 106.039 |  +6.73 % | +46.41 %  |
| w4a16_tp4_c2          | coding    |       89.225  | 104.233 | +16.82 % | +40.51 %  |
| w4a16_tp4_c4          | synthetic |      193.994  | 198.414 |  +2.28 % | +82.86 %  |
| w4a16_tp4_c4          | coding    |      161.981  | 191.955 | +18.50 % | +47.25 %  |

W4A16 Δ_tput stats: min = +2.28 %, max = +18.50 %, mean = +8.51 %.

**These cross-host W4A16 wins are exactly the pattern §3 retracts.** They do
not reflect a fused-kernel benefit on the M1-promoted code; they reflect
cross-host drift. Treat them as evidence of cross-host drift magnitude
(+2 % to +18 %), not as performance wins.

---

## 5. Quality Gates (M3-F2)

Source: `library/quality-gates.md`. Server: TP=1, single GPU,
`--gpu-memory-utilization 0.90`, `--max-model-len 32768`,
`--enable-prefix-caching`, `--block-size 32`, `--dtype float16`, ROCm 7.12,
AITER + skinny-gemm-off + TensileLite merged library + tuning JSONs.
Codepath: `mi100/redundant-silu-elimination` @ `7239a3cf8` (M1-promoted
placeholder-view default).

| Gate                                                                | Threshold              | This run                     | Verdict |
| ------------------------------------------------------------------- | ---------------------- | ---------------------------- | ------- |
| W8A8 perplexity (wikitext-2-raw-v1, 50×512 chunks, seed=0)          | ≤ +1 % over 9.6518     | 9.6936 (Δ = +0.43 %)         | PASS    |
| Coding-agent eval, fused-off (10 prompts, T=0, seed=0)              | ≥ 9 / 10               | 9 / 10                       | PASS    |
| Coding-agent eval, fused-on / M1-promoted (10 prompts, T=0, seed=0) | ≥ 9 / 10               | 8 / 10 (two independent runs)| FAIL    |
| Needle-in-haystack @ ctx 32 768 (5 needles)                         | 5 / 5                  | 5 / 5                        | PASS    |

### 5.1 Coding fused-on FAIL — `anagram` T=0 first-fence flip (M3-F2 diagnostic disclosure)

Two independent runs over `tests/eval/coding_prompts.json` against the same
warm M1-promoted server, each at `temperature=0, seed=0`, both produced 8/10.
Failures: `anagram` (NEW vs prior M6 baseline) and `max_subarray` (pre-existing
IndentationError, unchanged).

`anagram` failure body (`m6_coding_eval_m3_w8a8.json`):

```python
def is_anagram(a: str, b: str) -> bool:
    def normalize(s: str) -> str:
        return "".join(s.lower().split())
    return normalize(a) == normalize(b)
```

This implementation strips whitespace and lowercases but does **not** compare
character multisets, so `is_anagram("listen", "silent") → False`. The prior
M6 W8A8 run on the same prompt produced a correct sorted-string or
character-count comparison (9/10 overall, with only `max_subarray` failing).

The M3-F2 worker confirmed the regression is reproducible at T=0, seed=0
across two consecutive runs against the same warm server — i.e., the
M1-promoted placeholder-view path produces a deterministically *different
first-fenced code block* on this prompt than the prior M6 W8A8 baseline.
This is a real semantic regression in coding behavior, not a harness
artifact. Combined with the §2.2 win-bar failure, this fully justifies the
revert decision.

### 5.2 Pre-existing `max_subarray` failure (unchanged)

Same `IndentationError` already present in the M6 baseline
(`/root/bench-int8-w4a16/final/m6_coding_eval_m6_w8a8.json`:
`compiles=False, runs=False, correct=False`). Not introduced by this mission.

### 5.3 Disable-path smoke (M3-F4) — also FAIL

`scripts/mi100/disable_path_smoke.sh` w8a8_tp4_c4 canary measured 208.53 tok/s
vs M0 canary 224.76 tok/s (Δ = −7.22 %), outside the ±2 % gate. Tracked in
`/root/bench-int8-w4a16-redundant-silu/m3-disable-smoke/results.md`;
contributing rationale for revert is the M1-promoted code's measurable
regressive effect on the disable-path canary itself.

---

## 6. Win-Bar Attestation — RETRACTED

**Win-bar (mission AGENTS.md §7, moderate):** fused-on `FETCH_SIZE + WRITE_SIZE`
on `w8a8_tp1_c4_coding` MUST be ≤ fused-off; breakeven (Δ = 0 %) is acceptable;
positive Δ FAILS the mission.

**Result:** Δ = **+0.652 %** (fused-on = 2,004,304,435,328 bytes; fused-off =
1,991,320,185,664 bytes; +12,984,249,664 bytes regression).

**Win-bar verdict: RETRACTED (FAIL).**

**Rationale:** The M1-F4 directional reading of −0.72 % was measured against a
baseline that itself ran both compute paths redundantly. Re-measured against
the legitimate disable-path fused-off shape in M3-F3, the M1-promoted
placeholder-view path adds HBM traffic instead of saving it. Combined with the
W8A8 coding gate failure (§5.1, anagram regression) and the disable-path
canary failure (§5.3), there is no surviving evidence to support promotion of
the M1 code, and the mission's only honest publishable artifact is the
negative-result rocprofv3 evidence plus the W4A16 retraction.

---

## 7. Mission Outcome — Revert + Documentation-Only PR

The M1 redundant-silu-elimination code is **REVERTED** in M3-F6.

What lands in the bundled fork PR to `larkinwc/vllm-gfx908`:

1. **This report** (`BENCH_REDUNDANT_SILU.md`) — negative-result publication
   with rocprofv3 evidence, quality-gate disclosure, and win-bar retraction.
2. **rocprofv3 evidence** under
   `/root/bench-int8-w4a16-redundant-silu/m3-bench/rocprof/` — referenced
   from this report; the raw kernel_trace.csv / pmc.csv artifacts stay on the
   bench host (off-tree).
3. **W4A16 A/B retraction** (`library/w4a16-ab-audit.md`) — explicit retraction
   of all 12 prior W4A16 wins from PR #38.

What does NOT land in the fork PR:

- Any `vllm/` code change from this mission (M1 placeholder-view kernel-call
  rework is reverted; `silu_and_mul` continues to fire alongside the fused
  producer as in PR #38).
- The `VLLM_MI100_SILU_ELIMINATION_MODE` env switch (was removed in M1-F5,
  stays removed; only the legacy `VLLM_MI100_DISABLE_FUSED_ACT_QUANT`
  remains in the tree).
- Any quality-gate waiver. The coding 8/10 failure stands as documented in
  §5.1; production sits on the legacy redundant-silu path that scored 9/10
  on the M6 baseline.

Closes:

- `Closes #<new-redundant-silu-issue>` — closed by this report's negative-result
  publication (the redundant-silu byte-loss is **documented** rather than
  fixed; no fix-path on MI100 silicon was found that beats the legacy
  composition).
- `Closes #<new-w4a16-ab-issue>` — closed by `library/w4a16-ab-audit.md` and
  §3 retractions above.

Issue numbers are filled in by the M3-F6 mission-ops worker against
`library/issue-tracking.md`.

---

## 8. AI-Assistance Disclosure (per AGENTS.md §1)

This mission, including all benchmark orchestration, rocprofv3 capture,
quality-gate evaluation, W4A16 A/B audit, and this report, was performed with
AI assistance (Claude). Per AGENTS.md §1:

- **Duplicate-work check** was run before filing both fork issues and before
  opening the bundled M3-F6 PR. Output is captured in the M3-F6 PR body and
  in `library/issue-tracking.md`.
- **Test commands** run by the mission (numerics suite, quality gates,
  rocprofv3 capture, disable-path smoke) are documented in
  `library/quality-gates.md` and the M3-F6 PR body.
- **AI-assistance statement** is included verbatim in the M3-F6 PR body. The
  submitting human reviews every changed line in this report and the
  W4A16 retraction artifact, and is responsible for defending the change
  end-to-end per AGENTS.md §1.
- **Co-author trailers** (`Co-authored-by: Claude`) are applied to all
  mission commits on `mi100/redundant-silu-elimination`.

This report itself is the AI-assistance disclosure artifact required by
AGENTS.md §1 for the M3-F6 PR; it discloses both the negative result and the
AI assistance used to obtain it.
