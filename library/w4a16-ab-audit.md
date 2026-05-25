# W4A16 A/B Audit: fused-on (M4) vs fused-off (M2 redundant-silu)

Source fused-on JSONs: `/root/bench-int8-w4a16-fused/m4-bench/w4a16/<cell>.json`  
Source fused-off JSONs: `/root/bench-int8-w4a16-redundant-silu/m2-bench/w4a16/<cell>.json`  
Keys: top-level `output_throughput_toks_s` and `mean_ttft_ms` (long-form).

Δ = fused-off − fused-on. Positive Δ throughput / negative Δ TTFT = fused-off wins.

<!-- markdownlint-disable MD060 -->
| cell | tput_on (tok/s) | tput_off (tok/s) | Δ tput (tok/s) | Δ tput (%) | ttft_on (ms) | ttft_off (ms) | Δ ttft (ms) | Δ ttft (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w4a16_tp1_c1_coding | 30.395 | 30.785 | +0.391 | +1.29% | 353.83 | 340.61 | -13.22 | -3.74% |
| w4a16_tp1_c1_synthetic | 30.456 | 30.829 | +0.373 | +1.23% | 350.35 | 336.23 | -14.12 | -4.03% |
| w4a16_tp1_c2_coding | 55.754 | 55.659 | -0.095 | -0.17% | 426.55 | 388.43 | -38.12 | -8.94% |
| w4a16_tp1_c2_synthetic | 56.690 | 56.897 | +0.208 | +0.37% | 500.75 | 493.00 | -7.75 | -1.55% |
| w4a16_tp1_c4_coding | 100.205 | 99.066 | -1.140 | -1.14% | 441.96 | 420.80 | -21.16 | -4.79% |
| w4a16_tp1_c4_synthetic | 103.788 | 103.582 | -0.206 | -0.20% | 1000.98 | 1003.27 | +2.28 | +0.23% |
| w4a16_tp4_c1_coding | 54.996 | 54.994 | -0.002 | -0.00% | 148.80 | 147.53 | -1.27 | -0.85% |
| w4a16_tp4_c1_synthetic | 55.403 | 55.391 | -0.011 | -0.02% | 143.77 | 142.37 | -1.41 | -0.98% |
| w4a16_tp4_c2_coding | 104.633 | 104.254 | -0.378 | -0.36% | 170.13 | 171.13 | +0.99 | +0.58% |
| w4a16_tp4_c2_synthetic | 106.302 | 106.451 | +0.149 | +0.14% | 207.57 | 207.62 | +0.05 | +0.03% |
| w4a16_tp4_c4_coding | 192.597 | 193.217 | +0.620 | +0.32% | 182.26 | 184.37 | +2.10 | +1.15% |
| w4a16_tp4_c4_synthetic | 198.977 | 199.902 | +0.925 | +0.46% | 366.25 | 367.95 | +1.71 | +0.47% |
<!-- markdownlint-enable MD060 -->

## Retraction Rule

Prior-mission W4A16 wins (BENCH_FUSED_ACT_QUANT.md §(f) lines 223-234) were
measured fused-on (this host) vs a **cross-host** production baseline at
`/root/bench-int8-w4a16/baseline/raw_*/raw.json`. The same-host A/B above
re-runs the SAME fused-on artifact against a fused-off baseline on the SAME
host (`VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`).

A prior win is **retracted** when:

- prior Δ vs cross-host baseline was ≥ +2 % (throughput), AND
- same-host A/B Δ_tput (fused-off − fused-on) is ≤ +0.5 % — i.e., fused-on
  does NOT materially outperform fused-off on the same host (the prior
  "win" is indistinguishable from same-host run-to-run noise and is
  therefore attributable to cross-host drift, not the fused kernel).

A prior win is **confirmed** when:

- prior Δ vs cross-host baseline was ≥ +2 %, AND
- same-host A/B Δ_tput (fused-off − fused-on) is ≤ −2 % (fused-on
  materially faster than fused-off on the same host).

Cells failing both criteria (e.g., same-host Δ between −2 % and +0.5 %) are
**retracted by default** — there is no same-host evidence to support the
prior claim.

## Retracted Wins

All 12 prior W4A16 wins are retracted. Every cell had a prior cross-host
Δ_tput ≥ +2 % but a same-host A/B Δ_tput within ±1.29 % (well inside the
retraction band of ≤ +0.5 % and far above the confirmation band of
≤ −2 %).

<!-- markdownlint-disable MD060 -->
| cell | workload | prior Δ_tput vs cross-host | same-host A/B Δ_tput (off − on) | verdict |
|---|---|---:|---:|:-:|
| w4a16_tp1_c1 | synthetic | +5.72 % | +1.23 % | RETRACT |
| w4a16_tp1_c1 | coding    | +7.34 % | +1.29 % | RETRACT |
| w4a16_tp1_c2 | synthetic | +3.69 % | +0.37 % | RETRACT |
| w4a16_tp1_c2 | coding    | +6.45 % | −0.17 % | RETRACT |
| w4a16_tp1_c4 | synthetic | +2.36 % | −0.20 % | RETRACT |
| w4a16_tp1_c4 | coding    | +7.74 % | −1.14 % | RETRACT |
| w4a16_tp4_c1 | synthetic | +8.98 % | −0.02 % | RETRACT |
| w4a16_tp4_c1 | coding    | +12.76 % | −0.00 % | RETRACT |
| w4a16_tp4_c2 | synthetic | +6.99 % | +0.14 % | RETRACT |
| w4a16_tp4_c2 | coding    | +17.27 % | −0.36 % | RETRACT |
| w4a16_tp4_c4 | synthetic | +2.57 % | +0.46 % | RETRACT |
| w4a16_tp4_c4 | coding    | +18.90 % | +0.32 % | RETRACT |
<!-- markdownlint-enable MD060 -->

Prior-mission cross-host deltas pulled from
[BENCH_FUSED_ACT_QUANT.md](../BENCH_FUSED_ACT_QUANT.md) §(f) table,
lines 223-234. Same-host A/B deltas from the per-cell table above.

**Interpretation.** All 12 prior wins evaporate under same-host A/B. The
fused-on artifact and the fused-off artifact produce statistically
indistinguishable W4A16 throughput on this host (max |Δ| = 1.29 %, well
inside same-host run-to-run variance reported by M0-F4 canary at ±2 %).
The prior wins of +2.4 % to +18.9 % were therefore cross-host drift
between the 2026-05-19 production-baseline measurement window and the
2026-05-22/23 worker session, NOT a real fused-kernel benefit — exactly
the suspicion documented in [redundant-silu-mission-context.md] that
motivated this M2 audit.

## Confirmed Wins

None. Zero W4A16 cells meet the confirmation criteria (same-host
Δ_tput ≤ −2 %). The fused-act-quant path delivers no measurable W4A16
throughput benefit on this host versus the legacy
`silu_and_mul` + per-token-quant composition. Any continued W4A16
production recommendation should rest on the W8A8 HBM evidence (which
this mission's M3 rocprofv3 capture addresses separately) and NOT on
the retracted W4A16 throughput numbers.
