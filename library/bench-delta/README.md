# M3-F2 Per-Cell Delta Aggregates

Computed by `compute_deltas.py` comparing `library/bench-post-sync/` against `library/bench-baseline/`.

## Threshold gates (validation-contract D1–D4)

- D1 24-cell grid: regression if `Δ output_throughput_toks_s < −3 %`
- D2 fused-act-quant cell: regression if `Δ output_throughput_toks_s < −3 %`
- D3 CK-FA decode latency: regression if `Δ avg/p50/p90/p99 > +2 %`
- D4 HBM bytes per output token: regression if `Δ > +2 %`

## Top-line verdict

**`PASS`** — no cell regresses past any threshold. See `summary.json`.

| Surface | Verdict | Max Δ (worst direction) |
| --- | --- | --- |
| D1 grid (24 cells) | PASS | throughput −1.12 % (w8a8_tp4_c2/coding) |
| D2 fused-act-quant | PASS | throughput +0.46 % |
| D3 CK-FA latency | PASS | avg +0.15 %, p99 −0.77 % |
| D4 HBM bytes/output_token | PASS | −0.003 % |

## Files

| Path | Contents |
| --- | --- |
| `compute_deltas.py` | Aggregator entrypoint |
| `grid-deltas.json` | 24 rows: cell, baseline/post throughput, Δ %, verdict |
| `d2-fused-act-quant-delta.json` | Single-cell W8A8 TP1 C1 synthetic delta |
| `d3-ckfa-delta.json` | Avg + p50/p90/p99 deltas |
| `d4-hbm-delta.json` | total/per-output/per-total deltas |
| `summary.json` | Top-line verdict + thresholds + per-surface verdicts |

## No PR-bisection required

Because zero cells exceed any threshold (max throughput regression
is −1.12 %, max latency uplift is +0.19 %, HBM is −0.003 %), no
per-upstream-PR bisection was triggered. The hot-file conflict
upstream PR set referenced in `benchmark-worker` SKILL (#42095, #43660,
\#42080, #41434, #40327, #43731, #40687) all merged cleanly into the

sync branch without measurable MI100 perf impact under this harness.
