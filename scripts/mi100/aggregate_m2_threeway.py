#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Aggregate W8A8 cells across three grids:
  * M1 baseline (May-7)
      /root/bench-int8-w4a16/baseline/{synthetic,coding}/w8a8_*.json
  * M1 rebaselined (M2 env)
      /root/bench-int8-w4a16/m1-rebaseline/{synthetic,coding}/w8a8_*.json
  * M2 (M2 env + tuned)
      /root/bench-int8-w4a16/m2/{synthetic,coding}/w8a8_*.json

Emits a markdown table with the canonical metrics per cell, plus deltas:

  Δ_M1->M2_baseline  : M2  vs M1  (May-7)            [contains lib-swap noise]
  Δ_M1->M1_rebaseline: M1rb vs M1  (May-7)            [pure lib-swap delta]
  Δ_M1rb->M2         : M2  vs M1rb (same M2 env)      [pure tuning delta]

This is the comparison the m2-rebaseline-and-fill-grid feature
requires (see mission features.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS = [
    ("w8a8", tp, c, wl)
    for tp in (1, 4)
    for c in (1, 2, 4)
    for wl in ("synthetic", "coding")
]

ROOTS = {
    "M1":  Path("/root/bench-int8-w4a16/baseline"),
    "M1r": Path("/root/bench-int8-w4a16/m1-rebaseline"),
    "M2":  Path("/root/bench-int8-w4a16/m2"),
}

METRICS = [
    ("output_throughput_toks_s", "tput",     "higher_better"),
    ("request_throughput_req_s", "req_tput", "higher_better"),
    ("mean_ttft_ms",             "mean_ttft","lower_better"),
    ("p99_ttft_ms",              "p99_ttft", "lower_better"),
    ("mean_tpot_ms",             "mean_tpot","lower_better"),
    ("p99_tpot_ms",              "p99_tpot", "lower_better"),
]


def load_cell(root: Path, model: str, tp: int, c: int, wl: str) -> dict | None:
    p = root / wl / f"{model}_tp{tp}_c{c}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def pct(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return (a - b) / b * 100.0


def fmt_delta(d: float, direction: str, threshold: float = 3.0) -> str:
    """direction: 'higher_better' | 'lower_better'"""
    if d != d:  # NaN
        return "—"
    sign = "+" if d > 0 else ""
    s = f"{sign}{d:.2f}%"
    is_improvement = (
        (direction == "higher_better" and d >= threshold) or
        (direction == "lower_better" and d <= -threshold)
    )
    is_regression = (
        (direction == "higher_better" and d <= -1.0) or
        (direction == "lower_better" and d >= 1.0)
    )
    if is_improvement:
        return f"**{s}**"
    if is_regression:
        return f"_{s}_"
    return s


def main() -> int:
    out = []
    out.append("# M1 / M1-rebaselined / M2 three-way comparison (W8A8)")
    out.append("")
    out.append("> All three columns are walltime-measured on the same 4×MI100 host;")
    out.append("> M1 was measured 2026-05-07, M1-rebaseline + M2 measured")
    out.append("> 2026-05-09/10 with the M2-build `libhipblaslt.so` swapped in")
    out.append("> via `LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:...`.")
    out.append("> M2 additionally exports `HIPBLASLT_TENSILE_LIBPATH` to load")
    out.append("> the merged TensileLite logic; M1-rebaseline does NOT (so")
    out.append("> hipBLASLt uses its prebuilt I8I8 default kernel).")
    out.append("")
    out.append("Markup: **bold** = Pareto improvement ≥ 3% on this metric, "
               "_italic_ = regression > 1%.")
    out.append("")
    out.append("## Per-cell results (W8A8)")
    out.append("")
    out.append("| Cell | Workload | Metric | M1 (May-7) | M1-rebaseline | M2 | "
               "Δ M1→M2 | Δ M1→M1rb (lib-swap) | "
               "Δ M1rb→M2 (**tuning**) |")
    out.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    summary: list[dict] = []

    for model, tp, c, wl in CELLS:
        cells = {k: load_cell(r, model, tp, c, wl) for k, r in ROOTS.items()}
        if any(v is None for v in cells.values()):
            missing = [k for k, v in cells.items() if v is None]
            print(f"  [warn] {model}_tp{tp}_c{c}_{wl}: missing {missing}",
                  file=sys.stderr)
            continue
        cell_id = f"{model}_tp{tp}_c{c}"
        for key, label, direction in METRICS:
            m1 = cells["M1"][key]
            m1r = cells["M1r"][key]
            m2 = cells["M2"][key]
            d_m1_m2 = pct(m2, m1)
            d_m1_m1r = pct(m1r, m1)
            d_m1r_m2 = pct(m2, m1r)
            if direction == "lower_better":
                # invert sign for the human reader so "negative = improvement"
                pass  # we keep raw delta; format function does the inversion semantics
            out.append(
                f"| {cell_id} | {wl} | {label} | "
                f"{m1:.2f} | {m1r:.2f} | {m2:.2f} | "
                f"{fmt_delta(d_m1_m2, direction)} | "
                f"{fmt_delta(d_m1_m1r, direction)} | "
                f"{fmt_delta(d_m1r_m2, direction)} |"
            )
            summary.append({
                "cell": cell_id, "workload": wl, "metric": label,
                "direction": direction,
                "M1": m1, "M1r": m1r, "M2": m2,
                "d_M1_M2": d_m1_m2,
                "d_M1_M1r": d_m1_m1r,
                "d_M1r_M2": d_m1r_m2,
            })

    out.append("")
    out.append("## Headline analysis: tuning gain (Δ M1rb→M2)")
    out.append("")
    out.append("Cells where the **tuning** column shows ≥ +3% gain "
               "(throughput) or ≤ -3% latency reduction:")
    out.append("")
    out.append("| Cell | Workload | Metric | M1rb | M2 | Δ M1rb→M2 |")
    out.append("| --- | --- | --- | ---: | ---: | ---: |")
    n_pareto = 0
    for r in summary:
        d = r["d_M1r_M2"]
        is_improvement = (
            (r["direction"] == "higher_better" and d >= 3.0) or
            (r["direction"] == "lower_better" and d <= -3.0)
        )
        if is_improvement:
            n_pareto += 1
            out.append(
                f"| {r['cell']} | {r['workload']} | {r['metric']} | "
                f"{r['M1r']:.2f} | {r['M2']:.2f} | "
                f"{fmt_delta(d, r['direction'])} |"
            )
    if n_pareto == 0:
        out.append("| — | — | — | — | — | (none) |")
    out.append("")
    out.append(f"**Pareto-bar cells (≥ 3% improvement vs M1-rebaselined):** {n_pareto}")
    out.append("")

    # Identify prefill-dominated cells
    out.append("## Prefill-dominated improvement check (gate)")
    out.append("")
    out.append("Mission gate: at least one **prefill-dominated** cell improves ≥ 3% "
               "in the M2-vs-M1-rebaselined column. Prefill dominance is highest "
               "on TP=1 c≥2 synthetic (large M batches) and on TP=4 c=4 synthetic.")
    out.append("")
    prefill_cells = {
        "w8a8_tp1_c2", "w8a8_tp1_c4", "w8a8_tp4_c4"
    }
    prefill_metrics = {"mean_ttft", "p99_ttft"}
    prefill_pass = []
    for r in summary:
        if r["cell"] not in prefill_cells:
            continue
        if r["metric"] not in prefill_metrics:
            continue
        if r["workload"] != "synthetic":
            continue
        d = r["d_M1r_M2"]
        if r["direction"] == "lower_better" and d <= -3.0:
            prefill_pass.append(r)
    out.append("| Cell | Workload | Metric | M1rb | M2 | Δ M1rb→M2 |")
    out.append("| --- | --- | --- | ---: | ---: | ---: |")
    for r in prefill_pass:
        out.append(
            f"| {r['cell']} | {r['workload']} | {r['metric']} | "
            f"{r['M1r']:.2f} | {r['M2']:.2f} | "
            f"{fmt_delta(r['d_M1r_M2'], r['direction'])} |"
        )
    if not prefill_pass:
        out.append("| — | — | — | — | — | (no prefill-dominated cell ≥ 3%) |")
    out.append("")
    if prefill_pass:
        out.append(f"**GATE PASS:** {len(prefill_pass)} prefill-dominated metric(s) "
                   "improved ≥ 3% in M2 vs M1-rebaselined.")
    else:
        out.append("**GATE FAIL:** no prefill-dominated metric improved ≥ 3% "
                   "in M2 vs M1-rebaselined.")
    out.append("")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
