#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate an HBM-mission milestone grid against the production baseline.

Patterned after the prior mission's ``scripts/mi100/aggregate_m4.py``.

For each of the 24 per-cell raw.json files at
``/root/bench-int8-w4a16-hbm/<milestone>/{w8a8,w4a16}/``, this script joins
on (model, tp, concurrency, workload) against the per-cell production
baseline at ``/root/bench-int8-w4a16/final/final_grid.csv`` and emits both
a 144-row CSV (12 cells × 2 workloads × 6 metrics) and a markdown summary
table.

Per-row CSV columns::

    model, tp, concurrency, workload, metric,
    production_value, milestone_value, delta_pct, verdict

Verdicts:

* ``WIN``        — direction-correct improvement of ≥ 3 %
* ``HOLD``       — direction-relative change within ±3 %
* ``REGRESSION`` — direction-correct degradation of ≥ 1 %

A decode-dominated win-bar summary is also computed: per quant scheme, we
scan the four decode-dominated cells (``tp1_c1``, ``tp1_c2``, ``tp4_c1``,
``tp4_c2``) on either workload and check whether at least one cell shows
``output_throughput_toks_s`` delta_pct ≥ +3 %.

Outputs land at:

* ``/root/bench-int8-w4a16-hbm/<milestone>/<milestone-key>_pareto.csv``
* ``/root/bench-int8-w4a16-hbm/<milestone>/<milestone-key>_pareto.md``

The markdown is also printed to stdout so the caller may embed it.

Usage::

    /opt/vllm-env/bin/python3 scripts/mi100/aggregate_hbm.py \
        --milestone m1-kvint8

Optional ``--geomean`` adds a decode-dominated geomean throughput section
(used by the M4 final aggregate).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# CSV metric label  ->  raw.json field name  + improvement direction
METRICS: list[tuple[str, str, str]] = [
    ("tput", "output_throughput_toks_s", "higher_better"),
    ("req_tput", "request_throughput_req_s", "higher_better"),
    ("mean_ttft", "mean_ttft_ms", "lower_better"),
    ("p99_ttft", "p99_ttft_ms", "lower_better"),
    ("mean_tpot", "mean_tpot_ms", "lower_better"),
    ("p99_tpot", "p99_tpot_ms", "lower_better"),
]

QUANTS = ("w8a8", "w4a16")
TPS = (1, 4)
CONCS = (1, 2, 4)
WORKLOADS = ("synthetic", "coding")

# m3-tp covers only TP=4 cells (TP=1 is a no-op for NCCL topology).
TPS_BY_MILESTONE: dict[str, tuple[int, ...]] = {
    "m1-kvint8": (1, 4),
    "m2-chunked": (1, 4),
    "m3-tp": (4,),
    "m4-final": (1, 4),
    # The follow-on Triton flash-decoding tuning mission (M1 of the FA
    # mission) compares the tuned-lookup stack against the M4 baseline
    # cell JSONs (not against the production final_grid.csv), so all
    # 24 cells (TP=1 + TP=4) are in scope.
    "m1-flash-tune": (1, 4),
    # The Fused Activation-Quant mission's M4 grid covers all 24 cells
    # (TP=1 + TP=4); its baseline is the per-cell production raw.json
    # at /root/bench-int8-w4a16/baseline/raw_*/raw.json (NOT
    # final_grid.csv — see library/fused-act-quant-mission-context.md
    # §9 / AGENTS.md anti-pattern #10).
    "m4-fused-act-quant": (1, 4),
}

# Decode-dominated cells used by the win-bar evaluation.
DECODE_DOMINATED = (
    ("tp1", 1),
    ("tp1", 2),
    ("tp4", 1),
    ("tp4", 2),
)

WIN_THRESHOLD_PCT = 3.0
REGRESSION_THRESHOLD_PCT = 1.0

BENCH_ROOT = Path("/root/bench-int8-w4a16-hbm")
# The follow-on Triton flash-decoding tuning mission writes its cells under
# /root/bench-int8-w4a16-hbm-fa/m1-tuning/ (NOT the prior mission's
# /root/bench-int8-w4a16-hbm/<milestone>/ path).
BENCH_ROOT_FLASH_TUNE = Path("/root/bench-int8-w4a16-hbm-fa/m1-tuning")
# The Fused Activation-Quant mission's m4-bench grid lands under
# /root/bench-int8-w4a16-fused/m4-bench/{w8a8,w4a16}/<cell>_<wl>.json.
BENCH_ROOT_FUSED_M4 = Path("/root/bench-int8-w4a16-fused/m4-bench")
PROD_CSV = Path("/root/bench-int8-w4a16/final/final_grid.csv")
# Per-cell production baseline raw JSONs (anti-pattern #10: NOT
# final_grid.csv). Directory layout:
#   /root/bench-int8-w4a16/baseline/raw_<cell>_<wl>_<8-char-suffix>/raw.json
PROD_RAW_BASELINE_ROOT = Path("/root/bench-int8-w4a16/baseline")

# Map metric label -> raw.json field name (mirrors METRICS above; used when
# loading per-cell JSONs as the comparison baseline instead of a CSV).
FIELD_BY_LABEL = {label: field for label, field, _ in METRICS}


def bench_root_for(milestone: str) -> Path:
    """Return the bench-root path that contains <milestone>'s cell JSONs."""
    if milestone == "m1-flash-tune":
        return BENCH_ROOT_FLASH_TUNE
    if milestone == "m4-fused-act-quant":
        return BENCH_ROOT_FUSED_M4
    return BENCH_ROOT / milestone


def load_production_baseline() -> dict[tuple[str, str, str], float]:
    """Return mapping ``(cell, workload, metric) -> winner_value``."""
    prod: dict[tuple[str, str, str], float] = {}
    if not PROD_CSV.exists():
        raise FileNotFoundError(f"missing production baseline: {PROD_CSV}")
    with open(PROD_CSV) as f:
        for row in csv.DictReader(f):
            prod[(row["cell"], row["workload"], row["metric"])] = float(
                row["winner_value"]
            )
    return prod


def load_baseline_from_cells(
    baseline_root: Path,
) -> dict[tuple[str, str, str], float]:
    """Return ``(cell, workload, metric) -> value`` built from per-cell JSONs.

    The baseline_root layout matches the per-mission per-quant layout
    written by ``run_grid_hbm.sh``: ``<root>/{w8a8,w4a16}/<cell>_<wl>.json``.
    Used by m1-flash-tune to anchor against the prior M4 cells rather than
    the production-CSV winners.
    """
    base: dict[tuple[str, str, str], float] = {}
    for quant in QUANTS:
        for tp in TPS:
            for conc in CONCS:
                for wl in WORKLOADS:
                    cell_id = f"{quant}_tp{tp}_c{conc}"
                    p = baseline_root / quant / f"{cell_id}_{wl}.json"
                    if not p.exists():
                        # Not all milestones populate all 24 cells (e.g.,
                        # m3-tp is TP=4 only). Skip missing.
                        continue
                    with open(p) as f:
                        d = json.load(f)
                    for label, field, _direction in METRICS:
                        v = d.get(field)
                        if v is None:
                            continue
                        base[(cell_id, wl, label)] = float(v)
    return base


def load_cell(milestone: str, quant: str, tp: int, conc: int, workload: str) -> dict:
    if milestone == "m1-flash-tune":
        # m1-flash-tune writes under /root/bench-int8-w4a16-hbm-fa/m1-tuning/
        # (no milestone subdir below the bench root).
        p = BENCH_ROOT_FLASH_TUNE / quant / f"{quant}_tp{tp}_c{conc}_{workload}.json"
    elif milestone == "m4-fused-act-quant":
        p = BENCH_ROOT_FUSED_M4 / quant / f"{quant}_tp{tp}_c{conc}_{workload}.json"
    else:
        p = BENCH_ROOT / milestone / quant / f"{quant}_tp{tp}_c{conc}_{workload}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing milestone cell JSON: {p}")
    with open(p) as f:
        return json.load(f)


def load_production_raw_baseline() -> dict[tuple[str, str], dict]:
    """Return mapping ``(cell_id, workload) -> raw.json dict`` from the
    per-cell production baseline at /root/bench-int8-w4a16/baseline/raw_*/.

    Each ``raw_<cell_id>_<workload>_<8-char-suffix>/raw.json`` is the
    canonical per-cell baseline used by the Fused Activation-Quant mission
    (see library/fused-act-quant-mission-context.md §9). For cells with
    multiple raw_* directories (e.g., a re-run), the most recently modified
    raw.json wins.
    """
    base: dict[tuple[str, str], dict] = {}
    base_mtime: dict[tuple[str, str], float] = {}
    if not PROD_RAW_BASELINE_ROOT.exists():
        raise FileNotFoundError(
            f"missing per-cell production baseline: {PROD_RAW_BASELINE_ROOT}"
        )
    for raw_dir in sorted(PROD_RAW_BASELINE_ROOT.glob("raw_*")):
        # raw_<quant>_<tp>_<c>_<workload>_<suffix>
        parts = raw_dir.name.split("_")
        if len(parts) < 6:
            continue
        # parts[0] == 'raw'
        cell_id = "_".join(parts[1:4])  # e.g. w8a8_tp1_c1
        workload = parts[4]  # synthetic | coding
        raw_file = raw_dir / "raw.json"
        if not raw_file.exists():
            continue
        try:
            data = json.loads(raw_file.read_text())
        except Exception:  # noqa: BLE001
            continue
        mtime = raw_file.stat().st_mtime
        key = (cell_id, workload)
        if key not in base or mtime > base_mtime.get(key, -1.0):
            base[key] = data
            base_mtime[key] = mtime
    return base


# Prefill-dominated cells (see m4-bench-grid VAL-M4-003): TP=1 c=1 W8A8
# synthetic is the canonical prefill-dominated cell; TP=1 c=2 W8A8
# synthetic is the secondary candidate. Used for the win-bar check.
PREFILL_DOMINATED_CELLS: set[tuple[str, str]] = {
    ("w8a8_tp1_c1", "synthetic"),
    ("w8a8_tp1_c2", "synthetic"),
}


def fused_m4_vs_baseline_rows(
    raw_baseline: dict[tuple[str, str], dict],
) -> list[dict[str, object]]:
    """Build the m4_vs_baseline_grid.csv rows for the Fused mission.

    Each row corresponds to one of the 24 fused-mission m4 cells joined
    against the per-cell production raw.json baseline on
    ``(cell_id, workload)``. Columns:

        cell_id, workload, baseline_tput, fused_tput, delta_tput_pct,
        baseline_ttft, fused_ttft, delta_ttft_pct,
        prefill_dominated_bool

    where ``delta_tput_pct = (fused - baseline) / baseline * 100`` and
    ``delta_ttft_pct = (fused - baseline) / baseline * 100`` (negative
    delta_ttft_pct is an improvement).
    """
    rows: list[dict[str, object]] = []
    for quant in QUANTS:
        for tp in (1, 4):
            for conc in (1, 2, 4):
                cell_id = f"{quant}_tp{tp}_c{conc}"
                for wl in WORKLOADS:
                    try:
                        cell = load_cell("m4-fused-act-quant", quant, tp, conc, wl)
                    except FileNotFoundError:
                        cell = {}
                    base = raw_baseline.get((cell_id, wl), {})
                    # Production raw.json uses 'output_throughput'; cell
                    # JSON uses 'output_throughput_toks_s'.
                    base_tput = float(base.get("output_throughput") or 0.0)
                    fused_tput = float(cell.get("output_throughput_toks_s") or 0.0)
                    base_ttft = float(base.get("mean_ttft_ms") or 0.0)
                    fused_ttft = float(cell.get("mean_ttft_ms") or 0.0)
                    if base_tput > 0:
                        d_tput = (fused_tput - base_tput) / base_tput * 100.0
                    else:
                        d_tput = float("nan")
                    if base_ttft > 0:
                        d_ttft = (fused_ttft - base_ttft) / base_ttft * 100.0
                    else:
                        d_ttft = float("nan")
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "workload": wl,
                            "baseline_tput": base_tput,
                            "fused_tput": fused_tput,
                            "delta_tput_pct": d_tput,
                            "baseline_ttft": base_ttft,
                            "fused_ttft": fused_ttft,
                            "delta_ttft_pct": d_ttft,
                            "prefill_dominated_bool": (cell_id, wl)
                            in PREFILL_DOMINATED_CELLS,
                        }
                    )
    return rows


def write_fused_m4_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "cell_id",
        "workload",
        "baseline_tput",
        "fused_tput",
        "delta_tput_pct",
        "baseline_ttft",
        "fused_ttft",
        "delta_ttft_pct",
        "prefill_dominated_bool",
    ]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:

            def fmt(v: object) -> str:
                if isinstance(v, float):
                    if math.isnan(v):
                        return ""
                    return f"{v:.6f}"
                if isinstance(v, bool):
                    return "true" if v else "false"
                return str(v)

            w.writerow([fmt(r[c]) for c in cols])


def fused_m4_win_bar(
    rows: list[dict[str, object]],
    tput_thresh: float = 0.1,
    ttft_thresh: float = -0.1,
) -> tuple[bool, list[dict[str, object]]]:
    """Evaluate the research-mode win-bar (VAL-M4-003).

    The win-bar is met if at least one prefill-dominated cell shows:
      delta_tput_pct >= +0.1%  OR  delta_ttft_pct <= -0.1%

    Returns (met, list_of_winning_rows). The list will be empty if the
    win-bar is not met.
    """
    winning: list[dict[str, object]] = []
    for r in rows:
        if not r.get("prefill_dominated_bool"):
            continue
        d_t = r.get("delta_tput_pct")
        d_ttft = r.get("delta_ttft_pct")
        t_ok = isinstance(d_t, float) and not math.isnan(d_t) and d_t >= tput_thresh
        ttft_ok = (
            isinstance(d_ttft, float)
            and not math.isnan(d_ttft)
            and d_ttft <= ttft_thresh
        )
        if t_ok or ttft_ok:
            winning.append(r)
    return (len(winning) > 0, winning)


def delta_pct(milestone_value: float, production_value: float) -> float:
    if production_value == 0:
        return float("nan")
    return (milestone_value - production_value) / production_value * 100.0


def classify(delta: float, direction: str) -> str:
    """Return WIN / HOLD / REGRESSION based on direction and thresholds."""
    if math.isnan(delta):
        return "HOLD"
    if direction == "higher_better":
        if delta >= WIN_THRESHOLD_PCT:
            return "WIN"
        if delta <= -REGRESSION_THRESHOLD_PCT:
            return "REGRESSION"
        return "HOLD"
    # lower_better
    if delta <= -WIN_THRESHOLD_PCT:
        return "WIN"
    if delta >= REGRESSION_THRESHOLD_PCT:
        return "REGRESSION"
    return "HOLD"


def fmt_delta(delta: float, verdict: str) -> str:
    if math.isnan(delta):
        return "—"
    sign = "+" if delta > 0 else ""
    s = f"{sign}{delta:.2f}%"
    if verdict == "WIN":
        return f"**{s}**"
    if verdict == "REGRESSION":
        return f"_{s}_"
    return s


def model_for_quant(quant: str) -> str:
    return f"Qwen3.5-9B-{quant}"


def build_rows(
    milestone: str, prod: dict[tuple[str, str, str], float]
) -> list[dict[str, object]]:
    """Build the 144-row table (or fewer for milestones that subset cells)."""
    rows: list[dict[str, object]] = []
    tps = TPS_BY_MILESTONE.get(milestone, TPS)
    for quant in QUANTS:
        for tp in tps:
            for conc in CONCS:
                for wl in WORKLOADS:
                    cell_id = f"{quant}_tp{tp}_c{conc}"
                    cell = load_cell(milestone, quant, tp, conc, wl)
                    for metric_label, field, direction in METRICS:
                        prod_val = prod.get((cell_id, wl, metric_label))
                        m1_val = cell.get(field)
                        if prod_val is None or m1_val is None:
                            d = float("nan")
                            verdict = "MISSING"
                        else:
                            d = delta_pct(m1_val, prod_val)
                            verdict = classify(d, direction)
                        rows.append(
                            {
                                "model": model_for_quant(quant),
                                "tp": tp,
                                "concurrency": conc,
                                "workload": wl,
                                "metric": metric_label,
                                "production_value": prod_val,
                                "milestone_value": m1_val,
                                "delta_pct": d,
                                "verdict": verdict,
                                "direction": direction,
                                "cell_id": cell_id,
                                "quant": quant,
                            }
                        )
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "tp",
                "concurrency",
                "workload",
                "metric",
                "production_value",
                "milestone_value",
                "delta_pct",
                "verdict",
            ]
        )
        for r in rows:
            d = r["delta_pct"]
            pv = r["production_value"]
            mv = r["milestone_value"]
            w.writerow(
                [
                    r["model"],
                    r["tp"],
                    r["concurrency"],
                    r["workload"],
                    r["metric"],
                    "" if pv is None else f"{pv:.6f}",
                    "" if mv is None else f"{mv:.6f}",
                    ""
                    if (d is None or (isinstance(d, float) and math.isnan(d)))
                    else f"{d:.4f}",
                    r["verdict"],
                ]
            )


def win_bar(rows: list[dict[str, object]]) -> dict[str, dict]:
    """Per-quant win-bar evaluation on decode-dominated cells."""
    result: dict[str, dict] = {}
    for quant in QUANTS:
        decode_rows = []
        any_win = False
        for r in rows:
            if r["quant"] != quant:
                continue
            if r["metric"] != "tput":
                continue
            cell_tag = (f"tp{r['tp']}", r["concurrency"])
            if cell_tag not in DECODE_DOMINATED:
                continue
            decode_rows.append(r)
            if r["verdict"] == "WIN":
                any_win = True
        result[quant] = {
            "rows": decode_rows,
            "win_bar_met": any_win,
        }
    return result


def collect_regressions(rows: list[dict[str, object]]) -> list[dict]:
    return [r for r in rows if r["verdict"] == "REGRESSION"]


def decode_geomean(
    rows: list[dict[str, object]], quant: str
) -> tuple[float, float, float]:
    """Geomean of milestone tput / production tput on decode cells.

    Returns (geomean_ratio, geomean_milestone, geomean_production).
    """
    ratios: list[float] = []
    ms: list[float] = []
    ps: list[float] = []
    for r in rows:
        if r["quant"] != quant or r["metric"] != "tput":
            continue
        cell_tag = (f"tp{r['tp']}", r["concurrency"])
        if cell_tag not in DECODE_DOMINATED:
            continue
        # decode geomean uses synthetic-workload tput only to match the
        # prior mission's convention.
        if r["workload"] != "synthetic":
            continue
        pv = r["production_value"]
        mv = r["milestone_value"]
        if not pv or not mv:
            continue
        ratios.append(mv / pv)
        ms.append(mv)
        ps.append(pv)

    def gm(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    return gm(ratios), gm(ms), gm(ps)


def tp4_win_bar_count(rows: list[dict[str, object]]) -> tuple[int, list[dict]]:
    """For m3-tp: count TP=4 cells with `tput` WIN (≥+3%) vs production.

    Returns (n_cells_won, list_of_winning_rows). A "cell" here is one
    (quant, tp, concurrency, workload) tuple; the m3-tp grid has 12 such
    cells total (6 TP=4 cells × 2 workloads).
    """
    winning: list[dict] = []
    for r in rows:
        if r["tp"] != 4 or r["metric"] != "tput":
            continue
        if r["verdict"] == "WIN":
            winning.append(r)
    return len(winning), winning


def write_markdown(
    milestone: str,
    rows: list[dict[str, object]],
    win_bar_res: dict[str, dict],
    regressions: list[dict],
    include_geomean: bool,
) -> str:
    out: list[str] = []
    out.append(f"## {milestone} Pareto grid vs production baseline")
    out.append("")
    out.append(
        "Joined against `/root/bench-int8-w4a16/final/final_grid.csv` "
        "(`winner_value` per `(cell, workload, metric)`) on "
        "`(model, tp, concurrency, workload)`."
    )
    out.append("")
    out.append(
        "Verdicts: **WIN** ≥ +3 % direction-correct improvement; "
        "HOLD within ±3 %; _REGRESSION_ ≥ 1 % direction-correct "
        "degradation."
    )
    out.append("")
    out.append(
        "| Model | TP | Conc | Workload | Metric | Production | "
        f"{milestone} | Δ% | Verdict |"
    )
    out.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |")
    for r in rows:
        pv = r["production_value"]
        mv = r["milestone_value"]
        d = r["delta_pct"]
        verdict = r["verdict"]
        pv_s = "—" if pv is None else f"{pv:.4f}"
        mv_s = "—" if mv is None else f"{mv:.4f}"
        out.append(
            f"| {r['model']} | {r['tp']} | {r['concurrency']} | "
            f"{r['workload']} | {r['metric']} | "
            f"{pv_s} | {mv_s} | "
            f"{fmt_delta(d if isinstance(d, float) else float('nan'), verdict)} "
            f"| {verdict} |"
        )
    out.append("")

    # Per-quant decode-dominated win-bar summary
    out.append("## Decode-dominated win-bar (per-quant)")
    out.append("")
    out.append(
        "Decode-dominated cells = {`tp1_c1`, `tp1_c2`, `tp4_c1`, `tp4_c2`} "
        "on either workload. Win-bar = at least one cell with "
        "`output_throughput_toks_s` Δ ≥ +3 % vs production."
    )
    out.append("")
    for quant in QUANTS:
        out.append(f"### {quant}")
        out.append("")
        out.append(
            f"| Cell | Workload | Production tput | {milestone} tput | Δ% | Verdict |"
        )
        out.append("| --- | --- | ---: | ---: | ---: | --- |")
        for r in win_bar_res[quant]["rows"]:
            pv = r["production_value"]
            mv = r["milestone_value"]
            d = r["delta_pct"]
            out.append(
                f"| {r['cell_id']} | {r['workload']} | "
                f"{pv:.4f} | {mv:.4f} | "
                f"{fmt_delta(d, r['verdict'])} | {r['verdict']} |"
            )
        verdict = (
            "**WIN-BAR-MET**"
            if win_bar_res[quant]["win_bar_met"]
            else "**WIN-BAR-NOT-MET**"
        )
        out.append("")
        out.append(f"Verdict for `{quant}`: {verdict}")
        out.append("")

    # m3-tp specific: total TP=4 cell wins on tput (mission-level win-bar)
    if milestone == "m3-tp":
        n_won, winning = tp4_win_bar_count(rows)
        out.append("## m3-tp TP=4 cell win-bar (≥ 3 of 12 cells must WIN on tput)")
        out.append("")
        out.append(
            "Per mission VAL-TP-004: ≥ +3 % `output_throughput_toks_s` win on "
            "≥ 3 TP=4 cells vs the production `final_grid.csv` baseline. "
            "Total TP=4 grid is 6 cells × 2 workloads = 12 candidate rows."
        )
        out.append("")
        out.append(f"Cells won (Δ ≥ +3 %): **{n_won} / 12**")
        out.append("")
        if winning:
            out.append(f"| Cell | Workload | Production tput | {milestone} tput | Δ% |")
            out.append("| --- | --- | ---: | ---: | ---: |")
            for r in winning:
                pv = r["production_value"]
                mv = r["milestone_value"]
                d = r["delta_pct"]
                out.append(
                    f"| {r['cell_id']} | {r['workload']} | "
                    f"{pv:.4f} | {mv:.4f} | "
                    f"{fmt_delta(d, r['verdict'])} |"
                )
        verdict = "**WIN-BAR-MET**" if n_won >= 3 else "**WIN-BAR-NOT-MET**"
        out.append("")
        out.append(f"Mission win-bar verdict: {verdict}")
        out.append("")

    # Regressions
    out.append("## Regressions ≥ 1 % (any metric, any cell)")
    out.append("")
    if not regressions:
        out.append("None.")
    else:
        out.append(f"| Cell | Workload | Metric | Production | {milestone} | Δ% |")
        out.append("| --- | --- | --- | ---: | ---: | ---: |")
        for r in regressions:
            pv = r["production_value"]
            mv = r["milestone_value"]
            d = r["delta_pct"]
            out.append(
                f"| {r['cell_id']} | {r['workload']} | {r['metric']} | "
                f"{pv:.4f} | {mv:.4f} | "
                f"{fmt_delta(d, r['verdict'])} |"
            )
    out.append("")

    # Optional decode-geomean section (used by M4)
    if include_geomean:
        out.append("## Decode-dominated throughput geomean (synthetic only)")
        out.append("")
        out.append(
            "| Quant | Production geomean (tok/s) | "
            f"{milestone} geomean (tok/s) | Ratio |"
        )
        out.append("| --- | ---: | ---: | ---: |")
        for quant in QUANTS:
            ratio, mgeo, pgeo = decode_geomean(rows, quant)
            out.append(f"| {quant} | {pgeo:.4f} | {mgeo:.4f} | {ratio:.4f}× |")
        out.append("")

    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--milestone",
        required=True,
        choices=[
            "m1-kvint8",
            "m2-chunked",
            "m3-tp",
            "m4-final",
            "m1-flash-tune",
            "m4-fused-act-quant",
        ],
        help="Milestone identifier (used as both the bench subdir and the "
        "CSV/Markdown filename prefix).",
    )
    p.add_argument(
        "--geomean",
        action="store_true",
        help="Include a decode-dominated throughput geomean section "
        "(used by the M4 final aggregate).",
    )
    p.add_argument(
        "--baseline-root",
        default=None,
        help="When set, anchor the comparison against the per-cell JSONs "
        "under <baseline-root>/{w8a8,w4a16}/<cell>_<wl>.json instead "
        "of /root/bench-int8-w4a16/final/final_grid.csv. Required for "
        "milestone=m1-flash-tune (compare to M4 baselines).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    milestone = args.milestone
    milestone_key = milestone.replace("-", "_")

    # m4-fused-act-quant has its own per-cell baseline (production raw.json
    # files, NOT final_grid.csv) and its own CSV format (the spec defines
    # a 9-column m4_vs_baseline_grid.csv). Handle it ahead of the
    # general-milestone flow.
    if milestone == "m4-fused-act-quant":
        raw_baseline = load_production_raw_baseline()
        rows = fused_m4_vs_baseline_rows(raw_baseline)
        out_dir = bench_root_for(milestone)
        csv_out = out_dir / "m4_vs_baseline_grid.csv"
        write_fused_m4_csv(rows, csv_out)
        met, winning = fused_m4_win_bar(rows)
        # Emit a human-readable summary alongside the CSV.
        md_out = out_dir / "m4_vs_baseline_grid.md"
        md_lines: list[str] = [
            "## m4-fused-act-quant — 24-cell delta vs production baseline",
            "",
            (
                "Baseline source: per-cell raw.json under "
                "`/root/bench-int8-w4a16/baseline/raw_*/` (per-cell, "
                "NOT `final_grid.csv` — see "
                "`library/fused-act-quant-mission-context.md` §9)."
            ),
            "",
            (
                "| cell_id | workload | baseline_tput | fused_tput | "
                "Δ tput % | baseline_ttft | fused_ttft | Δ ttft % | "
                "prefill_dom |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rows:
            d_t = r["delta_tput_pct"]
            d_ttft = r["delta_ttft_pct"]
            d_t_s = "—" if isinstance(d_t, float) and math.isnan(d_t) else f"{d_t:+.2f}"
            d_ttft_s = (
                "—"
                if isinstance(d_ttft, float) and math.isnan(d_ttft)
                else f"{d_ttft:+.2f}"
            )
            md_lines.append(
                f"| {r['cell_id']} | {r['workload']} | "
                f"{r['baseline_tput']:.4f} | {r['fused_tput']:.4f} | "
                f"{d_t_s} | {r['baseline_ttft']:.2f} | "
                f"{r['fused_ttft']:.2f} | {d_ttft_s} | "
                f"{'yes' if r['prefill_dominated_bool'] else ''} |"
            )
        md_lines.append("")
        md_lines.append("### Win-bar (research-mode, VAL-M4-003)")
        md_lines.append("")
        md_lines.append(
            "At least one prefill-dominated cell must show "
            "`delta_tput_pct >= +0.1%` OR `delta_ttft_pct <= -0.1%`. "
            "Prefill-dominated cells: "
            + ", ".join(f"`{c}/{w}`" for c, w in sorted(PREFILL_DOMINATED_CELLS))
            + "."
        )
        md_lines.append("")
        md_lines.append(f"Result: **{'WIN-BAR-MET' if met else 'WIN-BAR-NOT-MET'}**")
        if winning:
            md_lines.append("")
            md_lines.append("Winning cells:")
            for r in winning:
                md_lines.append(
                    f"- {r['cell_id']}/{r['workload']}: "
                    f"Δtput={r['delta_tput_pct']:+.2f}%, "
                    f"Δttft={r['delta_ttft_pct']:+.2f}%"
                )
        md = "\n".join(md_lines)
        md_out.write_text(md + "\n")
        print(md)
        print()
        print(f"# CSV:  {csv_out}")
        print(f"# MD:   {md_out}")
        print(f"# WIN_BAR: {'MET' if met else 'NOT_MET'}")
        return 0

    # Decide baseline source: per-cell JSON root (preferred when --baseline-root
    # given or milestone=m1-flash-tune) vs the production CSV.
    if args.baseline_root is not None:
        baseline_root = Path(args.baseline_root)
        if not baseline_root.exists():
            raise FileNotFoundError(f"missing baseline root: {baseline_root}")
        prod = load_baseline_from_cells(baseline_root)
    elif milestone == "m1-flash-tune":
        # Default for m1-flash-tune: M4 cell JSONs.
        baseline_root = Path("/root/bench-int8-w4a16-hbm/m4-final")
        prod = load_baseline_from_cells(baseline_root)
    else:
        prod = load_production_baseline()

    rows = build_rows(milestone, prod)

    # Output dir tracks where the per-cell JSONs live.
    out_dir = bench_root_for(milestone)

    # m1-flash-tune mission convention names the output m1_vs_m4_grid.{csv,md}
    # so future workers can grep for the verdict by that filename.
    if milestone == "m1-flash-tune":
        csv_out = out_dir / "m1_vs_m4_grid.csv"
        md_out = out_dir / "m1_vs_m4_grid.md"
    else:
        csv_out = out_dir / f"{milestone_key.split('_')[0]}_pareto.csv"
        md_out = out_dir / f"{milestone_key.split('_')[0]}_pareto.md"

    write_csv(rows, csv_out)
    win_bar_res = win_bar(rows)
    regressions = collect_regressions(rows)
    md = write_markdown(
        milestone, rows, win_bar_res, regressions, include_geomean=args.geomean
    )
    md_out.write_text(md)

    print(md)
    print()
    print(f"# 144-row CSV: {csv_out}")
    print(f"# Markdown:    {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
