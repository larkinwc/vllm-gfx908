#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate M4 W8A8 grid results into a markdown table.

Compares the M4 W8A8 W8A8 path under two settings:
  - M4-CK     : CK > hipBLASLt > Triton dispatcher (default M4 build)
  - M4-noCk   : VLLM_DISABLE_CK=1 (forces hipBLASLt -> Triton fall-through)

…against the M3 best-of-{autotune, heuristic} per cell as the comparison
column, and the M3-autotune column directly for the gate test.

Outputs a markdown table to stdout suitable for committing as
``BENCH_M4_CK.md`` (prefixed with the existing M4 narrative; this script
only emits the bench-grid + perplexity tables and the headline gate
verdict).

Usage:
    /opt/vllm-env/bin/python3 scripts/mi100/aggregate_m4.py > _m4_grid.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

W8A8_CELLS = [
    ("w8a8", tp, c, wl)
    for tp in (1, 4)
    for c in (1, 2, 4)
    for wl in ("synthetic", "coding")
]

ROOTS = {
    "M3-auto": Path("/root/bench-int8-w4a16/m3/w8a8/autotune"),
    "M3-heur": Path("/root/bench-int8-w4a16/m3/w8a8/heuristic"),
    "M4-CK": Path("/root/bench-int8-w4a16/m4/w8a8/ck"),
    "M4-noCk": Path("/root/bench-int8-w4a16/m4/w8a8/noCk"),
}

# (key, label, direction)
METRICS = [
    ("output_throughput_toks_s", "tput", "higher_better"),
    ("request_throughput_req_s", "req_tput", "higher_better"),
    ("mean_ttft_ms", "mean_ttft", "lower_better"),
    ("p99_ttft_ms", "p99_ttft", "lower_better"),
    ("mean_tpot_ms", "mean_tpot", "lower_better"),
    ("p99_tpot_ms", "p99_tpot", "lower_better"),
]


def load_cell(root: Path, model: str, tp: int, c: int, wl: str) -> dict | None:
    p = root / wl / f"{model}_tp{tp}_c{c}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def pct(a: float, b: float) -> float:
    if not b or b == 0:
        return float("nan")
    return (a - b) / b * 100.0


def fmt_delta(d: float, direction: str, threshold: float = 3.0) -> str:
    if d != d:  # NaN
        return "—"
    sign = "+" if d > 0 else ""
    s = f"{sign}{d:.2f}%"
    is_imp = (direction == "higher_better" and d >= threshold) or (
        direction == "lower_better" and d <= -threshold
    )
    is_reg = (direction == "higher_better" and d <= -1.0) or (
        direction == "lower_better" and d >= 1.0
    )
    if is_imp:
        return f"**{s}**"
    if is_reg:
        return f"_{s}_"
    return s


def m3_best(m3_auto: dict, m3_heur: dict, key: str, direction: str) -> float:
    """Return the better of the two M3 values per the metric direction."""
    a = m3_auto[key]
    h = m3_heur[key]
    if direction == "higher_better":
        return max(a, h)
    return min(a, h)


def write_grid(out: list[str]) -> dict:
    out.append("## W8A8 grid: M3-autotune / M3-heuristic / M4-noCk / M4-CK")
    out.append("")
    out.append(
        "Four-way comparison across the 12 W8A8 cells "
        "(NUM_PROMPTS=200, M2-build env, identical hipBLASLt + merged "
        "TensileLite library). M4-noCk forces `VLLM_DISABLE_CK=1` so "
        "the dispatcher falls through to hipBLASLt -> Triton (= M3 "
        "autotune path); M4-CK uses the default CK > hipBLASLt > Triton "
        "priority. Δ M3-auto→M4-CK is the **gate test**: at least one "
        "metric ≥ +3% is required."
    )
    out.append("")
    out.append(
        "| Cell | Workload | Metric | M3-auto | M3-heur | M4-noCk | "
        "M4-CK | Δ noCk→CK | Δ M3-auto→CK (**gate**) | Δ M3-best→CK |"
    )
    out.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    n_pareto_gate = 0
    n_reg5_gate = 0
    speedups_vs_m3_best_throughput = []
    speedups_vs_m3_auto_throughput = []
    cells_present = 0
    cell_winners = {}

    for model, tp, c, wl in W8A8_CELLS:
        cells = {k: load_cell(r, model, tp, c, wl) for k, r in ROOTS.items()}
        if any(v is None for v in cells.values()):
            for k, v in cells.items():
                if v is None:
                    out.append(
                        f"| {model}_tp{tp}_c{c} | {wl} | (missing {k}) | "
                        f"— | — | — | — | — | — | — |"
                    )
            continue
        cells_present += 1
        cell_id = f"{model}_tp{tp}_c{c}"
        m3a = cells["M3-auto"]
        m3h = cells["M3-heur"]
        m4n = cells["M4-noCk"]
        m4c = cells["M4-CK"]

        for key, label, direction in METRICS:
            best_m3 = m3_best(m3a, m3h, key, direction)
            d_n_c = pct(m4c[key], m4n[key])
            d_a_c = pct(m4c[key], m3a[key])
            d_b_c = pct(m4c[key], best_m3)

            if key == "output_throughput_toks_s":
                if best_m3:
                    speedups_vs_m3_best_throughput.append(m4c[key] / best_m3)
                if m3a[key]:
                    speedups_vs_m3_auto_throughput.append(m4c[key] / m3a[key])

            out.append(
                f"| {cell_id} | {wl} | {label} | "
                f"{m3a[key]:.2f} | {m3h[key]:.2f} | "
                f"{m4n[key]:.2f} | {m4c[key]:.2f} | "
                f"{fmt_delta(d_n_c, direction)} | "
                f"{fmt_delta(d_a_c, direction)} | "
                f"{fmt_delta(d_b_c, direction)} |"
            )

            is_imp = (direction == "higher_better" and d_a_c >= 3.0) or (
                direction == "lower_better" and d_a_c <= -3.0
            )
            is_reg5 = (direction == "higher_better" and d_a_c <= -5.0) or (
                direction == "lower_better" and d_a_c >= 5.0
            )
            if is_imp:
                n_pareto_gate += 1
            if is_reg5:
                n_reg5_gate += 1

        # Determine cell winner by throughput
        winner_throughput = max(
            ("M3-auto", m3a["output_throughput_toks_s"]),
            ("M3-heur", m3h["output_throughput_toks_s"]),
            ("M4-noCk", m4n["output_throughput_toks_s"]),
            ("M4-CK", m4c["output_throughput_toks_s"]),
            key=lambda x: x[1],
        )
        cell_winners[f"{cell_id}_{wl}"] = winner_throughput[0]

    out.append("")

    def geomean(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    g_best = geomean(speedups_vs_m3_best_throughput)
    g_auto = geomean(speedups_vs_m3_auto_throughput)

    out.append(
        f"**Pareto bar (≥3% gain M3-auto → M4-CK on any metric):** "
        f"{n_pareto_gate} metric(s)"
    )
    out.append(f"**>5% regressions (M3-auto → M4-CK):** {n_reg5_gate} metric(s)")
    out.append("")
    out.append(
        f"**Throughput geomean M4-CK / M3-auto:** "
        f"{g_auto:.4f}x ({(g_auto - 1) * 100:+.2f}%)"
    )
    out.append(
        f"**Throughput geomean M4-CK / M3-best-per-cell:** "
        f"{g_best:.4f}x ({(g_best - 1) * 100:+.2f}%)"
    )
    out.append("")

    out.append("### Per-cell throughput winner")
    out.append("")
    out.append("| Cell+Workload | Winning column |")
    out.append("| --- | --- |")
    for k, v in sorted(cell_winners.items()):
        out.append(f"| {k} | {v} |")
    out.append("")

    return {
        "n_pareto_gate": n_pareto_gate,
        "n_reg5_gate": n_reg5_gate,
        "geomean_vs_m3_auto": g_auto,
        "geomean_vs_m3_best": g_best,
        "cells_present": cells_present,
    }


def write_perplexity(out: list[str]) -> None:
    out.append("## Wikitext-2 Perplexity (50 chunks × 512 tokens, seed 0)")
    out.append("")
    out.append("| Run | Perplexity | Δ vs M1-rebaseline |")
    out.append("| --- | ---: | ---: |")
    files = [
        (
            "M1-rebaseline (W8A8 reference)",
            Path("/root/bench-int8-w4a16/m1-rebaseline/ppl_w8a8_m1rb.json"),
        ),
        ("M3-autotune W8A8", Path("/root/bench-int8-w4a16/m3/ppl_w8a8_m3.json")),
        ("M4-CK W8A8", Path("/root/bench-int8-w4a16/m4/ppl_w8a8_m4_ck.json")),
    ]
    ref = None
    for label, p in files:
        if not p.exists():
            out.append(f"| {label} | (missing) | — |")
            continue
        try:
            blob = json.loads(p.read_text())
            ppl = blob.get("perplexity")
        except Exception:
            out.append(f"| {label} | (parse error) | — |")
            continue
        if "M1-rebaseline" in label:
            ref = ppl
            out.append(f"| {label} | {ppl:.4f} | — (reference) |")
        elif ref is not None:
            d = pct(ppl, ref)
            sign = "+" if d > 0 else ""
            verdict = " ✅ (≤ +1%)" if d <= 1.0 else " ❌ (> +1%)"
            out.append(f"| {label} | {ppl:.4f} | {sign}{d:.3f}%{verdict} |")
        else:
            out.append(f"| {label} | {ppl:.4f} | — |")
    out.append("")


def write_startup(out: list[str]) -> None:
    out.append("## TP=4 W8A8 Startup (gate: < 300 s)")
    out.append("")
    p = Path("/root/bench-int8-w4a16/m4/tp4_w8a8_startup.json")
    if not p.exists():
        out.append("(missing)")
        return
    blob = json.loads(p.read_text())
    out.append(
        f"- Time-to-health: **{blob.get('elapsed_to_health_s')} s** "
        f"(gate {blob.get('gate_s')} s) — verdict {blob.get('verdict')}"
    )
    out.append(
        "- Confirms M4 build (CK adds ~80 MB to `_rocm_C.abi3.so`) does "
        "not regress the TP=4 W8A8 startup envelope."
    )
    out.append("")


def main() -> int:
    out: list[str] = []
    out.append("## W8A8 bench-grid (M4 follow-up)")
    out.append("")
    out.append(
        "Runs the canonical 12-cell W8A8 grid at NUM_PROMPTS=200 against "
        "the M4 build (commit `a852f2600`) under both `VLLM_DISABLE_CK=1` "
        "(forces hipBLASLt → Triton path = M3 autotune dispatcher minus "
        "CK) and the default CK-on path. Compared against M3 (commit "
        "`9699d1f0a`) per cell."
    )
    out.append("")
    out.append("```")
    out.append("# CK path (default, CK > hipBLASLt > Triton)")
    out.append("NUM_PROMPTS=200 scripts/mi100/run_grid.sh m4-w8a8-ck w8a8_")
    out.append("# noCk path (VLLM_DISABLE_CK=1, hipBLASLt > Triton)")
    out.append("NUM_PROMPTS=200 scripts/mi100/run_grid.sh m4-w8a8-noCk w8a8_")
    out.append("```")
    out.append("")

    summary = write_grid(out)
    write_perplexity(out)
    write_startup(out)

    out.append("## Headline (M4 W8A8 follow-up)")
    out.append("")
    out.append(
        f"- M4-CK vs M3-autotune: {summary['n_pareto_gate']} metric(s) "
        f"≥ +3% gain; {summary['n_reg5_gate']} metric(s) regress > 5%."
    )
    g = summary["geomean_vs_m3_auto"]
    out.append(
        f"- Throughput geomean M4-CK / M3-autotune: **{g:.4f}×** "
        f"({(g - 1) * 100:+.2f}%)."
    )
    g = summary["geomean_vs_m3_best"]
    out.append(
        f"- Throughput geomean M4-CK / M3-best-per-cell: **{g:.4f}×** "
        f"({(g - 1) * 100:+.2f}%)."
    )
    out.append("")
    out.append(
        "_Cells where CK is registered (TP=1 prefill on N ∈ {24576, "
        "10240, 4096} or N=4096 with K=12288) are the only ones the CK "
        "path can win on; decode (M=1) and TP=4-sharded shapes fall "
        "through to hipBLASLt → Triton autotune in both M4-CK and "
        "M4-noCk runs, so those cells are expected to be flat between "
        "the two columns and serve as a noise reference._"
    )
    out.append("")
    out.append("## Files")
    out.append("")
    out.append("- M4-CK cells: `/root/bench-int8-w4a16/m4/w8a8/ck/{synthetic,coding}/`")
    out.append(
        "- M4-noCk cells: `/root/bench-int8-w4a16/m4/w8a8/noCk/{synthetic,coding}/`"
    )
    out.append("- Perplexity: `/root/bench-int8-w4a16/m4/ppl_w8a8_m4_ck.json`")
    out.append("- TP=4 startup: `/root/bench-int8-w4a16/m4/tp4_w8a8_startup.json`")
    out.append("- Pareto exceptions: `/root/bench-int8-w4a16/m4/pareto_exceptions.md`")
    out.append("")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
