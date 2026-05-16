#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate M3 grid results into a markdown table.

Compares:
  - W8A8 m3-autotune vs M1-rebaseline vs m3-heuristic
  - W4A16 m3-mi100 vs baseline vs m3-generic

Outputs a table per quant scheme + pareto gate verdict.

Usage:
    /opt/vllm-env/bin/python3 scripts/mi100/aggregate_m3.py > BENCH_INT8_W4A16_M3.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

W8A8_CELLS = [
    ("w8a8", tp, c, wl)
    for tp in (1, 4)
    for c in (1, 2, 4)
    for wl in ("synthetic", "coding")
]
W4A16_CELLS = [
    ("w4a16", tp, c, wl)
    for tp in (1, 4)
    for c in (1, 2, 4)
    for wl in ("synthetic", "coding")
]

ROOTS_W8A8 = {
    "M1rb":      Path("/root/bench-int8-w4a16/m1-rebaseline"),
    "M3-auto":   Path("/root/bench-int8-w4a16/m3/w8a8/autotune"),
    "M3-heur":   Path("/root/bench-int8-w4a16/m3/w8a8/heuristic"),
}
ROOTS_W4A16 = {
    "M1":         Path("/root/bench-int8-w4a16/baseline"),
    "M3-mi100":   Path("/root/bench-int8-w4a16/m3/w4a16/mi100"),
    "M3-generic": Path("/root/bench-int8-w4a16/m3/w4a16/generic"),
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
    if d != d:
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


def write_w8a8(out: list[str]) -> tuple[int, int]:
    out.append("## W8A8: M3-autotune vs M1-rebaseline vs M3-heuristic")
    out.append("")
    out.append(
        "Three-way comparison: M1-rebaseline (M2 build, prebuilt I8I8 default "
        "kernel) → M3 with VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1 (heuristic "
        "fallback Triton) → M3 with autotune configs active. Autotune "
        "configs persisted at `vllm/model_executor/kernels/configs/gfx908/"
        "mi100_int8_*.json`."
    )
    out.append("")
    out.append(
        "| Cell | Workload | Metric | M1-rb | M3-heur | M3-auto | "
        "Δ heur→auto | Δ M1rb→auto (**gate**) |"
    )
    out.append(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    n_pareto = 0
    n_regression = 0
    for model, tp, c, wl in W8A8_CELLS:
        cells = {k: load_cell(r, model, tp, c, wl)
                 for k, r in ROOTS_W8A8.items()}
        if any(v is None for v in cells.values()):
            continue
        cell_id = f"{model}_tp{tp}_c{c}"
        for key, label, direction in METRICS:
            m1rb = cells["M1rb"][key]
            heur = cells["M3-heur"][key]
            auto = cells["M3-auto"][key]
            d_h_a = pct(auto, heur)
            d_m1rb_auto = pct(auto, m1rb)
            out.append(
                f"| {cell_id} | {wl} | {label} | "
                f"{m1rb:.2f} | {heur:.2f} | {auto:.2f} | "
                f"{fmt_delta(d_h_a, direction)} | "
                f"{fmt_delta(d_m1rb_auto, direction)} |"
            )
            is_imp = (
                (direction == "higher_better" and d_m1rb_auto >= 3.0) or
                (direction == "lower_better" and d_m1rb_auto <= -3.0)
            )
            is_reg5 = (
                (direction == "higher_better" and d_m1rb_auto <= -5.0) or
                (direction == "lower_better" and d_m1rb_auto >= 5.0)
            )
            if is_imp:
                n_pareto += 1
            if is_reg5:
                n_regression += 1
    out.append("")
    out.append(f"**Pareto bar (≥3% gain M1rb→M3-autotune):** {n_pareto} metric(s)")
    out.append(f"**>5% regressions (M1rb→M3-autotune):** {n_regression} metric(s)")
    out.append("")
    return n_pareto, n_regression


def write_w4a16(out: list[str]) -> int:
    out.append("## W4A16: M3-mi100 vs Baseline vs M3-generic")
    out.append("")
    out.append(
        "Two-way M3 comparison: M3 with the new `mi100_w4a16` Triton "
        "kernel (autotune configs active) vs M3 with "
        "`VLLM_DISABLE_MI100_W4A16=1` forcing the generic Triton path. "
        "Baseline column is M1 (commit `fc20b6f4f`)."
    )
    out.append("")
    out.append(
        "| Cell | Workload | Metric | M1 | M3-generic | M3-mi100 | "
        "Δ generic→mi100 | Δ M1→mi100 |"
    )
    out.append(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    n_pareto = 0
    for model, tp, c, wl in W4A16_CELLS:
        cells = {k: load_cell(r, model, tp, c, wl)
                 for k, r in ROOTS_W4A16.items()}
        if any(v is None for v in cells.values()):
            continue
        cell_id = f"{model}_tp{tp}_c{c}"
        for key, label, direction in METRICS:
            m1 = cells["M1"][key]
            gen = cells["M3-generic"][key]
            mi = cells["M3-mi100"][key]
            d_g_m = pct(mi, gen)
            d_m1_m = pct(mi, m1)
            out.append(
                f"| {cell_id} | {wl} | {label} | "
                f"{m1:.2f} | {gen:.2f} | {mi:.2f} | "
                f"{fmt_delta(d_g_m, direction)} | "
                f"{fmt_delta(d_m1_m, direction)} |"
            )
            is_imp = (
                (direction == "higher_better" and d_g_m >= 3.0) or
                (direction == "lower_better" and d_g_m <= -3.0)
            )
            if is_imp:
                n_pareto += 1
    out.append("")
    out.append(
        f"**Pareto bar (≥3% gain generic→mi100):** {n_pareto} metric(s)"
    )
    out.append("")
    return n_pareto


def write_perplexity(out: list[str]) -> None:
    out.append("## Wikitext-2 Perplexity (50 chunks × 512 tokens, seed 0)")
    out.append("")
    out.append("| Run | Perplexity | Δ vs reference |")
    out.append("| --- | ---: | ---: |")
    files = [
        ("M1-rebaseline (W8A8 reference)",
         Path("/root/bench-int8-w4a16/m1-rebaseline/ppl_w8a8_m1rb.json")),
        ("M3-autotune W8A8",
         Path("/root/bench-int8-w4a16/m3/ppl_w8a8_m3.json")),
        ("M0 baseline (W4A16 reference)",
         Path("/root/bench-int8-w4a16/baseline/ppl_w4a16.json")),
        ("M3-mi100 W4A16",
         Path("/root/bench-int8-w4a16/m3/ppl_w4a16_m3.json")),
    ]
    ref_w8a8 = None
    ref_w4a16 = None
    rows = []
    for label, p in files:
        if not p.exists():
            rows.append((label, None, None))
            continue
        try:
            blob = json.loads(p.read_text())
            ppl = blob.get("perplexity")
            rows.append((label, ppl, p))
        except json.JSONDecodeError:
            rows.append((label, None, None))
    for label, ppl, p in rows:
        if ppl is None:
            out.append(f"| {label} | (missing) | — |")
            continue
        if "M1-rebaseline" in label:
            ref_w8a8 = ppl
            out.append(f"| {label} | {ppl:.4f} | — (reference) |")
        elif "W4A16 reference" in label or "baseline (W4A16" in label:
            ref_w4a16 = ppl
            out.append(f"| {label} | {ppl:.4f} | — (reference) |")
        elif "W8A8" in label and ref_w8a8 is not None:
            d = pct(ppl, ref_w8a8)
            sign = "+" if d > 0 else ""
            verdict = " ✅ (≤ +1%)" if d <= 1.0 else " ❌ (> +1%)"
            out.append(f"| {label} | {ppl:.4f} | {sign}{d:.3f}%{verdict} |")
        elif "W4A16" in label and ref_w4a16 is not None:
            d = pct(ppl, ref_w4a16)
            sign = "+" if d > 0 else ""
            verdict = " ✅ (≤ +1%)" if d <= 1.0 else " ❌ (> +1%)"
            out.append(f"| {label} | {ppl:.4f} | {sign}{d:.3f}%{verdict} |")
        else:
            out.append(f"| {label} | {ppl:.4f} | — |")
    out.append("")


def main() -> int:
    out: list[str] = []
    out.append("# BENCH M3 — Triton W8A8 / W4A16 grid + perplexity")
    out.append("")
    out.append(
        "Companion to `BENCH_INT8_W4A16_M2.md`. M3 lands the custom "
        "Triton W8A8 + W4A16 kernels (commit `9699d1f0a`) and re-runs "
        "the 24-cell grid against the M1-rebaseline column "
        "(`73f7e629a`)."
    )
    out.append("")
    out.append("## Reproduction")
    out.append("")
    out.append("```")
    out.append(
        "# Autotune configs already in"
        " vllm/model_executor/kernels/configs/gfx908/"
    )
    out.append("scripts/mi100/run_grid.sh m3-w8a8-autotune w8a8_")
    out.append("scripts/mi100/run_grid.sh m3-w8a8-heuristic w8a8_")
    out.append("scripts/mi100/run_grid.sh m3-w4a16-mi100 w4a16_")
    out.append("scripts/mi100/run_grid.sh m3-w4a16-generic w4a16_")
    out.append("```")
    out.append("")
    out.append("**Triage levers if M3 regresses:**")
    out.append("")
    out.append(
        "- W8A8: `VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1` reverts to "
        "the heuristic fallback path."
    )
    out.append(
        "- W4A16: `VLLM_DISABLE_MI100_W4A16=1` reverts to the generic "
        "Triton W4A16 path."
    )
    out.append("")

    n_pareto_w8a8, n_reg_w8a8 = write_w8a8(out)
    n_pareto_w4a16 = write_w4a16(out)
    write_perplexity(out)

    out.append("## Headline")
    out.append("")
    out.append(
        f"- W8A8 M3-autotune-vs-M1rb: {n_pareto_w8a8} metric(s) ≥ 3% gain; "
        f"{n_reg_w8a8} metric(s) regress > 5%."
    )
    out.append(
        f"- W4A16 mi100-vs-generic: {n_pareto_w4a16} metric(s) ≥ 3% gain."
    )
    out.append("")
    out.append("## Files")
    out.append("")
    out.append(
        "- W8A8 autotune cells: "
        "`/root/bench-int8-w4a16/m3/w8a8/autotune/{synthetic,coding}/`"
    )
    out.append(
        "- W8A8 heuristic cells: "
        "`/root/bench-int8-w4a16/m3/w8a8/heuristic/{synthetic,coding}/`"
    )
    out.append(
        "- W4A16 mi100 cells: "
        "`/root/bench-int8-w4a16/m3/w4a16/mi100/{synthetic,coding}/`"
    )
    out.append(
        "- W4A16 generic cells: "
        "`/root/bench-int8-w4a16/m3/w4a16/generic/{synthetic,coding}/`"
    )
    out.append(
        "- Pareto exceptions: `/root/bench-int8-w4a16/m3/pareto_exceptions.md`"
    )
    out.append("")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
