#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
M3-F1 per-cell delta aggregator.

Compares the redundant-silu mission's full 24-cell M3 grid against the
prior mission's production baselines at /root/bench-int8-w4a16/baseline/raw_*/raw.json
(most-recent mtime per (cell, workload)).

Emits a markdown delta table with:
  - output_throughput_toks_s (higher_better)
  - mean_ttft_ms (lower_better)

Cell layout:
  M3 cells: /root/bench-int8-w4a16-redundant-silu/m3-bench/{w8a8,w4a16}/<cell>_<wl>.json
            (harness-wrapped schema: top-level output_throughput_toks_s, mean_ttft_ms)
  Baseline: /root/bench-int8-w4a16/baseline/raw_<cell>_<wl>_<suffix>/raw.json
            (raw vLLM-bench schema: top-level output_throughput, mean_ttft_ms)
"""

from __future__ import annotations

import json
from pathlib import Path

M3_ROOT = Path("/root/bench-int8-w4a16-redundant-silu/m3-bench")
BASELINE_ROOT = Path("/root/bench-int8-w4a16/baseline")

QUANTS = ("w8a8", "w4a16")
TPS = (1, 4)
CONCS = (1, 2, 4)
WLS = ("synthetic", "coding")


def load_baseline() -> dict[tuple[str, str], dict]:
    """(cell_id, workload) -> raw.json (most-recent mtime wins)."""
    base: dict[tuple[str, str], dict] = {}
    mtimes: dict[tuple[str, str], float] = {}
    for d in sorted(BASELINE_ROOT.glob("raw_*")):
        parts = d.name.split("_")
        if len(parts) < 6:
            continue
        cell_id = "_".join(parts[1:4])  # w8a8_tp1_c1
        wl = parts[4]
        rj = d / "raw.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        m = rj.stat().st_mtime
        key = (cell_id, wl)
        if key not in base or m > mtimes[key]:
            base[key] = data
            mtimes[key] = m
    return base


def pct(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return (new - old) / old * 100.0


def fmt(d: float) -> str:
    if d != d:
        return "—"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.2f}%"


def main() -> int:
    base = load_baseline()
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for quant in QUANTS:
        for tp in TPS:
            for c in CONCS:
                cell_id = f"{quant}_tp{tp}_c{c}"
                for wl in WLS:
                    cell_path = M3_ROOT / quant / f"{cell_id}_{wl}.json"
                    if not cell_path.exists():
                        missing.append(f"{cell_id}_{wl}")
                        continue
                    cell = json.loads(cell_path.read_text())
                    b = base.get((cell_id, wl), {})
                    new_tput = float(cell.get("output_throughput_toks_s") or 0.0)
                    new_ttft = float(cell.get("mean_ttft_ms") or 0.0)
                    old_tput = float(b.get("output_throughput") or 0.0)
                    old_ttft = float(b.get("mean_ttft_ms") or 0.0)
                    rows.append(
                        {
                            "quant": quant,
                            "cell_id": cell_id,
                            "workload": wl,
                            "baseline_tput": old_tput,
                            "m3_tput": new_tput,
                            "d_tput": pct(new_tput, old_tput),
                            "baseline_ttft": old_ttft,
                            "m3_ttft": new_ttft,
                            "d_ttft": pct(new_ttft, old_ttft),
                            "failed": bool(cell.get("FAILED")) or new_tput == 0.0,
                        }
                    )

    out: list[str] = []
    out.append(
        "# M3-F1 per-cell delta table (full 24-cell grid vs production baseline)"
    )
    out.append("")
    out.append("Compares the M1-promoted (placeholder-view) M3 grid against the")
    out.append("prior mission's per-cell production baselines at")
    out.append(
        "`/root/bench-int8-w4a16/baseline/raw_*/raw.json` (most-recent mtime per cell)."
    )
    out.append("")
    out.append(
        "- Δ throughput = (M3 − baseline) / baseline × 100  (higher = improvement)"
    )
    out.append(
        "- Δ mean_ttft  = (M3 − baseline) / baseline × 100  (lower = improvement)"
    )
    out.append("- Baseline JSON shape: raw vLLM-bench (top-level `output_throughput`).")
    out.append(
        "- M3 JSON shape:       harness-wrapped (top-level `output_throughput_toks_s`)."
    )
    out.append("")
    if missing:
        out.append(f"**Missing M3 cells**: {len(missing)} — {', '.join(missing)}")
        out.append("")

    # W8A8 first, then W4A16
    for quant in QUANTS:
        qrows = [r for r in rows if r["quant"] == quant]
        if not qrows:
            continue
        out.append(f"## {quant.upper()} ({len(qrows)} cells)")
        out.append("")
        out.append(
            "| cell | wl | baseline_tput | M3_tput | Δ_tput | "
            "baseline_ttft | M3_ttft | Δ_ttft | status |"
        )
        out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in qrows:
            status = "FAILED" if r["failed"] else "ok"
            out.append(
                f"| {r['cell_id']} | {r['workload']} | "
                f"{r['baseline_tput']:.3f} | {r['m3_tput']:.3f} | {fmt(r['d_tput'])} | "
                f"{r['baseline_ttft']:.2f} | {r['m3_ttft']:.2f} | {fmt(r['d_ttft'])} | "
                f"{status} |"
            )
        out.append("")
        # Per-quant summary stats
        ok_rows = [r for r in qrows if not r["failed"]]
        if ok_rows:
            ds = [r["d_tput"] for r in ok_rows if r["d_tput"] == r["d_tput"]]
            if ds:
                out.append(
                    f"- {quant.upper()} Δ_tput stats: min={min(ds):+.2f}% "
                    f"max={max(ds):+.2f}% mean={sum(ds) / len(ds):+.2f}%"
                )
            dt = [r["d_ttft"] for r in ok_rows if r["d_ttft"] == r["d_ttft"]]
            if dt:
                out.append(
                    f"- {quant.upper()} Δ_ttft stats: min={min(dt):+.2f}% "
                    f"max={max(dt):+.2f}% mean={sum(dt) / len(dt):+.2f}%"
                )
            out.append("")

    # Notable regressions/wins
    bigwins = [r for r in rows if not r["failed"] and r["d_tput"] >= 3.0]
    bigregs = [r for r in rows if not r["failed"] and r["d_tput"] <= -3.0]
    out.append("## Notable cells")
    out.append("")
    out.append(f"- Wins ≥ +3% tput: {len(bigwins)}")
    for r in bigwins:
        out.append(
            f"  - {r['cell_id']}_{r['workload']}: "
            f"{fmt(r['d_tput'])} tput, {fmt(r['d_ttft'])} ttft"
        )
    out.append(f"- Regressions ≤ −3% tput: {len(bigregs)}")
    for r in bigregs:
        out.append(
            f"  - {r['cell_id']}_{r['workload']}: "
            f"{fmt(r['d_tput'])} tput, {fmt(r['d_ttft'])} ttft"
        )
    out.append("")

    out_path = M3_ROOT / "delta_table.md"
    out_path.write_text("\n".join(out) + "\n")
    print(f"wrote {out_path}")
    print(f"rows={len(rows)} missing={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
