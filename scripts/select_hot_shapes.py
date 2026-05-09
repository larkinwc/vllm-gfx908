#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Pick the top 1-2 hot GEMM shapes per (regime, quant scheme) from the
profile traces and write ``hot_shapes.json`` for downstream consumption
by the M2/M3/M4 kernel workers.

Regimes:
  tp1c1 = TP=1 concurrency=1 (decode-heavy)
  tp4c4 = TP=4 concurrency=4 (prefill+decode mix)

Quants:
  w8a8 = compressed_tensors_w8a8_int8
  w4a16 = gptq_marlin / mixed_precision

Trace files are named ``rocprof_<quant>_tp{1,4}_c{1,4}_kernel_trace.csv``
under ``--traces <dir>``.

Usage:
    /opt/vllm-env/bin/python3 scripts/select_hot_shapes.py \
        --traces /root/bench-int8-w4a16/baseline/profile/ \
        --top 2 --regimes tp1c1,tp4c4 \
        --out /root/bench-int8-w4a16/baseline/hot_shapes.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the GEMM regex
from top_gemm_shapes import GEMM_RE, parse_shape  # type: ignore[import-not-found]


def _trace_files(traces_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """
    Map (quant, regime) -> list of CSV paths.
    The trace filenames follow: rocprof_w8a8_tp1_c1_kernel_trace.csv etc.
    Also tolerate per-cell subdirs.
    """
    out: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in sorted(traces_dir.rglob("*kernel_trace*.csv")):
        name = p.name.lower() + " " + p.parent.name.lower()
        if "w8a8" in name:
            quant = "w8a8"
        elif "w4a16" in name:
            quant = "w4a16"
        else:
            continue
        if "tp1" in name and ("c1" in name or "_c1_" in name):
            regime = "tp1c1"
        elif "tp4" in name and ("c4" in name or "_c4_" in name):
            regime = "tp4c4"
        elif "tp1" in name and "c4" in name:
            regime = "tp1c4"
        elif "tp4" in name and "c1" in name:
            regime = "tp4c1"
        else:
            continue
        out[(quant, regime)].append(p)
    return out


def _aggregate(csv_paths: list[Path]) -> tuple[dict[str, dict], int]:
    stats: dict[str, dict] = defaultdict(
        lambda: {"ns_total": 0, "n_calls": 0, "shape": None}
    )
    total_busy = 0
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            name_col = (
                cols.get("kernel_name") or cols.get("name") or cols.get("kernelname")
            )
            start_col = cols.get("start_timestamp") or cols.get("start_ns")
            end_col = cols.get("end_timestamp") or cols.get("end_ns")
            if not name_col or not start_col or not end_col:
                continue
            for row in reader:
                try:
                    dur = int(row[end_col]) - int(row[start_col])
                except (KeyError, ValueError):
                    continue
                if dur <= 0:
                    continue
                kn = row[name_col]
                if not GEMM_RE.search(kn):
                    continue
                stats[kn]["ns_total"] += dur
                stats[kn]["n_calls"] += 1
                if stats[kn]["shape"] is None:
                    stats[kn]["shape"] = parse_shape(kn)
                total_busy += dur
    return stats, total_busy


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--regimes", default="tp1c1,tp4c4")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    files = _trace_files(args.traces)

    selections: list[dict] = []
    for regime in regimes:
        for quant in ("w8a8", "w4a16"):
            paths = files.get((quant, regime), [])
            if not paths:
                print(f"[warn] no traces for {quant}/{regime}", file=sys.stderr)
                continue
            stats, busy = _aggregate(paths)
            ranked = sorted(
                stats.items(), key=lambda kv: kv[1]["ns_total"], reverse=True
            )[: args.top]
            for rank, (kn, s) in enumerate(ranked, 1):
                selections.append(
                    {
                        "regime": regime,
                        "quant": quant,
                        "rank": rank,
                        "kernel_name": kn,
                        "shape": s["shape"],
                        "n_calls": s["n_calls"],
                        "total_ms": s["ns_total"] / 1e6,
                        "pct_of_regime_busy": 100.0 * s["ns_total"] / max(1, busy),
                        "trace_files": [str(p) for p in paths],
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"hot_shapes": selections}, indent=2))
    print(f"wrote {args.out} ({len(selections)} selections)")
    print(
        f"{'regime':>8} {'quant':>6} {'rank':>4} {'pct':>6} {'ms':>9}  "
        f"{'shape':>20}  kernel"
    )
    for s in selections:
        print(
            f"{s['regime']:>8} {s['quant']:>6} {s['rank']:>4} "
            f"{s['pct_of_regime_busy']:>5.1f}% {s['total_ms']:>8.2f}  "
            f"{s['shape'] or '':>20}  {s['kernel_name'][:80]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
