#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Aggregate top GEMM kernels from rocprofv3 kernel-trace CSVs.

`rocprofv3 --kernel-trace -o <name> -d <dir>` writes
``<dir>/<name>_kernel_trace.csv`` with at minimum:
    Kernel_Name, Start_Timestamp, End_Timestamp, [Grid_Size,
    Wave_Front_Size, ...]

We extract per-kernel total wall-clock (sum of
End_Timestamp - Start_Timestamp), filter to GEMM/MFMA/attn-style names,
group by kernel name, and report the top N kernels (N defaults to 12).

Usage:
    /opt/vllm-env/bin/python3 scripts/top_gemm_shapes.py \
        --traces /root/bench-int8-w4a16/baseline/profile/ \
        --out-md /root/bench-int8-w4a16/baseline/top_gemm_shapes.md \
        --out-csv /root/bench-int8-w4a16/baseline/top_gemm_shapes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import regex as re

GEMM_PATTERNS = [
    r"gemm",
    r"hgemm",
    r"i8gemm",
    r"int8_gemm",
    r"wmma",
    r"mfma",
    r"matmul",
    r"linear",
    r"fused_moe",
    r"flash_attn",
    r"attention",
    r"attn",
    r"_dot_kernel",
    r"split_k",
    r"awq_",
    r"gptq_",
    r"_fc",
    r"scaled_mm",
    r"int4",
    r"w4a16",
    r"matrix_mul",
    r"cijk_",  # TensileLite kernel-name prefix
]
GEMM_RE = re.compile("|".join(GEMM_PATTERNS), re.IGNORECASE)

# Try to fish (M, N, K) out of kernel names that include them.
SHAPE_PATTERNS = [
    re.compile(r"_M(\d+)N(\d+)K(\d+)"),
    re.compile(r"_(\d+)x(\d+)x(\d+)_"),
    re.compile(r"_BM(\d+)BN(\d+)BK(\d+)"),
]


def parse_shape(kernel_name: str) -> str | None:
    for r in SHAPE_PATTERNS:
        m = r.search(kernel_name)
        if m:
            return f"M={m.group(1)},N={m.group(2)},K={m.group(3)}"
    return None


def _candidate_csvs(d: Path) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(d.rglob("*kernel_trace*.csv")):
        if p.is_file() and p.stat().st_size > 0:
            paths.append(p)
    if not paths:
        # rocprof v3 sometimes drops a single 'agents.csv'+'kernel_trace.csv';
        # fall back to any csv with 'kernel' in name.
        for p in sorted(d.rglob("*.csv")):
            if p.is_file() and "kernel" in p.name.lower():
                paths.append(p)
    return paths


def aggregate_csv(path: Path) -> tuple[dict[str, dict], int]:
    """
    Returns:
      stats: kernel_name -> {ns_total, n_calls, shape}
      total_busy_ns: sum of all kernel durations in this csv
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"ns_total": 0, "n_calls": 0, "shape": None}
    )
    total_busy_ns = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # Detect rocprof v3 columns
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        name_col = cols.get("kernel_name") or cols.get("name") or cols.get("kernelname")
        start_col = cols.get("start_timestamp") or cols.get("start_ns")
        end_col = cols.get("end_timestamp") or cols.get("end_ns")
        if not name_col or not start_col or not end_col:
            print(
                f"  [warn] missing required columns in {path};"
                f" cols={list(cols.values())}",
                file=sys.stderr,
            )
            return stats, 0
        for row in reader:
            try:
                start = int(row[start_col])
                end = int(row[end_col])
            except (KeyError, ValueError):
                continue
            dur = end - start
            if dur <= 0:
                continue
            kname = row[name_col]
            stats[kname]["ns_total"] += dur
            stats[kname]["n_calls"] += 1
            if stats[kname]["shape"] is None:
                stats[kname]["shape"] = parse_shape(kname)
            total_busy_ns += dur
    return stats, total_busy_ns


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Per-trace structured JSON for downstream consumers",
    )
    args = p.parse_args()

    traces = _candidate_csvs(args.traces)
    if not traces:
        print(f"no kernel-trace CSVs under {args.traces}", file=sys.stderr)
        return 2

    print(f"found {len(traces)} kernel-trace CSV files")

    per_trace: dict[str, dict] = {}
    global_stats: dict[str, dict] = defaultdict(
        lambda: {"ns_total": 0, "n_calls": 0, "shape": None, "appears_in": set()}
    )
    grand_busy_ns = 0
    for t in traces:
        cell = t.parent.name + "/" + t.stem
        st, busy = aggregate_csv(t)
        per_trace[cell] = {
            "csv_path": str(t),
            "n_kernels_total": len(st),
            "busy_ns": busy,
            "n_records": sum(s["n_calls"] for s in st.values()),
        }
        grand_busy_ns += busy
        for kname, s in st.items():
            g = global_stats[kname]
            g["ns_total"] += s["ns_total"]
            g["n_calls"] += s["n_calls"]
            g["shape"] = g["shape"] or s["shape"]
            g["appears_in"].add(cell)

    # Filter to GEMM-like kernels
    gemm_only = {k: v for k, v in global_stats.items() if GEMM_RE.search(k)}
    if not gemm_only:
        # If nothing matches, fall back to all kernels (still report top N
        # so the worker can adjust the regex).
        print(
            "[warn] no kernels matched GEMM regex; falling back to all kernels",
            file=sys.stderr,
        )
        gemm_only = dict(global_stats)

    sorted_k = sorted(gemm_only.items(), key=lambda kv: kv[1]["ns_total"], reverse=True)
    top_n = sorted_k[: args.top]
    cum_ns = sum(v["ns_total"] for _, v in top_n)
    pct_top = 100.0 * cum_ns / max(1, grand_busy_ns)

    rows = []
    for kname, s in top_n:
        rows.append(
            {
                "kernel_name": kname,
                "shape": s["shape"] or "",
                "n_calls": s["n_calls"],
                "total_ms": s["ns_total"] / 1e6,
                "pct_of_busy": 100.0 * s["ns_total"] / max(1, grand_busy_ns),
                "appears_in": sorted(s["appears_in"]),
            }
        )

    # Stdout summary
    print(
        f"\n=== Top {args.top} GEMM kernels (across {len(traces)} traces, "
        f"grand busy={grand_busy_ns / 1e6:.1f} ms) ==="
    )
    print(f"{'rank':>4}  {'pct_busy':>8}  {'total_ms':>10}  {'n_calls':>7}  kernel")
    for i, r in enumerate(rows):
        print(
            f"{i + 1:>4}  {r['pct_of_busy']:>7.2f}%  {r['total_ms']:>9.2f}  "
            f"{r['n_calls']:>7}  {r['kernel_name'][:90]}"
        )
    print(f"\nTop-{args.top} cumulative coverage: {pct_top:.2f}% of GPU busy time")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "rank",
                    "kernel_name",
                    "shape",
                    "n_calls",
                    "total_ms",
                    "pct_of_busy",
                    "appears_in",
                ]
            )
            for i, r in enumerate(rows):
                w.writerow(
                    [
                        i + 1,
                        r["kernel_name"],
                        r["shape"],
                        r["n_calls"],
                        f"{r['total_ms']:.3f}",
                        f"{r['pct_of_busy']:.3f}",
                        ";".join(r["appears_in"]),
                    ]
                )
        print(f"wrote {args.out_csv}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Top {args.top} GEMM Shapes (M1 Baseline)",
            "",
            f"- Traces analyzed: {len(traces)}",
            f"- Grand GPU busy time: {grand_busy_ns / 1e6:.2f} ms",
            f"- Top-{args.top} cumulative coverage: **{pct_top:.2f}%**",
            "",
            "| Rank | %busy | Total ms | Calls | Shape | Kernel |",
            "|------|-------|----------|-------|-------|--------|",
        ]
        for i, r in enumerate(rows):
            kern = r["kernel_name"]
            kern = kern.replace("|", "\\|")
            lines.append(
                f"| {i + 1} | {r['pct_of_busy']:.2f}% | {r['total_ms']:.2f} | "
                f"{r['n_calls']} | {r['shape']} | `{kern[:80]}` |"
            )
        lines.append("")
        lines.append("## Per-trace busy summary")
        lines.append("")
        lines.append("| Cell | n_kernels | n_records | busy_ms |")
        lines.append("|------|-----------|-----------|---------|")
        for cell, info in sorted(per_trace.items()):
            lines.append(
                f"| {cell} | {info['n_kernels_total']} | "
                f"{info['n_records']} | {info['busy_ns'] / 1e6:.2f} |"
            )
        args.out_md.write_text("\n".join(lines))
        print(f"wrote {args.out_md}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        # serialize sets
        s_per_trace = per_trace
        s_top = []
        for r in rows:
            s_top.append({**r})
        args.out_json.write_text(
            json.dumps(
                {
                    "grand_busy_ns": grand_busy_ns,
                    "top_n_cumulative_pct": pct_top,
                    "top": s_top,
                    "per_trace": s_per_trace,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
