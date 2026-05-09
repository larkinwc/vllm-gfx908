#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Pick a small set of representative W8A8 (M, N, K) shapes to drive the
TensileLite tuning campaign and the dispatcher allowlist.

Strategy:
  - Trust the aggregated CSV from dump_gemm_shapes.py (M, N, K, n_calls).
  - Cluster by (N, K) since those are dictated by model architecture
    (qkv_proj / o_proj / gate_up_proj / down_proj for Qwen3.5-9B).
  - Per (N, K) cluster, keep the top-K M values by call count, but cap
    the total number of (M, N, K) tuples to keep tuning feasible.

Outputs:
  - tuning manifest JSON (sorted, keyed by (M, N, K))
  - human-readable shape catalog markdown
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=Path, required=True,
                    help="Aggregated CSV from dump_gemm_shapes.py")
    ap.add_argument("--out-json", type=Path, required=True,
                    help="Path to write the tuning shape manifest.")
    ap.add_argument("--catalog-md", type=Path, required=True,
                    help="Human-readable markdown catalog.")
    ap.add_argument("--top-per-cluster", type=int, default=2,
                    help="Number of top-M values to keep per (N, K).")
    ap.add_argument("--max-shapes", type=int, default=8,
                    help="Total cap on selected shapes.")
    args = ap.parse_args()

    rows = []
    with args.in_csv.open() as f:
        f.readline()  # discard header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                M, N, K, n = (int(parts[0]), int(parts[1]), int(parts[2]),
                              int(parts[3]))
            except ValueError:
                continue
            rows.append({"M": M, "N": N, "K": K, "n_calls": n})

    rows.sort(key=lambda r: r["n_calls"], reverse=True)

    # Cluster by (N, K) and pick the top M per cluster.
    clusters: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        clusters[(r["N"], r["K"])].append(r)

    selected: list[dict] = []
    cluster_summaries = []
    for (N, K), entries in sorted(clusters.items(),
                                  key=lambda kv: -sum(e["n_calls"] for e
                                                      in kv[1])):
        entries.sort(key=lambda e: -e["n_calls"])
        cluster_total_calls = sum(e["n_calls"] for e in entries)
        cluster_summaries.append({
            "N": N, "K": K,
            "total_calls": cluster_total_calls,
            "n_unique_M": len(entries),
        })
        for e in entries[: args.top_per_cluster]:
            selected.append(e)
        if len(selected) >= args.max_shapes:
            break

    selected = selected[: args.max_shapes]
    selected.sort(key=lambda e: (-e["n_calls"], e["M"], e["N"], e["K"]))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "schema_version": 1,
        "source_csv": str(args.in_csv),
        "shape_count": len(selected),
        "shapes": selected,
        "cluster_summary": cluster_summaries[:8],
    }, indent=2))

    md = ["# W8A8 TensileLite tuning shape catalog",
          "",
          f"Source: `{args.in_csv}`",
          f"Selected {len(selected)} (M, N, K) tuples for tuning.",
          "",
          "## Top (N, K) clusters by call count",
          "",
          "| N | K | total_calls | unique_M |",
          "| --- | --- | --- | --- |"]
    for c in cluster_summaries[:8]:
        md.append(f"| {c['N']} | {c['K']} | {c['total_calls']} | "
                  f"{c['n_unique_M']} |")
    md += ["", "## Selected shapes for tuning", "",
           "| M | N | K | n_calls |", "| --- | --- | --- | --- |"]
    for r in selected:
        md.append(f"| {r['M']} | {r['N']} | {r['K']} | {r['n_calls']} |")
    args.catalog_md.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_md.write_text("\n".join(md) + "\n")

    print(f"Selected {len(selected)} shapes -> {args.out_json}")
    for r in selected:
        print(f"  M={r['M']:5d} N={r['N']:5d} K={r['K']:5d} "
              f"n_calls={r['n_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
