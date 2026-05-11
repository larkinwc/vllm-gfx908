#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VAL-FINAL-001 / VAL-FINAL-007 — validate ``BENCH_INT8_W4A16_FINAL.md``.

Two modes:

1. Default (schema): assert the report contains the required sections,
   the full Pareto grid table with the right columns, the production-
   recommendations matrix, the quality-gate row, and a non-empty
   long-context section.

2. ``--check-links``: in addition, every linked file (relative to the
   repo) must exist, and every URL-style fragment of the form
   ``BENCH_M*.md`` must resolve to a file in the repo root.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path("/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4")
DEFAULT_REPORT = REPO / "BENCH_INT8_W4A16_FINAL.md"
GRID_CSV = Path("/root/bench-int8-w4a16/final/final_grid.csv")

REQUIRED_SECTIONS = [
    "Hardware",
    "Quality",
    "Pareto",
    "Production Recommendations",
    "Long-Context",
    "Reproducibility",
    "Cross-cutting",
]

PARETO_HEADER_RE = re.compile(
    r"\|\s*cell\s*\|\s*workload\s*\|\s*metric\s*\|\s*stock\s*\|\s*\+TensileLite\s*\|"
    r"\s*\+Triton\s*\|\s*\+CK\s*\|\s*\+ISA\s*\|\s*winner\s*\|",
    re.IGNORECASE,
)

RECOMMENDATION_HEADER_RE = re.compile(
    r"\|\s*model\s*shape.*\|\s*TP\s*\|\s*concurrency\s*\|.*recommend.*\|",
    re.IGNORECASE,
)

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
BENCH_REF_RE = re.compile(r"`?(BENCH_[A-Za-z0-9_]*\.md)`?")


def check_schema(report: str, errors: list[str]) -> None:
    for s in REQUIRED_SECTIONS:
        if s.lower() not in report.lower():
            errors.append(f"Missing required section/keyword: {s}")

    if not PARETO_HEADER_RE.search(report):
        errors.append(
            "Pareto grid table header not found (need columns: cell, workload, "
            "metric, stock, +TensileLite, +Triton, +CK, +ISA, winner)"
        )
    if not RECOMMENDATION_HEADER_RE.search(report):
        errors.append(
            "Production-recommendation matrix header not found"
        )

    # Cross-reference grid row count vs CSV row count (excluding header).
    if GRID_CSV.exists():
        with GRID_CSV.open() as f:
            csv_rows = sum(1 for _ in csv.DictReader(f))
    else:
        csv_rows = -1
    md_grid_rows = report.count("| w8a8_tp")
    md_grid_rows += report.count("| w4a16_tp")
    if csv_rows >= 0 and md_grid_rows < csv_rows:
        errors.append(
            f"Pareto-grid rows in markdown ({md_grid_rows}) "
            f"< CSV rows ({csv_rows}); some cells are missing from the table"
        )

    # 24 launch scripts must exist.
    missing_launches = []
    for model in ("w8a8", "w4a16"):
        for tp in (1, 4):
            for c in (1, 2, 4):
                p = REPO / "scripts" / f"launch_{model}_tp{tp}_c{c}.sh"
                if not p.exists():
                    missing_launches.append(p.name)
    if missing_launches:
        errors.append(
            f"Missing per-cell launch scripts: {missing_launches}"
        )

    # Quality numbers must appear.
    for s in ("perplexity", "9/10", "5/5", "needle"):
        if s.lower() not in report.lower():
            errors.append(f"Missing quality-gate keyword: {s}")


def check_links(report: str, errors: list[str]) -> None:
    # Markdown link targets relative to repo or /root/bench-int8-w4a16/
    for label, target in LINK_RE.findall(report):
        # Skip anchors and external URLs.
        if target.startswith("#") or target.startswith("http"):
            continue
        # Strip optional anchor.
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        # Resolve relative to repo root (where the .md lives).
        cand = (REPO / path_part).resolve()
        if not cand.exists():
            # Try absolute path interpretation.
            if not Path(path_part).exists():
                errors.append(f"Broken link [{label}] → {target}")
    # Bench-cross-reference style: must resolve relative to repo.
    for ref in set(BENCH_REF_RE.findall(report)):
        if ref == "BENCH_INT8_W4A16_FINAL.md":
            continue
        if not (REPO / ref).exists():
            errors.append(f"Cross-reference {ref} not present in repo")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--check-links", action="store_true")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"FATAL: report {args.path} missing")
        return 2
    report = args.path.read_text()

    errors: list[str] = []
    check_schema(report, errors)
    if args.check_links:
        check_links(report, errors)

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        print(f"\n{args.path}: {len(errors)} validation error(s)")
        return 1

    out_log = Path("/root/bench-int8-w4a16/final/m6_link_check.txt")
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text(
        f"validate_final_report.py {('--check-links ' if args.check_links else '')}{args.path}: PASS\n"
    )
    print(f"{args.path}: schema OK; link-check {'ON' if args.check_links else 'OFF'}; "
          f"wrote {out_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
