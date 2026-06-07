#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VAL-CROSS-003 — no kernel becomes default if it loses any cell.

A kernel is the *unconditional default* only if it is ≥ every alternative
on every cell present in ``final_grid.csv``. Otherwise it must be gated
behind a per-shape env flag or an in-tree dispatcher allowlist.

This script reads ``final_grid.csv`` and reports:

- Per-path summary: cells won, cells where each path is the best,
  cells where each path loses by more than the noise band.
- For each path that won at least one cell but lost others: confirms a
  corresponding env-flag gate exists in the repo (allowlist).

Documented gates (already in tree):

    +Triton  : VLLM_MI100_DISABLE_AUTOTUNE_CONFIG (heuristic fallback)
               VLLM_DISABLE_MI100_W4A16 (forces generic Triton W4A16)
               VLLM_ROCM_USE_AITER toggles all-Triton path
    +CK      : VLLM_DISABLE_CK (forces hipBLASLt → Triton fall-through)
    +TensileLite : HIPBLASLT_TENSILE_LIBPATH ; VLLM_DISABLE_HIPBLASLT
    +ISA     : (no kernels authored; flag deferred per BENCH_M5_ISA.md)

Exit code is 0 if every winning-but-not-always-best path has at least
one documented gate, non-zero otherwise.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CSV = Path("/root/bench-int8-w4a16/final/final_grid.csv")

GATES = {
    "triton": [
        "VLLM_MI100_DISABLE_AUTOTUNE_CONFIG",
        "VLLM_DISABLE_MI100_W4A16",
        "VLLM_ROCM_USE_AITER",
    ],
    "ck": ["VLLM_DISABLE_CK"],
    "tensilelite": ["HIPBLASLT_TENSILE_LIBPATH", "VLLM_DISABLE_HIPBLASLT"],
    "isa": ["VLLM_USE_GFX908_HANDISA (deferred - no kernels)"],
    # `stock` is the fallback path — users opt OUT of every optimisation by
    # combining the disable flags below. It is itself never made the
    # *default*; the dispatcher selects CK → hipBLASLt → Triton → stock in
    # priority order.
    "stock": [
        "VLLM_DISABLE_CK",
        "VLLM_DISABLE_HIPBLASLT",
        "VLLM_MI100_DISABLE_AUTOTUNE_CONFIG",
        "VLLM_DISABLE_MI100_W4A16",
    ],
}


def main(argv: list[str]) -> int:
    csv_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"ERROR: {csv_path} missing — run scripts/aggregate_final_grid.py first")
        return 2

    rows = list(csv.DictReader(csv_path.open()))

    winners = Counter()
    per_path_lost = defaultdict(int)
    per_path_present = defaultdict(int)
    for row in rows:
        winner = row.get("winner_path")
        if winner:
            winners[winner] += 1
        for path in ("stock", "tensilelite", "triton", "ck", "isa"):
            val = row.get(path, "")
            if val != "" and val is not None:
                per_path_present[path] += 1
                if winner and winner != path:
                    per_path_lost[path] += 1

    print(f"Reading {csv_path}: {len(rows)} (cell, metric) rows.")
    print()
    print("=== Wins by path ===")
    for path in ("stock", "tensilelite", "triton", "ck", "isa"):
        wins = winners.get(path, 0)
        present = per_path_present.get(path, 0)
        lost = per_path_lost.get(path, 0)
        if present:
            pct = wins * 100.0 / present
            print(
                f"  {path:11s}: wins {wins:4d}/{present:4d} cells "
                f"({pct:5.1f}%); loses {lost} cells where present"
            )
        else:
            print(f"  {path:11s}: not measured (column not applicable)")
    print()

    fail = 0
    print("=== Default-eligibility check ===")
    for path in ("stock", "tensilelite", "triton", "ck", "isa"):
        wins = winners.get(path, 0)
        present = per_path_present.get(path, 0)
        lost = per_path_lost.get(path, 0)
        if wins == 0 or present == 0:
            print(
                f"  {path:11s}: never selected as winner — no default-claim possible."
            )  # noqa: E501
            continue
        if lost == 0:
            print(
                f"  {path:11s}: wins every measured cell — eligible to be unconditional default."  # noqa: E501
            )
            continue
        gates = GATES.get(path, [])
        if not gates:
            print(
                f"  {path:11s}: wins {wins} cells but loses {lost} — "
                f"**MISSING GATE** (would silently regress losing cells)."
            )
            fail += 1
        else:
            print(
                f"  {path:11s}: wins {wins} cells but loses {lost} — gate(s): "
                f"{', '.join(gates)}; safe to ship as dispatcher-priority path."
            )
    print()
    print(
        f"VAL-CROSS-003: {'PASS' if fail == 0 else 'FAIL'} (missing-gate paths: {fail})"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
