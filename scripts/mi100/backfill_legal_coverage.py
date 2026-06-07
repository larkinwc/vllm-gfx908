#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backfill legal-coverage metadata into existing autotune config JSONs.

The original M3 autotune sweep emitted ``autotune_runs_evaluated`` and
``autotune_runs_pruned`` but did not distinguish between configs eliminated
by hard kernel invariants (e.g. ``BLOCK_K > group_size`` for W4A16) and
configs eliminated by shape-specific filters. VAL-TRITON-003 amend (2026-
05-10) requires the coverage denominator to be the LEGAL subset rather
than the full cartesian.

This script re-derives the legal-subset metadata from the same prune
predicates used in ``scripts/mi100/autotune_sweep.py`` and writes the
extra fields into each config JSON without re-running the costly bench
loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "mi100"))

from autotune_sweep import (  # noqa: E402
    _enumerate_configs,
    _prune_w4a16,
    _prune_w8a8,
)

CONFIG_DIR = REPO / "vllm/model_executor/kernels/configs/gfx908"

W8A8_HARD = ("lds_overflow", "nonkdim32_needs_>=32_tiles")
W4A16_HARD = (
    "block_k_gt_group",
    "block_n_not_multiple_of_8",
    "lds_overflow",
    "nonkdim32_needs_>=32_tiles",
)


def _legal_for_shape(
    kernel: str, M: int, N: int, K: int, group_size: int | None
) -> tuple[int, int, int]:
    cart_total = 0
    legal = 0
    eliminated_hard = 0
    full_cart = _enumerate_configs()
    cart_total = len(full_cart)
    hard = W8A8_HARD if kernel == "mi100_int8" else W4A16_HARD
    for cfg in full_cart:
        if kernel == "mi100_int8":
            r = _prune_w8a8(cfg, M, N, K)
        else:
            r = _prune_w4a16(cfg, M, N, K, group_size or 128)
        if r is None:
            legal += 1
            continue
        if any(r.startswith(h) for h in hard):
            eliminated_hard += 1
        else:
            legal += 1
    return cart_total, legal, eliminated_hard


def main() -> int:
    updated = 0
    for path in sorted(CONFIG_DIR.glob("mi100_*.json")):
        if path.name == "hipblaslt_tuned_shapes.json":
            continue
        blob = json.loads(path.read_text())
        kernel = blob.get("kernel")
        shape = blob.get("shape", {})
        M = shape.get("M")
        N = shape.get("N")
        K = shape.get("K")
        gs = shape.get("group_size")
        if not (kernel and M and N and K):
            continue
        cart_total, legal, hard_eliminated = _legal_for_shape(kernel, M, N, K, gs)
        evaluated = blob.get("autotune_runs_evaluated", 0)
        legal_pct = (evaluated / legal * 100.0) if legal else 0.0
        blob["autotune_cartesian_total"] = cart_total
        blob["autotune_legal_subset"] = legal
        blob["autotune_legal_coverage_pct"] = legal_pct
        blob["autotune_hard_invariant_eliminated"] = hard_eliminated
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(
            f"{path.name}: cart={cart_total} legal={legal} "
            f"evaluated={evaluated} legal_cov={legal_pct:.1f}% "
            f"hard_elim={hard_eliminated}"
        )
        updated += 1
    print(f"updated {updated} files in {CONFIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
