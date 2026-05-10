#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regenerate sweep_*_summary.json with the legal-coverage block.

The original summary files predate the VAL-TRITON-003 amendment. This
regenerates them from the per-shape JSONs in configs/gfx908/ so the
coverage block is available without rerunning the bench loop.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "vllm/model_executor/kernels/configs/gfx908"
LOG_DIR = Path("/root/bench-int8-w4a16/triton/autotune")


def _shapes_for(kernel: str) -> list[dict]:
    rows = []
    for path in sorted(CONFIG_DIR.glob(f"{kernel}_M*.json")):
        blob = json.loads(path.read_text())
        shape = blob["shape"]
        rows.append({
            "shape": [
                shape["M"], shape["N"], shape["K"],
                shape.get("group_size"),
            ],
            "best_ms": blob.get("measured_ms_per_iter"),
            "best_cfg": {
                k: v for k, v in blob["config"].items()
                if k not in {"GROUP_SIZE_M", "num_warps"}
            },
            "evaluated": blob.get("autotune_runs_evaluated", 0),
            "pruned": blob.get("autotune_runs_pruned", 0),
            "cartesian_total": blob.get("autotune_cartesian_total", 0),
            "evaluated_pct_of_cart": (
                blob.get("autotune_runs_evaluated", 0)
                / blob.get("autotune_cartesian_total", 1) * 100.0
            ),
            "legal_subset": blob.get("autotune_legal_subset", 0),
            "legal_coverage_pct": blob.get(
                "autotune_legal_coverage_pct", 0.0
            ),
            "hard_invariant_eliminated": blob.get(
                "autotune_hard_invariant_eliminated", 0
            ),
            "out_path": str(path),
        })
    return rows


def _emit(kernel: str) -> None:
    rows = _shapes_for(kernel)
    if not rows:
        return
    cart_total_global = rows[0]["cartesian_total"]
    legal_avg = sum(r["legal_subset"] for r in rows) / len(rows)
    cov_avg = sum(r["legal_coverage_pct"] for r in rows) / len(rows)
    coverage_summary = {
        "cartesian_total": cart_total_global,
        "shapes_processed": len(rows),
        "average_legal_subset": legal_avg,
        "average_legal_coverage_pct": cov_avg,
        "per_shape": [
            {
                "shape": r["shape"],
                "evaluated": r["evaluated"],
                "cartesian_total": r["cartesian_total"],
                "legal_subset": r["legal_subset"],
                "legal_coverage_pct": r["legal_coverage_pct"],
                "hard_invariant_eliminated": r["hard_invariant_eliminated"],
            }
            for r in rows
        ],
    }
    name_map = {"mi100_int8": "w8a8", "mi100_w4a16": "w4a16"}
    sweep_kernel = name_map[kernel]
    out = LOG_DIR / f"sweep_{sweep_kernel}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kernel": sweep_kernel,
        "shapes": rows,
        "cartesian_total": cart_total_global,
        "coverage_summary": coverage_summary,
        "regenerated_from": "per-shape config JSONs",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2) + "\n")
    print(f"wrote {out}")
    # Pretty-print summary line for the validator log.
    for r in rows:
        shape_str = "x".join(str(x) for x in r["shape"] if x is not None)
        print(
            f"  shape {shape_str}: legal subset = {r['legal_subset']} "
            f"configs; evaluated = {r['evaluated']}; "
            f"legal-coverage = {r['evaluated']}/{r['legal_subset']} = "
            f"{r['legal_coverage_pct']:.1f}%"
        )
    print(
        f"  average legal-coverage = {cov_avg:.1f}% across {len(rows)} shapes"
    )


def main() -> int:
    _emit("mi100_int8")
    _emit("mi100_w4a16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
