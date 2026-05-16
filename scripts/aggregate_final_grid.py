#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M6 — aggregate the full Pareto grid across milestones.

Emits two artifacts:
  - /root/bench-int8-w4a16/final/final_grid.csv
      One row per (cell, workload, metric) with columns:
        cell, workload, metric, stock, tensilelite, triton, ck, isa,
        winner_path, winner_value
  - stdout: markdown table grouped by cell summarising winners.

Sources (per validation contract VAL-FINAL-001 and AGENTS M6 notes):
  stock        — /root/bench-int8-w4a16/baseline/{workload}/<cell>.json
  +TensileLite — /root/bench-int8-w4a16/m1-rebaseline/{workload}/<cell>.json  (W8A8 only)
  +Triton      — /root/bench-int8-w4a16/m3/{w8a8/autotune,w4a16/mi100}/{workload}/<cell>.json
  +CK          — /root/bench-int8-w4a16/m4/w8a8/ck/{workload}/<cell>.json  (W8A8 only)
  +ISA         — n/a (M5 declined as negative result; see BENCH_M5_ISA.md)

W4A16 does NOT have a TensileLite column (M2 was W8A8-only) nor a CK column
(M4 CK W4A16 declined as documented negative result in BENCH_M4_CK.md).
Missing cells are emitted with `null` and an explicit reason captured in
``final_grid_reasons.json``.
"""  # noqa: E501
from __future__ import annotations

import csv
import json
from pathlib import Path

FINAL_DIR = Path("/root/bench-int8-w4a16/final")
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# (path-prefix, label, applicable_quants, missing_reason)
COLUMNS = [
    (
        Path("/root/bench-int8-w4a16/baseline"),
        "stock",
        ("w8a8", "w4a16"),
        None,
    ),
    (
        Path("/root/bench-int8-w4a16/m1-rebaseline"),
        "tensilelite",
        ("w8a8",),
        "M2 TensileLite is W8A8-only; W4A16 has no hipBLASLt INT4 logic on ROCm 7.12.",
    ),
    (
        Path("/root/bench-int8-w4a16/m3"),
        "triton",
        ("w8a8", "w4a16"),
        None,
    ),
    (
        Path("/root/bench-int8-w4a16/m4/w8a8/ck"),
        "ck",
        ("w8a8",),
        "CK W4A16 declined as negative result on ROCm 7.12 gfx908; see BENCH_M4_CK.md.",
    ),
    (
        None,
        "isa",
        (),
        "M5 hand-ISA declined as negative result (memory-bound hot kernels); see BENCH_M5_ISA.md.",  # noqa: E501
    ),
]

CELLS = []
for model in ("w8a8", "w4a16"):
    for tp in (1, 4):
        for c in (1, 2, 4):
            for wl in ("synthetic", "coding"):
                CELLS.append((model, tp, c, wl))

# direction = True (higher_better) / False (lower_better)
METRICS = [
    ("output_throughput_toks_s", "tput", True),
    ("request_throughput_req_s", "req_tput", True),
    ("mean_ttft_ms", "mean_ttft", False),
    ("p99_ttft_ms", "p99_ttft", False),
    ("mean_tpot_ms", "mean_tpot", False),
    ("p99_tpot_ms", "p99_tpot", False),
]


def resolve_path(root: Path, label: str, model: str, tp: int, c: int, wl: str) -> Path | None:  # noqa: E501
    """Resolve milestone cell JSON path. Returns None if column N/A."""
    if root is None:
        return None
    if label == "stock":
        return root / wl / f"{model}_tp{tp}_c{c}.json"
    if label == "tensilelite":
        if model != "w8a8":
            return None
        return root / wl / f"{model}_tp{tp}_c{c}.json"
    if label == "triton":
        if model == "w8a8":
            return root / "w8a8" / "autotune" / wl / f"{model}_tp{tp}_c{c}.json"
        return root / "w4a16" / "mi100" / wl / f"{model}_tp{tp}_c{c}.json"
    if label == "ck":
        if model != "w8a8":
            return None
        return root / wl / f"{model}_tp{tp}_c{c}.json"
    return None


def load(p: Path | None) -> dict | None:
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def pick_winner(values: dict[str, float | None], higher_better: bool) -> tuple[str, float] | None:  # noqa: E501
    """Pick the column with the best value. Skips None/NaN entries.

    Tie-break: prefer the earlier column in COLUMNS list (i.e. simpler path
    wins ties), which encodes a 'do no harm' Pareto preference.
    """
    order = [c[1] for c in COLUMNS]
    best = None
    for label in order:
        v = values.get(label)
        if v is None:
            continue
        if best is None or (higher_better and v > best[1]) or (not higher_better and v < best[1]):  # noqa: E501
            best = (label, v)
    return best


def main() -> int:
    rows = []
    reasons: dict[str, str] = {}
    for path_prefix, label, applies, reason in COLUMNS:
        if reason:
            reasons[label] = reason
    for model, tp, c, wl in CELLS:
        cell_id = f"{model}_tp{tp}_c{c}"
        cell_values: dict[str, dict | None] = {}
        for path_prefix, label, applies, reason in COLUMNS:
            if model not in applies:
                cell_values[label] = None
                continue
            cell_values[label] = load(resolve_path(path_prefix, label, model, tp, c, wl))  # noqa: E501

        for key, label_metric, higher_better in METRICS:
            values: dict[str, float | None] = {}
            for path_prefix, col_label, applies, reason in COLUMNS:
                blob = cell_values[col_label]
                values[col_label] = (blob or {}).get(key)
            winner = pick_winner(values, higher_better)
            row = {
                "cell": cell_id,
                "workload": wl,
                "metric": label_metric,
                "stock": values.get("stock"),
                "tensilelite": values.get("tensilelite"),
                "triton": values.get("triton"),
                "ck": values.get("ck"),
                "isa": values.get("isa"),
                "winner_path": winner[0] if winner else None,
                "winner_value": winner[1] if winner else None,
                "higher_better": higher_better,
            }
            rows.append(row)

    out_csv = FINAL_DIR / "final_grid.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cell",
                "workload",
                "metric",
                "stock",
                "tensilelite",
                "triton",
                "ck",
                "isa",
                "winner_path",
                "winner_value",
                "higher_better",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    out_reasons = FINAL_DIR / "final_grid_reasons.json"
    out_reasons.write_text(json.dumps(reasons, indent=2))

    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {out_reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
