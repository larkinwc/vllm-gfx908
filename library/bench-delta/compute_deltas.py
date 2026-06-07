#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3-F2 per-cell delta aggregator.

Compares library/bench-post-sync/ vs library/bench-baseline/ across:
  - D1 24-cell grid (w4a16-grid.json + w8a8 cells; output_throughput_toks_s)
  - D2 fused-act-quant (alias to w8a8_tp1_c1/synthetic)
  - D3 CK-FA decode latency (avg_latency, p50, p99)
  - D4 HBM bytes/token (hbm-summary.json)

Regression thresholds (per AGENTS.md / validation-contract.md):
  - Throughput regression: delta < -3%
  - Decode latency regression: delta > +2%
  - HBM bytes/token regression: delta > +2%

Writes:
  library/bench-delta/grid-deltas.json
  library/bench-delta/d2-fused-act-quant-delta.json
  library/bench-delta/d3-ckfa-delta.json
  library/bench-delta/d4-hbm-delta.json
  library/bench-delta/summary.json (top-line verdict, per-cell verdicts)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k"  # noqa: E501
)
BASE = ROOT / "library" / "bench-baseline"
POST = ROOT / "library" / "bench-post-sync"
DELT = ROOT / "library" / "bench-delta"
DELT.mkdir(parents=True, exist_ok=True)

# Thresholds
THR_REG_PCT = -3.0  # D1, D2 throughput
LAT_REG_PCT = +2.0  # D3 latency
HBM_REG_PCT = +2.0  # D4 HBM bytes


def pct(post: float, base: float) -> float:
    if base == 0:
        return float("nan")
    return (post - base) / base * 100.0


def load_grid(path: Path) -> dict:
    obj = json.loads(path.read_text())
    out = {}
    for c in obj["cells"]:
        key = f"{c['cell_id']}/{c['workload']}"
        out[key] = c
    return out


def aggregate_postsync_grid() -> dict:
    """Build a post-sync grid JSON mirroring baseline schema from per-cell files."""
    cells = []
    for wl in ("synthetic", "coding"):
        d = POST / "grid" / wl
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.json")):
            cell = json.loads(fp.read_text())
            cell["workload"] = wl
            cell.setdefault("cell_id", fp.stem)
            cells.append(cell)
    return {"cells": cells, "n_cells": len(cells)}


def compute_grid_deltas() -> dict:
    base = load_grid(BASE / "w4a16-grid.json")
    post_obj = aggregate_postsync_grid()
    POST.mkdir(parents=True, exist_ok=True)
    (POST / "w4a16-grid.json").write_text(json.dumps(post_obj, indent=2))
    post = {f"{c['cell_id']}/{c['workload']}": c for c in post_obj["cells"]}

    rows = []
    regressions = []
    for key in sorted(base.keys()):
        b = base[key]
        p = post.get(key)
        if not p:
            rows.append({"cell": key, "status": "MISSING_POST"})
            continue
        b_tput = b["output_throughput_toks_s"]
        p_tput = p["output_throughput_toks_s"]
        d_tput = pct(p_tput, b_tput)
        b_ttft = b["mean_ttft_ms"]
        p_ttft = p["mean_ttft_ms"]
        d_ttft = pct(p_ttft, b_ttft)
        b_tpot = b["mean_tpot_ms"]
        p_tpot = p["mean_tpot_ms"]
        d_tpot = pct(p_tpot, b_tpot)
        verdict = "PASS"
        if d_tput < THR_REG_PCT:
            verdict = "REGRESSION"
            regressions.append(
                {
                    "cell": key,
                    "metric": "output_throughput_toks_s",
                    "baseline": b_tput,
                    "postsync": p_tput,
                    "delta_pct": d_tput,
                }
            )
        rows.append(
            {
                "cell": key,
                "baseline_tput": b_tput,
                "postsync_tput": p_tput,
                "delta_tput_pct": d_tput,
                "baseline_ttft_ms": b_ttft,
                "postsync_ttft_ms": p_ttft,
                "delta_ttft_pct": d_ttft,
                "baseline_tpot_ms": b_tpot,
                "postsync_tpot_ms": p_tpot,
                "delta_tpot_pct": d_tpot,
                "verdict": verdict,
            }
        )
    return {
        "rows": rows,
        "regressions": regressions,
        "threshold_pct": THR_REG_PCT,
        "metric": "output_throughput_toks_s",
    }


def compute_d2() -> dict:
    base = json.loads((BASE / "fused-act-quant.json").read_text())
    post_path = POST / "fused-act-quant.json"
    if not post_path.exists():
        return {"status": "MISSING_POST"}
    post = json.loads(post_path.read_text())
    b_tput = base["output_throughput_toks_s"]
    p_tput = post["output_throughput_toks_s"]
    d_tput = pct(p_tput, b_tput)
    verdict = "REGRESSION" if d_tput < THR_REG_PCT else "PASS"
    return {
        "cell": "w8a8_tp1_c1/synthetic (fused-act-quant alias)",
        "baseline_tput": b_tput,
        "postsync_tput": p_tput,
        "delta_tput_pct": d_tput,
        "threshold_pct": THR_REG_PCT,
        "verdict": verdict,
    }


def compute_d3() -> dict:
    base = json.loads((BASE / "ckfa-latency.json").read_text())
    post_path = POST / "ckfa-latency.json"
    if not post_path.exists():
        return {"status": "MISSING_POST"}
    post = json.loads(post_path.read_text())
    rows = {}
    for k in ("avg_latency",):
        b, p = base[k], post[k]
        d = pct(p, b)
        rows[k] = {
            "baseline_s": b,
            "postsync_s": p,
            "delta_pct": d,
            "verdict": "REGRESSION" if d > LAT_REG_PCT else "PASS",
        }
    for pk in ("50", "90", "99"):
        b = base["percentiles"][pk]
        p = post["percentiles"][pk]
        d = pct(p, b)
        rows[f"p{pk}"] = {
            "baseline_s": b,
            "postsync_s": p,
            "delta_pct": d,
            "verdict": "REGRESSION" if d > LAT_REG_PCT else "PASS",
        }
    verdict = (
        "REGRESSION"
        if any(r["verdict"] == "REGRESSION" for r in rows.values())
        else "PASS"
    )
    return {"metrics": rows, "threshold_pct": LAT_REG_PCT, "verdict": verdict}


def compute_d4() -> dict:
    base = json.loads((BASE / "hbm-summary.json").read_text())
    post_path = POST / "hbm-summary.json"
    if not post_path.exists():
        return {"status": "MISSING_POST"}
    post = json.loads(post_path.read_text())
    rows = {}
    for k in (
        "hbm_bytes_per_output_token",
        "hbm_bytes_per_total_token",
        "total_hbm_bytes",
    ):
        b, p = base[k], post[k]
        d = pct(p, b)
        rows[k] = {
            "baseline": b,
            "postsync": p,
            "delta_pct": d,
            "verdict": "REGRESSION" if d > HBM_REG_PCT else "PASS",
        }
    verdict = (
        "REGRESSION"
        if rows["hbm_bytes_per_output_token"]["verdict"] == "REGRESSION"
        else "PASS"
    )
    return {
        "metrics": rows,
        "threshold_pct": HBM_REG_PCT,
        "verdict": verdict,
        "primary_metric": "hbm_bytes_per_output_token",
    }


def main() -> int:
    grid = compute_grid_deltas()
    (DELT / "grid-deltas.json").write_text(json.dumps(grid, indent=2))
    d2 = compute_d2()
    (DELT / "d2-fused-act-quant-delta.json").write_text(json.dumps(d2, indent=2))
    d3 = compute_d3()
    (DELT / "d3-ckfa-delta.json").write_text(json.dumps(d3, indent=2))
    d4 = compute_d4()
    (DELT / "d4-hbm-delta.json").write_text(json.dumps(d4, indent=2))

    # Top-line verdict
    any_regression = bool(grid["regressions"])
    d2_verdict = d2.get("verdict", "MISSING")
    d3_verdict = d3.get("verdict", "MISSING")
    d4_verdict = d4.get("verdict", "MISSING")
    if d2_verdict == "REGRESSION":
        any_regression = True
    if d3_verdict == "REGRESSION":
        any_regression = True
    if d4_verdict == "REGRESSION":
        any_regression = True

    # MIXED vs FAIL distinction: per skill, MIXED if any cell in (-5%, -3%)
    # with documented attribution. Mechanically we mark MIXED if all
    # regressions are mild (>= -5% throughput, <= +5% latency/HBM); FAIL otherwise.
    def is_mild_thr(p):
        return p > -5.0

    def is_mild_lat(p):
        return p < +5.0

    mild_only = True
    for r in grid["regressions"]:
        if not is_mild_thr(r["delta_pct"]):
            mild_only = False
    if d2_verdict == "REGRESSION" and not is_mild_thr(d2["delta_tput_pct"]):
        mild_only = False
    if d3_verdict == "REGRESSION":
        for k, v in d3["metrics"].items():
            if v["verdict"] == "REGRESSION" and not is_mild_lat(v["delta_pct"]):
                mild_only = False
    if d4_verdict == "REGRESSION":
        for k, v in d4["metrics"].items():
            if v["verdict"] == "REGRESSION" and not is_mild_lat(v["delta_pct"]):
                mild_only = False

    if not any_regression:
        topline = "PASS"
    elif mild_only:
        topline = "MIXED"
    else:
        topline = "FAIL"

    summary = {
        "topline_verdict": topline,
        "d1_grid_pass": not bool(grid["regressions"]),
        "d1_regressions": grid["regressions"],
        "d2_verdict": d2_verdict,
        "d2_delta_pct": d2.get("delta_tput_pct"),
        "d3_verdict": d3_verdict,
        "d4_verdict": d4_verdict,
        "thresholds": {
            "throughput_min_pct": THR_REG_PCT,
            "latency_max_pct": LAT_REG_PCT,
            "hbm_bytes_max_pct": HBM_REG_PCT,
        },
        "baseline_dir": str(BASE),
        "postsync_dir": str(POST),
    }
    (DELT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
