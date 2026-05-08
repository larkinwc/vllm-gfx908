#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare two W8A8 TP=1 c=1 synthetic result JSONs and emit
canary_diff.json with the throughput delta percent.

Usage:
    canary_diff.py --runA <path> --runB <path> \
        --out /root/bench-int8-w4a16/baseline/canary_diff.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runA", type=Path, required=True)
    p.add_argument("--runB", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold-pct", type=float, default=5.0)
    args = p.parse_args()

    a = json.loads(args.runA.read_text())
    b = json.loads(args.runB.read_text())

    runs = []
    for label, d in (("runA", a), ("runB", b)):
        runs.append({
            "label": label,
            "source": str(args.runA if label == "runA" else args.runB),
            "output_throughput_toks_s": float(d["output_throughput_toks_s"]),
            "p50_ttft_ms": float(d["p50_ttft_ms"]),
            "p50_tpot_ms": float(d["p50_tpot_ms"]),
            "vllm_commit": d.get("vllm_commit"),
            "harness_commit": d.get("harness_commit"),
            "timestamp": d.get("timestamp"),
        })

    a_t = runs[0]["output_throughput_toks_s"]
    b_t = runs[1]["output_throughput_toks_s"]
    delta_pct = 100.0 * (b_t - a_t) / a_t if a_t else float("nan")

    out = {
        "metric": "output_throughput_toks_s",
        "cell": "w8a8_tp1_c1_synthetic",
        "runs": runs,
        "delta_pct_throughput": delta_pct,
        "pass_threshold_pct": args.threshold_pct,
        "passed": abs(delta_pct) <= args.threshold_pct,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    verdict = "PASS" if out["passed"] else "FAIL"
    print(
        f"canary delta = {delta_pct:+.2f}%  "
        f"(threshold ±{args.threshold_pct}%) -> {verdict}"
    )
    print(f"wrote {args.out}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
