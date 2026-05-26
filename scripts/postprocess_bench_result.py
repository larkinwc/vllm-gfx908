#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Wrap the JSON file written by `vllm bench serve --save-result` and emit
a schema-conformant result file at the canonical path.

Usage:
    postprocess_bench_result.py \
        --raw <path-to-vllm-bench-json> \
        --cell w8a8_tp1_c1 \
        --model /models/Qwen3.5-9B-w8a8 \
        --quant w8a8-int8 \
        --tp 1 \
        --concurrency 1 \
        --request-rate 1 \
        --workload synthetic \
        --num-prompts 200 \
        --launch-command "<command-string>" \
        --env-file <path-to-env.json> \
        --out <path-to-cell.json>
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# Pin REPO to the env-var when set so workers in sibling worktrees can override
# the original mission's hardcoded path. Fall back to the script's own
# repository root (two parents above this file) which always exists.
REPO = Path(os.environ.get("REPO") or Path(__file__).resolve().parent.parent)


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(path), text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _detect_versions() -> tuple[str, str, str, str]:
    rocm = os.environ.get("ROCM_VERSION", "7.12")
    py = "/opt/vllm-env/bin/python3"
    try:
        torch_version = subprocess.check_output(
            [py, "-c", "import torch; print(torch.__version__)"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        torch_version = "unknown"
    try:
        triton_version = subprocess.check_output(
            [py, "-c", "import triton; print(triton.__version__)"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        triton_version = "unknown"
    try:
        vllm_version = subprocess.check_output(
            [py, "-c", "import vllm; print(vllm.__version__)"],
            cwd=str(REPO),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        vllm_version = "unknown"
    return rocm, torch_version, triton_version, vllm_version


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--cell", required=True, help="e.g. w8a8_tp1_c1_synthetic")
    p.add_argument("--model", required=True)
    p.add_argument("--quant", required=True, choices=["w8a8-int8", "w4a16"])
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--request-rate", required=True, help="Numeric or 'inf'.")
    p.add_argument("--workload", required=True, choices=["synthetic", "coding"])
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--launch-command", required=True)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kernel-backend", default="stock")
    p.add_argument("--tuning-json-sha256", default=None)
    args = p.parse_args()

    raw_data = json.loads(args.raw.read_text())
    env = json.loads(args.env_file.read_text())

    rocm, torch_version, triton_version, vllm_version = _detect_versions()
    harness_commit = _git_sha(REPO)

    # vLLM bench serve produces these field names
    def _f(name: str, default: float = 0.0) -> float:
        v = raw_data.get(name)
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return default
        return float(v)

    out = {
        "model": args.model,
        "quant": args.quant,
        "tp": args.tp,
        "concurrency": args.concurrency,
        "request_rate": (
            float(args.request_rate)
            if args.request_rate not in ("inf", "Inf", "INF")
            else "inf"
        ),
        "workload": args.workload,
        "num_prompts": args.num_prompts,
        "mean_ttft_ms": _f("mean_ttft_ms"),
        "p50_ttft_ms": _f("median_ttft_ms"),
        "p99_ttft_ms": _f("p99_ttft_ms"),
        "mean_tpot_ms": _f("mean_tpot_ms"),
        "p50_tpot_ms": _f("median_tpot_ms"),
        "p99_tpot_ms": _f("p99_tpot_ms"),
        "output_throughput_toks_s": _f("output_throughput"),
        "request_throughput_req_s": _f("request_throughput"),
        "harness_commit": harness_commit,
        "vllm_commit": vllm_version,
        "rocm_version": rocm,
        "torch_version": torch_version,
        "triton_version": triton_version,
        "tuning_json_sha256": args.tuning_json_sha256,
        "kernel_backend": args.kernel_backend,
        "launch_command": args.launch_command,
        "env": env,
        "timestamp": datetime.datetime.now(datetime.UTC)
        .replace(tzinfo=None)
        .isoformat()
        + "Z",
        "raw_vllm_bench": raw_data,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(
        f"wrote cell {args.cell}: tput={out['output_throughput_toks_s']:.2f} tok/s "
        f"ttft_p50={out['p50_ttft_ms']:.1f} ms "
        f"tpot_p50={out['p50_tpot_ms']:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
