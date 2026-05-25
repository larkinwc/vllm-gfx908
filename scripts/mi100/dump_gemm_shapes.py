#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
M2 helper: launch a short vLLM serving session against the W8A8 model with
VLLM_LOG_GEMM_SHAPES=1, hit it with a few prompts, then aggregate the per-call
shape CSVs into a unique-shape catalog.

This is a lightweight wrapper around the instrumentation already added to
vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py. The kernel itself
appends one row per W8A8 GEMM call into the path given by
VLLM_GEMM_SHAPES_OUT.

Usage:
    /opt/vllm-env/bin/python3 scripts/mi100/dump_gemm_shapes.py \\
        --model /models/Qwen3.5-9B-w8a8 --tp 1 --duration-sec 60 \\
        --out /root/bench-int8-w4a16/tensilelite/gemm_shapes_w8a8_tp1.csv

The launcher is `nohup vllm.entrypoints.openai.api_server` so we can attach
client requests; once the script exits, the server is killed and the CSV
contains every (M,N,K) recorded.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/"
    "emdash/cold-points-sit-rancb"
)
PY = "/opt/vllm-env/bin/python3"


def _kill_orphans() -> None:
    subprocess.run(
        ["pkill", "-9", "-f", "vllm.entrypoints"],
        check=False,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["pkill", "-9", "-f", "VLLM::"],
        check=False,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)


def _wait_health(port: int, timeout_sec: int = 600) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        rc = subprocess.run(
            ["curl", "-sf", f"http://127.0.0.1:{port}/health"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        if rc == 0:
            return True
        time.sleep(5)
    return False


def _bench_serve_once(
    model: str, port: int, n_prompts: int, max_concurrency: int
) -> int:
    cmd = [
        PY,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--model",
        model,
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--num-prompts",
        str(n_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(max_concurrency),
        "--seed",
        "42",
        "--dataset-name",
        "random",
        "--random-input-len",
        "512",
        "--random-output-len",
        "64",
        "--ignore-eos",
        "--trust-remote-code",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode


def _aggregate(raw_csv: Path, out_csv: Path) -> dict:
    """Aggregate raw per-call rows into unique-shape rows with counts."""
    if not raw_csv.exists():
        return {"total_calls": 0, "unique_shapes": 0}
    counts: Counter[tuple[int, int, int]] = Counter()
    total = 0
    with raw_csv.open() as f:
        header = f.readline()
        if not header.startswith("M,N,K"):
            # No header; rewind
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                M, N, K = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            counts[(M, N, K)] += 1
            total += 1
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("M,N,K,n_calls\n")
        for (M, N, K), n in counts.most_common():
            f.write(f"{M},{N},{K},{n}\n")
    return {"total_calls": total, "unique_shapes": len(counts), "raw_csv": str(raw_csv)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.5-9B-w8a8")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--duration-sec",
        type=int,
        default=60,
        help="Wall time the server stays up for shape capture.",
    )
    ap.add_argument(
        "--n-prompts",
        type=int,
        default=80,
        help="Per-concurrency prompt count for the warmup hits.",
    )
    ap.add_argument(
        "--concurrencies",
        default="1,2,4",
        help="Comma-separated concurrency levels to exercise.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/root/bench-int8-w4a16/tensilelite/gemm_shapes_w8a8.csv"),
    )
    ap.add_argument(
        "--raw-out",
        type=Path,
        default=Path("/root/bench-int8-w4a16/tensilelite/gemm_shapes_w8a8_raw.csv"),
    )
    ap.add_argument(
        "--server-log",
        default="/root/bench-int8-w4a16/tensilelite/shape_dump_server.log",
    )
    args = ap.parse_args()

    raw_out = args.raw_out
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    # Truncate the per-call CSV before this run.
    if raw_out.exists():
        raw_out.unlink()

    env = os.environ.copy()
    env.update(
        {
            "VLLM_LOG_GEMM_SHAPES": "1",
            "VLLM_GEMM_SHAPES_OUT": str(raw_out),
            "LD_LIBRARY_PATH": "/opt/rocm/core-7.12/lib",
            "ROCM_PATH": "/opt/rocm/core-7.12",
            "PATH": "/opt/rocm/core-7.12/bin:" + env.get("PATH", "/usr/bin"),
            "PYTORCH_ROCM_ARCH": "gfx908",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_SKINNY_GEMM": "0",
            "TORCH_COMPILE_DISABLE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    if args.tp > 1:
        env["VLLM_MI100_DISABLE_CUSTOM_AR"] = "1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"

    _kill_orphans()
    server_args = [
        PY,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        "8192",
        "--block-size",
        "32",
        "--enable-prefix-caching",
        "--language-model-only",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.93",
        "--port",
        str(args.port),
    ]
    if args.tp > 1:
        server_args.append("--disable-custom-all-reduce")

    log_handle = open(args.server_log, "w")  # noqa: SIM115 — long-lived proc
    proc = subprocess.Popen(
        server_args,
        cwd=str(REPO),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"[dump_gemm_shapes] vLLM PID={proc.pid}, log={args.server_log}")

    summary = {
        "model": args.model,
        "tp": args.tp,
        "raw_csv": str(raw_out),
    }
    try:
        if not _wait_health(args.port, timeout_sec=args.duration_sec + 600):
            print("[dump_gemm_shapes] HEALTH TIMEOUT", file=sys.stderr)
            return 2

        print(
            f"[dump_gemm_shapes] server healthy, hitting concurrencies "
            f"{args.concurrencies}"
        )
        for c in [int(x) for x in args.concurrencies.split(",")]:
            print(f"[dump_gemm_shapes]   concurrency={c}")
            _bench_serve_once(args.model, args.port, args.n_prompts, c)
    finally:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(2)
        _kill_orphans()
        log_handle.close()

    summary.update(_aggregate(raw_out, args.out))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
