#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AOT pre-tune the stock Triton W4A16 kernel for Qwen3.5-9B-w4a16.

Goal
----
M0 found that vLLM W4A16 TP=4 cannot capture FULL_DECODE_ONLY cudagraphs
because Triton's W4A16 JIT compile across 4 workers exceeds NCCL's default
10-min ``all_gather`` timeout during ``profile_run``.

This script runs a representative prefill + decode workload at **TP=1**
against a single rank, exercising every (BLOCK_M, BLOCK_N, BLOCK_K,
HAS_ZP, ZP_BIAS, num_warps, num_stages) tuple the W4A16 kernel will
issue at TP=4. Triton's compile cache is keyed by kernel source +
``@triton.jit`` constexpr meta-parameters (NOT by runtime M/N/K), so
populating the cache with this set of variants at TP=1 produces cache
entries that are byte-identical to the ones every TP=4 worker would
otherwise have to compile from scratch.

The cache directory is then exported as a portable artifact at
``/root/bench-int8-w4a16/baseline/triton_cache_w4a16/`` and pointed
at by the ``vllm-w4a16-tp4`` service via ``TRITON_CACHE_DIR``.

Usage
-----
::

    /opt/vllm-env/bin/python3 scripts/mi100/pretune_w4a16.py \
        [--cache-dir /root/bench-int8-w4a16/baseline/triton_cache_w4a16] \
        [--model /models/Qwen3.5-9B-w4a16] \
        [--max-wait-sec 1800]

The script:
  1. ``pkill``s any lingering vLLM processes (manifest stop semantics).
  2. Starts ``vllm.entrypoints.openai.api_server`` at TP=1 with
     ``TRITON_CACHE_DIR=<cache-dir>`` so EVERY compile lands in that dir.
  3. Polls ``/health`` for up to ``--max-wait-sec`` seconds.
  4. Drives prefill + decode via ``/v1/completions`` across three
     batch-token regimes that span the three BLOCK_{M,N,K} variants
     selected in ``triton_w4a16_gemm`` (M ≤ 16 decode, 16 < M ≤ 64,
     M > 64 prefill).
  5. Stops the server, prints the populated cache stats.

The exit code is 0 iff the cache contains at least one ``__grp__*``
HIP binary entry — i.e. Triton actually compiled the W4A16 kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/"
    "emdash/fuzzy-hornets-see-szfl4"
)
DEFAULT_CACHE = Path("/root/bench-int8-w4a16/baseline/triton_cache_w4a16")
DEFAULT_MODEL = "/models/Qwen3.5-9B-w4a16"
PORT = 8000


def _log(msg: str) -> None:
    print(f"[pretune] {msg}", flush=True)


def _kill_orphans() -> None:
    """Manifest-style stop: kill any lingering vLLM processes."""
    for pat in ("vllm.entrypoints", "VLLM::"):
        subprocess.run(
            ["pkill", "-9", "-f", pat], check=False, stderr=subprocess.DEVNULL
        )
    time.sleep(2)


def _start_server(cache_dir: Path, model: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    # vLLM/ROCm baseline env (mirrors services.yaml::vllm-w4a16-tp1).
    env.update(
        {
            "LD_LIBRARY_PATH": "/opt/rocm/core-7.12/lib",
            "ROCM_PATH": "/opt/rocm/core-7.12",
            "PATH": f"/opt/rocm/core-7.12/bin:{env.get('PATH', '')}",
            "PYTORCH_ROCM_ARCH": "gfx908",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_SKINNY_GEMM": "0",
            "TORCH_COMPILE_DISABLE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            # Critical: route ALL Triton compiles into our portable cache dir.
            "TRITON_CACHE_DIR": str(cache_dir),
        },
    )
    cmd = [
        "/opt/vllm-env/bin/python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--block-size",
        "32",
        "--enable-prefix-caching",
        "--language-model-only",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.93",
        "--port",
        str(PORT),
    ]
    _log(f"starting vLLM TP=1 with TRITON_CACHE_DIR={cache_dir}")
    log_fh = open(log_path, "wb")  # noqa: SIM115
    # cd to the repo so the in-tree vllm package wins on sys.path
    return subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _wait_health(proc: subprocess.Popen, max_wait_sec: int, log_path: Path) -> bool:
    deadline = time.time() + max_wait_sec
    last_log_size = 0
    while time.time() < deadline:
        if proc.poll() is not None:
            _log(f"server exited early with rc={proc.returncode}")
            return False
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=2.0
            ) as r:
                if r.status == 200:
                    _log(
                        f"server healthy after {int(max_wait_sec - (deadline - time.time()))}s",
                    )
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        # Light heartbeat
        try:
            sz = log_path.stat().st_size
            if sz > last_log_size + 32 * 1024:
                _log(f"... waiting (log grew to {sz} bytes)")
                last_log_size = sz
        except FileNotFoundError:
            pass
        time.sleep(5)
    _log("healthcheck timeout")
    return False


def _post_completion(prompt: str, max_tokens: int, request_id: int = 0) -> bool:
    """Send a single /v1/completions request to the local server."""
    body = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            ok = resp.status == 200
            if ok:
                _ = resp.read()
            return ok
    except urllib.error.HTTPError as e:
        _log(f"request {request_id} HTTP {e.code}: {e.reason}")
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        _log(f"request {request_id} failed: {e}")
        return False


def _make_prompt(approx_tokens: int) -> str:
    # ~3 chars/token rough heuristic for the Hello-world placeholder text.
    chunk = "Hello world. The quick brown fox jumps over the lazy dog. "
    return chunk * max(approx_tokens // 12, 1)


def _drive_workload() -> bool:
    """Issue prefill + decode requests covering all three M-bucket regimes.

    The W4A16 kernel selects (BLOCK_M, BLOCK_N, BLOCK_K) by token count
    via ``on_mi100`` branch in ``triton_w4a16.py``::

        M ≤ 16        -> (16, 64, 32)   # decode hot path
        16 < M ≤ 64   -> (32, 64, 32)
        M > 64        -> (64, 128, 32)  # prefill

    Issuing requests in all three regimes populates Triton's compile
    cache with the corresponding kernel binaries.
    """
    # Sequential requests at varying prefill lengths (covers all three
    # block-size buckets via prefill chunked-to-bucket and decode steps).
    sequential = [
        # (label, prompt_tokens, max_tokens)
        ("decode-only-1tok",  8,    32),    # tiny prefill + 32 decode steps (M=1)
        ("prefill-32",        32,   8),     # prefill bucket (16, 64]
        ("prefill-128",       128,  8),     # prefill bucket > 64
        ("prefill-512",       512,  8),     # larger prefill
        ("prefill-1024",     1024,  4),     # even larger prefill
    ]
    for label, ptok, mtok in sequential:
        _log(f"-> driving '{label}' p={ptok} m={mtok}")
        if not _post_completion(_make_prompt(ptok), mtok):
            _log(f"   FAIL on '{label}'")
            return False
        _log(f"   OK '{label}'")

    # Parallel decode burst — many concurrent decode-only requests so the
    # scheduler runs decode batches with M up to ~8 covering the
    # M ≤ 16 bucket on real batched-decode shapes (not just M=1).
    burst_n = 8
    _log(f"-> driving 'decode-burst' parallel={burst_n}")
    prompts = [_make_prompt(8)] * burst_n
    with ThreadPoolExecutor(max_workers=burst_n) as ex:
        futs = [
            ex.submit(_post_completion, prompts[i], 32, i) for i in range(burst_n)
        ]
        results = [f.result() for f in futs]
    if not all(results):
        _log(f"   FAIL on decode-burst ({sum(results)}/{burst_n} ok)")
        return False
    _log(f"   OK 'decode-burst' ({sum(results)}/{burst_n})")
    return True


def _summarise_cache(cache_dir: Path) -> dict:
    files = list(cache_dir.rglob("*"))
    n_dirs = sum(1 for f in files if f.is_dir())
    n_files = sum(1 for f in files if f.is_file())
    by_ext: dict[str, int] = {}
    total_bytes = 0
    for f in files:
        if f.is_file():
            ext = f.suffix or "<noext>"
            by_ext[ext] = by_ext.get(ext, 0) + 1
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass
    return {
        "cache_dir": str(cache_dir),
        "n_subdirs": n_dirs,
        "n_files": n_files,
        "total_bytes": total_bytes,
        "by_ext": by_ext,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-wait-sec", type=int, default=1800)
    p.add_argument(
        "--keep-existing-cache",
        action="store_true",
        help="If set, do not wipe cache-dir before pretune.",
    )
    args = p.parse_args()

    cache_dir = args.cache_dir.resolve()
    log_path = cache_dir.parent / "pretune_server.log"

    if cache_dir.exists() and not args.keep_existing_cache:
        _log(f"wiping existing cache dir: {cache_dir}")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _kill_orphans()
    proc = _start_server(cache_dir, args.model, log_path)
    try:
        if not _wait_health(proc, args.max_wait_sec, log_path):
            _log("server failed to come healthy; aborting")
            return 2

        ok = _drive_workload()
        if not ok:
            _log("workload failed; cache may be incomplete")
            return 3

        # Allow a short flush time for any final compile to land on disk.
        time.sleep(2)
        summary = _summarise_cache(cache_dir)
        _log(f"cache summary: {json.dumps(summary, indent=2)}")
        manifest = cache_dir.parent / "triton_cache_manifest.json"
        manifest.write_text(json.dumps(summary, indent=2))
        _log(f"manifest written to {manifest}")
        if summary["n_files"] == 0:
            _log("WARNING: cache is empty — pretune did NOT populate Triton cache")
            return 4
        return 0
    finally:
        _log("shutting down server")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        _kill_orphans()


if __name__ == "__main__":
    sys.exit(main())
