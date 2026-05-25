#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M2 chunked-prefill chunk-size sweep (VAL-CHUNKED-001).

For the prefill-dominated cell ``<quant>_tp1_c4_coding`` (TP=1, c=4 on the
coding-agent workload — heavy mixed input lengths 256-8k, output 64-1024),
sweep ``--max-num-batched-tokens`` ∈ {512, 1024, 2048, 4096} with
``--enable-chunked-prefill`` enabled. The sweep stacks on top of the M1
KV-INT8 winner (``KV_CACHE_DTYPE=int8_per_token_head``) per orchestrator
stacking decision.

For each ``chunk_size`` value:

  1. Kill any orphan vLLM processes.
  2. Start ``scripts/launch_<model>_tp1_c4.sh --serve-only`` with the
     env overrides ``KV_CACHE_DTYPE=int8_per_token_head``,
     ``ENABLE_CHUNKED_PREFILL=1``, ``MAX_NUM_BATCHED_TOKENS=<chunk_size>``.
     Wait for ``/health``.
  3. Run ``vllm.entrypoints.cli.main bench serve`` on the coding-agent
     dataset with ``NUM_PROMPTS=200``, ``--request-rate inf``,
     ``--max-concurrency 4``, ``--seed 42``. Capture raw.json under
     ``/root/bench-int8-w4a16-hbm/m2-chunked/sweep/<quant>/cs<chunk>/``.
  4. Kill the server before the next chunk.

After all 4 chunks complete (per --quant), update
``/root/bench-int8-w4a16-hbm/m2-chunked/chunk_sweep.json`` with the new
``cells`` rows + a top-level ``optimum_per_quant`` block keyed by
``request_throughput_req_s`` (coding-only this iteration; synthetic
geomean is added at the m2-chunked-bench-grid milestone). The optimum
per quant is the chunk size with the highest request_throughput_req_s.

Re-render ``chunk_sweep_summary.md`` for human review on every write.

Usage:
    scripts/mi100/chunk_size_sweep.py --quant {w8a8,w4a16}
    scripts/mi100/chunk_size_sweep.py --quant w8a8 --chunks 1024,2048

Per VAL-CHUNKED-001, this script does NOT push to git.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/"
    "emdash/cold-points-sit-rancb"
)
PY = "/opt/vllm-env/bin/python3"
OUT_ROOT = Path("/root/bench-int8-w4a16-hbm/m2-chunked")
SWEEP_ROOT = OUT_ROOT / "sweep"
SWEEP_JSON = OUT_ROOT / "chunk_sweep.json"
SUMMARY_MD = OUT_ROOT / "chunk_sweep_summary.md"
CODING_DATASET = Path("/root/bench-int8-w4a16/datasets/coding_agent.jsonl")
HEALTH_URL = "http://127.0.0.1:8000/health"

DEFAULT_CHUNKS = (512, 1024, 2048, 4096)
HEALTH_WAIT_SECS = 1800  # FULL graph capture for chunked prefill can be slow
HEALTH_POLL_SECS = 5

QUANT_MODELS = {
    "w8a8": "Qwen3.5-9B-w8a8",
    "w4a16": "Qwen3.5-9B-w4a16",
}


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    print(f"[{ts}] {msg}", flush=True)


def _kill_orphans() -> None:
    for pat in ("vllm.entrypoints", "VLLM::"):
        subprocess.run(
            ["pkill", "-9", "-f", pat],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(2)


def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001
        return False


def _wait_for_health(server: subprocess.Popen, log_path: Path) -> bool:
    deadline = time.time() + HEALTH_WAIT_SECS
    while time.time() < deadline:
        if _health_ok():
            return True
        if server.poll() is not None:
            _log(
                f"  FATAL: server PID exited rc={server.returncode}; "
                f"tail of {log_path}:"
            )
            try:
                tail = log_path.read_text().splitlines()[-60:]
                print("\n".join(tail), flush=True)
            except Exception:  # noqa: BLE001
                pass
            return False
        time.sleep(HEALTH_POLL_SECS)
    return False


# ---------------------------------------------------------------------------
# Per-chunk launch + bench
# ---------------------------------------------------------------------------
def _start_server(
    quant: str,
    chunk_size: int,
    server_log: Path,
) -> subprocess.Popen | None:
    """Launch scripts/launch_<model>_tp1_c4.sh --serve-only with the M1+M2
    env overrides. Returns the Popen handle on healthy startup, else None.
    """
    launch_script = REPO / "scripts" / f"launch_{quant}_tp1_c4.sh"
    if not launch_script.is_file():
        _log(f"  FATAL: launch script missing: {launch_script}")
        return None

    env = os.environ.copy()
    env["KV_CACHE_DTYPE"] = "int8_per_token_head"
    env["ENABLE_CHUNKED_PREFILL"] = "1"
    env["MAX_NUM_BATCHED_TOKENS"] = str(chunk_size)
    # First-time cudagraph capture for chunked-prefill can exceed the 300 s
    # default; bump the launch script's health-wait ceiling so it doesn't
    # bail before the engine becomes healthy.
    env["LAUNCH_HEALTH_WAIT_SECS"] = str(HEALTH_WAIT_SECS)

    server_log.parent.mkdir(parents=True, exist_ok=True)
    f_log = server_log.open("w")
    _log(f"  starting {launch_script.name} --serve-only chunk_size={chunk_size}")
    proc = subprocess.Popen(
        [str(launch_script), "--serve-only"],
        cwd=str(REPO),
        env=env,
        stdout=f_log,
        stderr=subprocess.STDOUT,
        # Put in a new process group so we can clean up the whole tree later.
        start_new_session=True,
    )

    if not _wait_for_health(proc, server_log):
        _log("  health timeout / engine init failure; killing server group")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        _kill_orphans()
        return None
    _log("  server healthy")
    return proc


def _detect_failure_reason(quant: str, cell_dir: Path) -> tuple[str, Path | None]:
    """Inspect the vLLM api_server log to classify the failure.

    Returns (reason, archived_log_path). archived_log_path is the per-chunk
    copy we make so the engine log survives subsequent overwrites by the
    shared launch-script log path.
    """
    shared_log = Path(
        f"/root/bench-int8-w4a16/final/launch_smoke/{quant}_tp1_c4/server.log"
    )
    archived: Path | None = None
    text = ""
    if shared_log.is_file():
        archived = cell_dir / "engine.log"
        try:
            shutil.copy2(shared_log, archived)
        except Exception:  # noqa: BLE001
            archived = None
        try:
            text = shared_log.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            text = ""

    if "must be <= max_num_batched_tokens" in text:
        return (
            "Mamba block_size constraint: attention block_size must be "
            "<= max_num_batched_tokens (vllm/config/vllm.py "
            "validate_block_size). This chunk_size is infeasible on "
            "Qwen3.5 with Mamba layers; the smallest feasible chunk "
            "is determined by the model's mamba-aligned block_size."
        ), archived
    if "EngineCore failed to start" in text:
        return ("EngineCore failed to start; see engine.log for traceback",
                archived)
    return ("server startup / healthcheck failure", archived)


def _run_bench(
    quant: str,
    chunk_size: int,
    raw_dir: Path,
) -> Path | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    model_path = f"/models/{QUANT_MODELS[quant]}"
    cell_id = f"{quant}_tp1_c4"
    cmd = [
        PY,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--model",
        model_path,
        "--base-url",
        "http://127.0.0.1:8000",
        "--num-prompts",
        "200",
        "--request-rate",
        "inf",
        "--max-concurrency",
        "4",
        "--seed",
        "42",
        "--save-result",
        "--result-dir",
        str(raw_dir),
        "--result-filename",
        "raw.json",
        "--trust-remote-code",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--metadata",
        f"cell_id={cell_id}",
        "workload=coding",
        "tp=1",
        "concurrency=4",
        "num_prompts=200",
        "kv_cache_dtype=int8_per_token_head",
        f"max_num_batched_tokens={chunk_size}",
        "enable_chunked_prefill=1",
        "milestone=m2-chunked",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(CODING_DATASET),
        "--custom-output-len",
        "256",
        "--skip-chat-template",
    ]
    _log(f"  bench serve chunk_size={chunk_size} -> {raw_dir}/raw.json")
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    raw_json = raw_dir / "raw.json"
    if rc != 0 or not raw_json.is_file():
        _log(f"  bench serve exit={rc}; raw.json present={raw_json.is_file()}")
        return None
    return raw_json


def _stop_server(proc: subprocess.Popen | None) -> None:
    if proc is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    _kill_orphans()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _extract_metrics(raw_json: Path) -> dict[str, float]:
    """Pull the required schema fields from vllm bench serve raw.json.

    Schema target (VAL-CHUNKED-001):
        mean_ttft_ms, p99_ttft_ms, mean_tpot_ms,
        request_throughput_req_s, output_throughput_toks_s
    """
    raw = json.loads(raw_json.read_text())

    def _f(name: str, default: float = 0.0) -> float:
        v = raw.get(name)
        if v is None or (isinstance(v, float) and v != v):
            return default
        return float(v)

    return {
        "mean_ttft_ms": _f("mean_ttft_ms"),
        "p99_ttft_ms": _f("p99_ttft_ms"),
        "mean_tpot_ms": _f("mean_tpot_ms"),
        "p50_tpot_ms": _f("median_tpot_ms"),
        "p99_tpot_ms": _f("p99_tpot_ms"),
        "request_throughput_req_s": _f("request_throughput"),
        "output_throughput_toks_s": _f("output_throughput"),
    }


def _load_existing() -> dict:
    if SWEEP_JSON.is_file():
        try:
            return json.loads(SWEEP_JSON.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {
        "schema_version": "m2-chunk-sweep-v1",
        "harness": "scripts/mi100/chunk_size_sweep.py",
        "workload": "coding",
        "num_prompts": 200,
        "request_rate": "inf",
        "max_concurrency": 4,
        "dataset_path": str(CODING_DATASET),
        "kv_cache_dtype": "int8_per_token_head",
        "enable_chunked_prefill": True,
        "tp": 1,
        "cells": [],
        "optimum_per_quant": {},
    }


def _upsert_cell(state: dict, row: dict) -> None:
    cells = state.setdefault("cells", [])
    for i, existing in enumerate(cells):
        if (
            existing.get("quant") == row["quant"]
            and int(existing.get("chunk_size", -1)) == int(row["chunk_size"])
        ):
            cells[i] = row
            return
    cells.append(row)


def _compute_optimum(state: dict) -> dict[str, int]:
    """Optimum per quant = chunk_size with highest request_throughput_req_s
    on the coding workload. Per task spec: "this iteration uses coding
    request_throughput as the primary metric"."""
    by_quant: dict[str, list[dict]] = {}
    for c in state.get("cells", []):
        # Skip explicit FAILED entries from the optimum selection.
        if c.get("status") == "FAILED":
            continue
        by_quant.setdefault(c["quant"], []).append(c)

    optimum: dict[str, int] = {}
    for quant, rows in by_quant.items():
        if not rows:
            continue
        winner = max(
            rows,
            key=lambda r: float(r.get("request_throughput_req_s") or 0.0),
        )
        optimum[quant] = int(winner["chunk_size"])
    return optimum


def _render_summary_md(state: dict) -> str:
    lines = []
    lines.append("# M2 Chunk-Size Sweep — Per-Quant Optimum")
    lines.append("")
    lines.append(
        "Generated by `scripts/mi100/chunk_size_sweep.py` "
        f"(written at {datetime.datetime.now(datetime.timezone.utc).isoformat()})."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("- Cell: `<quant>_tp1_c4_coding` (prefill-dominated)")
    lines.append(
        "- Workload: coding-agent "
        "(`/root/bench-int8-w4a16/datasets/coding_agent.jsonl`)"
    )
    lines.append(
        "- NUM_PROMPTS: 200, --request-rate inf, "
        "--max-concurrency 4, --seed 42"
    )
    lines.append("- KV-cache: `int8_per_token_head` (M1 winner stacked)")
    lines.append("- chunked-prefill: enabled")
    lines.append(
        "- Primary metric: `request_throughput_req_s` "
        "(coding-only this iteration)"
    )
    lines.append("")

    cells = state.get("cells", [])
    by_quant: dict[str, list[dict]] = {}
    for c in cells:
        by_quant.setdefault(c["quant"], []).append(c)
    for q in by_quant:
        by_quant[q].sort(key=lambda r: int(r["chunk_size"]))

    lines.append("## Per-quant results")
    lines.append("")
    for quant in sorted(by_quant):
        rows = by_quant[quant]
        lines.append(f"### {quant}")
        lines.append("")
        lines.append(
            "| chunk_size | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | "
            "req_tput_req/s | out_tput_tok/s | status |"
        )
        lines.append(
            "|-----------:|-------------:|------------:|-------------:|"
            "---------------:|---------------:|:------:|"
        )
        for r in rows:
            status = r.get("status", "OK")
            lines.append(
                f"| {int(r['chunk_size'])} "
                f"| {float(r.get('mean_ttft_ms') or 0):.2f} "
                f"| {float(r.get('p99_ttft_ms') or 0):.2f} "
                f"| {float(r.get('mean_tpot_ms') or 0):.2f} "
                f"| {float(r.get('request_throughput_req_s') or 0):.4f} "
                f"| {float(r.get('output_throughput_toks_s') or 0):.2f} "
                f"| {status} |"
            )
        lines.append("")

    opt = state.get("optimum_per_quant", {}) or {}
    lines.append("## optimum_per_quant (by `request_throughput_req_s`, coding)")
    lines.append("")
    if opt:
        lines.append("| quant | optimum_chunk_size |")
        lines.append("|:------|-------------------:|")
        for q in sorted(opt):
            lines.append(f"| {q} | {int(opt[q])} |")
    else:
        lines.append("_no quant has a measured row yet_")
    lines.append("")

    # Surface any unique FAILED reasons so reviewers understand which chunk
    # sizes are model-infeasible vs which were a runtime hiccup.
    seen_reasons: set[str] = set()
    failed_rows = [c for c in cells if c.get("status") == "FAILED"]
    if failed_rows:
        lines.append("## FAILED rows")
        lines.append("")
        for c in sorted(
            failed_rows, key=lambda r: (r["quant"], int(r["chunk_size"]))
        ):
            reason = (c.get("reason") or "").strip()
            key = f"{c['quant']}/{c['chunk_size']}/{reason[:80]}"
            if key in seen_reasons:
                continue
            seen_reasons.add(key)
            lines.append(
                f"- **{c['quant']} chunk_size={int(c['chunk_size'])}**: {reason}"
            )
        lines.append("")

    lines.append("Persisted machine-readable copy: "
                 "`/root/bench-int8-w4a16-hbm/m2-chunked/chunk_sweep.json`.")
    lines.append("")
    return "\n".join(lines)


def _persist(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    state["optimum_per_quant"] = _compute_optimum(state)
    state["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state["cells"].sort(key=lambda r: (r["quant"], int(r["chunk_size"])))
    SWEEP_JSON.write_text(json.dumps(state, indent=2))
    SUMMARY_MD.write_text(_render_summary_md(state))


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------
def sweep(quant: str, chunks: tuple[int, ...]) -> int:
    if quant not in QUANT_MODELS:
        _log(f"FATAL: unknown quant {quant}")
        return 2
    if not CODING_DATASET.is_file():
        _log(f"FATAL: coding dataset missing at {CODING_DATASET}")
        return 2

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    state = _load_existing()
    _kill_orphans()

    failures = 0
    for chunk_size in chunks:
        cell_dir = SWEEP_ROOT / quant / f"cs{chunk_size}"
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True, exist_ok=True)
        server_log = cell_dir / "server.log"

        _log(f"== {quant} chunk_size={chunk_size} ==")
        proc = _start_server(quant, chunk_size, server_log)
        if proc is None:
            failures += 1
            reason, archived = _detect_failure_reason(quant, cell_dir)
            failure_row = {
                "quant": quant,
                "chunk_size": chunk_size,
                "workload": "coding",
                "tp": 1,
                "concurrency": 4,
                "num_prompts": 200,
                "kv_cache_dtype": "int8_per_token_head",
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": chunk_size,
                "status": "FAILED",
                "reason": reason,
                "server_log": str(server_log),
            }
            if archived is not None:
                failure_row["engine_log"] = str(archived)
            _upsert_cell(state, failure_row)
            _persist(state)
            continue

        try:
            raw_json = _run_bench(quant, chunk_size, cell_dir)
            if raw_json is None:
                failures += 1
                _upsert_cell(
                    state,
                    {
                        "quant": quant,
                        "chunk_size": chunk_size,
                        "status": "FAILED",
                        "reason": "vllm bench serve produced no raw.json",
                        "server_log": str(server_log),
                    },
                )
                _persist(state)
                continue

            metrics = _extract_metrics(raw_json)
            row = {
                "quant": quant,
                "chunk_size": chunk_size,
                "workload": "coding",
                "tp": 1,
                "concurrency": 4,
                "num_prompts": 200,
                "kv_cache_dtype": "int8_per_token_head",
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": chunk_size,
                "raw_json": str(raw_json),
                "server_log": str(server_log),
                "status": "OK",
                **metrics,
            }
            _upsert_cell(state, row)
            _log(
                f"  OK chunk_size={chunk_size}: "
                f"req_tput={metrics['request_throughput_req_s']:.4f} req/s "
                f"out_tput={metrics['output_throughput_toks_s']:.2f} tok/s "
                f"mean_ttft={metrics['mean_ttft_ms']:.1f} ms "
                f"mean_tpot={metrics['mean_tpot_ms']:.2f} ms"
            )
            _persist(state)
        finally:
            _stop_server(proc)

    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quant",
        required=True,
        choices=sorted(QUANT_MODELS),
        help="quant scheme to sweep",
    )
    ap.add_argument(
        "--chunks",
        default=",".join(str(c) for c in DEFAULT_CHUNKS),
        help="comma-separated chunk sizes (default: 512,1024,2048,4096)",
    )
    args = ap.parse_args()

    try:
        chunks = tuple(int(c) for c in args.chunks.split(",") if c.strip())
    except ValueError:
        _log(f"FATAL: invalid --chunks value: {args.chunks!r}")
        return 2
    if not chunks:
        _log("FATAL: --chunks resolved to empty list")
        return 2

    return sweep(args.quant, chunks)


if __name__ == "__main__":
    sys.exit(main())
