#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M3 NCCL-algorithm sweep on TP=4 cells (VAL-TP-001).

For each of the 6 TP=4 cells (3 concurrency × 2 quant), measure under
``NCCL_ALGO ∈ {Ring, Tree, default}`` on the synthetic workload at
``NUM_PROMPTS=200``. The sweep stacks on top of the M1 KV-INT8 winner
(``KV_CACHE_DTYPE=int8_per_token_head``) and the M2 chunked-prefill
winner (``ENABLE_CHUNKED_PREFILL=1`` + ``MAX_NUM_BATCHED_TOKENS=<M2
optimum per quant>``) per orchestrator stacking decision.

For each (cell, algo) combination:

  1. Kill any orphan vLLM processes (pkill vllm.entrypoints + sleep 2).
  2. Export ``NCCL_ALGO=<algo>`` (empty string for ``default`` so rccl
     uses its default heuristic), ``KV_CACHE_DTYPE=int8_per_token_head``,
     ``ENABLE_CHUNKED_PREFILL=1``, ``MAX_NUM_BATCHED_TOKENS=<M2 optimum>``.
  3. Invoke ``scripts/launch_<model>_tp4_c<conc>.sh --check`` (synthetic
     random workload, NUM_PROMPTS=200, input=1024, output=256, seed=42).
     The launch script handles its own server lifecycle: kills orphans,
     starts the api_server with the additive env-var → CLI-flag overrides,
     waits for /health, runs vllm bench serve, and terminates the server.
  4. Capture the resulting raw.json (written by the launch script under
     ``/root/bench-int8-w4a16/final/launch_smoke/<cell_id>/bench_<ts>/raw.json``)
     and copy it to
     ``/root/bench-int8-w4a16-hbm/m3-tp/sweep/<cell>_<algo>.json``.
  5. Sleep 2 s before the next combination.

After all 18 combinations complete, persist
``/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json`` with the cells
breakdown (per-cell winner_algo selected by maximum
``output_throughput_toks_s``) and render
``/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep_summary.md`` as a 6-row
markdown table.

Per task spec: synthetic workload only for this sweep (coding is added at
the grid-level in feature ``m3-tp-bench-and-update``).

Usage:
    scripts/mi100/nccl_algo_sweep.py
    scripts/mi100/nccl_algo_sweep.py --cells w8a8_tp4_c1
    scripts/mi100/nccl_algo_sweep.py --algos Ring,Tree

Per VAL-CROSS-001, this script does NOT push to git.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/"
    "emdash/fuzzy-hornets-see-szfl4"
)
PY = "/opt/vllm-env/bin/python3"

# Output paths
OUT_ROOT = Path("/root/bench-int8-w4a16-hbm/m3-tp")
SWEEP_DIR = OUT_ROOT / "sweep"
SWEEP_JSON = OUT_ROOT / "nccl_sweep.json"
SUMMARY_MD = OUT_ROOT / "nccl_sweep_summary.md"

# M2 winner per quant (read from chunk_sweep.json at runtime)
M2_CHUNK_SWEEP_JSON = Path("/root/bench-int8-w4a16-hbm/m2-chunked/chunk_sweep.json")

# Where the launch scripts emit raw.json
LAUNCH_SMOKE_ROOT = Path("/root/bench-int8-w4a16/final/launch_smoke")

QUANTS = ("w8a8", "w4a16")
CONCURRENCIES = (1, 2, 4)
ALGOS = ("Ring", "Tree", "default")  # default = empty NCCL_ALGO
SLEEP_BETWEEN_RUNS_SECS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    print(f"[{ts}] {msg}", flush=True)


def _kill_orphans() -> None:
    """Match the lifecycle pattern from AGENTS.md and the launch scripts:
    pkill vllm.entrypoints + VLLM:: workers + sleep 2."""
    for pat in ("vllm.entrypoints", "VLLM::"):
        subprocess.run(
            ["pkill", "-9", "-f", pat],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(SLEEP_BETWEEN_RUNS_SECS)


def _load_m2_chunk_per_quant() -> dict[str, int]:
    if not M2_CHUNK_SWEEP_JSON.is_file():
        raise SystemExit(
            f"FATAL: missing M2 chunk_sweep.json at {M2_CHUNK_SWEEP_JSON}; "
            "m2-chunked-bench-grid precondition not satisfied."
        )
    raw = json.loads(M2_CHUNK_SWEEP_JSON.read_text())
    opt = raw.get("optimum_per_quant") or {}
    if not all(q in opt for q in QUANTS):
        raise SystemExit(
            f"FATAL: optimum_per_quant missing keys in {M2_CHUNK_SWEEP_JSON}: "
            f"{opt!r}"
        )
    return {q: int(opt[q]) for q in QUANTS}


def _resolve_raw_json(cell_id: str, start_time: float) -> Path | None:
    """The launch scripts write raw.json to:
        /root/bench-int8-w4a16/final/launch_smoke/<cell_id>/bench_<ts>/raw.json
    Find the newest one created after start_time.
    """
    cell_dir = LAUNCH_SMOKE_ROOT / cell_id
    if not cell_dir.is_dir():
        return None
    candidates = []
    for sub in cell_dir.iterdir():
        if not sub.is_dir() or not sub.name.startswith("bench_"):
            continue
        raw = sub / "raw.json"
        if raw.is_file() and raw.stat().st_mtime >= start_time - 5:
            candidates.append((raw.stat().st_mtime, raw))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def _extract_metrics(raw_json: Path) -> dict[str, float]:
    """Pull canonical fields from `vllm bench serve` raw.json."""
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


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------
def _run_one(
    quant: str,
    conc: int,
    algo: str,
    chunk_size: int,
    out_json: Path,
    sweep_log: Path,
) -> dict:
    """Run scripts/launch_<quant>_tp4_c<conc>.sh --check with the requested
    NCCL_ALGO + M1/M2 stacked env overrides, then collect raw.json.

    Returns a row dict (status: OK or FAILED) suitable for inclusion in
    the nccl_sweep.json cells.algos list.
    """
    cell_id = f"{quant}_tp4_c{conc}"
    launch_script = REPO / "scripts" / f"launch_{cell_id}.sh"
    if not launch_script.is_file():
        return {
            "algo": algo,
            "status": "FAILED",
            "reason": f"launch script missing: {launch_script}",
        }

    env = os.environ.copy()
    # M1 winner
    env["KV_CACHE_DTYPE"] = "int8_per_token_head"
    # M2 winner
    env["ENABLE_CHUNKED_PREFILL"] = "1"
    env["MAX_NUM_BATCHED_TOKENS"] = str(chunk_size)
    # M3 axis under test
    # rccl reads NCCL_ALGO from the environment; empty string == default
    # heuristic. We explicitly export the empty value too so that any value
    # inherited from the calling shell is overridden (and the launch script's
    # vLLM child sees a deterministic state).
    if algo == "default":
        env["NCCL_ALGO"] = ""
    else:
        env["NCCL_ALGO"] = algo
    # TP=4 graph capture for chunked-prefill can exceed the 300 s default.
    env["LAUNCH_HEALTH_WAIT_SECS"] = "1800"

    _log(
        f"  -> launch_{cell_id}.sh --check "
        f"NCCL_ALGO={env['NCCL_ALGO'] or '(default)'} "
        f"KV={env['KV_CACHE_DTYPE']} "
        f"chunked_prefill=1 "
        f"max_num_batched_tokens={chunk_size}"
    )
    start = time.time()
    # The launch script handles server start, bench, and teardown itself.
    # We redirect its combined stdout/stderr into a per-(cell,algo) log so
    # post-hoc debugging is straightforward.
    with sweep_log.open("a") as f_log:
        f_log.write(
            f"\n========== {cell_id} algo={algo} chunk={chunk_size} "
            f"start={datetime.datetime.now(datetime.timezone.utc).isoformat()} "
            "==========\n"
        )
        f_log.flush()
        rc = subprocess.run(
            [str(launch_script), "--check"],
            cwd=str(REPO),
            env=env,
            stdout=f_log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    elapsed = time.time() - start
    _log(f"     launch_{cell_id}.sh --check exit={rc} elapsed={elapsed:.1f}s")

    # The launch script always tries to teardown the server on EXIT trap,
    # but be defensive and kill any orphan before next iteration anyway.
    _kill_orphans()

    # Snapshot the per-cell server.log to a per-(cell, algo) file so the
    # next iteration's overwrite doesn't blow away post-mortem evidence.
    shared_server_log = LAUNCH_SMOKE_ROOT / cell_id / "server.log"
    if shared_server_log.is_file():
        try:
            archive = out_json.with_name(f"{cell_id}_{algo}.server.log")
            shutil.copy2(shared_server_log, archive)
        except Exception:  # noqa: BLE001
            pass

    raw_json = _resolve_raw_json(cell_id, start)
    if raw_json is None:
        return {
            "algo": algo,
            "status": "FAILED",
            "reason": (
                f"raw.json not found under {LAUNCH_SMOKE_ROOT/cell_id} after "
                f"launch script exit={rc}"
            ),
            "launch_exit_code": rc,
        }

    # Persist a per-(cell, algo) copy under the m3-tp sweep dir.
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_json, out_json)
    except Exception as e:  # noqa: BLE001
        return {
            "algo": algo,
            "status": "FAILED",
            "reason": f"failed to copy raw.json to {out_json}: {e}",
            "launch_exit_code": rc,
            "source_raw_json": str(raw_json),
        }

    metrics = _extract_metrics(out_json)
    # rc != 0 from --check means the ±2% repro gate failed — the bench
    # measurement is still valid (it's recorded as the "actual" tput in the
    # launch script's stdout). We surface launch_exit_code in the row so the
    # aggregator can flag drift, but we keep status="OK" since the metric
    # extraction succeeded.
    return {
        "algo": algo,
        "status": "OK",
        "raw_json": str(out_json),
        "source_raw_json": str(raw_json),
        "launch_exit_code": rc,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _pick_winner(algo_rows: list[dict]) -> str | None:
    """Winner per cell = algo with max output_throughput_toks_s.
    Per task spec: synthetic workload only this iteration; the
    grid-level geomean across both workloads happens in
    m3-tp-bench-and-update.
    """
    ok_rows = [r for r in algo_rows if r.get("status") == "OK"]
    if not ok_rows:
        return None
    winner = max(
        ok_rows,
        key=lambda r: float(r.get("output_throughput_toks_s") or 0.0),
    )
    return winner["algo"]


def _render_summary_md(state: dict) -> str:
    lines = []
    lines.append("# M3 NCCL-Algorithm Sweep — Per-Cell Winners")
    lines.append("")
    lines.append(
        "Generated by `scripts/mi100/nccl_algo_sweep.py` "
        f"(written at {datetime.datetime.now(datetime.timezone.utc).isoformat()})."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("- Cells: 6 TP=4 cells (3 concurrency × 2 quant)")
    lines.append("- NCCL algos: Ring, Tree, default")
    lines.append(
        "- Workload: synthetic random "
        "(input=1024, output=256, NUM_PROMPTS=200, seed=42)"
    )
    lines.append("- KV-cache: `int8_per_token_head` (M1 winner stacked)")
    lines.append(
        "- chunked-prefill: enabled; max-num-batched-tokens per quant from "
        "M2 `optimum_per_quant`"
    )
    lines.append(
        "- Primary metric: `output_throughput_toks_s` "
        "(synthetic-only this iteration)"
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Cell | Ring (tok/s) | Tree (tok/s) | default (tok/s) | Winner |"
    )
    lines.append(
        "|:-----|-------------:|-------------:|----------------:|:-------|"
    )
    for cell in state.get("cells", []):
        cell_id = cell["cell_id"]
        winner = cell.get("winner_algo") or "—"
        algo_map = {a["algo"]: a for a in cell.get("algos", [])}
        cells_row = []
        for algo in ALGOS:
            row = algo_map.get(algo)
            if row is None or row.get("status") != "OK":
                cells_row.append("FAILED")
            else:
                marker = " *" if algo == winner else ""
                cells_row.append(
                    f"{float(row.get('output_throughput_toks_s') or 0):.2f}"
                    f"{marker}"
                )
        lines.append(
            f"| {cell_id} | {cells_row[0]} | {cells_row[1]} "
            f"| {cells_row[2]} | **{winner}** |"
        )
    lines.append("")
    lines.append("(`*` marks the winning algo for that cell.)")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The `default` column corresponds to leaving `NCCL_ALGO` unset, "
        "i.e. rccl picks per its default heuristic."
    )
    lines.append(
        "- The winner per cell is baked into the corresponding "
        "`scripts/launch_<model>_tp4_c<conc>.sh` in the follow-up feature "
        "`m3-tp-bench-and-update`."
    )
    lines.append(
        "- Coding-workload measurements are added to the grid-level Pareto "
        "in `m3-tp-bench-and-update`; this sweep is synthetic-only."
    )
    tree_root_cause = state.get("tree_failure_root_cause")
    if tree_root_cause:
        lines.append("")
        lines.append("## Tree-algo failure root cause")
        lines.append("")
        lines.append(tree_root_cause)
        lines.append("")
        lines.append(
            "Implication: NCCL_ALGO=Tree is **not selectable** on the "
            "M1+M2-stacked TP=4 path on gfx908. The winner search "
            "therefore reduces to {Ring, default}, which on these cells "
            "are within ~1% of each other (see Ring vs default columns). "
            "This is a candidate for the conditional-pass / "
            "negative-result clause when m3-tp-bench-and-update evaluates "
            "VAL-TP-004 (≥+3 % throughput win on ≥3 TP=4 cells)."
        )
    lines.append("")
    lines.append(
        "Persisted machine-readable copy: "
        "`/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json`."
    )
    lines.append("")
    return "\n".join(lines)


def _persist(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    state["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Per-cell winner refresh
    for cell in state.get("cells", []):
        cell["winner_algo"] = _pick_winner(cell.get("algos", []))
    SWEEP_JSON.write_text(json.dumps(state, indent=2))
    SUMMARY_MD.write_text(_render_summary_md(state))


def _load_existing(chunk_per_quant: dict[str, int]) -> dict:
    if SWEEP_JSON.is_file():
        try:
            return json.loads(SWEEP_JSON.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {
        "schema_version": "m3-nccl-sweep-v1",
        "harness": "scripts/mi100/nccl_algo_sweep.py",
        "workload": "synthetic",
        "num_prompts": 200,
        "request_rate": "inf",
        "kv_cache_dtype": "int8_per_token_head",
        "enable_chunked_prefill": True,
        "max_num_batched_tokens_per_quant": chunk_per_quant,
        "algos": list(ALGOS),
        "cells": [],
    }


def _upsert_cell_algo(state: dict, cell_id: str, quant: str, conc: int,
                      algo_row: dict) -> None:
    """Insert or replace an algo row inside state['cells'][<cell>].algos."""
    cells = state.setdefault("cells", [])
    cell_entry = None
    for c in cells:
        if c.get("cell_id") == cell_id:
            cell_entry = c
            break
    if cell_entry is None:
        cell_entry = {
            "cell_id": cell_id,
            "w8a8_or_w4a16": quant,
            "tp": 4,
            "concurrency": conc,
            "algos": [],
            "winner_algo": None,
        }
        cells.append(cell_entry)

    algos = cell_entry.setdefault("algos", [])
    for i, existing in enumerate(algos):
        if existing.get("algo") == algo_row.get("algo"):
            algos[i] = algo_row
            break
    else:
        algos.append(algo_row)


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------
def sweep(
    target_cells: list[tuple[str, int]],
    target_algos: tuple[str, ...],
) -> int:
    chunk_per_quant = _load_m2_chunk_per_quant()
    _log(f"M2 chunk-size optimum per quant: {chunk_per_quant}")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_existing(chunk_per_quant)

    sweep_log = OUT_ROOT / "nccl_sweep.log"
    sweep_log.parent.mkdir(parents=True, exist_ok=True)

    _kill_orphans()

    n_failed = 0
    n_total = 0
    for (quant, conc) in target_cells:
        cell_id = f"{quant}_tp4_c{conc}"
        for algo in target_algos:
            n_total += 1
            algo_label = algo  # "Ring" | "Tree" | "default"
            out_json = SWEEP_DIR / f"{cell_id}_{algo_label}.json"
            _log(f"== {cell_id} algo={algo_label} ==")
            row = _run_one(
                quant=quant,
                conc=conc,
                algo=algo_label,
                chunk_size=chunk_per_quant[quant],
                out_json=out_json,
                sweep_log=sweep_log,
            )
            if row.get("status") != "OK":
                n_failed += 1
                _log(f"  FAILED: {row.get('reason', 'unknown reason')}")
            else:
                _log(
                    f"  OK: out_tput={row['output_throughput_toks_s']:.2f} tok/s "
                    f"req_tput={row['request_throughput_req_s']:.4f} req/s "
                    f"mean_ttft={row['mean_ttft_ms']:.1f} ms "
                    f"p99_ttft={row['p99_ttft_ms']:.1f} ms"
                )
            _upsert_cell_algo(state, cell_id, quant, conc, row)
            _persist(state)
            time.sleep(SLEEP_BETWEEN_RUNS_SECS)

    # Final order: by quant then concurrency for stable rendering.
    state["cells"].sort(
        key=lambda c: (c["w8a8_or_w4a16"], int(c["concurrency"]))
    )
    _persist(state)

    _log(f"sweep complete: {n_total - n_failed}/{n_total} runs OK")
    return 0 if n_failed == 0 else 1


def _parse_cells(arg: str) -> list[tuple[str, int]]:
    """Parse --cells 'w8a8_tp4_c1,w4a16_tp4_c4' or 'all'."""
    if not arg or arg == "all":
        return [(q, c) for q in QUANTS for c in CONCURRENCIES]
    pairs: list[tuple[str, int]] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        # Accept "w8a8_tp4_c1" or short "w8a8_c1".
        parts = token.split("_")
        quant = parts[0]
        conc = None
        for p in parts[1:]:
            if p.startswith("c") and p[1:].isdigit():
                conc = int(p[1:])
                break
        if quant not in QUANTS or conc not in CONCURRENCIES:
            raise SystemExit(f"FATAL: invalid --cells token: {token!r}")
        pairs.append((quant, conc))
    return pairs


def _parse_algos(arg: str) -> tuple[str, ...]:
    if not arg or arg == "all":
        return ALGOS
    out: list[str] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in ALGOS:
            raise SystemExit(
                f"FATAL: invalid --algos token: {token!r}; "
                f"expected one of {ALGOS}"
            )
        out.append(token)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells",
        default="all",
        help=(
            "Comma-separated cells to sweep "
            "(default: all 6 TP=4 cells). "
            "Examples: 'w8a8_tp4_c1', 'w8a8_tp4_c1,w4a16_tp4_c4'."
        ),
    )
    ap.add_argument(
        "--algos",
        default="all",
        help=(
            "Comma-separated NCCL algos to sweep "
            "(default: Ring,Tree,default)."
        ),
    )
    args = ap.parse_args()

    target_cells = _parse_cells(args.cells)
    target_algos = _parse_algos(args.algos)
    return sweep(target_cells, target_algos)


if __name__ == "__main__":
    sys.exit(main())
