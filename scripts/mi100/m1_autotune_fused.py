#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1-F2 autotune sweep for the three MI100 fused Triton kernels.

Covers the surveyed shapes from m1-shape-survey and persists the per-shape
best config under ``vllm/model_executor/kernels/configs/gfx908/``.

Kernels in scope:
    * ``mi100_int8_scaled_mm_kernel`` (both EMIT_INT8_NEXT={False, True}).
      Pinned filename: ``mi100_int8_M<M>_N<N>_K<K>.json`` (or ``_e1.json``
      for the EMIT_INT8_NEXT=True variant).
    * ``_fused_silu_quant_int8_kernel`` (M1 silu+mul + per-token int8 quant).
      Pinned filename: ``fused_silu_quant_int8_M<M>_H<H>.json``.
    * ``fused_int8_quant`` (candidate per-token int8 quantizer; not on the
      production hot path in this worktree).
      Pinned filename: ``fused_int8_quant_M<M>_H<N>.json``.

The sweep grids are intentionally small (sub-cartesian) so each (kernel,
shape) cell stays inside the 10-minute wall-clock budget set by the
m1-autotune-jsons feature description.

Usage::

    PYTHONPATH=<repo> /opt/vllm-env/bin/python3 \\
        scripts/mi100/m1_autotune_fused.py \\
        --shapes-json <survey.json> \\
        --out-dir vllm/model_executor/kernels/configs/gfx908 \\
        --log-dir <log-dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Resolve repo root from this file's location so the script works in any
# worktree (no hard-coded absolute path).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_SHAPES = Path(
    "/root/bench-int8-w4a16-m2-producer/m1-shapes/qwen3p5_9b_w8a8_shapes.json"
)
DEFAULT_OUT = REPO / "vllm/model_executor/kernels/configs/gfx908"
DEFAULT_LOG = Path("/root/bench-int8-w4a16-m2-producer/m1-shapes")

WARMUP_ITERS = 3
TIMING_ITERS = 5

logger = logging.getLogger("m1_autotune_fused")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _time_launch(launch, trials: int) -> float | None:
    """Return median ms across ``trials`` warm-launches, or None on failure."""
    sync = torch.accelerator.synchronize
    try:
        for _ in range(WARMUP_ITERS):
            launch()
        sync()
        timings: list[float] = []
        for _ in range(trials):
            sync()
            t0 = time.perf_counter()
            launch()
            sync()
            timings.append((time.perf_counter() - t0) * 1000.0)
        timings.sort()
        return timings[len(timings) // 2]
    except Exception as exc:  # noqa: BLE001
        logger.debug("launch failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Kernel 1: mi100_int8_scaled_mm_kernel                                       #
# --------------------------------------------------------------------------- #


# Pruned sub-grid (legal-only, hand-picked for sub-10-minute budget per shape).
# Kept small but representative across BLOCK_M / BLOCK_K / waves_per_eu.
_W8A8_GRID = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, matrix_instr_nonkdim, kpack,
    #  waves_per_eu, num_warps, num_stages)
    (16, 64, 128, 8, 16, 2, 1, 4, 2),
    (16, 64, 128, 8, 16, 2, 2, 4, 2),
    (16, 64, 128, 8, 16, 2, 1, 4, 1),
    (32, 64, 128, 8, 16, 2, 2, 4, 2),
    (32, 64, 128, 8, 32, 2, 2, 4, 2),
    (32, 128, 64, 8, 32, 2, 2, 4, 2),
    (32, 128, 128, 8, 32, 2, 2, 4, 2),
    (64, 64, 64, 8, 32, 2, 2, 4, 2),
    (64, 128, 64, 8, 32, 2, 2, 4, 2),
    (64, 128, 64, 8, 32, 2, 0, 4, 2),
    (64, 128, 64, 8, 32, 2, 3, 4, 2),
    (128, 128, 64, 8, 32, 2, 2, 4, 2),
    (128, 128, 64, 8, 32, 2, 0, 4, 2),
    (128, 64, 64, 8, 32, 2, 2, 4, 2),
]


def _w8a8_lds_ok(block_m: int, block_n: int, block_k: int) -> bool:
    """gfx908 has 64 KiB LDS/CU; A+B int8 tile must fit."""
    return block_m * block_k + block_k * block_n <= 65536


def _bench_w8a8_one(
    *,
    M: int,
    N: int,
    K: int,
    cfg: tuple,
    emit_int8_next: bool,
    trials: int,
) -> tuple[float | None, str | None]:
    """Benchmark one W8A8 config; returns (median ms, err or None)."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm_kernel,
    )
    from vllm.triton_utils import triton

    (
        block_m,
        block_n,
        block_k,
        group_size_m,
        matrix_instr_nonkdim,
        kpack,
        waves_per_eu,
        num_warps,
        num_stages,
    ) = cfg

    # For EMIT_INT8_NEXT=True the kernel requires BLOCK_N >= N (per the
    # wrapper's correctness guard). Force the next pow2 of N and keep
    # BLOCK_M / BLOCK_K small enough that the (A + B) int8 LDS tile stays
    # under the 64 KiB budget. We pre-filter so the sweep does not waste
    # time on shapes that are guaranteed to OOM LDS.
    if emit_int8_next:
        block_n = _next_pow2(N)
        # The wrapper's BLOCK_N >= N assertion is already satisfied.
        if not _w8a8_lds_ok(block_m, block_n, block_k):
            return None, f"lds_overflow_{block_m}x{block_n}x{block_k}"
        if matrix_instr_nonkdim == 32 and (block_m < 32 or block_n < 32):
            return None, "nonkdim32_needs>=32"
    else:
        if not _w8a8_lds_ok(block_m, block_n, block_k):
            return None, "lds_overflow"
        if matrix_instr_nonkdim == 32 and (block_m < 32 or block_n < 32):
            return None, "nonkdim32_needs>=32"
        if block_n > N:
            return None, "block_n_gt_N"

    torch.manual_seed(0)
    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device="cuda")
    sa = torch.rand((M, 1), device="cuda", dtype=torch.float32) * 0.01
    sb = torch.rand((N, 1), device="cuda", dtype=torch.float32) * 0.01
    if emit_int8_next:
        out_int8 = torch.empty((M, N), dtype=torch.int8, device="cuda")
        out_scale = torch.empty((M, 1), dtype=torch.float32, device="cuda")
        result = out_int8  # store target for stride math
    else:
        result = torch.empty((M, N), dtype=torch.float16, device="cuda")
        out_int8 = result
        out_scale = result

    block_size_sa = block_m
    block_size_sb = block_n

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    extra = {
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "kpack": kpack,
        "waves_per_eu": waves_per_eu,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def _launch():
        mi100_int8_scaled_mm_kernel[grid](
            a,
            b,
            sa,
            sb,
            result,
            None,
            out_int8,
            out_scale,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            result.stride(0),
            result.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_SCALE_A=block_size_sa,
            BLOCK_SIZE_SCALE_B=block_size_sb,
            GROUP_SIZE_M=group_size_m,
            EMIT_INT8_NEXT=emit_int8_next,
            **extra,
        )

    ms = _time_launch(_launch, trials)
    if ms is None:
        return None, "launch_exception"
    return ms, None


def _autotune_w8a8(
    shape: dict, trials: int, budget_s: float
) -> tuple[dict | None, dict]:
    M = int(shape["M"])
    N = int(shape["N"])
    K = int(shape["K"])
    emit_int8_next = bool(shape.get("EMIT_INT8_NEXT", False))

    start = time.time()
    evaluated = 0
    pruned = 0
    best: tuple[float, tuple] | None = None

    for cfg in _W8A8_GRID:
        if time.time() - start > budget_s:
            logger.warning(
                "budget exhausted for M=%d N=%d K=%d e1=%s after %d evaluated",
                M,
                N,
                K,
                emit_int8_next,
                evaluated,
            )
            break
        ms, err = _bench_w8a8_one(
            M=M,
            N=N,
            K=K,
            cfg=cfg,
            emit_int8_next=emit_int8_next,
            trials=trials,
        )
        if err is not None:
            pruned += 1
            continue
        evaluated += 1
        if ms is None:
            continue
        if best is None or ms < best[0]:
            best = (ms, cfg)

    if best is None:
        return None, {
            "evaluated": evaluated,
            "pruned": pruned,
            "grid_total": len(_W8A8_GRID),
        }

    ms, cfg = best
    (
        block_m,
        block_n,
        block_k,
        group_size_m,
        matrix_instr_nonkdim,
        kpack,
        waves_per_eu,
        num_warps,
        num_stages,
    ) = cfg
    # For EMIT_INT8_NEXT=True the actual launched BLOCK_N is the next pow2 of
    # N (the wrapper enforces this). Persist that effective value so the
    # runtime loader returns a config that matches what was measured.
    if emit_int8_next:
        block_n = _next_pow2(N)

    cfg_dict = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_SIZE_M": group_size_m,
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "kpack": kpack,
        "waves_per_eu": waves_per_eu,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    return cfg_dict, {
        "best_ms": ms,
        "evaluated": evaluated,
        "pruned": pruned,
        "grid_total": len(_W8A8_GRID),
    }


# --------------------------------------------------------------------------- #
# Kernel 2: _fused_silu_quant_int8_kernel                                     #
# --------------------------------------------------------------------------- #


# Small sub-grid: (BLOCK_M, BLOCK_H, num_warps, num_stages). BLOCK_H is
# capped at 1024 by the kernel's docstring (see fused_silu_quant_int8.py).
_SILU_GRID = [
    (16, 256, 2, 2),
    (16, 512, 4, 2),
    (16, 1024, 4, 2),
    (32, 256, 2, 2),
    (32, 512, 4, 2),
    (32, 1024, 4, 2),
    (64, 256, 2, 2),
    (64, 512, 4, 2),
    (64, 1024, 4, 2),
]


def _bench_silu_one(
    *,
    M: int,
    H: int,
    cfg: tuple,
    trials: int,
) -> tuple[float | None, str | None]:
    from vllm.model_executor.kernels.quantization.fused_silu_quant_int8 import (
        _fused_silu_quant_int8_kernel,
    )
    from vllm.triton_utils import triton

    block_m, block_h, num_warps, num_stages = cfg
    # Skip configs whose BLOCK_H is much larger than H — they only waste
    # masked-off lanes and do not surface a representative timing.
    if block_h > _next_pow2(max(H, 32)) * 4:
        return None, "block_h_far_above_H"

    torch.manual_seed(0)
    x = torch.randn((M, 2 * H), dtype=torch.float16, device="cuda")
    q = torch.empty((M, H), dtype=torch.int8, device="cuda")
    scale = torch.empty((M, 1), dtype=torch.float32, device="cuda")

    grid = (triton.cdiv(M, block_m),)

    def _launch():
        _fused_silu_quant_int8_kernel[grid](
            x,
            q,
            scale,
            M,
            H,
            x.stride(0),
            x.stride(1),
            q.stride(0),
            q.stride(1),
            BLOCK_M=block_m,
            BLOCK_H=block_h,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    ms = _time_launch(_launch, trials)
    if ms is None:
        return None, "launch_exception"
    return ms, None


def _autotune_silu(
    shape: dict,
    trials: int,
    budget_s: float,
) -> tuple[dict | None, dict]:
    M = int(shape["M"])
    # The survey records H under ``N`` (kernel's output hidden) and ``K`` is
    # the inferred 2*H (silu_and_mul layout). We tune on the per-row
    # reduction width H. Fall back to N when both are present.
    H = int(shape.get("N", 0))
    if H == 0:
        H = int(shape["K"]) // 2

    start = time.time()
    evaluated = 0
    pruned = 0
    best: tuple[float, tuple] | None = None

    for cfg in _SILU_GRID:
        if time.time() - start > budget_s:
            logger.warning(
                "budget exhausted for silu M=%d H=%d after %d evaluated",
                M,
                H,
                evaluated,
            )
            break
        ms, err = _bench_silu_one(M=M, H=H, cfg=cfg, trials=trials)
        if err is not None:
            pruned += 1
            continue
        evaluated += 1
        if ms is None:
            continue
        if best is None or ms < best[0]:
            best = (ms, cfg)

    if best is None:
        return None, {
            "evaluated": evaluated,
            "pruned": pruned,
            "grid_total": len(_SILU_GRID),
            "H": H,
        }

    ms, cfg = best
    block_m, block_h, num_warps, num_stages = cfg
    cfg_dict = {
        "BLOCK_M": block_m,
        "BLOCK_H": block_h,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    return cfg_dict, {
        "best_ms": ms,
        "evaluated": evaluated,
        "pruned": pruned,
        "grid_total": len(_SILU_GRID),
        "H": H,
    }


# --------------------------------------------------------------------------- #
# Kernel 3: fused_int8_quant                                                  #
# --------------------------------------------------------------------------- #


# Small sub-grid: (BLOCK_M, num_warps, num_stages). The kernel's BLOCK_N is
# next_power_of_2(N) and is not part of the per-call autotune surface (the
# kernel is a single-tile reduction across the hidden dim).
_QUANT_GRID = [
    (1, 4, 1),
    (1, 8, 1),
    (2, 4, 1),
    (2, 8, 1),
    (4, 4, 1),
    (4, 8, 1),
]


def _bench_quant_one(
    *,
    M: int,
    N: int,
    cfg: tuple,
    trials: int,
) -> tuple[float | None, str | None]:
    from vllm.model_executor.kernels.linear.mixed_precision.fused_int8_quant import (
        _fused_int8_quant_kernel,
    )
    from vllm.triton_utils import triton

    block_m, num_warps, num_stages = cfg
    block_n = _next_pow2(N)
    if block_n > 8192:
        return None, "block_n_above_cap"

    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float16, device="cuda")
    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M, 1), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(M, block_m),)

    def _launch():
        _fused_int8_quant_kernel[grid](
            x,
            x_q,
            scales,
            M,
            N,
            x.stride(0),
            x.stride(1),
            x_q.stride(0),
            x_q.stride(1),
            EPS=1e-10,
            INV_INT8_MAX=1.0 / 127.0,
            INT8_MAX=127.0,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    ms = _time_launch(_launch, trials)
    if ms is None:
        return None, "launch_exception"
    return ms, None


def _autotune_quant(
    shape: dict,
    trials: int,
    budget_s: float,
) -> tuple[dict | None, dict]:
    M = int(shape["M"])
    # fused_int8_quant has no K-dim; its hidden dim is what the survey
    # recorded under ``N``.
    N = int(shape["N"])

    start = time.time()
    evaluated = 0
    pruned = 0
    best: tuple[float, tuple] | None = None

    for cfg in _QUANT_GRID:
        if time.time() - start > budget_s:
            logger.warning(
                "budget exhausted for quant M=%d N=%d after %d evaluated",
                M,
                N,
                evaluated,
            )
            break
        ms, err = _bench_quant_one(M=M, N=N, cfg=cfg, trials=trials)
        if err is not None:
            pruned += 1
            continue
        evaluated += 1
        if ms is None:
            continue
        if best is None or ms < best[0]:
            best = (ms, cfg)

    if best is None:
        return None, {
            "evaluated": evaluated,
            "pruned": pruned,
            "grid_total": len(_QUANT_GRID),
        }

    ms, cfg = best
    block_m, num_warps, num_stages = cfg
    cfg_dict = {
        "BLOCK_M": block_m,
        "BLOCK_N": _next_pow2(N),
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    return cfg_dict, {
        "best_ms": ms,
        "evaluated": evaluated,
        "pruned": pruned,
        "grid_total": len(_QUANT_GRID),
    }


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def _persist(
    out_dir: Path,
    fname: str,
    kernel: str,
    shape: dict,
    cfg: dict,
    meta: dict,
    kernel_src: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / fname
    M = int(shape["M"])
    best_ms = float(meta.get("best_ms", 0.0))
    measured_tok_s = (M / (best_ms / 1000.0)) if best_ms > 0 else 0.0
    payload = {
        "schema_version": 1,
        "kernel": kernel,
        "shape": {
            "M": M,
            "N": int(shape.get("N", 0)) or None,
            "K": int(shape.get("K", 0)) or None,
            "H": meta.get("H"),
            "group_size": None,
            "emit_int8_next": bool(shape.get("EMIT_INT8_NEXT", False)),
        },
        "config": cfg,
        "measured_ms_per_iter": best_ms,
        "measured_tok_s": measured_tok_s,
        "autotune_runs_evaluated": int(meta.get("evaluated", 0)),
        "autotune_runs_pruned": int(meta.get("pruned", 0)),
        "autotune_cartesian_total": int(meta.get("grid_total", 0)),
        "autotune_legal_subset": int(meta.get("grid_total", 0))
        - int(meta.get("pruned", 0)),
        "autotune_legal_coverage_pct": (
            (
                int(meta.get("evaluated", 0))
                / max(1, int(meta.get("grid_total", 0)) - int(meta.get("pruned", 0)))
            )
            * 100.0
        ),
        "autotune_hard_invariant_eliminated": int(meta.get("pruned", 0)),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vllm_commit": _git_sha(),
        "rocm_version": os.environ.get("ROCM_VERSION", "7.12"),
        "kernel_source_sha256": _sha_of(kernel_src),
    }
    fpath.write_text(json.dumps(payload, indent=2))
    return fpath


def _w8a8_fname(M: int, N: int, K: int, emit_int8_next: bool) -> str:
    return (
        f"mi100_int8_M{M}_N{N}_K{K}_e1.json"
        if emit_int8_next
        else f"mi100_int8_M{M}_N{N}_K{K}.json"
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes-json", default=str(DEFAULT_SHAPES))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--log-dir", default=str(DEFAULT_LOG))
    p.add_argument("--trials-per-config", type=int, default=TIMING_ITERS)
    p.add_argument(
        "--budget-s",
        type=float,
        default=600.0,
        help="Per-shape wall-clock budget in seconds (default 10 min).",
    )
    p.add_argument(
        "--kernels",
        nargs="+",
        default=["w8a8", "silu", "quant"],
        choices=["w8a8", "silu", "quant"],
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[m1-autotune] %(asctime)s %(levelname)s %(message)s",
    )

    shapes_path = Path(args.shapes_json)
    if not shapes_path.exists():
        logger.error("shapes JSON not found: %s", shapes_path)
        return 2
    survey = json.loads(shapes_path.read_text())

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "m1_autotune_summary.json"

    w8a8_src = REPO / "vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py"
    silu_src = (
        REPO / "vllm/model_executor/kernels/quantization/fused_silu_quant_int8.py"
    )
    quant_src = (
        REPO / "vllm/model_executor/kernels/linear/mixed_precision/fused_int8_quant.py"
    )

    results: dict = {
        "mi100_int8_scaled_mm_kernel": [],
        "_fused_silu_quant_int8_kernel": [],
        "fused_int8_quant": [],
    }

    if "w8a8" in args.kernels:
        for shape in survey.get("mi100_int8_scaled_mm_kernel", {}).get("shapes", []):
            M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
            emit = bool(shape.get("EMIT_INT8_NEXT", False))
            logger.info("=== mi100_int8 M=%d N=%d K=%d e1=%s ===", M, N, K, emit)
            cfg, meta = _autotune_w8a8(shape, args.trials_per_config, args.budget_s)
            if cfg is None:
                logger.warning(
                    "no winner for mi100_int8 M=%d N=%d K=%d e1=%s "
                    "(evaluated=%d pruned=%d/%d)",
                    M,
                    N,
                    K,
                    emit,
                    meta["evaluated"],
                    meta["pruned"],
                    meta["grid_total"],
                )
                results["mi100_int8_scaled_mm_kernel"].append(
                    {"M": M, "N": N, "K": K, "emit_int8_next": emit, **meta}
                )
                continue
            fname = _w8a8_fname(M, N, K, emit)
            path = _persist(out_dir, fname, "mi100_int8", shape, cfg, meta, w8a8_src)
            logger.info(
                "best mi100_int8 M=%d N=%d K=%d e1=%s -> %s (ms=%.4f)",
                M,
                N,
                K,
                emit,
                fname,
                meta["best_ms"],
            )
            results["mi100_int8_scaled_mm_kernel"].append(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "emit_int8_next": emit,
                    "path": str(path),
                    **meta,
                }
            )

    if "silu" in args.kernels:
        for shape in survey.get("_fused_silu_quant_int8_kernel", {}).get("shapes", []):
            M = int(shape["M"])
            logger.info("=== fused_silu_quant_int8 M=%d (H from survey) ===", M)
            cfg, meta = _autotune_silu(shape, args.trials_per_config, args.budget_s)
            if cfg is None:
                logger.warning(
                    "no winner for silu M=%d (evaluated=%d pruned=%d/%d)",
                    M,
                    meta["evaluated"],
                    meta["pruned"],
                    meta["grid_total"],
                )
                results["_fused_silu_quant_int8_kernel"].append({"M": M, **meta})
                continue
            H = meta["H"]
            fname = f"fused_silu_quant_int8_M{M}_H{H}.json"
            path = _persist(
                out_dir,
                fname,
                "fused_silu_quant_int8",
                shape,
                cfg,
                meta,
                silu_src,
            )
            logger.info(
                "best silu M=%d H=%d -> %s (ms=%.4f)",
                M,
                H,
                fname,
                meta["best_ms"],
            )
            results["_fused_silu_quant_int8_kernel"].append(
                {"M": M, "H": H, "path": str(path), **meta}
            )

    if "quant" in args.kernels:
        for shape in survey.get("fused_int8_quant", {}).get("shapes", []):
            M = int(shape["M"])
            N = int(shape["N"])
            logger.info("=== fused_int8_quant M=%d N=%d ===", M, N)
            cfg, meta = _autotune_quant(shape, args.trials_per_config, args.budget_s)
            if cfg is None:
                logger.warning(
                    "no winner for quant M=%d N=%d (evaluated=%d pruned=%d/%d)",
                    M,
                    N,
                    meta["evaluated"],
                    meta["pruned"],
                    meta["grid_total"],
                )
                results["fused_int8_quant"].append({"M": M, "N": N, **meta})
                continue
            # The fused_int8_quant loader keys on H (=N for this kernel).
            fname = f"fused_int8_quant_M{M}_H{N}.json"
            path = _persist(
                out_dir,
                fname,
                "fused_int8_quant",
                shape,
                cfg,
                {**meta, "H": N},
                quant_src,
            )
            logger.info(
                "best quant M=%d N=%d -> %s (ms=%.4f)",
                M,
                N,
                fname,
                meta["best_ms"],
            )
            results["fused_int8_quant"].append(
                {"M": M, "N": N, "path": str(path), **meta}
            )

    summary_path.write_text(
        json.dumps(
            {
                "shapes_json": str(shapes_path),
                "out_dir": str(out_dir),
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "vllm_commit": _git_sha(),
                "trials_per_config": args.trials_per_config,
                "budget_s": args.budget_s,
                "results": results,
            },
            indent=2,
        )
    )
    logger.info("summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
