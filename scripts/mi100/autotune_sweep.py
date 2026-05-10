#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3 autotune sweep for the MI100 (gfx908) custom Triton GEMMs.

Usage::

    /opt/vllm-env/bin/python3 scripts/mi100/autotune_sweep.py \\
        --kernel {w8a8,w4a16} \\
        [--shapes-json /root/bench-int8-w4a16/baseline/hot_shapes.json] \\
        [--out-dir vllm/model_executor/kernels/configs/gfx908] \\
        [--log-dir /root/bench-int8-w4a16/triton/autotune] \\
        [--trials-per-config 5] \\
        [--max-configs N]            # debug: limit cartesian product
        [--shape M,N,K[,group]]      # debug: tune one shape inline

Sweep space (per the M3 brief):

* ``BLOCK_M ∈ {16, 32, 64, 128}``
* ``BLOCK_N ∈ {32, 64, 128, 256}``
* ``BLOCK_K ∈ {32, 64, 128}``
* ``matrix_instr_nonkdim ∈ {16, 32}``
* ``kpack = 2``
* ``waves_per_eu ∈ {0, 1, 2, 3}``
* ``num_stages ∈ {1, 2}``

Pruning rules (logged with reason):
  - LDS overflow: A-tile + B-tile + acc > 64 KB
  - BLOCK_M > M (next-pow2): irrelevant for the shape
  - BLOCK_K > group_size on W4A16 (would cross quant groups)
  - num_warps × wavefront-size > BLOCK_M × BLOCK_N (illegal)

VAL-TRITON-003 requires >= 80% of the cartesian product to be evaluated
per shape. The sweep records per-config measurement results into a JSONL
log, and the resulting "best config" gets persisted to:

    vllm/model_executor/kernels/configs/gfx908/<kernel>_M<M>_N<N>_K<K>[_g<g>].json

with full provenance (timestamp, vllm git SHA, ROCm version, hash of the
kernel source file).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# Resolve repo root from this file's location (scripts/mi100/autotune_sweep.py)
# so the script works in any worktree (no hard-coded absolute path).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Defaults
DEFAULT_OUT = REPO / "vllm/model_executor/kernels/configs/gfx908"
DEFAULT_LOG = Path("/root/bench-int8-w4a16/triton/autotune")
DEFAULT_HOT = Path("/root/bench-int8-w4a16/baseline/hot_shapes.json")

# Sweep grid (per M3 brief)
BLOCK_M_VALUES = [16, 32, 64, 128]
BLOCK_N_VALUES = [32, 64, 128, 256]
BLOCK_K_VALUES = [32, 64, 128]
MATRIX_INSTR_NONKDIM = [16, 32]
KPACK_VALUES = [2]
WAVES_PER_EU_VALUES = [0, 1, 2, 3]
NUM_STAGES_VALUES = [1, 2]

WARMUP_ITERS = 3
TIMING_ITERS = 10  # default; overridden by --trials-per-config

logger = logging.getLogger("autotune_sweep")


@dataclass
class SweepConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    matrix_instr_nonkdim: int
    kpack: int
    waves_per_eu: int
    num_stages: int

    def as_launch_kwargs(self) -> dict:
        return {
            "BLOCK_M": self.BLOCK_M,
            "BLOCK_N": self.BLOCK_N,
            "BLOCK_K": self.BLOCK_K,
            "matrix_instr_nonkdim": self.matrix_instr_nonkdim,
            "kpack": self.kpack,
            "waves_per_eu": self.waves_per_eu,
            "num_stages": self.num_stages,
        }


def _enumerate_configs() -> list[SweepConfig]:
    return [
        SweepConfig(*combo)
        for combo in itertools.product(
            BLOCK_M_VALUES,
            BLOCK_N_VALUES,
            BLOCK_K_VALUES,
            MATRIX_INSTR_NONKDIM,
            KPACK_VALUES,
            WAVES_PER_EU_VALUES,
            NUM_STAGES_VALUES,
        )
    ]


def _prune_w8a8(cfg: SweepConfig, M: int, N: int, K: int) -> str | None:
    """Prune only configs that are illegal or guaranteed-OOM.

    VAL-TRITON-003 demands >=80% of the cartesian product evaluated per
    shape, so we err on the side of evaluating rather than pruning.
    Concretely, we prune only:
      - LDS overflow (A-tile + B-tile > 64 KB)
      - matrix_instr_nonkdim=32 with tiles smaller than 32x32
      - BLOCK_N > N (would launch zero-work tiles).
      - BLOCK_K > K (same).
    All other shape-vs-tile mismatches are evaluated; the autotuner
    decides the winner.
    """
    a_bytes = cfg.BLOCK_M * cfg.BLOCK_K
    b_bytes = cfg.BLOCK_K * cfg.BLOCK_N
    if a_bytes + b_bytes > 65536:
        return f"lds_overflow_{a_bytes + b_bytes}B"
    if cfg.BLOCK_N > N:
        return "block_n_gt_N"
    if cfg.BLOCK_K > K:
        return "block_k_gt_K"
    if cfg.matrix_instr_nonkdim == 32 and (
        cfg.BLOCK_M < 32 or cfg.BLOCK_N < 32
    ):
        return "nonkdim32_needs_>=32_tiles"
    return None


def _prune_w4a16(
    cfg: SweepConfig, M: int, N: int, K: int, group_size: int
) -> str | None:
    """Same philosophy as _prune_w8a8: only illegal configs are dropped."""
    if group_size < cfg.BLOCK_K:
        return f"block_k_gt_group_{cfg.BLOCK_K}_{group_size}"
    if cfg.BLOCK_N % 8 != 0:
        return "block_n_not_multiple_of_8"
    a_bytes = cfg.BLOCK_M * cfg.BLOCK_K * 2
    b_bytes = cfg.BLOCK_K * cfg.BLOCK_N * 2
    if a_bytes + b_bytes > 65536:
        return f"lds_overflow_{a_bytes + b_bytes}B"
    if cfg.BLOCK_N > N:
        return "block_n_gt_N"
    if cfg.matrix_instr_nonkdim == 32 and (
        cfg.BLOCK_M < 32 or cfg.BLOCK_N < 32
    ):
        return "nonkdim32_needs_>=32_tiles"
    return None


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


# --------------------------- W8A8 path ---------------------------------------

def _bench_w8a8_one_config(
    cfg: SweepConfig, M: int, N: int, K: int, trials: int
) -> tuple[float | None, str | None]:
    """Benchmark one config; return (median_ms_per_op, error or None)."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm_kernel,
    )
    from vllm.triton_utils import triton

    torch.manual_seed(0)
    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device="cuda")
    sa = torch.rand((M, 1), device="cuda", dtype=torch.float32) * 0.01
    sb = torch.rand((N, 1), device="cuda", dtype=torch.float32) * 0.01
    out = torch.empty((M, N), dtype=torch.float16, device="cuda")

    block_size_sa = cfg.BLOCK_M
    block_size_sb = cfg.BLOCK_N

    # GROUP_SIZE_M for L2 swizzle: same as runtime heuristic.
    num_pid_n = triton.cdiv(N, cfg.BLOCK_N)
    group_size_m = max(1, min(8, 120 // num_pid_n)) if num_pid_n else 1

    grid = (triton.cdiv(M, cfg.BLOCK_M) * triton.cdiv(N, cfg.BLOCK_N),)

    extra = {}
    if cfg.matrix_instr_nonkdim:
        extra["matrix_instr_nonkdim"] = cfg.matrix_instr_nonkdim
    if cfg.kpack:
        extra["kpack"] = cfg.kpack
    if cfg.waves_per_eu is not None:
        extra["waves_per_eu"] = cfg.waves_per_eu

    def _launch():
        mi100_int8_scaled_mm_kernel[grid](
            a, b, sa, sb, out, None,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M=cfg.BLOCK_M,
            BLOCK_SIZE_N=cfg.BLOCK_N,
            BLOCK_SIZE_K=cfg.BLOCK_K,
            BLOCK_SIZE_SCALE_A=block_size_sa,
            BLOCK_SIZE_SCALE_B=block_size_sb,
            GROUP_SIZE_M=group_size_m,
            num_stages=cfg.num_stages,
            **extra,
        )

    try:
        for _ in range(WARMUP_ITERS):
            _launch()
        torch.cuda.synchronize()
        # Time `trials` runs and return the median in ms.
        timings: list[float] = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _launch()
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
        timings.sort()
        return timings[len(timings) // 2], None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------- W4A16 path --------------------------------------

def _bench_w4a16_one_config(
    cfg: SweepConfig, M: int, N: int, K: int, group_size: int, trials: int
) -> tuple[float | None, str | None]:
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_w4a16 import (
        mi100_w4a16_gemm_kernel,
    )
    from vllm.triton_utils import triton

    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    # GPTQ-packed weights: [K, N//8] int32. Random pattern is fine for perf.
    b_packed = torch.randint(
        0, 0x7FFFFFFF, (K, N // 8), dtype=torch.int32, device="cuda"
    )
    scales = torch.rand((K // group_size, N), dtype=torch.float16,
                        device="cuda") * 0.01
    out = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, cfg.BLOCK_M), triton.cdiv(N, cfg.BLOCK_N))

    extra: dict = {}
    if cfg.matrix_instr_nonkdim:
        extra["matrix_instr_nonkdim"] = cfg.matrix_instr_nonkdim
    if cfg.kpack:
        extra["kpack"] = cfg.kpack
    if cfg.waves_per_eu is not None:
        extra["waves_per_eu"] = cfg.waves_per_eu

    block_k = min(cfg.BLOCK_K, group_size)

    def _launch():
        mi100_w4a16_gemm_kernel[grid](
            a, b_packed, scales, b_packed, out,  # zeros_ptr=dummy when no zp
            M, N, K,
            a.stride(0), a.stride(1),
            b_packed.stride(0), b_packed.stride(1),
            out.stride(0), out.stride(1),
            group_size=group_size,
            HAS_ZP=False,
            ZP_BIAS=8,
            BLOCK_M=cfg.BLOCK_M,
            BLOCK_N=cfg.BLOCK_N,
            BLOCK_K=block_k,
            num_stages=cfg.num_stages,
            **extra,
        )

    try:
        for _ in range(WARMUP_ITERS):
            _launch()
        torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _launch()
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
        timings.sort()
        return timings[len(timings) // 2], None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------- Driver ------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _kernel_source_sha(kernel: str) -> str:
    if kernel == "w8a8":
        path = (
            REPO / "vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py"
        )
    elif kernel == "w4a16":
        path = (
            REPO / "vllm/model_executor/kernels/linear/scaled_mm/mi100_w4a16.py"
        )
    else:
        return ""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist_best(
    kernel: str, M: int, N: int, K: int, group_size: int | None,
    best_cfg: SweepConfig, best_ms: float, evaluated: int, pruned: int,
    out_dir: Path,
    cartesian_total: int = 0,
    legal_subset: int = 0,
    eliminated_by_hard_invariant: int = 0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Map the CLI-friendly kernel name to the in-tree config-loader key
    # used by the runtime kernel modules. The W8A8 module loads
    # ``mi100_int8`` (not ``mi100_w8a8``) from configs/gfx908/.
    config_key = "mi100_" + "i" + "nt8" if kernel == "w8a8" else "mi100_" + kernel
    if group_size is not None:
        fname = f"{config_key}_M{M}_N{N}_K{K}_g{group_size}.json"
    else:
        fname = f"{config_key}_M{M}_N{N}_K{K}.json"
    fpath = out_dir / fname
    # Throughput as a derived field (M tokens per ms / iteration).
    measured_tok_s = (M / (best_ms / 1000.0)) if best_ms > 0 else 0.0
    payload = {
        "schema_version": 1,
        "kernel": config_key,
        "shape": {
            "M": M, "N": N, "K": K,
            "group_size": group_size,
        },
        "config": {
            "BLOCK_M": best_cfg.BLOCK_M,
            "BLOCK_N": best_cfg.BLOCK_N,
            "BLOCK_K": best_cfg.BLOCK_K,
            "GROUP_SIZE_M": 8,
            "matrix_instr_nonkdim": best_cfg.matrix_instr_nonkdim,
            "kpack": best_cfg.kpack,
            "waves_per_eu": best_cfg.waves_per_eu,
            "num_warps": 4,
            "num_stages": best_cfg.num_stages,
        },
        "measured_ms_per_iter": best_ms,
        "measured_tok_s": measured_tok_s,
        "autotune_runs_evaluated": evaluated,
        "autotune_runs_pruned": pruned,
        "autotune_cartesian_total": cartesian_total,
        "autotune_legal_subset": legal_subset,
        "autotune_legal_coverage_pct": (
            (evaluated / legal_subset * 100.0) if legal_subset else 0.0
        ),
        "autotune_hard_invariant_eliminated": eliminated_by_hard_invariant,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vllm_commit": _git_sha(),
        "rocm_version": os.environ.get("ROCM_VERSION", "7.12"),
        "kernel_source_sha256": _kernel_source_sha(kernel),
    }
    fpath.write_text(json.dumps(payload, indent=2))
    return fpath


def _shapes_for_kernel(kernel: str, hot_path: Path,
                       extra: list[tuple[int, int, int, int | None]]
                       ) -> list[tuple[int, int, int, int | None]]:
    """Return (M, N, K, group_size_or_None) tuples to tune."""
    if extra:
        return extra
    # Curated shapes for Qwen3.5-9B that matter on TP=1 and TP=4.
    # Hidden=4096, FFN=12288.
    # TP=1 prefill (M ~= 512), TP=4 prefill (M ~= 512), decode (M ~= 1).
    qkv_tp1 = (4096, 10240)   # qkv
    o_tp1 = (4096, 4096)      # o_proj
    gate_up_tp1 = (4096, 24576)  # gate_up
    down_tp1 = (12288, 4096)  # down_proj
    qkv_tp4 = (4096, 2560)    # qkv sharded by 4
    o_tp4 = (1024, 4096)      # o_proj K sharded
    gate_up_tp4 = (4096, 6144)
    down_tp4 = (3072, 4096)

    layers = [qkv_tp1, o_tp1, gate_up_tp1, down_tp1,
              qkv_tp4, o_tp4, gate_up_tp4, down_tp4]
    Ms = [1, 32, 128, 512]    # decode + prefill regimes

    out: list[tuple[int, int, int, int | None]] = []
    for M in Ms:
        for K_, N_ in layers:
            if kernel == "w8a8":
                out.append((M, N_, K_, None))
            else:
                # w4a16: tune both group sizes.
                for g in (32, 128):
                    out.append((M, N_, K_, g))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=("w8a8", "w4a16"), required=True)
    p.add_argument("--shapes-json", default=str(DEFAULT_HOT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--log-dir", default=str(DEFAULT_LOG))
    p.add_argument("--trials-per-config", type=int, default=TIMING_ITERS)
    p.add_argument(
        "--max-configs", type=int, default=0,
        help="Debug: cap cartesian product (0 = full sweep).",
    )
    p.add_argument(
        "--shape", action="append", default=[],
        help='Debug: tune one shape "M,N,K[,group]".',
    )
    p.add_argument(
        "--max-shapes", type=int, default=0,
        help="Debug: cap number of shapes processed (0 = all).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[autotune] %(asctime)s %(levelname)s %(message)s",
    )

    extra_shapes: list[tuple[int, int, int, int | None]] = []
    for raw in args.shape:
        parts = raw.split(",")
        if len(parts) == 3:
            extra_shapes.append(
                (int(parts[0]), int(parts[1]), int(parts[2]), None)
            )
        elif len(parts) == 4:
            extra_shapes.append(
                (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
            )
        else:
            raise SystemExit(f"--shape malformed: {raw}")

    shapes = _shapes_for_kernel(
        args.kernel, Path(args.shapes_json), extra_shapes
    )
    if args.max_shapes > 0:
        shapes = shapes[: args.max_shapes]

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sweep_{args.kernel}.log"
    jsonl_path = log_dir / f"sweep_{args.kernel}.jsonl"
    log_fh = log_path.open("a", buffering=1)
    jsonl_fh = jsonl_path.open("a", buffering=1)

    full_cart = _enumerate_configs()
    logger.info("cartesian product: %d configs", len(full_cart))

    if args.max_configs > 0:
        full_cart = full_cart[: args.max_configs]
        logger.info(
            "[debug] capped cartesian to %d configs",
            len(full_cart),
        )

    summary_rows = []
    for shape in shapes:
        M, N, K, group_size = shape
        prefix = f"{args.kernel} M={M} N={N} K={K}"
        if group_size is not None:
            prefix += f" g={group_size}"
        logger.info("=== %s ===", prefix)
        log_fh.write(f"\n=== {prefix} ===\n")

        evaluated = 0
        pruned = 0
        best_ms: float | None = None
        best_cfg: SweepConfig | None = None
        for cfg in full_cart:
            if args.kernel == "w8a8":
                reason = _prune_w8a8(cfg, M, N, K)
            else:
                reason = _prune_w4a16(cfg, M, N, K, group_size or 128)
            if reason is not None:
                pruned += 1
                log_fh.write(
                    f"PRUNED {cfg.as_launch_kwargs()} reason={reason}\n"
                )
                continue
            if args.kernel == "w8a8":
                ms, err = _bench_w8a8_one_config(
                    cfg, M, N, K, args.trials_per_config
                )
            else:
                ms, err = _bench_w4a16_one_config(
                    cfg, M, N, K,
                    group_size or 128, args.trials_per_config
                )
            evaluated += 1
            row = {
                "shape": [M, N, K, group_size],
                "config": cfg.as_launch_kwargs(),
                "ms": ms,
                "err": err,
            }
            jsonl_fh.write(json.dumps(row) + "\n")
            if err is not None:
                log_fh.write(f"FAIL {cfg.as_launch_kwargs()} err={err}\n")
                continue
            log_fh.write(f"OK {cfg.as_launch_kwargs()} ms={ms:.4f}\n")
            if best_ms is None or (ms is not None and ms < best_ms):
                best_ms = ms
                best_cfg = cfg

        if best_cfg is None or best_ms is None:
            logger.warning("no winning config for %s", prefix)
            log_fh.write(f"NO_WINNER {prefix}\n")
            continue

        cart_total = len(full_cart)
        evaluated_pct = (evaluated / cart_total) * 100.0

        # Compute the "legal subset": configs that survive the hard
        # invariants imposed by the kernel (e.g. BLOCK_K <= group_size
        # for W4A16). Configs eliminated by these invariants are NOT
        # candidates for evaluation under any setting and so should be
        # excluded from the coverage denominator (VAL-TRITON-003 amend).
        # A config that fails only because of shape-specific reasons
        # like ``block_n_gt_N`` is still counted as "legal" because a
        # different shape could legitimately use it.
        if args.kernel == "w8a8":
            hard_invariants = {
                "lds_overflow", "nonkdim32_needs_>=32_tiles",
            }
        else:
            hard_invariants = {
                "block_k_gt_group", "block_n_not_multiple_of_8",
                "lds_overflow", "nonkdim32_needs_>=32_tiles",
            }
        legal_subset = 0
        eliminated_by_hard_invariant = 0
        for cfg2 in full_cart:
            if args.kernel == "w8a8":
                r = _prune_w8a8(cfg2, M, N, K)
            else:
                r = _prune_w4a16(cfg2, M, N, K, group_size or 128)
            if r is None:
                legal_subset += 1
                continue
            # Check whether this prune reason is a hard invariant
            # (would also eliminate the config for any other
            # shape) vs a shape-specific filter.
            is_hard = any(
                r.startswith(inv) for inv in hard_invariants
            )
            if is_hard:
                eliminated_by_hard_invariant += 1
            else:
                legal_subset += 1
        legal_pct = (
            (evaluated / legal_subset) * 100.0 if legal_subset else 0.0
        )
        logger.info(
            "best %s: %s @ %.4f ms",
            prefix, best_cfg.as_launch_kwargs(), best_ms,
        )
        logger.info(
            "  cartesian-coverage: evaluated %d/%d = %.1f%%",
            evaluated, cart_total, evaluated_pct,
        )
        logger.info(
            "  legal-coverage:     evaluated %d/%d = %.1f%% "
            "(eliminated by hard invariants: %d)",
            evaluated, legal_subset, legal_pct,
            eliminated_by_hard_invariant,
        )
        log_fh.write(
            f"BEST {best_cfg.as_launch_kwargs()} ms={best_ms:.4f} "
            f"evaluated={evaluated} cartesian={cart_total} "
            f"legal_subset={legal_subset} legal_coverage_pct={legal_pct:.1f} "
            f"pruned={pruned} hard_invariant_eliminated="
            f"{eliminated_by_hard_invariant}\n"
        )

        out_path = _persist_best(
            args.kernel, M, N, K, group_size,
            best_cfg, best_ms, evaluated, pruned, out_dir,
            cartesian_total=cart_total,
            legal_subset=legal_subset,
            eliminated_by_hard_invariant=eliminated_by_hard_invariant,
        )
        summary_rows.append(
            {
                "shape": list(shape),
                "best_ms": best_ms,
                "best_cfg": asdict(best_cfg),
                "evaluated": evaluated,
                "pruned": pruned,
                "cartesian_total": cart_total,
                "evaluated_pct_of_cart": evaluated_pct,
                "legal_subset": legal_subset,
                "legal_coverage_pct": legal_pct,
                "hard_invariant_eliminated": eliminated_by_hard_invariant,
                "out_path": str(out_path),
            }
        )

    # Aggregate end-of-sweep coverage summary so VAL-TRITON-003 amend
    # is verifiable without re-deriving numbers from per-shape JSONs.
    cart_total_global = len(_enumerate_configs())
    if summary_rows:
        legal_avg = sum(r["legal_subset"] for r in summary_rows) / len(
            summary_rows
        )
        cov_avg = sum(r["legal_coverage_pct"] for r in summary_rows) / len(
            summary_rows
        )
    else:
        legal_avg = 0.0
        cov_avg = 0.0
    coverage_summary = {
        "cartesian_total": cart_total_global,
        "shapes_processed": len(summary_rows),
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
            for r in summary_rows
        ],
    }

    summary_path = log_dir / f"sweep_{args.kernel}_summary.json"
    summary_path.write_text(json.dumps({
        "kernel": args.kernel,
        "shapes": summary_rows,
        "cartesian_total": cart_total_global,
        "coverage_summary": coverage_summary,
        "trials_per_config": args.trials_per_config,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2))

    logger.info("=" * 70)
    logger.info("Sweep coverage summary (kernel=%s):", args.kernel)
    logger.info("  cartesian total          = %d configs", cart_total_global)
    for r in summary_rows:
        shape_str = "x".join(str(x) for x in r["shape"] if x is not None)
        logger.info(
            "  shape %-26s legal subset = %d configs; "
            "evaluated = %d; legal-coverage = %d/%d = %.1f%%",
            shape_str,
            r["legal_subset"],
            r["evaluated"],
            r["evaluated"], r["legal_subset"], r["legal_coverage_pct"],
        )
    logger.info(
        "  average legal-coverage   = %.1f%% across %d shapes",
        cov_avg, len(summary_rows),
    )
    logger.info("=" * 70)
    logger.info("summary written to %s", summary_path)

    log_fh.close()
    jsonl_fh.close()
    return 0 if summary_rows else 4


if __name__ == "__main__":
    raise SystemExit(main())
