# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-shape autotune config loader for gfx908 (MI100) Triton kernels.

Workers in milestone M3 persist per-shape autotune results as JSON files in
this directory using the naming convention:

    mi100_int8_M<M>_N<N>_K<K>.json
    mi100_int8_M<M>_N<N>_K<K>_e1.json            # EMIT_INT8_NEXT=True variant
    mi100_w4a16_M<M>_N<N>_K<K>_g<group_size>.json
    fused_silu_quant_int8_M<M>_H<H>.json         # M1 fused silu+quant
    fused_int8_quant_M<M>_H<H>.json              # candidate per-token quant

Each JSON has the schema:
    {
        "kernel": "mi100_int8" | "mi100_w4a16" | "fused_silu_quant_int8"
                  | "fused_int8_quant",
        "shape": {"M": int, "N": int, "K": int, "group_size": int|null,
                  "emit_int8_next": bool|null, "H": int|null},
        "config": {
            "BLOCK_M": int, "BLOCK_N": int, "BLOCK_K": int,
            "GROUP_SIZE_M": int,
            "matrix_instr_nonkdim": int,
            "kpack": int,
            "waves_per_eu": int,
            "num_warps": int,
            "num_stages": int
        },
        "measured_tok_s": float,
        "autotune_runs_evaluated": int,
        "autotune_runs_pruned": int,
        "timestamp": iso8601 str,
        "vllm_commit": str|null,
        "rocm_version": str|null
    }

Lookup priority:
    1. Exact (M, N, K[, group_size][, emit_int8_next]) match.
    2. Fallback to a "default" config keyed on M-bucket so callers always
       get something runnable.

The loader is tolerant of missing files — when no config is found for a
shape, ``None`` is returned and callers fall back to their hard-coded
heuristic (preserving the existing pre-M3 behaviour).
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def _config_path(
    kernel: str,
    M: int,
    N: int,
    K: int,
    group_size: int | None = None,
    emit_int8_next: bool = False,
) -> Path:
    if group_size is not None:
        fname = f"{kernel}_M{M}_N{N}_K{K}_g{group_size}.json"
    elif emit_int8_next:
        fname = f"{kernel}_M{M}_N{N}_K{K}_e1.json"
    else:
        fname = f"{kernel}_M{M}_N{N}_K{K}.json"
    return CONFIG_DIR / fname


def _fused_act_config_path(kernel: str, M: int, H: int) -> Path:
    """Filename convention for the M1 fused activation/quant kernels.

    These kernels do not carry an explicit ``K`` dim — only ``M`` (tokens)
    and ``H`` (hidden / per-row reduction width). ``fused_silu_quant_int8``
    additionally has the silu_and_mul ``[M, 2*H]`` input layout; we key on
    the per-row ``H`` (=``input.shape[-1] // 2``) so the filename matches
    the kernel's natural shape parameter.
    """
    return CONFIG_DIR / f"{kernel}_M{M}_H{H}.json"


@lru_cache(maxsize=512)
def _load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[mi100_config] failed to parse %s: %s", p, exc)
        return None


def load_config(
    kernel: str,
    M: int,
    N: int,
    K: int,
    group_size: int | None = None,
    emit_int8_next: bool = False,
) -> dict | None:
    """Return the autotune-selected ``config`` dict for the given shape, or
    ``None`` if no config has been persisted yet.

    The returned dict is the inner ``config`` field of the JSON schema:
    callers can splat it directly into a Triton launch as kwargs.

    ``emit_int8_next=True`` looks up the ``_e1`` variant filename, which
    is the EMIT_INT8_NEXT=True specialization of the W8A8 GEMM (issue #26
    / M2 wire-in). The variant filename is checked first; if the variant
    has not been pinned for this shape, the base config (if any) is
    returned so the caller can still benefit from the surveyed GEMM tile
    sizes (the wrapper applies an additional LDS-budget clamp on top).
    """
    if os.environ.get("VLLM_MI100_DISABLE_AUTOTUNE_CONFIG") == "1":
        return None
    if emit_int8_next:
        # Variant lookup first; fall through to the base config when the
        # EMIT_INT8_NEXT=True specialization has not been pinned yet.
        variant_path = _config_path(kernel, M, N, K, group_size, emit_int8_next=True)
        blob = _load_json(str(variant_path))
        if blob is not None:
            cfg = blob.get("config")
            if isinstance(cfg, dict):
                return cfg
    path = _config_path(kernel, M, N, K, group_size, emit_int8_next=False)
    blob = _load_json(str(path))
    if blob is None:
        return None
    cfg = blob.get("config")
    if not isinstance(cfg, dict):
        return None
    return cfg


def load_fused_act_config(
    kernel: str,
    M: int,
    H: int,
) -> dict | None:
    """Return the autotune-selected ``config`` dict for an M1 fused
    activation/quant kernel (``fused_silu_quant_int8`` /
    ``fused_int8_quant``).

    Returns ``None`` when no JSON has been pinned for the shape; the
    caller then falls back to its static heuristic.

    Set ``VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1`` to disable the loader
    globally (same escape hatch as :func:`load_config`).
    """
    if os.environ.get("VLLM_MI100_DISABLE_AUTOTUNE_CONFIG") == "1":
        return None
    path = _fused_act_config_path(kernel, M, H)
    blob = _load_json(str(path))
    if blob is None:
        return None
    cfg = blob.get("config")
    if not isinstance(cfg, dict):
        return None
    return cfg


def list_configs(kernel: str | None = None) -> list[dict]:
    """List every config JSON in this directory (used by tests & tooling)."""
    out: list[dict] = []
    for p in sorted(CONFIG_DIR.glob("*.json")):
        if p.name == "hipblaslt_tuned_shapes.json":
            continue  # M2 manifest, not an autotune config
        if kernel is not None and not p.name.startswith(kernel + "_"):
            continue
        blob = _load_json(str(p))
        if blob is not None:
            out.append(blob)
    return out


def reset_cache() -> None:
    """Test helper: clear the lru_cache so newly written configs pick up."""
    _load_json.cache_clear()


__all__ = [
    "load_config",
    "load_fused_act_config",
    "list_configs",
    "reset_cache",
    "CONFIG_DIR",
    "SCHEMA_VERSION",
]
