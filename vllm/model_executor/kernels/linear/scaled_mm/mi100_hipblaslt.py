# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Milestone-2 hipBLASLt INT8 W8A8 path for gfx908 (MI100).

The W8A8 dispatcher in this repo originally always routes through the
Triton ``mi100_int8_scaled_mm`` kernel. M2 introduces an alternative:
for a curated set of (M, N, K) shapes — those tuned by the TensileLite
campaign at ``vllm/model_executor/kernels/configs/gfx908/hipblaslt_tuned_shapes.json``
— we dispatch through ``torch._int_mm`` (which is a thin PyTorch wrapper
around hipBLASLt's INT8 GEMM on ROCm) and apply per-token / per-channel
scales in a simple fp32 epilogue.

The TensileLite tuning pipeline emits per-shape logic YAMLs at
``$HIPBLASLT_TENSILE_LIBPATH``; hipBLASLt loads them at process start
and selects the tuned ``Cijk_Ailk_Bljk_I8_MT*`` kernel for matching
problem sizes when ``torch._int_mm`` is invoked.

This module is intentionally small: the per-shape decision lives in
``mi100_hipblaslt_supports`` and the actual call in
``mi100_hipblaslt_scaled_mm``. The Triton kernel remains the fallback
for everything else (and for the ``VLLM_DISABLE_HIPBLASLT=1`` override).
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Default location of the tuned-shapes manifest. May be overridden at
# runtime via VLLM_HIPBLASLT_TUNED_SHAPES for tests.
_DEFAULT_TUNED_SHAPES_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "gfx908"
    / "hipblaslt_tuned_shapes.json"
)


@lru_cache(maxsize=1)
def _load_tuned_shapes_set() -> tuple[frozenset[tuple[int, int, int]], dict]:
    """Return the (M, N, K) tuples + raw manifest from the tuning JSON.

    Returns an empty set when the manifest is missing so the dispatcher
    cleanly falls back to the Triton path with no tuning evidence yet.
    """
    path_str = os.environ.get("VLLM_HIPBLASLT_TUNED_SHAPES")
    path = Path(path_str) if path_str else _DEFAULT_TUNED_SHAPES_PATH
    if not path.exists():
        logger.debug(
            "[MI100_HIPBLASLT] tuned-shapes manifest missing at %s; "
            "all shapes route to Triton fallback",
            path,
        )
        return frozenset(), {"shapes": [], "version": None}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[MI100_HIPBLASLT] failed to parse %s: %s; routing to Triton",
            path,
            exc,
        )
        return frozenset(), {"shapes": [], "version": None}
    shapes = manifest.get("shapes", [])
    out: set[tuple[int, int, int]] = set()
    for entry in shapes:
        if not isinstance(entry, dict):
            continue
        try:
            M = int(entry["M"])
            N = int(entry["N"])
            K = int(entry["K"])
        except (KeyError, TypeError, ValueError):
            continue
        out.add((M, N, K))
    logger.debug(
        "[MI100_HIPBLASLT] loaded %d tuned shapes from %s",
        len(out),
        path,
    )
    return frozenset(out), manifest


def is_hipblaslt_disabled() -> bool:
    """``VLLM_DISABLE_HIPBLASLT=1`` forces 100% Triton fallback."""
    return os.environ.get("VLLM_DISABLE_HIPBLASLT", "0") == "1"


def mi100_hipblaslt_supports(M: int, N: int, K: int) -> bool:
    """Return True if (M, N, K) is in the tuned-shapes manifest and the
    hipBLASLt path is not disabled.
    """
    if is_hipblaslt_disabled():
        return False
    tuned, _ = _load_tuned_shapes_set()
    return (M, N, K) in tuned


def reset_tuned_shape_cache() -> None:
    """Test helper: drop the lru_cache so a new manifest is picked up."""
    _load_tuned_shapes_set.cache_clear()


def mi100_hipblaslt_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """hipBLASLt-routed INT8 W8A8 scaled matmul.

    Shapes match ``mi100_int8_scaled_mm``:
        input:  [M, K] int8  (row-major contiguous, layout='K')
        weight: [K, N] int8  (col-major contiguous, layout='K')
        scale_a: [1,1] or [M,1] floating
        scale_b: [1,1] or [N,1] floating

    Implementation:
        1. ``torch._int_mm(input, weight) -> [M, N] int32``. On ROCm this
           is dispatched through hipBLASLt; with HIPBLASLT_TENSILE_LIBPATH
           pointing at our tuned logic dir, hipBLASLt selects a
           ``Cijk_Ailk_Bljk_I8_MT*`` kernel for the tuned (M, N, K).
        2. Apply per-token (M-axis) and per-channel (N-axis) scales in
           fp32 to preserve numerical accuracy comparable to the Triton
           epilogue.
        3. Cast to ``out_dtype`` and optionally add bias.
    """
    M, K = input.shape
    N = weight.shape[1]

    # torch._int_mm needs both operands int8 and contiguous on the inner dim.
    # Our tensors are produced contiguous in mi100_int8_scaled_mm caller; double
    # check the K-stride invariant that hipBLASLt expects.
    a = input.contiguous() if not input.is_contiguous() else input
    b = weight.contiguous() if not weight.is_contiguous() else weight

    # int8 x int8 -> int32 (hipBLASLt under the hood on ROCm).
    acc = torch._int_mm(a, b)

    # Scales in fp32 for fidelity, then cast.
    sa = scale_a.to(torch.float32).reshape(-1, 1)
    sb = scale_b.to(torch.float32).reshape(1, -1)
    if sa.shape[0] == 1:
        sa = sa.expand(M, 1)
    if sb.shape[1] == 1:
        sb = sb.expand(1, N)
    out_f32 = acc.to(torch.float32) * sa * sb

    out = out_f32.to(out_dtype)
    if bias is not None:
        out = out + bias.to(out_dtype)
    return out
