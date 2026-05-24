# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 backend dispatcher for MI100 (gfx908).

Selects one of {ck, hipblaslt, triton} for a given (M, N, K, tp_rank). The
spec'd order is **CK > hipBLASLt > Triton**, where:
- CK is preferred whenever an instance is registered for the shape, since
  it usually owns the lowest latency on prefill-class shapes that match
  the M4 hot-shape catalog.
- hipBLASLt is the next-best option for shapes covered by the M2
  TensileLite tuning JSON.
- Triton (``mi100_int8.py``) is the universal fallback.

The dispatcher is intentionally small and pure-Python so it can be
unit-tested without launching a kernel; the corresponding wiring inside
``mi100_int8.mi100_int8_scaled_mm`` calls ``choose_backend`` and routes
to the matching path.

When ``VLLM_LOG_GEMM_BACKEND=1`` is set, every dispatch is appended to a
ring buffer for offline inspection (``get_recent_dispatch_log``).
"""

from __future__ import annotations

import collections
import logging
import os
import threading
from typing import Literal

logger = logging.getLogger(__name__)

Backend = Literal["ck", "hipblaslt", "triton"]

_DISPATCH_LOG: collections.deque[tuple[int, int, int, int, str]] = collections.deque(
    maxlen=1024
)
_DISPATCH_LOG_LOCK = threading.Lock()


def _ck_supports(M: int, N: int, K: int, tp_rank: int) -> bool:
    """Probe ``torch.ops._rocm_C.ck_int8_gemm_supports`` if available."""
    try:
        import torch  # local import to keep module import cheap

        op = getattr(torch.ops._rocm_C, "ck_int8_gemm_supports", None)
        if op is None:
            return False
        return bool(op(M, N, K, tp_rank))
    except Exception:  # pragma: no cover - defensive: ROCm absent
        return False


def _hipblaslt_supports(M: int, N: int, K: int) -> bool:
    """Probe the M2 hipBLASLt tuned-shape manifest."""
    try:
        from .mi100_hipblaslt import mi100_hipblaslt_supports
    except Exception:  # pragma: no cover - hipBLASLt path absent
        return False
    try:
        return bool(mi100_hipblaslt_supports(M, N, K))
    except Exception:
        return False


def choose_backend(M: int, N: int, K: int, tp_rank: int = 1) -> Backend:
    """Return the backend that should service this GEMM.

    Priority: CK > hipBLASLt > Triton. The ``VLLM_DISABLE_CK`` and
    ``VLLM_DISABLE_HIPBLASLT`` env vars short-circuit each layer for
    debugging or reproducibility experiments.
    """
    backend: Backend
    if os.environ.get("VLLM_DISABLE_CK", "0") != "1" and _ck_supports(M, N, K, tp_rank):
        backend = "ck"
    elif os.environ.get("VLLM_DISABLE_HIPBLASLT", "0") != "1" and _hipblaslt_supports(
        M, N, K
    ):
        backend = "hipblaslt"
    else:
        backend = "triton"

    if os.environ.get("VLLM_LOG_GEMM_BACKEND") == "1":
        with _DISPATCH_LOG_LOCK:
            _DISPATCH_LOG.append((M, N, K, tp_rank, backend))
        # Emit one stderr/log line per dispatch so server logs are
        # grep-able for integration verification (e.g.
        # ``grep 'choose_backend: ck' server.log``). The ring buffer
        # remains for in-process introspection.
        logger.info(
            "choose_backend: %s (M=%d,N=%d,K=%d,tp_rank=%d)",
            backend,
            M,
            N,
            K,
            tp_rank,
        )
    return backend


def get_recent_dispatch_log() -> list[tuple[int, int, int, int, str]]:
    """Return the most recent dispatches recorded under
    ``VLLM_LOG_GEMM_BACKEND=1``."""
    with _DISPATCH_LOG_LOCK:
        return list(_DISPATCH_LOG)


def clear_dispatch_log() -> None:
    with _DISPATCH_LOG_LOCK:
        _DISPATCH_LOG.clear()


# ---------------------------------------------------------------------------
# M2 wire-in: per-call EMIT_INT8_NEXT decision (issue #26)
# ---------------------------------------------------------------------------
#
# The W8A8 Triton kernel ``mi100_int8_scaled_mm_kernel`` exposes a
# kernel-level ``EMIT_INT8_NEXT`` constexpr (commit a64742222) that fuses
# the per-token int8 quantization of the GEMM output into the store
# epilogue. Activating it eliminates one HBM round-trip when the next op
# in the model graph is a known fusable int8 consumer (a subsequent W8A8
# linear, or the M1-wired silu/mul-quant path).
#
# The decision is intentionally a pure function so it can be unit-tested
# without launching a kernel. The wrapper ``mi100_int8_scaled_mm``
# additionally enforces the BLOCK_SIZE_N >= N correctness guard
# (commit 3f397bd95 + wrapper fix) so that an ``emit_int8_next=True``
# call cannot silently corrupt output even if the dispatcher mispredicts.
#
# Activation rule — ALL conditions must hold:
#   (a) ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT`` is unset / not truthy AND
#       legacy alias ``VLLM_DISABLE_FUSED_ACT_QUANT`` is also not truthy;
#   (b) device is gfx908 (``current_platform`` reports MI100);
#   (c) the layer's prefix matches a known fusable producer→consumer
#       pattern (e.g. ``qkv_proj.next == o_proj``,
#       ``gate_up_proj.next == down_proj``);
#   (d) ``N`` is small enough that forcing ``BLOCK_SIZE_N = next_pow2(N)``
#       stays under Triton's 2**20 per-tile numel cap (cap N at 32768 — a
#       comfortable headroom over Qwen3.5-9B's largest N of 18944).
#
# When any condition is false we return False (today's behavior) so the
# disable flag, non-fusable layers, and pathological shapes silently fall
# through to the legacy fp16-store path.

# Cap on N for forcing BLOCK_SIZE_N=N. Above this, the kernel's per-tile
# numel would risk exceeding Triton's 2**20 cap at common BLOCK_SIZE_M
# values; route to emit_int8_next=False in that regime.
EMIT_INT8_NEXT_MAX_N: int = 32768

# Layer-name (suffix) whitelist for the fusable producer → next-op map.
# The key is the layer prefix substring identifying the *producer*
# W8A8 linear; the value is the expected *consumer* substring in the
# same module. Maintained as a small, easily-auditable table so the M3
# perplexity / coding evals can attribute any regression to a specific
# wire-in cell.
FUSABLE_LAYER_NEXT_OP: dict[str, str] = {
    # Attention: QKV-proj → O-proj sits across the attention op, NOT a
    # direct chain through int8. Listed here because issue #26 explicitly
    # calls it out and the dispatcher allows the *decision* to fire even
    # if the kernel-level consumer plumbing is added later.
    "qkv_proj": "o_proj",
    # MLP: gate_up_proj → silu_and_mul → down_proj. The M1 fused path
    # (issue #33) is the int8 consumer between gate_up_proj and down_proj.
    "gate_up_proj": "down_proj",
}


def _fused_act_quant_env_disabled() -> bool:
    """Return True iff the user disabled fused-act-quant via env."""

    def _truthy(val: str | None) -> bool:
        if val is None:
            return False
        return val.strip().lower() in {"1", "true", "yes", "on"}

    return _truthy(os.environ.get("VLLM_MI100_DISABLE_FUSED_ACT_QUANT")) or _truthy(
        os.environ.get("VLLM_DISABLE_FUSED_ACT_QUANT")
    )


def _on_gfx908() -> bool:
    """Return True iff the current platform reports an MI100 (gfx908) GPU.

    Kept defensive so the unit tests can mock this out on non-ROCm hosts.
    """
    try:
        from vllm.platforms.rocm import on_mi100

        return bool(on_mi100())
    except Exception:  # pragma: no cover - non-ROCm import path
        return False


def _layer_name_is_fusable(layer_name: str | None) -> bool:
    """Return True iff ``layer_name`` matches a known fusable producer.

    The producer naming convention follows vLLM's standard
    ``<model>.layers.<idx>.<submodule>.<linear>`` prefix scheme. We match
    on a suffix substring against :data:`FUSABLE_LAYER_NEXT_OP` so the
    decision survives both ``model.`` and ``language_model.model.``
    prefix variants.
    """
    if not layer_name:
        return False
    for producer in FUSABLE_LAYER_NEXT_OP:
        # Match ``.<producer>`` or ``<producer>`` as a path component so
        # ``qkv_proj_extra`` does not accidentally activate.
        if layer_name.endswith(producer) or layer_name.endswith(f".{producer}"):
            return True
    return False


def choose_emit_int8_next(
    M: int,
    N: int,
    K: int,
    *,
    layer_name: str | None = None,
    on_gfx908: bool | None = None,
    env_disabled: bool | None = None,
) -> bool:
    """Decide whether a W8A8 GEMM call should activate ``EMIT_INT8_NEXT``.

    Returns True iff every condition in the activation rule above holds.
    The ``on_gfx908`` and ``env_disabled`` keyword arguments are present
    for the unit tests so they can override the runtime probes.

    The function emits a single ``DEBUG``-level log line per call so a
    server run at ``VLLM_LOGGING_LEVEL=DEBUG`` records the per-layer
    decision for offline inspection (mission step 5 of m2-wire-in).
    """
    if env_disabled is None:
        env_disabled = _fused_act_quant_env_disabled()
    if on_gfx908 is None:
        on_gfx908 = _on_gfx908()

    # Track why the decision came out False so the debug log is useful.
    if env_disabled:
        decision, reason = False, "env-disabled"
    elif not on_gfx908:
        decision, reason = False, "not gfx908"
    elif int(N) > EMIT_INT8_NEXT_MAX_N:
        decision, reason = False, f"N={N} exceeds cap {EMIT_INT8_NEXT_MAX_N}"
    elif not _layer_name_is_fusable(layer_name):
        decision, reason = False, f"layer={layer_name!r} not in fusable whitelist"
    else:
        decision, reason = True, f"fusable producer layer={layer_name!r}"

    # Single-line debug log so server.log records per-layer fusion choices
    # under ``VLLM_LOGGING_LEVEL=DEBUG``. The line format is grep-friendly:
    # callers / tests rely on the literal substrings
    # ``emit_int8_next=True`` and ``emit_int8_next=False``.
    logger.debug(
        "emit_int8_next=%s dispatch on layer=%s shape=(M=%d,N=%d,K=%d) reason=%s",
        decision,
        layer_name,
        M,
        N,
        K,
        reason,
    )
    return decision
