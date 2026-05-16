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
