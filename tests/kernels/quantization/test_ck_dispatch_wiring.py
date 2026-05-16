# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M4 follow-up: regression test for the CK dispatcher *wiring* into the
production W8A8 entry point.

Scrutiny of the original M4 landing surfaced that
``mi100_int8_dispatch.choose_backend`` lived in isolation — the production
``mi100_int8_scaled_mm`` never imported nor consulted it, so CK was
silently bypassed during real inference. This test guards against that
regression by:

1. Monkey-patching the CK custom op (``torch.ops._rocm_C.ck_int8_gemm``)
   with a counter that records each invocation and returns a dummy fp16
   tensor of the expected shape.
2. Stubbing ``ck_int8_gemm_supports`` to report support for one specific
   shape (so the test does not depend on which CK instances were
   compiled into the build).
3. Calling ``mi100_int8_scaled_mm`` on the supported CK shape and
   asserting that the counter incremented exactly once.
4. Re-running the same call under ``VLLM_DISABLE_CK=1`` and asserting
   the CK counter stayed at zero (i.e. control fell through to the
   hipBLASLt / Triton path).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("vllm.model_executor.kernels.linear.scaled_mm.mi100_int8")

from vllm.model_executor.kernels.linear.scaled_mm import (  # noqa: E402
    mi100_int8,
    mi100_int8_dispatch,
)

# CK-registered shape used throughout the wiring test. Matches one of the
# instances declared in csrc/quantization/w8a8/int8/ck/instances/. We
# stub support for it so the test does not require the C++ extension to
# be built — the goal is to exercise the *Python* dispatcher wiring.
CK_SHAPE = (64, 24576, 4096)  # (M, N, K)


def _make_int8_inputs(M: int, N: int, K: int):
    """Build (input, weight, scale_a, scale_b) shaped exactly the way
    ``mi100_int8_scaled_mm`` expects them. Tensors live on CPU when no
    GPU is available; the spies short-circuit kernel launches before
    any real device dispatch occurs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_q = torch.zeros((M, K), dtype=torch.int8, device=device)
    # weight in mi100_int8_scaled_mm is [K, N] (post process_weights_after_loading)
    weight_q = torch.zeros((K, N), dtype=torch.int8, device=device)
    scale_a = torch.full((M, 1), 0.01, dtype=torch.float32, device=device)
    scale_b = torch.full((N, 1), 0.02, dtype=torch.float32, device=device)
    return input_q, weight_q, scale_a, scale_b


@pytest.fixture
def fake_ck_ops(monkeypatch):
    """Replace ``torch.ops._rocm_C`` with a stand-in exposing
    ``ck_int8_gemm`` (counter spy) and ``ck_int8_gemm_supports`` (returns
    True only for ``CK_SHAPE`` at tp_rank=0).

    Yields the counter list so tests can assert call counts.
    """
    counter: list[tuple[int, int, int, int]] = []
    M_target, N_target, K_target = CK_SHAPE

    def fake_ck_int8_gemm(a, b, scale_a, scale_b, bias, tp_rank):
        # Record (M, N, K, tp_rank) for each invocation. The shapes come
        # from ``a`` (M, K) and ``b`` ([N, K] per the CK ABI).
        M = int(a.shape[0])
        K = int(a.shape[1])
        N = int(b.shape[0])
        counter.append((M, N, K, int(tp_rank)))
        # Return a dummy fp16 tensor matching the CK output shape so the
        # caller's downstream `.to(out_dtype)` cast works.
        return torch.zeros((M, N), dtype=torch.float16, device=a.device)

    def fake_ck_supports(M, N, K, tp_rank):
        # tp_rank=1 matches the world-size convention used to register
        # the M4 CK instances (see ``_get_tp_rank`` in mi100_int8.py).
        return (int(M), int(N), int(K)) == (M_target, N_target, K_target) and int(
            tp_rank
        ) == 1

    fake_namespace = SimpleNamespace(
        ck_int8_gemm=fake_ck_int8_gemm,
        ck_int8_gemm_supports=fake_ck_supports,
    )
    # Patch ``torch.ops._rocm_C`` for the duration of the test.
    # ``raising=False`` so the patch installs even when ``_rocm_C`` is not
    # built into this torch image (the wiring under test only does
    # ``getattr`` lookups against ``torch.ops._rocm_C``).
    monkeypatch.setattr(torch.ops, "_rocm_C", fake_namespace, raising=False)
    # Make sure no stale env override hides the CK branch and that
    # tp_rank lookup deterministically returns 0.
    monkeypatch.delenv("VLLM_DISABLE_CK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    # Force ``_get_tp_rank`` to deterministically return 1 (single-process
    # convention used by the CK instance registry).
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield counter


def test_mi100_int8_scaled_mm_invokes_ck_for_registered_shape(fake_ck_ops):
    """Single invocation on a CK-registered shape must hit the CK op
    exactly once. Catches the bug where ``mi100_int8_scaled_mm`` never
    imported ``choose_backend`` and silently bypassed CK in production.
    """
    M, N, K = CK_SHAPE
    a, w, sa, sb = _make_int8_inputs(M, N, K)

    out = mi100_int8.mi100_int8_scaled_mm(
        a, w, sa, sb, out_dtype=torch.float16, bias=None
    )

    assert len(fake_ck_ops) == 1, (
        f"Expected ck_int8_gemm to be called exactly once, "
        f"got {len(fake_ck_ops)} calls: {fake_ck_ops}"
    )
    seen_M, seen_N, seen_K, seen_tp = fake_ck_ops[0]
    assert (seen_M, seen_N, seen_K) == (M, N, K), (
        f"CK was called with wrong (M, N, K): "
        f"got ({seen_M}, {seen_N}, {seen_K}), expected {CK_SHAPE}"
    )
    assert seen_tp == 1
    assert out.shape == (M, N)
    assert out.dtype == torch.float16


def test_vllm_disable_ck_routes_around_ck(fake_ck_ops):
    """``VLLM_DISABLE_CK=1`` must bypass the CK branch entirely. The
    counter stays at zero because control flows through the hipBLASLt
    fallback (which we stub) instead of CK.
    """
    M, N, K = CK_SHAPE
    a, w, sa, sb = _make_int8_inputs(M, N, K)

    fallback_calls = {"hipblaslt": 0}

    def fake_hipblaslt_supports(*args, **kwargs):
        # Force the dispatch to take the hipBLASLt fallback so we never
        # reach the Triton kernel (which can't run on CPU in CI).
        return True

    def fake_hipblaslt_scaled_mm(*args, **kwargs):
        fallback_calls["hipblaslt"] += 1
        return torch.zeros((M, N), dtype=torch.float16, device=a.device)

    with (
        mock.patch.dict(os.environ, {"VLLM_DISABLE_CK": "1"}, clear=False),
        mock.patch.object(
            mi100_int8, "mi100_hipblaslt_supports", fake_hipblaslt_supports
        ),
        mock.patch.object(
            mi100_int8, "mi100_hipblaslt_scaled_mm", fake_hipblaslt_scaled_mm
        ),
    ):
        out = mi100_int8.mi100_int8_scaled_mm(
            a, w, sa, sb, out_dtype=torch.float16, bias=None
        )

    assert len(fake_ck_ops) == 0, (
        f"VLLM_DISABLE_CK=1 must not hit ck_int8_gemm; got "
        f"{len(fake_ck_ops)} calls: {fake_ck_ops}"
    )
    assert fallback_calls["hipblaslt"] == 1, (
        "Expected the hipBLASLt fallback to be invoked exactly once when "
        f"CK is disabled, got {fallback_calls}"
    )
    assert out.shape == (M, N)


def test_choose_backend_symbol_is_imported_from_dispatcher_module():
    """Sanity check that ``mi100_int8`` exposes the very same
    ``choose_backend`` callable as ``mi100_int8_dispatch``. This catches
    the regression where the production module is missing the import
    (and therefore silently never consults the dispatcher).
    """
    assert hasattr(mi100_int8, "choose_backend"), (
        "mi100_int8 must import choose_backend so production code can "
        "consult the CK > hipBLASLt > Triton selector."
    )
    assert mi100_int8.choose_backend is mi100_int8_dispatch.choose_backend
