# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamo trace-break regression test for ``try_stash_fused_silu_quant_int8``.

Covers the fix for the confirmed gfx900 ``VLLM_COMPILE`` defect: Dynamo
cannot fake-tensor-propagate ``torch.cuda.is_current_stream_capturing()``
(it returns a plain Python bool from a C++ extension binding), so any call
reached while Dynamo is tracing a ``torch.compile``'d region raises
``torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor``. This hit
Qwen3.5's compiled MoE MLP forward (``qwen2_moe.py``'s ``Qwen2MoeMLP``,
wrapped by ``@support_torch_compile``), which calls
``try_stash_fused_silu_quant_int8`` unconditionally between the
``gate_up_proj`` GEMM and ``act_fn`` / ``down_proj``.

The fix adds a ``torch.compiler.is_compiling()`` guard *before* the
``is_current_stream_capturing()`` call so the producer becomes a no-op
(falls through to the legacy ``act_fn`` + ``down_proj`` composition) for the
duration of Dynamo tracing/compilation, while remaining fully active at
eager runtime (including the existing CUDA-graph-capture skip it must not
weaken).

This test is intentionally hardware-free: it never touches a Triton kernel
or a real CUDA stream. ``try_stash_fused_silu_quant_int8`` only reads
``gate_up.dim()`` / ``.dtype`` / ``.shape`` / ``.is_cuda`` before reaching
the guard under test, so a small duck-typed stand-in is used in place of a
real CUDA tensor, and ``torch.compiler.is_compiling`` /
``torch.cuda.is_current_stream_capturing`` are monkeypatched directly (no
GPU required to observe whether either was invoked).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.quantization import fused_silu_quant_int8 as m
from vllm.model_executor.kernels.quantization.fused_silu_quant_int8 import (
    try_stash_fused_silu_quant_int8,
)


class _FakeGateUp:
    """Duck-types the subset of ``torch.Tensor`` that
    ``try_stash_fused_silu_quant_int8`` reads before it would reach the
    ``is_current_stream_capturing`` / kernel-dispatch branches: a 2-D fp16
    tensor with an even last dim, reporting ``is_cuda = True``. Using a
    plain stand-in (instead of a real CUDA tensor) is what lets these tests
    run without a GPU.
    """

    dtype = torch.float16
    shape = (4, 16)
    is_cuda = True

    def dim(self) -> int:
        return 2


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts from the default (fused-on) env state."""
    monkeypatch.delenv("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", raising=False)
    monkeypatch.delenv("VLLM_DISABLE_FUSED_ACT_QUANT", raising=False)


def test_dynamo_compiling_short_circuits_before_capture_check() -> None:
    """``torch.compiler.is_compiling() == True`` -> no-op, and the call that
    breaks Dynamo tracing (``is_current_stream_capturing``) is never made.

    This is the core regression assertion: previously the producer called
    ``torch.cuda.is_current_stream_capturing()`` unconditionally, which
    Dynamo cannot trace through a bool-returning torch op. The fix must
    return ``False`` (fall through to the legacy path) *without* ever
    reaching that call while compiling.
    """
    gate_up = _FakeGateUp()
    down_proj = object()  # no `.scheme` — would also fail the W8A8 check

    with (
        patch.object(
            torch.compiler, "is_compiling", return_value=True
        ) as mock_compiling,
        patch.object(
            torch.cuda,
            "is_current_stream_capturing",
            side_effect=AssertionError(
                "is_current_stream_capturing() must not be called while "
                "torch.compiler.is_compiling() is True"
            ),
        ) as mock_capturing,
        patch.object(
            m,
            "fused_silu_quant_int8",
            side_effect=AssertionError(
                "fused_silu_quant_int8() must not run while Dynamo is tracing"
            ),
        ) as mock_kernel,
    ):
        result = try_stash_fused_silu_quant_int8(gate_up, down_proj)

    assert result is False, (
        "try_stash_fused_silu_quant_int8 must return False (no-op) while "
        "torch.compiler.is_compiling() is True."
    )
    mock_compiling.assert_called_once()
    mock_capturing.assert_not_called()
    mock_kernel.assert_not_called()
    assert not hasattr(down_proj, "_mi100_fused_silu_cache")


def test_eager_mode_still_checks_cuda_graph_capture() -> None:
    """Not compiling -> the pre-existing CUDA-graph-capture skip still runs
    unchanged (the new guard must not swallow or replace it).
    """
    gate_up = _FakeGateUp()
    down_proj = object()

    with (
        patch.object(torch.compiler, "is_compiling", return_value=False),
        patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ) as mock_capturing,
        patch.object(m, "fused_silu_quant_int8") as mock_kernel,
    ):
        result = try_stash_fused_silu_quant_int8(gate_up, down_proj)

    assert result is False, (
        "try_stash_fused_silu_quant_int8 must still return False during an "
        "active CUDA-graph capture (pre-existing behavior)."
    )
    mock_capturing.assert_called_once()
    mock_kernel.assert_not_called()


def test_eager_mode_not_capturing_reaches_downstream_dispatch_check() -> None:
    """Not compiling, not capturing -> execution reaches the normal
    ``down_proj`` kernel-type dispatch check (proving the new guard is a
    pure short-circuit and does not alter the rest of the control flow).
    """
    gate_up = _FakeGateUp()
    down_proj = object()  # no `.scheme` -> _down_proj_is_mi100_w8a8_int8 -> False

    with (
        patch.object(torch.compiler, "is_compiling", return_value=False),
        patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ) as mock_capturing,
        patch.object(m, "fused_silu_quant_int8") as mock_kernel,
    ):
        result = try_stash_fused_silu_quant_int8(gate_up, down_proj)

    assert result is False, (
        "Expected fall-through to False via the down_proj W8A8-kernel-type "
        "check (down_proj has no `.scheme`)."
    )
    mock_capturing.assert_called_once()
    mock_kernel.assert_not_called()


def test_env_disable_short_circuits_before_compiling_check() -> None:
    """Env-disable flag remains the very first check — the new Dynamo guard
    must be additive, not a replacement for the existing kill-switch.
    """
    gate_up = _FakeGateUp()
    down_proj = object()

    with (
        patch.object(torch.compiler, "is_compiling") as mock_compiling,
        patch.object(torch.cuda, "is_current_stream_capturing") as mock_capturing,
        patch.dict("os.environ", {"VLLM_MI100_DISABLE_FUSED_ACT_QUANT": "1"}),
    ):
        result = try_stash_fused_silu_quant_int8(gate_up, down_proj)

    assert result is False
    mock_compiling.assert_not_called()
    mock_capturing.assert_not_called()
