# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamo trace-break regression test for ``should_use_atomic_add_reduce``.

Companion to ``test_fused_silu_quant_int8_compile_guard.py``: same failure
class (Dynamo cannot fake-tensor-propagate a bool/tuple-returning torch.cuda
op reached inside a traced region), found in
``vllm/model_executor/layers/quantization/utils/marlin_utils.py`` during an
independent Dynamo-safety audit.

``should_use_atomic_add_reduce()`` is called from ``apply_gptq_marlin_linear``
/ ``apply_awq_marlin_linear`` (``marlin_utils.py``), which run inline inside
``MarlinLinearKernel.apply_weights`` — i.e. inside a ``@support_torch_compile``
model's forward, same architecture pattern as the fused-silu producer. Its
``torch.cuda.get_device_capability(device)`` call (marlin_utils.py, just
before the sm8x/bfloat16 check) returns a plain Python tuple, not a Tensor,
and was unguarded — while the sibling ``maybe_warn_marlin_atomic_add()`` a
few lines above already guards the *identical* call with
``torch.compiler.is_dynamo_compiling()``. The fix mirrors that existing
guard.

Scope note: this fast path is only reached when ``n < 2048``, ``k >= 2048``,
``device.type == "cuda"``, and the opt-in ``VLLM_MARLIN_USE_ATOMIC_ADD=1``
env flag is set (default off) — ``MarlinLinearKernel`` itself is CUDA-only
(``can_implement`` requires ``current_platform.is_cuda()``), so this is
unrelated to the gfx900 ROCm AWQ+TP4 investigation, which dispatches through
``TritonW4A16LinearKernel`` instead and never reaches this function.

Hardware-free: never touches a real CUDA device or a Marlin kernel.
``torch.compiler.is_dynamo_compiling`` / ``torch.cuda.get_device_capability``
are monkeypatched directly to observe call counts.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    should_use_atomic_add_reduce,
)


@pytest.fixture(autouse=True)
def _enable_atomic_add(monkeypatch: pytest.MonkeyPatch) -> None:
    """The atomic-add fast path is opt-in and default-off; enable it so
    these tests actually reach the guard under test (matching the shape
    gate: n < 2048, k >= 2048)."""
    monkeypatch.setenv("VLLM_MARLIN_USE_ATOMIC_ADD", "1")


class _FakeCudaDevice:
    """Duck-types the subset of ``torch.device`` that
    ``should_use_atomic_add_reduce`` reads before it would reach
    ``torch.cuda.get_device_capability`` — lets these tests exercise the
    Dynamo-guard branch without a real CUDA device."""

    type = "cuda"


def test_dynamo_compiling_short_circuits_before_capability_check() -> None:
    """``torch.compiler.is_dynamo_compiling() == True`` -> conservatively
    returns False (no atomic add), and the call that breaks Dynamo tracing
    (``get_device_capability``) is never made.
    """
    with (
        patch.object(
            torch.compiler, "is_dynamo_compiling", return_value=True
        ) as mock_compiling,
        patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError(
                "get_device_capability() must not be called while "
                "torch.compiler.is_dynamo_compiling() is True"
            ),
        ) as mock_capability,
    ):
        result = should_use_atomic_add_reduce(
            m=4, n=64, k=4096, device=_FakeCudaDevice(), dtype=torch.bfloat16
        )

    assert result is False, (
        "should_use_atomic_add_reduce must conservatively return False "
        "while torch.compiler.is_dynamo_compiling() is True."
    )
    mock_compiling.assert_called_once()
    mock_capability.assert_not_called()


def test_eager_mode_still_checks_device_capability_for_bf16() -> None:
    """Not compiling -> the pre-existing sm8x+bfloat16 capability check
    still runs unchanged (the new guard must not swallow or replace it).

    ``get_device_capability`` is expected to be called twice here: once by
    ``should_use_atomic_add_reduce`` itself, and once more by the
    pre-existing ``maybe_warn_marlin_atomic_add`` logging helper it calls
    on the sm8x+bfloat16 branch — that double call is unrelated to the new
    guard and predates this fix.
    """
    with (
        patch.object(torch.compiler, "is_dynamo_compiling", return_value=False),
        patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ) as mock_capability,
    ):
        result = should_use_atomic_add_reduce(
            m=4, n=64, k=4096, device=_FakeCudaDevice(), dtype=torch.bfloat16
        )

    assert result is False, (
        "sm8x (capability major < 9) + bfloat16 must still disable atomic "
        "add (pre-existing behavior)."
    )
    mock_capability.assert_called()


def test_eager_mode_sm90_bf16_uses_atomic_add() -> None:
    """Not compiling, capability >= sm90 -> atomic add is enabled for
    bfloat16 (proves the guard is a pure short-circuit and does not alter
    the rest of the control flow / final True branch).
    """
    with (
        patch.object(torch.compiler, "is_dynamo_compiling", return_value=False),
        patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ) as mock_capability,
    ):
        result = should_use_atomic_add_reduce(
            m=4, n=64, k=4096, device=_FakeCudaDevice(), dtype=torch.bfloat16
        )

    assert result is True
    mock_capability.assert_called_once()


def test_env_disabled_short_circuits_before_compiling_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-off env flag remains the very first check reached — the new
    Dynamo guard must be additive, not a replacement for the existing
    opt-in gate. ``get_device_capability`` (the call the new guard exists
    to protect) must never be reached from this branch.

    Note: the pre-existing ``maybe_warn_marlin_atomic_add_env`` logging
    helper, called on this same branch, has its own independent
    ``is_dynamo_compiling()`` guard (unrelated to this fix) — so that mock
    is expected to be invoked once here, not zero times.
    """
    monkeypatch.setenv("VLLM_MARLIN_USE_ATOMIC_ADD", "0")

    with (
        patch.object(
            torch.compiler, "is_dynamo_compiling", return_value=False
        ) as mock_compiling,
        patch.object(torch.cuda, "get_device_capability") as mock_capability,
    ):
        result = should_use_atomic_add_reduce(
            m=4, n=64, k=4096, device=_FakeCudaDevice(), dtype=torch.bfloat16
        )

    assert result is False
    mock_compiling.assert_called_once()
    mock_capability.assert_not_called()


def test_shape_gate_short_circuits_before_compiling_check() -> None:
    """The n/k/device-type shape gate remains the very first check —
    reached even before the env-flag check, unaffected by the new guard.
    """
    with (
        patch.object(torch.compiler, "is_dynamo_compiling") as mock_compiling,
        patch.object(torch.cuda, "get_device_capability") as mock_capability,
    ):
        # n >= 2048 fails the shape gate regardless of env/compiling state.
        result = should_use_atomic_add_reduce(
            m=4, n=4096, k=4096, device=_FakeCudaDevice(), dtype=torch.bfloat16
        )

    assert result is False
    mock_compiling.assert_not_called()
    mock_capability.assert_not_called()
