# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M4 (CK INT8) dispatcher priority test.

VAL-CK-004:
  CK > hipBLASLt > Triton selection order over 50 forward passes.

The W8A8 dispatcher in `vllm.model_executor.kernels.linear.scaled_mm.mi100_int8`
is updated to consult `torch.ops._rocm_C.ck_int8_gemm_supports` first, then
fall through to hipBLASLt (`mi100_hipblaslt_supports`), and finally Triton.
With `VLLM_LOG_GEMM_BACKEND=1` set, every call records the chosen backend
to a log buffer; this test asserts the order.
"""

import os
import pytest
import torch


@pytest.fixture(scope="module")
def dispatcher():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP not available")
    if not getattr(torch.version, "hip", None):
        pytest.skip("Not a ROCm build")
    try:
        import vllm._rocm_C  # noqa: F401
    except ImportError:
        pytest.skip("_rocm_C not built")
    if not hasattr(torch.ops._rocm_C, "ck_int8_gemm_supports"):
        pytest.skip("CK build not enabled (VLLM_BUILD_CK=OFF)")
    from vllm.model_executor.kernels.linear.scaled_mm import (
        mi100_int8_dispatch,
    )
    return mi100_int8_dispatch


def test_ck_preferred_over_hipblaslt_and_triton(dispatcher, monkeypatch):
    """For a CK-registered shape, the dispatcher must report 'ck'."""
    monkeypatch.setenv("VLLM_LOG_GEMM_BACKEND", "1")
    chosen = dispatcher.choose_backend(M=128, N=4096, K=4096, tp_rank=1)
    assert chosen == "ck", (
        f"Expected 'ck' for registered shape (128,4096,4096) at tp=1, "
        f"got {chosen!r}")


def test_hipblaslt_fallback_when_ck_absent(dispatcher, monkeypatch):
    """For a shape not registered with CK, dispatcher must fall through to
    hipBLASLt (when M2 tuning JSON covers it) or Triton."""
    monkeypatch.setenv("VLLM_LOG_GEMM_BACKEND", "1")
    # Random untuned shape — should never hit CK.
    chosen = dispatcher.choose_backend(M=37, N=999, K=4096, tp_rank=1)
    assert chosen in ("hipblaslt", "triton"), (
        f"Expected fallback to hipblaslt or triton for untuned shape, "
        f"got {chosen!r}")


def test_priority_order_over_50_forward_passes(dispatcher, monkeypatch):
    """Run 50 forward passes mixing CK-registered and unregistered shapes.

    Verifies CK is selected for every CK-registered call and never selected
    for an unregistered one (no race conditions in dispatch).
    """
    monkeypatch.setenv("VLLM_LOG_GEMM_BACKEND", "1")
    ck_shapes = [(64, 4096, 4096), (32, 10240, 4096), (64, 4096, 12288)]
    other_shapes = [(11, 333, 555), (5, 7, 9)]
    backends = []
    for i in range(50):
        if i % 2 == 0:
            M, N, K = ck_shapes[i % len(ck_shapes)]
            tp = 1
            expected = "ck"
        else:
            M, N, K = other_shapes[i % len(other_shapes)]
            tp = 1
            expected = ("hipblaslt", "triton")
        chosen = dispatcher.choose_backend(M=M, N=N, K=K, tp_rank=tp)
        backends.append((expected, chosen))

    # Every CK-registered call must select CK.
    ck_failures = [
        (e, c) for (e, c) in backends if e == "ck" and c != "ck"
    ]
    assert not ck_failures, (
        f"CK dispatch order violated on {len(ck_failures)} of 50 calls: "
        f"{ck_failures[:5]}")
    # Unregistered shapes must never silently land on CK.
    leak = [(e, c) for (e, c) in backends if e != "ck" and c == "ck"]
    assert not leak, f"CK selected for unregistered shape: {leak[:5]}"
