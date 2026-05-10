# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M4 (CK INT8) smoke test.

Verifies that the `_rocm_C` extension exposes `ck_int8_gemm` and that a
1×1 invocation returns without raising.

Skipped on:
- non-ROCm hosts (no _rocm_C extension)
- ROCm builds where VLLM_BUILD_CK was OFF (op symbol absent)
"""

import pytest
import torch


@pytest.fixture(scope="module")
def ck_int8_gemm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP runtime not available")
    if not getattr(torch.version, "hip", None):
        pytest.skip("Not a ROCm build; CK INT8 GEMM is gfx908-only")
    try:
        import vllm._rocm_C  # noqa: F401
    except ImportError:
        pytest.skip("vllm._rocm_C extension is not built")
    op = getattr(torch.ops._rocm_C, "ck_int8_gemm", None)
    if op is None:
        pytest.skip(
            "torch.ops._rocm_C.ck_int8_gemm is unavailable. The vllm "
            "build was made with VLLM_BUILD_CK=OFF or on a non-gfx908 "
            "target. Re-run install with GPU_TARGETS=gfx908 + "
            "VLLM_BUILD_CK=ON to enable.")
    return op


def test_ck_int8_gemm_op_is_importable(ck_int8_gemm):
    # The op must be a callable OpOverload.
    assert callable(ck_int8_gemm)


def test_ck_int8_gemm_supports_query_does_not_raise():
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
    # Known M4-registered shape.
    res = torch.ops._rocm_C.ck_int8_gemm_supports(64, 24576, 4096, 1)
    assert isinstance(res, bool)


def test_ck_int8_gemm_smoke_1x1(ck_int8_gemm):
    """Run a 1×1 invocation through CK to exercise the binding.

    Uses one of the registered hot shapes; should not raise.
    """
    M, N, K = 64, 4096, 4096  # one of the registered M4 shapes
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
    scale_a = torch.full((M,), 0.01, dtype=torch.float32, device="cuda")
    scale_b = torch.full((N,), 0.02, dtype=torch.float32, device="cuda")
    out = ck_int8_gemm(a, b, scale_a, scale_b, None, 1)
    assert out.shape == (M, N)
    assert out.dtype == torch.float16
    # No NaN/Inf in output.
    assert torch.isfinite(out).all().item()
