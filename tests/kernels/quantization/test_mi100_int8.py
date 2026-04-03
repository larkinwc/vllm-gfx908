# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MI100 INT8 W8A8 scaled matmul kernel.

Run: pytest tests/kernels/quantization/test_mi100_int8.py -v
"""

import pytest
import torch

from vllm.platforms import current_platform

device = "cuda"

MNK_FACTORS = [
    (1, 128, 64),
    (1, 256, 128),
    (4, 256, 256),
    (32, 1024, 512),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 4096, 4096),
    (128, 6144, 4096),
    (512, 256, 496),
    (512, 4096, 4096),
    (1024, 4096, 6144),
]


def torch_int8_scaled_mm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation: dequant to float32, matmul, requant."""
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias
    return out


@pytest.mark.parametrize("M,N,K", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_mi100_int8_scaled_mm(
    M, N, K, out_dtype, use_scalar_scale_a, use_scalar_scale_b, use_bias
):
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    torch.manual_seed(0)

    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device=device)

    if use_scalar_scale_a:
        scale_a = torch.rand((1, 1), device=device, dtype=torch.float32)
    else:
        scale_a = 0.25 * torch.rand((M, 1), device=device, dtype=torch.float32)

    if use_scalar_scale_b:
        scale_b = torch.rand((1, 1), device=device, dtype=torch.float32)
    else:
        scale_b = 0.25 * torch.rand((N, 1), device=device, dtype=torch.float32)

    bias = None
    if use_bias:
        bias = torch.rand((N,), device=device, dtype=out_dtype)

    result = mi100_int8_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    reference = torch_int8_scaled_mm_ref(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(result, reference, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (64, 4096, 6144)])
def test_mi100_int8_matches_triton_generic(M, N, K):
    """Verify MI100 INT8 kernel matches the generic Triton INT8 kernel."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    torch.manual_seed(42)

    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device=device)
    scale_a = torch.rand((1, 1), device=device, dtype=torch.float32)
    scale_b = torch.rand((1, 1), device=device, dtype=torch.float32)

    mi100_out = mi100_int8_scaled_mm(a, b, scale_a, scale_b, torch.float16)
    triton_out = triton_scaled_mm(a, b, scale_a, scale_b, torch.float16)

    torch.testing.assert_close(mi100_out, triton_out, rtol=1e-1, atol=1e-1)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MI100 kernel registration test"
)
def test_mi100_int8_kernel_selection():
    """Verify MI100Int8 kernel is selected on gfx908."""
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_INT8_KERNELS,
        Int8ScaledMMLinearLayerConfig,
        MI100Int8ScaledMMLinearKernel,
        choose_scaled_mm_linear_kernel,
    )

    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )

    is_supported, _ = MI100Int8ScaledMMLinearKernel.is_supported()
    if is_supported:
        kernel_type = choose_scaled_mm_linear_kernel(config, _POSSIBLE_INT8_KERNELS)
        assert kernel_type == MI100Int8ScaledMMLinearKernel, (
            f"Expected MI100Int8ScaledMMLinearKernel, got {kernel_type.__name__}"
        )


@pytest.mark.parametrize("M", [1, 4, 32, 128, 512])
def test_mi100_int8_decode_shapes(M):
    """Test shapes typical of decode (small M, large K/N)."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    K, N = 2048, 1536  # Qwen3.5-9B QKV proj TP=4
    torch.manual_seed(0)

    a = torch.randint(-64, 64, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-64, 64, (K, N), dtype=torch.int8, device=device)
    scale_a = 0.01 * torch.rand((M, 1), device=device, dtype=torch.float32)
    scale_b = 0.01 * torch.rand((N, 1), device=device, dtype=torch.float32)

    result = mi100_int8_scaled_mm(a, b, scale_a, scale_b, torch.float16)
    reference = torch_int8_scaled_mm_ref(a, b, scale_a, scale_b, torch.float16)

    torch.testing.assert_close(result, reference, rtol=1e-1, atol=1e-1)
