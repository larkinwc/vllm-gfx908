# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused Triton per-token int8 quant kernel.

Run on MI100 with:

    /opt/vllm-env/bin/python3 -m pytest \
        tests/kernels/test_fused_int8_quant.py -v
"""

import pytest
import torch

from vllm.model_executor.kernels.linear.mixed_precision.fused_int8_quant import (
    fused_per_token_quant_int8,
)

M_VALUES = [1, 8, 64, 256]
K_VALUES = [768, 3072, 6144]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _reference_per_token_int8(
    x: torch.Tensor, eps: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.to(torch.float32)
    absmax = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_(min=eps)
    scale = absmax / 127.0
    x_q = (x_fp32 / scale).round().clamp_(-127, 127).to(torch.int8)
    return x_q, scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_int8_quant_matches_reference(M: int, K: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=dtype) * 3.0

    x_q, scales = fused_per_token_quant_int8(x)
    ref_q, ref_scale = _reference_per_token_int8(x)

    assert x_q.dtype == torch.int8
    assert x_q.shape == (M, K)
    assert scales.dtype == torch.float32
    assert scales.shape == (M, 1)

    # Scale: fp32 from a single-path reduction; expect bitwise-level match
    # modulo rounding in the max-abs search order. 1 ULP is the tight bound.
    torch.testing.assert_close(scales, ref_scale, rtol=0, atol=1e-6)

    # Quantized values: allow off-by-one for half-to-even vs round-half-away
    # discrepancies between Triton and PyTorch rounding modes.
    diff = (x_q.to(torch.int32) - ref_q.to(torch.int32)).abs()
    assert diff.max().item() <= 1, (
        f"int8 quant diverges by >1 at M={M}, K={K}, dtype={dtype}: "
        f"max|diff|={diff.max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fused_int8_quant_preserves_shape_3d():
    # (batch, seq, hidden) input — should be reshaped back on return
    x = torch.randn(2, 5, 3072, device="cuda", dtype=torch.bfloat16)
    x_q, scales = fused_per_token_quant_int8(x)
    assert x_q.shape == (2, 5, 3072)
    assert scales.shape == (2, 5, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fused_int8_quant_all_zero_row_safe():
    # All-zero row must not produce NaN/Inf or divide-by-zero
    x = torch.zeros(4, 3072, device="cuda", dtype=torch.bfloat16)
    x[1, 100] = 1.0  # one non-zero element in row 1
    x_q, scales = fused_per_token_quant_int8(x)
    assert torch.isfinite(scales).all()
    assert (x_q[0] == 0).all()  # zero row stays zero
    assert (x_q[2] == 0).all()
    assert (x_q[3] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fused_int8_quant_non_contiguous_input():
    # Slicing creates a non-contiguous view; the wrapper must restore contiguity
    y = torch.randn(3072, 8, device="cuda", dtype=torch.float16).t()
    assert not y.is_contiguous()
    x_q, scales = fused_per_token_quant_int8(y)
    ref_q, ref_scale = _reference_per_token_int8(y.contiguous())
    torch.testing.assert_close(scales, ref_scale, rtol=0, atol=1e-6)
    diff = (x_q.to(torch.int32) - ref_q.to(torch.int32)).abs()
    assert diff.max().item() <= 1
