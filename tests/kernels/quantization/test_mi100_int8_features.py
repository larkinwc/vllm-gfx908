# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3 feature tests for the extended MI100 W8A8 Triton kernel.

Verifies VAL-TRITON-001:
  - Kernel takes per-channel weight_scale [N] and per-token act_scale [M].
  - Accumulates via tl.dot with INT32 out_dtype.
  - Fuses dequant * scales + bias -> fp16 in the store epilogue (no
    separate dequant launch).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import regex as re
import torch

from vllm.platforms import current_platform


def _kernel_source() -> str:
    path = Path("vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py")
    return path.read_text()


def test_kernel_signature_and_epilogue():
    """Static checks on the kernel source & runtime signature."""
    source = _kernel_source()

    # 1) tl.dot is invoked with int32 out_dtype.
    assert re.search(r"tl\.dot\([^)]*out_dtype\s*=\s*tl\.int32", source), (
        "expected tl.dot(..., out_dtype=tl.int32) in the W8A8 kernel"
    )

    # 2) Per-token / per-channel scales: scale_a is along M, scale_b along N.
    assert "scale_a" in source and "scale_b" in source

    # 3) The store epilogue casts after the scale multiply (fused, no
    #    separate dequant launch).
    # Look for the order: scale_a multiply -> scale_b multiply -> cast -> store.
    pattern = re.compile(
        r"scale_a \* accumulator.*scale_b\.T \* result.*"
        r"\.to\(c_ptr\.type\.element_ty\).*tl\.store",
        re.DOTALL,
    )
    assert pattern.search(source), (
        "expected fused dequant+cast+store epilogue in mi100_int8 kernel"
    )

    # 4) Bias must be added BEFORE the store and cast applied to a fp32
    #    intermediate so per-channel bias has full fp32 fidelity.
    assert "bias_vec[None, :]" in source, (
        "expected fused fp32 bias addition in the epilogue, not a "
        "separate post-cast add"
    )

    # 5) The public function signature is unchanged but documented as
    #    per-token / per-channel.
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    sig = inspect.signature(mi100_int8_scaled_mm)
    params = list(sig.parameters)
    assert params[:6] == [
        "input",
        "weight",
        "scale_a",
        "scale_b",
        "out_dtype",
        "bias",
    ], f"unexpected signature: {params}"


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)
def test_w8a8_per_channel_per_token_runs_and_matches_reference():
    """Smoke: per-token act_scale [M,1] + per-channel weight_scale [N,1]
    + bias should match an FP32 reference."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    torch.manual_seed(123)
    M, N, K = 64, 4096, 4096
    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device="cuda")
    scale_a = (0.01 * torch.rand((M, 1), device="cuda")).to(torch.float32)
    scale_b = (0.01 * torch.rand((N, 1), device="cuda")).to(torch.float32)
    bias = torch.randn((N,), dtype=torch.float16, device="cuda") * 0.1

    out = mi100_int8_scaled_mm(a, b, scale_a, scale_b, torch.float16, bias)

    ref = (scale_a * a.to(torch.float32) @ b.to(torch.float32)) * scale_b.T
    ref = ref + bias.to(torch.float32)
    ref = ref.to(torch.float16)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=5e-2)
