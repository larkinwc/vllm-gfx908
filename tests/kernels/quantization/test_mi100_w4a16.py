# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3 feature tests for the new MI100 W4A16 Triton kernel.

Verifies VAL-TRITON-002:
  - Kernel exists at the expected import path.
  - Loads packed-int4 weights and unpacks at the register level (via
    bitshift+mask, no global-mem hop).
  - Group-size {32, 128} correctness against an FP32 PyTorch reference.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import regex as re
import torch

from vllm.platforms import current_platform


def _kernel_source() -> str:
    return Path(
        "vllm/model_executor/kernels/linear/scaled_mm/mi100_w4a16.py"
    ).read_text()


def test_module_exists_with_register_unpack_pattern():
    src = _kernel_source()
    # Register-level unpack pattern: bitwise_and-style mask with 0xF and a
    # right-shift indexed by per-column shifts. We use Triton's & / >>
    # operator overloads so the source contains "& 0xF" and ">> shifts".
    assert re.search(r">>\s*shifts", src), (
        "expected register-level shift unpack in mi100_w4a16 kernel"
    )
    assert "& 0xF" in src, "expected 4-bit mask after shift"
    # No global memory round-trip of the unpacked weights — i.e., we
    # do not write the unpacked B back into HBM. Heuristic: there is no
    # tl.store of `b`/`b_fp` anywhere (only the final accumulator gets
    # stored).
    assert "tl.store(c_ptrs" in src, "expected only the C tile stored"
    assert "tl.store(b_ptrs" not in src, (
        "register-level unpack must not store the unpacked weights"
    )


def _pytorch_w4a16_reference(
    a: torch.Tensor,  # [M, K] fp16
    b_packed: torch.Tensor,  # [K, N//8] int32
    scales: torch.Tensor,  # [K//G, N] fp16
    group_size: int,
    zp_bias: int = 8,
) -> torch.Tensor:
    """FP32 reference: dequantize then matmul."""
    K, N_packed = b_packed.shape
    N = N_packed * 8

    shifts = torch.arange(8, device=b_packed.device, dtype=torch.int32) * 4
    # [K, N//8, 8] -> [K, N]
    nibbles = ((b_packed.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N).to(torch.int32)
    nibbles_minus_z = nibbles - zp_bias
    # Broadcast scales [K//G, N] -> [K, N] by repeating each group.
    scales_full = scales.repeat_interleave(group_size, dim=0)
    b_fp = nibbles_minus_z.to(torch.float32) * scales_full.to(torch.float32)
    out = a.to(torch.float32) @ b_fp
    return out.to(a.dtype)


def _make_packed_b(K: int, N: int, seed: int = 0) -> torch.Tensor:
    """Random GPTQ-packed [K, N//8] int32 weight tensor."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nibbles = torch.randint(
        0, 16, (K, N), dtype=torch.int32, device="cuda", generator=g
    )
    shifts = torch.arange(8, device="cuda", dtype=torch.int32) * 4
    packed = (nibbles.view(K, N // 8, 8) << shifts).sum(dim=-1).to(torch.int32)
    return packed.contiguous()


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)
@pytest.mark.parametrize("group_size", [32, 128])
def test_groupwise_unpack(group_size: int):
    """Smoke test: dequant matches the PyTorch reference at small shape."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_w4a16 import (
        mi100_w4a16_gemm,
    )

    M, K, N = 16, 256, 64
    torch.manual_seed(group_size)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.1
    b_packed = _make_packed_b(K, N, seed=group_size)
    scales = (0.01 * torch.rand((K // group_size, N), device="cuda")).to(torch.float16)

    out = mi100_w4a16_gemm(
        a,
        b_packed,
        scales,
        qzeros=None,
        group_size=group_size,
        zp_bias=8,
    )
    ref = _pytorch_w4a16_reference(a, b_packed, scales, group_size, zp_bias=8)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=5e-2)
