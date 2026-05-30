# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-trip + pre-fuse tests for the MI100 Marlin-style W4A16 repack.

This covers the OFFLINE host-side repack (F-M1-repack-fn): the GEMM kernel
that consumes the layout is a separate feature. Two checks:

  - ``-k round_trip``: repack then logically unpack reproduces the ORIGINAL
    nibbles bit-exactly (max abs diff == 0). sym + asym, g in {32, 128}.
  - ``-k prefuse``: pre-fused scale/zero math equals element-wise
    ``(q - zero) * scale`` within atol=1e-2, rtol=5e-2. sym + asym.
"""
from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

# ROCm/MI100 specific; skip at import on non-ROCm.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.kernels.linear.scaled_mm.mi100_w4a16_marlin import (  # noqa: E501
    marlin_repack_w4a16,
    unpack_repacked_to_kn,
)

device = "cuda"


def _pack_int4_along_n(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack int4 values along N: [K, N] -> [K, N//8] int32 (GPTQ N-packing)."""
    assert w_int4_kn.dtype == torch.int32
    K, N = w_int4_kn.shape
    assert N % 8 == 0
    shifts = torch.arange(8, device=w_int4_kn.device, dtype=torch.int32) * 4
    return torch.sum(
        (w_int4_kn.view(K, N // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()


def _unpack_int4_along_n(w_packed_kn8: torch.Tensor) -> torch.Tensor:
    """Unpack int4 values along N: [K, N//8] -> [K, N] int32."""
    assert w_packed_kn8.dtype == torch.int32
    K, N8 = w_packed_kn8.shape
    shifts = torch.arange(8, device=w_packed_kn8.device, dtype=torch.int32) * 4
    nibbles = (w_packed_kn8.unsqueeze(-1) >> shifts) & 0xF
    return nibbles.reshape(K, N8 * 8)


# Small-but-nontrivial shapes (K divisible by 8 and by both group sizes).
SHAPES = [
    (256, 256),
    (512, 256),
    (256, 512),
]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("has_zp", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("K,N", SHAPES)
def test_round_trip(K, N, has_zp, group_size, seed):
    if K % group_size != 0:
        pytest.skip(f"K={K} not divisible by group_size={group_size}")

    torch.manual_seed(seed)
    num_groups = K // group_size

    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    b_q = _pack_int4_along_n(w_int4_kn)
    scales = (0.05 * torch.rand(
        (num_groups, N), device=device, dtype=torch.float32)).to(torch.float16)

    qzeros = None
    if has_zp:
        zeros_int4 = torch.randint(
            0, 16, (num_groups, N), device=device, dtype=torch.int32)
        qzeros = _pack_int4_along_n(zeros_int4)

    packed = marlin_repack_w4a16(
        b_q, scales, qzeros, group_size=group_size, zp_bias=8)

    assert tuple(packed.qweight.shape) == (K // 8, N)
    assert packed.qweight.dtype == torch.int32

    recovered = unpack_repacked_to_kn(packed.qweight)  # [K, N] int4
    max_abs = int((recovered - w_int4_kn).abs().max().item())
    assert max_abs == 0, f"round-trip nibble mismatch: max abs diff={max_abs}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("has_zp", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("K,N", SHAPES)
def test_prefuse(K, N, has_zp, group_size, seed):
    if K % group_size != 0:
        pytest.skip(f"K={K} not divisible by group_size={group_size}")

    torch.manual_seed(seed)
    num_groups = K // group_size
    zp_bias = 8

    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    b_q = _pack_int4_along_n(w_int4_kn)
    scales = (0.05 * torch.rand(
        (num_groups, N), device=device, dtype=torch.float32)).to(torch.float16)

    qzeros = None
    if has_zp:
        zeros_int4 = torch.randint(
            0, 16, (num_groups, N), device=device, dtype=torch.int32)
        qzeros = _pack_int4_along_n(zeros_int4)
        zero_raw = zeros_int4.to(torch.float32)  # [K//G, N]
    else:
        zero_raw = torch.full(
            (num_groups, N), float(zp_bias),
            device=device, dtype=torch.float32)

    packed = marlin_repack_w4a16(
        b_q, scales, qzeros, group_size=group_size, zp_bias=zp_bias)

    assert tuple(packed.scale.shape) == (num_groups, N)
    assert tuple(packed.zero.shape) == (num_groups, N)

    # Per-element nibble grid (full [K, N]) and its group index.
    q = w_int4_kn.to(torch.float32)                                  # [K, N]
    g_idx = torch.arange(K, device=device) // group_size            # [K]
    scale_full = packed.scale.to(torch.float32)[g_idx]              # [K, N]
    zero_fused_full = packed.zero.to(torch.float32)[g_idx]         # [K, N]

    # Pre-fused form:  q*scale - zero_fused  (zero_fused == zero_raw*scale)
    got = q * scale_full - zero_fused_full

    # Canonical reference:  (q - zero_raw) * scale
    zero_raw_full = zero_raw[g_idx]                                  # [K, N]
    ref = (q - zero_raw_full) * scales.to(torch.float32)[g_idx]

    torch.testing.assert_close(got, ref, atol=1e-2, rtol=5e-2)
