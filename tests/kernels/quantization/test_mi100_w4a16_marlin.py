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

import itertools

import pytest
import torch

from vllm.platforms import current_platform

# ROCm/MI100 specific; skip at import on non-ROCm.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.kernels.configs.gfx908 import (  # noqa: E501
    config_loader,
)
from vllm.model_executor.kernels.linear.scaled_mm import (  # noqa: E501
    mi100_w4a16_marlin as marlin_mod,
)
from vllm.model_executor.kernels.linear.scaled_mm.mi100_w4a16_marlin import (  # noqa: E501
    marlin_repack_w4a16,
    mi100_w4a16_marlin_gemm,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
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
    scales = (
        0.05 * torch.rand((num_groups, N), device=device, dtype=torch.float32)
    ).to(torch.float16)

    qzeros = None
    if has_zp:
        zeros_int4 = torch.randint(
            0, 16, (num_groups, N), device=device, dtype=torch.int32
        )
        qzeros = _pack_int4_along_n(zeros_int4)

    packed = marlin_repack_w4a16(b_q, scales, qzeros, group_size=group_size, zp_bias=8)

    assert tuple(packed.qweight.shape) == (K // 8, N)
    assert packed.qweight.dtype == torch.int32

    recovered = unpack_repacked_to_kn(packed.qweight)  # [K, N] int4
    max_abs = int((recovered - w_int4_kn).abs().max().item())
    assert max_abs == 0, f"round-trip nibble mismatch: max abs diff={max_abs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
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
    scales = (
        0.05 * torch.rand((num_groups, N), device=device, dtype=torch.float32)
    ).to(torch.float16)

    qzeros = None
    if has_zp:
        zeros_int4 = torch.randint(
            0, 16, (num_groups, N), device=device, dtype=torch.int32
        )
        qzeros = _pack_int4_along_n(zeros_int4)
        zero_raw = zeros_int4.to(torch.float32)  # [K//G, N]
    else:
        zero_raw = torch.full(
            (num_groups, N), float(zp_bias), device=device, dtype=torch.float32
        )

    packed = marlin_repack_w4a16(
        b_q, scales, qzeros, group_size=group_size, zp_bias=zp_bias
    )

    assert tuple(packed.scale.shape) == (num_groups, N)
    assert tuple(packed.zero.shape) == (num_groups, N)

    # Per-element nibble grid (full [K, N]) and its group index.
    q = w_int4_kn.to(torch.float32)  # [K, N]
    g_idx = torch.arange(K, device=device) // group_size  # [K]
    scale_full = packed.scale.to(torch.float32)[g_idx]  # [K, N]
    zero_fused_full = packed.zero.to(torch.float32)[g_idx]  # [K, N]

    # Pre-fused form:  q*scale - zero_fused  (zero_fused == zero_raw*scale)
    got = q * scale_full - zero_fused_full

    # Canonical reference:  (q - zero_raw) * scale
    zero_raw_full = zero_raw[g_idx]  # [K, N]
    ref = (q - zero_raw_full) * scales.to(torch.float32)[g_idx]

    torch.testing.assert_close(got, ref, atol=1e-2, rtol=5e-2)


# ===========================================================================
#  GEMM tests (F-M1-gemm-kernel): repack -> mi100_w4a16_marlin_gemm vs a
#  pure-FP32 dequant -> matmul reference.
# ===========================================================================

# Hot-shape catalog (M, K, N). K divisible by 8 and by both group sizes.
GEMM_SHAPES = [
    (16, 256, 256),
    (64, 512, 256),
    (32, 256, 512),
]


def _make_w4a16_case(
    M, K, N, group_size, has_zp, seed, scales_override=None, zeros_override=None
):
    """Build a random W4A16 case + its pure-FP32 dequant->matmul reference.

    Returns ``(a, b_q, scales, qzeros, ref_c)`` where ``ref_c`` is computed
    entirely in FP32 from the *unpacked* int4 grid (no kernel involvement).
    """
    torch.manual_seed(seed)
    num_groups = K // group_size
    zp_bias = 8

    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    b_q = _pack_int4_along_n(w_int4_kn)

    if scales_override is not None:
        scales = scales_override
    else:
        scales = (
            0.05 * torch.rand((num_groups, N), device=device, dtype=torch.float32)
        ).to(torch.float16)

    qzeros = None
    if has_zp:
        if zeros_override is not None:
            zeros_int4 = zeros_override
        else:
            zeros_int4 = torch.randint(
                0, 16, (num_groups, N), device=device, dtype=torch.int32
            )
        qzeros = _pack_int4_along_n(zeros_int4)
        zero_raw = zeros_int4.to(torch.float32)
    else:
        zero_raw = torch.full(
            (num_groups, N), float(zp_bias), device=device, dtype=torch.float32
        )

    a = (0.1 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )

    # Pure-FP32 reference: dequant then matmul.
    g_idx = torch.arange(K, device=device) // group_size  # [K]
    scale_full = scales.to(torch.float32)[g_idx]  # [K, N]
    zero_raw_full = zero_raw[g_idx]  # [K, N]
    w_fp = (w_int4_kn.to(torch.float32) - zero_raw_full) * scale_full
    ref_c = a.to(torch.float32) @ w_fp  # [M, N]

    return a, b_q, scales, qzeros, ref_c


def _run_gemm(a, b_q, scales, qzeros, group_size):
    packed = marlin_repack_w4a16(b_q, scales, qzeros, group_size=group_size, zp_bias=8)
    return mi100_w4a16_marlin_gemm(
        a, packed.qweight, packed.scale, packed.zero, group_size
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("M,K,N", GEMM_SHAPES)
def test_symmetric(M, K, N, group_size, seed):
    a, b_q, scales, qzeros, ref_c = _make_w4a16_case(
        M, K, N, group_size, has_zp=False, seed=seed
    )
    c = _run_gemm(a, b_q, scales, qzeros, group_size)
    torch.testing.assert_close(c.to(torch.float32), ref_c, atol=1e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("M,K,N", GEMM_SHAPES)
def test_asymmetric(M, K, N, group_size, seed):
    # PRODUCTION-CRITICAL: qzeros present, full asymmetric dequant.
    a, b_q, scales, qzeros, ref_c = _make_w4a16_case(
        M, K, N, group_size, has_zp=True, seed=seed
    )
    c = _run_gemm(a, b_q, scales, qzeros, group_size)
    torch.testing.assert_close(c.to(torch.float32), ref_c, atol=1e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("group_size", [32, 128])
def test_group(group_size):
    # Both supported group sizes produce correct results.
    a, b_q, scales, qzeros, ref_c = _make_w4a16_case(
        32, 256, 256, group_size, has_zp=True, seed=0
    )
    c = _run_gemm(a, b_q, scales, qzeros, group_size)
    torch.testing.assert_close(c.to(torch.float32), ref_c, atol=1e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_group_unsupported_rejected():
    # Unsupported group_size must raise in both repack and GEMM.
    num_groups = 256 // 48 if 256 % 48 == 0 else 4
    with pytest.raises(ValueError):
        # repack rejects unsupported group_size
        b_q = torch.zeros((256, 256 // 8), device=device, dtype=torch.int32)
        scales = torch.ones((num_groups, 256), device=device, dtype=torch.float16)
        marlin_repack_w4a16(b_q, scales, None, group_size=48)

    with pytest.raises(ValueError):
        # GEMM also rejects unsupported group_size
        a = torch.zeros((16, 256), device=device, dtype=torch.float16)
        qweight = torch.zeros((256 // 8, 256), device=device, dtype=torch.int32)
        scale = torch.ones((4, 256), device=device, dtype=torch.float16)
        zero = torch.zeros((4, 256), device=device, dtype=torch.float16)
        mi100_w4a16_marlin_gemm(a, qweight, scale, zero, group_size=48)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_boundary():
    # g=32 with sharply DISTINCT per-group scales+zeros: a cross-group scale
    # leak would corrupt the result at group boundaries. K spans 8 groups.
    M, K, N, group_size = 32, 256, 128, 32
    num_groups = K // group_size  # 8

    # Distinct, well-separated scale per group (one row per group).
    scales = (
        (
            torch.arange(1, num_groups + 1, device=device, dtype=torch.float32).reshape(
                num_groups, 1
            )
            * 0.01
        )
        .expand(num_groups, N)
        .contiguous()
        .to(torch.float16)
    )
    # Distinct zero per group: 0,2,4,... (mod 16).
    zeros_int4 = (
        ((torch.arange(num_groups, device=device, dtype=torch.int32) * 2) % 16)
        .reshape(num_groups, 1)
        .expand(num_groups, N)
        .contiguous()
    )

    a, b_q, scales, qzeros, ref_c = _make_w4a16_case(
        M,
        K,
        N,
        group_size,
        has_zp=True,
        seed=7,
        scales_override=scales,
        zeros_override=zeros_int4,
    )
    c = _run_gemm(a, b_q, scales, qzeros, group_size)
    torch.testing.assert_close(c.to(torch.float32), ref_c, atol=1e-2, rtol=5e-2)


# --- Source-pattern tests: enforce shift-only / no-interleave discipline. ---


def test_source_no_interleave():
    import inspect

    src = inspect.getsource(marlin_mod)
    assert "tl.interleave" not in src, (
        "kernel must be shift-only: tl.interleave is forbidden"
    )


def test_source_only_output_store():
    import inspect

    import regex as re

    src = inspect.getsource(marlin_mod)
    targets = re.findall(r"tl\.store\(\s*([A-Za-z_][A-Za-z0-9_]*)", src)
    assert targets == ["c_ptrs"], (
        f"the only tl.store target must be the output (c_ptrs); got {targets}"
    )


# ===========================================================================
#  F-M2 (autotune config seeding): the 48 seeded
#  ``mi100_w4a16_marlin_M*_N*_K*_g*.json`` configs must resolve via the real
#  ``config_loader`` and every selected tile must fit the gfx908 LDS budget.
# ===========================================================================

# Hot-shape catalog mirrored from the seeded JSONs (M x N x K x g = 48).
F_M2_MS = [1, 32, 128, 512]
F_M2_NS = [4096, 10240, 24576]
F_M2_KS = [4096, 12288]
F_M2_GS = [32, 128]
F_M2_SHAPES = list(itertools.product(F_M2_MS, F_M2_NS, F_M2_KS, F_M2_GS))

MARLIN_KERNEL_KEY = "mi100_w4a16_marlin"
_LDS_A_BYTES = 2  # fp16/bf16 activation tile element size
_LDS_BUDGET = 64 * 1024  # gfx908 LDS budget


def _marlin_lds_bytes(bm: int, bn: int, bk: int, ns: int) -> int:
    """Conservative gfx908 LDS estimate; matches autotune-marlin.md.

    bytes = ns * BLOCK_K * (BLOCK_M + BLOCK_N) * 2      # fp16 act term
          + ns * (BLOCK_K // 8) * BLOCK_N * 4           # packed int4 B
    """
    return ns * bk * (bm + bn) * _LDS_A_BYTES + ns * (bk // 8) * bn * 4


def test_seeded_configs_load_for_all_hot_shapes():
    """All 48 (M, N, K, g) resolve to a seeded JSON via the real loader and
    every selected tile is within the 64 KiB LDS budget (CPU-only)."""
    assert len(F_M2_SHAPES) == 48, f"expected 48 hot shapes, got {len(F_M2_SHAPES)}"
    config_loader.reset_cache()
    for m, n, k, g in F_M2_SHAPES:
        cfg = config_loader.load_config(MARLIN_KERNEL_KEY, M=m, N=n, K=k, group_size=g)
        assert cfg is not None, f"no seeded config for (M={m}, N={n}, K={k}, g={g})"
        bm = int(cfg["BLOCK_M"])
        bn = int(cfg["BLOCK_N"])
        bk = int(cfg["BLOCK_K"])
        ns = int(cfg["num_stages"])
        assert bk % 8 == 0, f"BLOCK_K={bk} not a multiple of 8"
        assert g % bk == 0, f"BLOCK_K={bk} does not divide group_size={g}"
        assert ns <= 2, f"num_stages={ns} > 2 on gfx908"
        est = _marlin_lds_bytes(bm, bn, bk, ns)
        assert est <= _LDS_BUDGET, (
            f"over-budget tile for (M={m}, N={n}, K={k}, g={g}): {est} > {_LDS_BUDGET}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize(
    "M,N,K,group_size",
    [
        (1, 4096, 4096, 128),  # decode
        (32, 4096, 4096, 32),  # prefill-ish small
    ],
)
def test_marlin_gemm_correct_with_seeded_config_active(M, N, K, group_size):
    """End-to-end correctness with the seeded config ACTIVE: repack -> GEMM
    matches the pure-FP32 dequant->matmul reference (symmetric path)."""
    # Confirm the seeded config is the one that will be selected.
    config_loader.reset_cache()
    cfg = config_loader.load_config(
        MARLIN_KERNEL_KEY, M=M, N=N, K=K, group_size=group_size
    )
    assert cfg is not None, "seeded config must be active for this shape"

    a, b_q, scales, qzeros, ref_c = _make_w4a16_case(
        M, K, N, group_size, has_zp=False, seed=0
    )
    c = _run_gemm(a, b_q, scales, qzeros, group_size)
    torch.testing.assert_close(c.to(torch.float32), ref_c, atol=1e-2, rtol=5e-2)
