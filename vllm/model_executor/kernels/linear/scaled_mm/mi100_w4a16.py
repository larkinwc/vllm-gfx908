# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MI100 (gfx908) custom Triton W4A16 GEMM kernel.

Computes ``C[M,N] = A[M,K] @ dequant(B)[K,N]`` where:

- ``A`` is fp16/bf16, contiguous ``[M, K]``.
- ``B`` is GPTQ-packed int4: ``[K, N//8] int32``, eight N-values per int32 at
  bit offsets ``[0, 4, 8, ..., 28]``.
- ``scales`` are ``[K//G, N]`` in the activation dtype, ``zeros`` (when
  asymmetric) are ``[K//G, N//8] int32`` packed identically to ``B``.

The unpack is **register-level**: a single ``int32`` tile is loaded from
HBM, replicated to the full N-width via ``tl.interleave``, then masked +
shifted via ``tl.bitwise_*`` so the int4 nibble for every output column is
present in registers. There is no global-memory round-trip of the
unpacked weights — this is what attacks the M1-omniperf finding that the
W4A16 hot kernel is HBM-bandwidth-bound (~32% of peak) on this model.

Tuning knobs and their gfx908 implications (per user M3 brief):
- ``BLOCK_M``: token-tile size. Decode hot path is M ≤ 16; prefill hits
  M ∈ {16..512+}.
- ``BLOCK_N`` / ``BLOCK_K``: must divide ``group_size`` (we clamp BLOCK_K).
- ``matrix_instr_nonkdim ∈ {16, 32}``: selects MFMA instruction variant
  ``v_mfma_f32_16x16x16`` vs ``v_mfma_f32_32x32x8`` (for fp16 input).
- ``kpack``: packs two K-elements per 32-bit load to match MFMA's
  K-major LDS layout.
- ``waves_per_eu``: occupancy lever; 0 = compiler default, 1..3 force
  fewer/more waves per execution unit.
- ``num_stages``: software pipelining depth. gfx908 has no async-LDS so
  num_stages > 2 yields no benefit.

Per-shape autotune configs are read from
``vllm/model_executor/kernels/configs/gfx908/mi100_w4a16_M*_N*_K*_g*.json``.
"""

from __future__ import annotations

import logging

import torch

from vllm.model_executor.kernels.configs.gfx908.config_loader import (
    load_config as _load_mi100_autotune_config,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)
_SUPPORTED_GROUP_SIZES = (32, 64, 128)


@triton.jit
def mi100_w4a16_gemm_kernel(
    a_ptr,  # [M, K] fp16/bf16
    b_ptr,  # [K, N//8] int32 (GPTQ sequential-packed N)
    scales_ptr,  # [K//G, N] same dtype as a
    zeros_ptr,  # [K//G, N//8] int32 (only used when HAS_ZP)
    c_ptr,  # [M, N] fp16/bf16
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,  # 8 for uint4b8 (symmetric), 0 for asymmetric
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """W4A16 GEMM with register-level int4 unpack.

    Each program computes a [BLOCK_M, BLOCK_N] tile of C. ``BLOCK_K``
    must divide ``group_size`` so a single tile loop iteration consumes
    exactly one quant group's scales/zeros (avoids per-row scale-mux
    inside the inner loop).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Packed-N column offsets: BLOCK_N // 8 packed int32s per row.
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    # Per-column shift table for the GPTQ sequential nibble layout.
    # Column j gets shift (j % 8) * 4. Build via broadcast+reshape so
    # Triton's compile-time shape inference is happy.
    shifts_row = tl.arange(0, 8) * 4  # [8]
    shifts_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))  # [N//8, 8]
    shifts_1d = tl.reshape(shifts_2d, (BLOCK_N,))  # [BLOCK_N]
    shifts = tl.broadcast_to(
        shifts_1d[None, :], (BLOCK_K, BLOCK_N)
    )  # [BLOCK_K, BLOCK_N]

    # Scales column offsets (one scalar per output column per group).
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_idx in range(0, n_k_tiles):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Activations: [BLOCK_M, BLOCK_K] fp16/bf16 ----
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Packed-int4 weights: [BLOCK_K, BLOCK_N//8] int32 ----
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # Register-level unpack: replicate each int32 8x along N then
        # extract the right nibble per column. No global-memory hop.
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        # bitwise_and with 0xF after a right-shift by per-column shifts.
        b = (b >> shifts) & 0xF

        # ---- Per-group scales / zeros (one group per BLOCK_K tile) ----
        g_idx = (k_idx * BLOCK_K) // group_size

        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        if HAS_ZP:
            zero_offset = g_idx * (N // 8) + offs_bn
            zero_mask = offs_bn < N // 8
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        # Dequant in registers: (q - zero) * scale -> fp16/bf16.
        b_fp = (b - z).to(a.dtype) * scales

        # MFMA: v_mfma_f32_*x*x*16f16 (or 32x32x8 with kpack=2) on gfx908.
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def _default_block_sizes(M: int) -> tuple[int, int, int, int, int]:
    """Heuristic fallback when no autotune JSON exists yet.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).
    Matches the existing in-tree W4A16 kernel branches for gfx908.
    """
    if M <= 16:
        return 16, 64, 32, 4, 2
    if M <= 64:
        return 32, 64, 32, 4, 2
    return 64, 128, 32, 4, 2


def mi100_w4a16_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
) -> torch.Tensor:
    """Public entry point matching ``triton_w4a16_gemm``'s contract.

    The MI100 path adds:
      - Per-shape autotune-config lookup (``configs/gfx908/mi100_w4a16_*``).
      - Default ``matrix_instr_nonkdim`` / ``kpack`` / ``waves_per_eu``
        forwarded to Triton when present in the config.
      - Sanity-clamps BLOCK_K to ``group_size`` so each tile consumes
        exactly one quant group (otherwise the dequant would silently
        mix two groups' scales, corrupting the output).
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"mi100_w4a16: unsupported group_size={group_size}; "
            f"supported: {_SUPPORTED_GROUP_SIZES}"
        )

    M, K = a.shape
    N = b_q.shape[1] * 8

    assert b_q.shape == (K, N // 8), (
        f"b_q shape mismatch: {b_q.shape} vs ({K}, {N // 8})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 8), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    zeros_ptr_arg = qzeros if has_zp else b_q  # dummy ptr when unused

    extra_launch: dict = {}
    cfg = _load_mi100_autotune_config(
        "mi100_w4a16", M=M, N=N, K=K, group_size=group_size
    )

    if cfg is not None:
        BLOCK_M = int(cfg["BLOCK_M"])
        BLOCK_N = int(cfg["BLOCK_N"])
        BLOCK_K = int(cfg["BLOCK_K"])
        num_warps = int(cfg.get("num_warps", 4))
        num_stages = int(cfg.get("num_stages", 2))
        if "matrix_instr_nonkdim" in cfg:
            extra_launch["matrix_instr_nonkdim"] = int(cfg["matrix_instr_nonkdim"])
        if "kpack" in cfg:
            extra_launch["kpack"] = int(cfg["kpack"])
        if "waves_per_eu" in cfg:
            extra_launch["waves_per_eu"] = int(cfg["waves_per_eu"])
    else:
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _default_block_sizes(M)

    # Each tile must lie within a single quant group.
    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    mi100_w4a16_gemm_kernel[grid](
        a,
        b_q,
        scales,
        zeros_ptr_arg,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_launch,
    )
    return c


def is_supported() -> bool:
    """True iff this kernel will run on the current platform."""
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_mi100
    except ImportError:
        return False
    return on_mi100()
