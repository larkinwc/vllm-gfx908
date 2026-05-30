# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MI100 (gfx908) Marlin-style W4A16 weight repack (OFFLINE / host-side).

This module performs the **offline** weight transformation that prepares a
GPTQ-packed int4 weight for a Marlin-style MFMA GEMM on gfx908. It does
*not* contain a Triton kernel and performs *no* HBM round-trip beyond the
single one-time repack at load. The GEMM kernel that consumes this layout
is implemented in a *separate* feature.

Input (matches the in-tree ``triton_w4a16`` kernel layout):
    b_q     : [K, N//8] int32  — GPTQ sequential N-packing. The int32 at
              ``b_q[k, n8]`` holds nibbles for output columns
              ``n8*8 + i`` (i in 0..7) at bit offset ``4*i``.
    scales  : [K//G, N] fp16/bf16
    qzeros  : [K//G, N//8] int32 (asymmetric only; same N-packing as b_q),
              or ``None`` for symmetric uint4b8 (zero == zp_bias == 8).

================================ OUTPUT LAYOUT CONTRACT ========================
``marlin_repack_w4a16`` returns a ``MarlinRepackedW4A16`` with:

  qweight : [K//8, N] int32   — **K-packed, N-lane-adjacent**.
            ``qweight[kb, n]`` packs the eight K-rows ``kb*8 + i`` (i in 0..7)
            for column ``n`` at bit offset ``4*i``:
                nibble(K = kb*8 + i, N = n) == (qweight[kb, n] >> (4*i)) & 0xF
            N is the contiguous (last) axis, so columns adjacent in the MFMA
            N-output dimension are adjacent in memory ("lane-adjacent"); the
            packing direction is K (the MFMA contraction dim), so a single
            int32 column-load yields an 8-deep K run for one N column.

  scale   : [K//G, N] (scales dtype)  — per-group scale, unchanged width,
            N-contiguous (one scalar per output column per group).

  zero    : [K//G, N] (scales dtype)  — **pre-fused** ``zero * scale``.
            The GEMM dequant becomes ``w_fp = q * scale - zero`` (note: the
            ``zero`` here is already multiplied by ``scale``), equivalent to
            the canonical ``(q - zero_raw) * scale``. For the symmetric path
            ``zero == zp_bias * scale`` (== ``8 * scale``).

  group_size, K, N, has_zp : metadata for the GEMM dispatcher.

Nibble ordering summary (3 lines):
  - qweight is [K//8, N] int32; bits [4i : 4i+4] of qweight[kb, n] == int4 of
    weight row (kb*8 + i), column n.
  - K is packed (contraction), N is the contiguous lane axis.
  - dequant: w_fp[k, n] = q[k, n] * scale[g, n] - zero[g, n], g = k // G.
===============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.kernels.configs.gfx908.config_loader import (
    load_config as _load_mi100_autotune_config,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_SUPPORTED_GROUP_SIZES = (32, 64, 128)


@dataclass
class MarlinRepackedW4A16:
    """Container for the offline-repacked W4A16 weight + pre-fused params.

    See the module docstring for the full layout contract.
    """

    qweight: torch.Tensor  # [K//8, N] int32, K-packed, N-lane-adjacent
    scale: torch.Tensor    # [K//G, N] scales dtype
    zero: torch.Tensor     # [K//G, N] scales dtype, pre-fused (zero_raw*scale)
    group_size: int
    K: int
    N: int
    has_zp: bool


def _shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(8, device=device, dtype=torch.int32) * 4


def _unpack_int4_along_n(packed_kn8: torch.Tensor) -> torch.Tensor:
    """[*, N//8] int32 (N-packed) -> [*, N] int32 nibbles in [0, 16)."""
    assert packed_kn8.dtype == torch.int32
    rows, n8 = packed_kn8.shape
    shifts = _shifts(packed_kn8.device)
    nibbles = (packed_kn8.unsqueeze(-1) >> shifts) & 0xF
    return nibbles.reshape(rows, n8 * 8)


def _pack_int4_along_k(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """[K, N] int32 nibbles -> [K//8, N] int32 (K-packed).

    ``out[kb, n]`` packs rows ``kb*8 + i`` at bit offset ``4*i``.
    """
    assert w_int4_kn.dtype == torch.int32
    K, N = w_int4_kn.shape
    assert K % 8 == 0, f"K={K} must be divisible by 8 for K-packing"
    shifts = _shifts(w_int4_kn.device)
    # [K//8, 8, N] -> shift along the size-8 axis -> reduce.
    grouped = w_int4_kn.view(K // 8, 8, N) & 0xF
    return torch.sum(
        grouped << shifts[None, :, None],
        dim=1,
        dtype=torch.int32,
    ).contiguous()


def unpack_repacked_to_kn(qweight_kb_n: torch.Tensor) -> torch.Tensor:
    """Logically UNPACK the repacked layout back to [K, N] int4.

    Inverse of the K-packing in :func:`marlin_repack_w4a16`. Used by the
    round-trip test and as the canonical reference for the GEMM feature.

    Args:
        qweight_kb_n: [K//8, N] int32 (K-packed, N-lane-adjacent).
    Returns:
        [K, N] int32 with values in [0, 16); ``out[kb*8 + i, n]`` is the
        nibble at bit offset ``4*i`` of ``qweight_kb_n[kb, n]``.
    """
    assert qweight_kb_n.dtype == torch.int32
    kb, N = qweight_kb_n.shape
    shifts = _shifts(qweight_kb_n.device)
    # [K//8, N, 8] nibbles -> [K//8, 8, N] -> [K, N]
    nibbles = (qweight_kb_n.unsqueeze(-1) >> shifts) & 0xF  # [kb, N, 8]
    nibbles = nibbles.permute(0, 2, 1).contiguous()          # [kb, 8, N]
    return nibbles.reshape(kb * 8, N)


def marlin_repack_w4a16(
    b_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
) -> MarlinRepackedW4A16:
    """Offline repack of GPTQ-packed int4 weights to the MI100 Marlin layout.

    Pure-tensor (torch ops only); no Triton kernel, no HBM round-trip beyond
    the one-time transform. Handles BOTH symmetric (``qzeros=None``,
    ``zp_bias=8``) and asymmetric (``qzeros`` present, ``HAS_ZP=True``).

    Args:
        b_q:        [K, N//8] int32, GPTQ N-packed int4 weights.
        scales:     [K//G, N] fp16/bf16 per-group scales.
        qzeros:     [K//G, N//8] int32 N-packed zeros, or None (symmetric).
        group_size: quant group size (must divide K).
        zp_bias:    constant zero for the symmetric path (default 8).

    Returns:
        :class:`MarlinRepackedW4A16` — see module docstring for the layout
        contract (shapes + nibble ordering).
    """
    assert b_q.dtype == torch.int32, f"b_q must be int32, got {b_q.dtype}"
    assert b_q.dim() == 2, f"b_q must be 2D [K, N//8], got {b_q.shape}"

    K, n8 = b_q.shape
    N = n8 * 8

    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"marlin_repack_w4a16: unsupported group_size={group_size}; "
            f"supported: {_SUPPORTED_GROUP_SIZES}"
        )
    assert K % group_size == 0, (
        f"K={K} not divisible by group_size={group_size}"
    )
    assert K % 8 == 0, f"K={K} must be divisible by 8 for K-packing"

    num_groups = K // group_size
    assert scales.shape == (num_groups, N), (
        f"scales shape mismatch: {scales.shape} vs ({num_groups}, {N})"
    )

    has_zp = qzeros is not None

    # ---- Weight: unpack N-packing, repack along K ----
    w_int4_kn = _unpack_int4_along_n(b_q)          # [K, N] int32
    qweight = _pack_int4_along_k(w_int4_kn)        # [K//8, N] int32

    # ---- Pre-fuse scales / zeros: w_fp = q*scale - (zero_raw*scale) ----
    scale_f32 = scales.to(torch.float32)
    if has_zp:
        assert qzeros.dtype == torch.int32, "qzeros must be int32"
        assert qzeros.shape == (num_groups, N // 8), (
            f"qzeros shape mismatch: {qzeros.shape} vs "
            f"({num_groups}, {N // 8})"
        )
        zero_raw = _unpack_int4_along_n(qzeros).to(torch.float32)  # [K//G, N]
    else:
        zero_raw = torch.full(
            (num_groups, N), float(zp_bias),
            dtype=torch.float32, device=scales.device,
        )

    zero_fused = (zero_raw * scale_f32).to(scales.dtype).contiguous()
    scale_out = scales.contiguous()

    return MarlinRepackedW4A16(
        qweight=qweight,
        scale=scale_out,
        zero=zero_fused,
        group_size=group_size,
        K=K,
        N=N,
        has_zp=has_zp,
    )


# ============================================================================
#  Marlin-style W4A16 GEMM (consumes the repacked layout above).
# ============================================================================
#
# Layout consumed (see MarlinRepackedW4A16 / module docstring):
#   qweight : [K//8, N] int32  — K-packed, N-lane-adjacent. The eight K-rows
#             ``kb*8 + i`` (i in 0..7) of column ``n`` live at bit offset
#             ``4*i`` of ``qweight[kb, n]``:
#                 q(K = kb*8 + i, N = n) == (qweight[kb, n] >> (4*i)) & 0xF
#   scale   : [K//G, N] (act dtype) — one scalar per output column per group.
#   zero    : [K//G, N] (act dtype) — PRE-FUSED (zero_raw * scale).
#   dequant : w_fp[k, n] = q[k, n] * scale[g, n] - zero[g, n],  g = k // G.
#
# The inner K-loop is SHIFT-ONLY: a BLOCK_K tile loads ``BLOCK_K//8`` packed
# int32 rows per N column (each int32 column-load reused across its 8 K-rows
# via a right-shift). No nibble-replication intrinsic is used anywhere, and
# the only weight-extraction op is a right-shift + mask. The sole output
# write is the ``c_ptr`` store. ``BLOCK_K`` is a multiple of 8 that divides
# ``group_size`` (clamped to ``group_size``), so a whole BLOCK_K tile maps to
# exactly one quant group.


@triton.jit
def _mi100_w4a16_marlin_gemm_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    group_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)

    # qweight is [K//8, N] int32, K-packed. Within a BLOCK_K tile the int32 row
    # for local k is ``k // 8`` and the int4 lives at bit offset ``(k % 8)*4``.
    # SHIFT-ONLY extraction: load the packed int32 (reused for its 8 K-rows),
    # then right-shift + mask. No nibble-replication intrinsic; no weight
    # store (only the output is stored).
    k_pack = offs_k // 8                       # [BLOCK_K] packed-row offset
    k_shift = (offs_k % 8) * 4                 # [BLOCK_K] nibble bit shift
    b_ptrs = b_ptr + (k_pack[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0)
        b_nibble = (b_packed >> k_shift[:, None]) & 0xF

        # One group per BLOCK_K tile (BLOCK_K divides group_size).
        g = (k * BLOCK_K) // group_size
        scales = tl.load(scales_ptr + g * stride_scales_g +
                         offs_bn * stride_scales_n)
        zeros = tl.load(zeros_ptr + g * stride_zeros_g +
                        offs_bn * stride_zeros_n)

        # Pre-fused dequant: w_fp = q * scale - zero  (zero == zero_raw*scale).
        b_fp = b_nibble.to(tl.float32) * scales[None, :].to(tl.float32) - \
            zeros[None, :].to(tl.float32)

        accumulator += tl.dot(a.to(tl.float32), b_fp)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _default_marlin_config(M: int) -> dict:
    """Heuristic fallback when no autotune JSON exists yet for the shape.

    Mirrors the legacy ``mi100_w4a16._default_block_sizes`` branches.
    ``num_stages`` is kept <= 2 (gfx908 has no async-LDS pipelining).
    """
    if M <= 16:
        return {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 4, "num_stages": 2}
    if M <= 64:
        return {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 4, "num_stages": 2}
    return {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32,
            "GROUP_M": 8, "num_warps": 4, "num_stages": 2}


def _select_marlin_config(M: int, N: int, K: int, group_size: int) -> dict:
    """Per-shape autotune lookup (key ``mi100_w4a16_marlin``), else heuristic.

    Reads ``configs/gfx908/mi100_w4a16_marlin_M*_N*_K*_g*.json`` via the
    shared gfx908 ``config_loader``; falls back to :func:`_default_marlin_config`
    when no JSON is pinned for the shape.
    """
    cfg = _load_mi100_autotune_config(
        "mi100_w4a16_marlin", M=M, N=N, K=K, group_size=group_size)
    if cfg is None:
        return _default_marlin_config(M)
    out = _default_marlin_config(M)
    out.update(cfg)
    # The loader schema uses GROUP_SIZE_M; map it onto our GROUP_M knob.
    if "GROUP_SIZE_M" in cfg:
        out["GROUP_M"] = int(cfg["GROUP_SIZE_M"])
    return out


def _resolve_block_k(cfg_block_k: int, group_size: int) -> int:
    """A valid BLOCK_K: multiple of 8, divides group_size, <= group_size."""
    block_k = min(int(cfg_block_k), group_size)
    block_k = max(8, (block_k // 8) * 8)
    while group_size % block_k != 0:
        block_k -= 8
    return block_k


def mi100_w4a16_marlin_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Marlin-style W4A16 GEMM: ``C[M,N] = A[M,K] @ dequant(W)[K,N]``.

    Consumes the offline-repacked layout produced by
    :func:`marlin_repack_w4a16` (pass the fields of the returned
    :class:`MarlinRepackedW4A16` directly).

    Args:
        a:          [M, K] activations (fp16/bf16).
        qweight:    [K//8, N] int32, K-packed N-lane-adjacent
                    (``MarlinRepackedW4A16.qweight``).
        scale:      [K//G, N] per-group scales
                    (``MarlinRepackedW4A16.scale``).
        zero:       [K//G, N] PRE-FUSED zero (``zero_raw * scale``)
                    (``MarlinRepackedW4A16.zero``).
        group_size: quant group size (must be in ``_SUPPORTED_GROUP_SIZES``).
        out_dtype:  output dtype (defaults to ``a.dtype``).
    Returns:
        [M, N] output tensor.
    """
    assert a.dim() == 2, f"a must be 2D [M, K], got {a.shape}"
    assert qweight.dim() == 2, f"qweight must be 2D, got {qweight.shape}"
    assert qweight.dtype == torch.int32, "qweight must be int32"

    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"mi100_w4a16_marlin_gemm: unsupported group_size={group_size}; "
            f"supported: {_SUPPORTED_GROUP_SIZES}"
        )

    M, K = a.shape
    N = scale.shape[1]
    assert qweight.shape == (K // 8, N), (
        f"qweight shape {tuple(qweight.shape)} != ({K // 8}, {N})"
    )
    assert K % group_size == 0, (
        f"K={K} not divisible by group_size={group_size}"
    )

    out_dtype = out_dtype or a.dtype
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    cfg = _select_marlin_config(M, N, K, group_size)
    block_k = _resolve_block_k(cfg["BLOCK_K"], group_size)
    num_stages = min(int(cfg.get("num_stages", 2)), 2)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _mi100_w4a16_marlin_gemm_kernel[grid](
        a, qweight, c, scale, zero,
        M, N, K,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scale.stride(0), scale.stride(1),
        zero.stride(0), zero.stride(1),
        group_size,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=block_k,
        GROUP_M=cfg["GROUP_M"],
        num_warps=cfg["num_warps"], num_stages=num_stages,
    )
    return c


def is_supported() -> bool:
    """True iff this layout's target kernel will run on the current platform.

    Mirrors the legacy ``mi100_w4a16.is_supported`` / ``on_mi100()`` guard.
    """
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_mi100
    except ImportError:
        return False
    return on_mi100()
