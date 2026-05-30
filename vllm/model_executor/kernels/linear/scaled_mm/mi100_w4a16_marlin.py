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

from vllm.platforms import current_platform

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
