# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MI100 (gfx908) fused SiLU-and-mul + per-token INT8 quant Triton kernel.

Issue #33 — eliminates one HBM round-trip per MLP layer on the W8A8 path.

Today's unfused composition is:
    silu_and_mul(x)            # csrc; writes fp16 [M, H] to HBM
    scaled_int8_quant(y)       # csrc; reads fp16, writes int8 [M, H] + fp32 scale

This module replaces those two passes with a single Triton kernel that:
    1. Loads BLOCK_M x BLOCK_H tiles of the fp16 input ``x`` of shape
       ``[M, 2*H]`` (silu_and_mul layout: ``x[:, :H]`` is the gate, ``x[:, H:]``
       is the up projection).
    2. Computes ``y = silu(gate) * up`` row-wise.
    3. Per-row-reduces ``tl.max(tl.abs(y), axis=-1)`` across the H dimension to
       derive a per-token absmax scale ``scale = absmax / 127``.
    4. Emits ``int8(y / scale)`` plus the fp32 ``[M, 1]`` scale tensor in the
       same launch — no intermediate fp16 ``[M, H]`` write to HBM.

The Python entry point :func:`fused_silu_quant_int8` is unconditionally
available; the env-gated default-on / disable-path wiring into the dispatcher
is handled separately by the ``m1-wire-in`` feature.

Tile-size / num_warps / num_stages selection mirrors the pattern used by
``vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py`` (BLOCK_M=64
default, with smaller M-tiles for decode-like shapes).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# Per-token int8 symmetric quantization range. INT8 is signed [-128, 127];
# using 127.0 leaves the asymmetric ``-128`` slot unused so the scale is
# symmetric and matches the existing ``ops.scaled_int8_quant`` convention.
# Inlined as a literal in the @triton.jit kernel because Triton requires
# kernel-side globals to be ``tl.constexpr``-wrapped at module load (which is
# not stable on Triton 3.5.1 / ROCm); the Python launcher uses ``_INT8_QMAX``.
_INT8_QMAX = 127.0


@triton.jit
def _fused_silu_quant_int8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    H,
    stride_xm,
    stride_xh,
    stride_qm,
    stride_qh,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-row fused silu_and_mul + INT8 absmax quant.

    Grid: ``(triton.cdiv(M, BLOCK_M),)`` — one program per row tile. ``BLOCK_H``
    is sized to cover the full hidden dim ``H`` in a single tile (Qwen3.5-9B
    intermediate widths are powers-of-two-rounded ≤ 32k, well within Triton's
    addressable tile budget on gfx908).

    The per-row reduction over ``BLOCK_H`` runs entirely in registers / LDS;
    no HBM round-trip happens between ``silu(gate) * up`` and the int8 store.
    """
    pid_m = tl.program_id(axis=0)

    # --- row offsets within the tile ---------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask_m = offs_m < M

    # --- column offsets for the gate / up halves of the fp16 input ----------
    offs_h = tl.arange(0, BLOCK_H).to(tl.int64)
    mask_h = offs_h < H
    mask_2d = mask_m[:, None] & mask_h[None, :]

    # x has shape [M, 2*H]; gate = x[:, :H], up = x[:, H:].
    gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh
    up_ptrs = x_ptr + offs_m[:, None] * stride_xm + (offs_h[None, :] + H) * stride_xh

    gate = tl.load(gate_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    # SiLU is x * sigmoid(x); compute in fp32 for numerical fidelity then
    # multiply by the up projection.
    y = (gate * tl.sigmoid(gate)) * up

    # --- per-row absmax reduction -> per-token scale -----------------------
    # Out-of-bounds H lanes were loaded as 0.0 above so they do not skew the
    # absmax reduction.
    absmax = tl.max(tl.abs(y), axis=-1)
    # Clamp tiny absmax to avoid division by zero on all-zero rows; the
    # resulting int8 output is zero either way.
    absmax = tl.maximum(absmax, 1e-12)
    scale = absmax / 127.0  # fp32 [BLOCK_M]

    # Quantize: int8 = round(y / scale). Cast from fp32 -> int8 below uses
    # the LLVM-default rounding mode (truncate toward zero on ROCm), so we
    # round explicitly via ``+0.5 * sign(x)`` before the clamp+cast.
    inv_scale = 127.0 / absmax  # fp32 [BLOCK_M]
    q = y * inv_scale[:, None]
    # Round-to-nearest, away from zero (matches scaled_int8_quant in csrc).
    q = tl.where(q >= 0.0, q + 0.5, q - 0.5)
    # Clamp to int8 range before cast; protects against rounding overshoot at
    # the boundary (e.g. 127.5 -> 128 would wrap to -128 without the clamp).
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)
    q_i8 = q.to(tl.int8)

    # --- stores ------------------------------------------------------------
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_h[None, :] * stride_qh
    tl.store(q_ptrs, q_i8, mask=mask_2d)

    # Per-row scale: shape [M, 1]; store via flat index along M.
    s_ptrs = s_ptr + offs_m
    tl.store(s_ptrs, scale, mask=mask_m)


def _next_power_of_2(x: int) -> int:
    """Smallest power-of-two ≥ x; matches ``triton.next_power_of_2`` semantics
    without requiring the helper to be importable at module load time."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def fused_silu_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU-and-mul + per-token INT8 absmax quantization.

    Args:
        x: fp16 tensor of shape ``[M, 2*H]`` in silu_and_mul layout. The first
            half along the last dim is silu-activated; the second half is the
            multiplicative gate.

    Returns:
        Tuple ``(q, scale)`` where ``q`` is an ``int8`` tensor of shape
        ``[M, H]`` and ``scale`` is an ``fp32`` tensor of shape ``[M, 1]``
        carrying the per-row absmax scale (i.e. ``y ≈ q * scale``).

    Mirrors the per-token output contract of
    ``vllm._custom_ops.scaled_int8_quant`` so the dispatcher can swap this
    fused path in for the existing ``silu_and_mul`` + ``scaled_int8_quant``
    composition without any downstream call-site changes.
    """
    assert x.is_cuda, "fused_silu_quant_int8 requires a CUDA/ROCm tensor."
    assert x.dim() == 2, f"expected 2-D input [M, 2*H], got shape {tuple(x.shape)}."
    assert x.dtype == torch.float16, (
        f"fused_silu_quant_int8 expects fp16 input, got {x.dtype}."
    )
    M, two_h = x.shape
    assert two_h % 2 == 0, (
        f"input last dim must be even (silu_and_mul layout), got {two_h}."
    )
    H = two_h // 2

    # Contiguous input keeps the gate/up halves stride-friendly; the kernel
    # also handles non-contiguous strides explicitly via stride_xm / stride_xh
    # but contiguous is the hot-path that vLLM activations produce.
    if not x.is_contiguous():
        x = x.contiguous()

    q = torch.empty((M, H), dtype=torch.int8, device=x.device)
    scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)

    # Mirror the BLOCK_M=64 default from mi100_int8.py with the same
    # decode-like / prefill-like split. BLOCK_H is sized to cover the full
    # hidden dim in one tile (Qwen3.5-9B widths are ≤ 18944).
    block_m_default = 64
    next_pow2_m = _next_power_of_2(max(1, M))
    if next_pow2_m <= 16:
        block_m = 16
    elif next_pow2_m <= 32:
        block_m = 32
    elif next_pow2_m <= 64:
        block_m = block_m_default
    else:
        block_m = 128

    block_h = _next_power_of_2(H)
    # Cap BLOCK_H so we don't blow the register budget for the per-row
    # reduction on pathologically wide intermediates; rows larger than the cap
    # would need a multi-tile reduction, which is intentionally out of scope
    # for this kernel — Qwen3.5-9B's widest MLP intermediate is 18944, which
    # rounds up to 32768 and fits comfortably in a single tile on gfx908.
    block_h = max(block_h, 32)

    # Launch heuristics mirror mi100_int8.py: 2 warps for small reductions,
    # 4 warps once the tile reaches a full wavefront-64 worth of work.
    num_warps = 2 if block_h <= 1024 else 4
    num_stages = 2

    grid = (triton.cdiv(M, block_m),)
    _fused_silu_quant_int8_kernel[grid](
        x,
        q,
        scale,
        M,
        H,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return q, scale
