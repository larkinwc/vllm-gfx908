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
    3. Per-row-reduces ``tl.max(tl.abs(y))`` across the H dimension to derive a
       per-token absmax scale ``scale = absmax / 127``.
    4. Emits ``int8(y / scale)`` plus the fp32 ``[M, 1]`` scale tensor in the
       same launch — no intermediate fp16 ``[M, H]`` write to HBM.

Why a multi-tile H reduction (this rework, see m1-kernel-rework feature)
-----------------------------------------------------------------------
The original single-tile design (commit aa358dbb9) used
``BLOCK_H = next_pow2(H)``. That works for Qwen3.5-9B's narrower MLP widths
(H ∈ {3584, 5120} round up to 4096 / 8192), but it FAILS for the wider
intermediates H ∈ {12544, 18944} which round up to 16384 / 32768:

  * Triton enforces a per-tile ``numel`` cap (2**20 elements). With
    BLOCK_M ≥ 32 the 2-D ``[BLOCK_M, BLOCK_H]`` tile exceeds the cap; shrinking
    BLOCK_M just trades the cap violation for catastrophic register pressure
    on the BLOCK_H-wide absmax reduction tree.
  * Empirically (m1-numerics-test session 6370d5b5) H=12544 never compiled
    within a 600 s cold-compile budget on Triton 3.5.1 + ROCm 7.12; the
    expected ``"BLOCK_H": "32768"`` cache artifact was never produced.

This rework keeps the public Python entry signature identical so the committed
M1 numerics test (commit 07ffbd1a8) re-runs unmodified, but replaces the
single-tile reduction with an in-kernel **sequential h-tile loop** sized to a
small BLOCK_H (≤ 1024) and performs the work in two sweeps over h-tiles:

  Sweep 1 (absmax accumulation): load gate/up tile, compute silu(gate)*up in
      fp32, take ``tl.max(tl.abs(.))`` along the h-axis, fold into a per-row
      running max held in registers across the h-tile loop.
  Sweep 2 (quantize + store): for each h-tile, recompute silu(gate)*up (so we
      never materialize a full fp16 ``[BLOCK_M, H]`` intermediate), apply the
      now-known per-row scale, round-half-away-from-zero, clamp to int8 range,
      and store.

Recomputing silu*up is cheap relative to HBM traffic on gfx908 (compute-bound
math vs memory-bound HBM bytes — the original PR's whole point is to remove an
HBM round-trip). The per-tile numel stays well under 2**20 for the worst-case
(BLOCK_M=128, BLOCK_H=1024 → 128k elements ≪ 1M), and the per-row reduction
tree is small enough (≤ 1024 lanes) that cold-compile times stay in the
single-digit-seconds range on Triton 3.5.1 / ROCm 7.12.

Tile-size / num_warps / num_stages selection mirrors the pattern used by
``vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py`` (BLOCK_M=64
default, with smaller M-tiles for decode-like shapes and BLOCK_M=128 for
prefill-like ones). Autotune is intentionally NOT used here — the static
heuristic compiles 12 (M, H) cells in <60 s cold each, whereas an autotune
sweep over 4 BLOCK_M × N BLOCK_H choices would blow the worker wall-clock
budget for this milestone.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# Per-token int8 symmetric quantization range. INT8 is signed [-128, 127];
# using 127.0 leaves the asymmetric ``-128`` slot unused so the scale is
# symmetric and matches the existing ``ops.scaled_int8_quant`` convention.
_INT8_QMAX = 127.0

# Static cap on BLOCK_H. The kernel sweeps h in steps of BLOCK_H, so the
# per-program ``[BLOCK_M, BLOCK_H]`` tile stays small (well under Triton's
# 2**20 numel cap) regardless of the model's H. See module docstring for why
# this cap exists.
_BLOCK_H_DEFAULT = 1024


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
    """Per-row fused silu_and_mul + INT8 absmax quant (multi-tile H reduction).

    Grid: ``(triton.cdiv(M, BLOCK_M),)`` — one program per row tile. Each
    program walks the H axis in steps of ``BLOCK_H`` twice:

      1. First sweep accumulates a per-row absmax over the h-tiles.
      2. Second sweep recomputes ``silu(gate)*up`` per h-tile, divides by the
         (now-known) scale, rounds, clamps, and stores int8.

    Keeping BLOCK_H small (≤ ~1024) bounds the per-tile numel and the
    per-row reduction tree so the kernel cold-compiles in seconds on
    Triton 3.5.1 / ROCm 7.12 even for the widest H ∈ {12544, 18944}.
    """
    pid_m = tl.program_id(axis=0)

    # --- row offsets within the tile ---------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask_m = offs_m < M

    num_h_tiles = tl.cdiv(H, BLOCK_H)

    # --- Sweep 1: per-row absmax across all h-tiles ------------------------
    # Hold a per-row running absmax in a small register vector of shape
    # [BLOCK_M]. Initialized to 0; ``tl.maximum`` against tile-local absmax
    # folds the running value across h-tiles.
    absmax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for h_tile in range(0, num_h_tiles):
        offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
        mask_h = offs_h < H
        mask_2d = mask_m[:, None] & mask_h[None, :]

        gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh
        up_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + (offs_h[None, :] + H) * stride_xh
        )

        gate = tl.load(gate_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

        # SiLU is x * sigmoid(x); compute in fp32 for numerical fidelity then
        # multiply by the up projection. Out-of-bounds H lanes were loaded as
        # 0.0 above so they do not skew the absmax reduction.
        y = (gate * tl.sigmoid(gate)) * up

        tile_absmax = tl.max(tl.abs(y), axis=-1)  # [BLOCK_M]
        absmax = tl.maximum(absmax, tile_absmax)

    # Clamp tiny absmax to avoid division by zero on all-zero rows; the
    # resulting int8 output is zero either way.
    absmax = tl.maximum(absmax, 1e-12)
    scale = absmax / 127.0  # fp32 [BLOCK_M]
    inv_scale = 127.0 / absmax  # fp32 [BLOCK_M]

    # --- Sweep 2: recompute silu*up per h-tile, quantize, and store --------
    for h_tile in range(0, num_h_tiles):
        offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
        mask_h = offs_h < H
        mask_2d = mask_m[:, None] & mask_h[None, :]

        gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh
        up_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + (offs_h[None, :] + H) * stride_xh
        )

        gate = tl.load(gate_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

        y = (gate * tl.sigmoid(gate)) * up

        # Quantize with the per-row scale derived in sweep 1.
        q = y * inv_scale[:, None]
        # Round-half-away-from-zero (matches scaled_int8_quant in csrc).
        q = tl.where(q >= 0.0, q + 0.5, q - 0.5)
        # Clamp to int8 range before cast; protects against rounding overshoot
        # at the boundary (e.g. 127.5 -> 128 would wrap to -128 without the
        # clamp).
        q = tl.maximum(tl.minimum(q, 127.0), -127.0)
        q_i8 = q.to(tl.int8)

        q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_h[None, :] * stride_qh
        tl.store(q_ptrs, q_i8, mask=mask_2d)

    # --- per-row scale store -----------------------------------------------
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

    # Mirror the BLOCK_M default from mi100_int8.py with the same
    # decode-like / prefill-like split. BLOCK_M=64 default; smaller for tiny
    # M, larger for prefill batches.
    next_pow2_m = _next_power_of_2(max(1, M))
    if next_pow2_m <= 16:
        block_m = 16
    elif next_pow2_m <= 32:
        block_m = 32
    elif next_pow2_m <= 64:
        block_m = 64
    else:
        block_m = 128

    # Static BLOCK_H cap (see module docstring): keeps per-program numel and
    # the per-row reduction tree small enough for sub-60s cold compile across
    # the full Qwen3.5-9B H grid {3584, 5120, 12544, 18944}. For very small H
    # (e.g. test cases), round down to next_pow2(H) to avoid a single h-tile
    # holding more masked-off lanes than live ones.
    block_h = min(_BLOCK_H_DEFAULT, _next_power_of_2(H))
    block_h = max(block_h, 32)

    # Launch heuristics mirror mi100_int8.py: 2 warps for small reductions,
    # 4 warps once the tile reaches a full wavefront-64 worth of work.
    num_warps = 2 if block_h <= 256 else 4
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
