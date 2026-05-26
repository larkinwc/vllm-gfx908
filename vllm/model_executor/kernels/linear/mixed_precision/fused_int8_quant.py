# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton per-token symmetric int8 dynamic quantization.

Consumed by the W4A8 / W8A8 linear paths that need activations quantized
once-per-token before an INT8 MFMA dot. The Python-wrapper version
(``x.abs().amax().clamp().div().round().clamp_().to(int8)``) issues ~6
eager kernel launches per activation, which costs ~100 us *per call*
regardless of shape — compounding to ~18 ms per decode step on a 62-layer
model with 4 linears per block.

This kernel folds the reduction, scale, round-to-int, and clamp into a
single launch. Parameters are tuned for gfx908 (MI100) — see the
`num_stages=2` and `num_warps=4` defaults — but the kernel is portable.
"""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.configs.gfx908.config_loader import (
    load_fused_act_config as _load_fused_act_config,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Upper bound on the per-row block along the hidden dim. Covers all typical
# transformer hidden sizes (<= 8192) in a single-tile pass. Rows with N above
# this bound fall back to the reference path (see `fused_per_token_quant_int8`).
_MAX_BLOCK_N = 8192


if current_platform.is_rocm():

    @triton.jit
    def _round_to_int8(x):
        return tl.extra.hip.libdevice.round(x).to(tl.int8)

elif current_platform.is_xpu():

    @triton.jit
    def _round_to_int8(x):
        return tl.extra.intel.libdevice.round(x).to(tl.int8)

else:

    @triton.jit
    def _round_to_int8(x):
        return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _fused_int8_quant_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_qm,
    stride_qn,
    EPS: tl.constexpr,
    INV_INT8_MAX: tl.constexpr,  # 1.0 / 127.0
    INT8_MAX: tl.constexpr,  # 127.0
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    mask2d = row_mask[:, None] & col_mask[None, :]

    x_offs = rows[:, None] * stride_xm + cols[None, :] * stride_xn
    x = tl.load(x_ptr + x_offs, mask=mask2d, other=0.0).to(tl.float32)

    absmax = tl.max(tl.abs(x), axis=1)
    absmax = tl.maximum(absmax, EPS)

    # scale = absmax / 127; inv_scale = 127 / absmax (reuse the division)
    inv_scale = INT8_MAX / absmax
    scale = absmax * INV_INT8_MAX

    x_q = _round_to_int8(x * inv_scale[:, None])

    q_offs = rows[:, None] * stride_qm + cols[None, :] * stride_qn
    tl.store(xq_ptr + q_offs, x_q, mask=mask2d)
    tl.store(scale_ptr + rows, scale, mask=row_mask)


def _pick_num_warps(block_n: int) -> int:
    # On gfx908 the wavefront is 64 lanes; 4 warps = 256 lanes is the sweet
    # spot for single-row reductions up through N=8192.
    if block_n <= 512:
        return 2
    if block_n <= 2048:
        return 4
    return 8


def fused_per_token_quant_int8(
    x: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 quantization: ``x_i8 = round(x * 127 / amax)``.

    Args:
        x: Activations of shape ``(..., N)`` in fp16/bf16/fp32.
        eps: Floor applied to per-row absmax to avoid divide-by-zero on
            all-zero rows.

    Returns:
        ``(x_q, scales)`` where ``x_q`` has the same shape as ``x`` in int8
        and ``scales`` has shape ``(..., 1)`` in fp32. ``scale = absmax / 127``
        matches the dequant convention of ``_custom_ops.scaled_int8_quant``
        and ``per_token_quant_int8`` in ``int8_utils.py``.
    """
    assert x.is_cuda, "fused int8 quant kernel is cuda/rocm only"
    original_shape = x.shape
    N = original_shape[-1]

    if not x.is_contiguous():
        x = x.contiguous()

    x_2d = x.view(-1, N)
    M = x_2d.shape[0]

    x_q = torch.empty_like(x_2d, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)

    if M == 0:
        return x_q.view(*original_shape), scales.view(*original_shape[:-1], 1)

    block_n = triton.next_power_of_2(N)
    if block_n > _MAX_BLOCK_N:
        # Large hidden sizes are uncommon for this kernel's callers; fall back
        # to the reference path rather than paying the multi-pass reduction
        # complexity cost.
        absmax = x_2d.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        absmax = absmax.clamp_(min=eps)
        scales_ref = absmax / 127.0
        x_q = (x_2d / scales_ref).round().clamp_(-127, 127).to(torch.int8)
        return (
            x_q.view(*original_shape),
            scales_ref.view(*original_shape[:-1], 1),
        )

    # Prefer the per-shape pinned config when present
    # (vllm/model_executor/kernels/configs/gfx908/fused_int8_quant_M<M>_H<N>.json).
    # The fused_int8_quant kernel uses ``N`` as the per-row reduction width,
    # which the loader keys on as ``H`` to match the fused-activation
    # naming convention. Missing-file lookups return ``None`` and the
    # caller falls back to the static heuristic below — preserving the
    # existing pre-M3 behaviour byte-identically.
    pinned_cfg = _load_fused_act_config("fused_int8_quant", M=M, H=N)
    if pinned_cfg is not None:
        block_m = int(pinned_cfg["BLOCK_M"])
        # The pinned config may legitimately request a smaller BLOCK_N
        # than the next_power_of_2(N) heuristic; honor it as long as it
        # still covers N (the kernel's single-tile reduction requires
        # BLOCK_N >= N).
        pinned_block_n = int(pinned_cfg.get("BLOCK_N", block_n))
        if pinned_block_n >= N:
            block_n = pinned_block_n
        num_warps = int(pinned_cfg.get("num_warps", _pick_num_warps(block_n)))
        num_stages = int(pinned_cfg.get("num_stages", 1))
    else:
        # One program per row. MI100 has 120 CUs, and with M tokens we want
        # at least that many programs to fill the machine. Multi-row tiling
        # was tried and hurts — parallelism matters more than per-program
        # work here because the kernel is already bandwidth-bound.
        block_m = 1
        num_warps = _pick_num_warps(block_n)
        # This kernel has no K-dim loop (single-tile reduction across the
        # hidden dim), so num_stages=1 avoids the pipeline prologue cost.
        # The "gfx908 needs num_stages=2" rule applies to kernels that *do*
        # loop over K.
        num_stages = 1

    grid = (triton.cdiv(M, block_m),)
    _fused_int8_quant_kernel[grid](
        x_2d,
        x_q,
        scales,
        M,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        x_q.stride(0),
        x_q.stride(1),
        EPS=eps,
        INV_INT8_MAX=1.0 / 127.0,
        INT8_MAX=127.0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q.view(*original_shape), scales.view(*original_shape[:-1], 1)
