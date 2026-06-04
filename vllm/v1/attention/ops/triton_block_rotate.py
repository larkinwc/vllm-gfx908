# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused block-diagonal rotation for the RotorQuant TurboQuant presets.

Replaces the dense ``y = x @ PiT`` GEMM (O(N·D²), cuBLAS) used by the Hadamard
path with an in-kernel block rotation (O(N·D)). Each row is rotated by D/2
independent 2×2 Givens blocks (``planar``) or D/4 independent 4×4 quaternion
blocks (``iso``). The kernel is pure elementwise arithmetic over strided loads:
no warp shuffles, so it is correct on wave64 hardware (CDNA / gfx908) where the
``__shfl_sync(width=32)`` codebook trick used by other ports breaks.

Forward map matches ``rotations.block_rotate`` exactly:
    planar : y[2i]   =  c[i]*x[2i] + s[i]*x[2i+1]
             y[2i+1] = -s[i]*x[2i] + c[i]*x[2i+1]
    iso    : y = m_i @ x_block   (m_i the 4×4 rotation for quaternion q_i)
"""

import torch

from vllm.triton_utils import tl, triton

# Row-count crossover (rows = tokens × heads). Below this the dense D×D rocBLAS
# GEMM beats the fused kernel because the fused launch has a fixed ~40µs floor
# on ROCm/gfx908 while the small GEMM dispatches in ~15µs. Above it the O(D)
# fused rotation wins (measured ~1.7× at 16k rows, ~3.5× at 64k). Decode query
# rotation (rows = B×Hq) stays on the GEMM; prefill/store keys cross over.
ROTATE_FUSED_MIN_ROWS = 8192


def should_use_fused_rotation(num_rows: int) -> bool:
    return num_rows >= ROTATE_FUSED_MIN_ROWS


@triton.jit
def _block_rotate_planar_kernel(
    X_ptr,  # [N, D] float32
    Cos_ptr,  # [D/2] float32
    Sin_ptr,  # [D/2] float32
    Y_ptr,  # [N, D] float32
    stride_xn,
    stride_yn,
    D: tl.constexpr,
    HALF: tl.constexpr,  # D // 2
    BLOCK_H: tl.constexpr,  # next_pow2(D//2)
):
    n = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    hm = h < HALF
    xb = X_ptr + n * stride_xn
    x_even = tl.load(xb + 2 * h, mask=hm, other=0.0)
    x_odd = tl.load(xb + 2 * h + 1, mask=hm, other=0.0)
    c = tl.load(Cos_ptr + h, mask=hm, other=0.0)
    s = tl.load(Sin_ptr + h, mask=hm, other=0.0)
    y_even = c * x_even + s * x_odd
    y_odd = -s * x_even + c * x_odd
    yb = Y_ptr + n * stride_yn
    tl.store(yb + 2 * h, y_even, mask=hm)
    tl.store(yb + 2 * h + 1, y_odd, mask=hm)


@triton.jit
def _block_rotate_iso_kernel(
    X_ptr,  # [N, D] float32
    Quat_ptr,  # [D/4, 4] float32  (w, x, y, z)
    Y_ptr,  # [N, D] float32
    stride_xn,
    stride_yn,
    stride_q,  # row stride of Quat (=4)
    D: tl.constexpr,
    QUARTER: tl.constexpr,  # D // 4
    BLOCK_Q: tl.constexpr,  # next_pow2(D//4)
):
    n = tl.program_id(0)
    q = tl.arange(0, BLOCK_Q)
    qm = q < QUARTER
    xb = X_ptr + n * stride_xn
    a = tl.load(xb + 4 * q + 0, mask=qm, other=0.0)
    b = tl.load(xb + 4 * q + 1, mask=qm, other=0.0)
    c = tl.load(xb + 4 * q + 2, mask=qm, other=0.0)
    e = tl.load(xb + 4 * q + 3, mask=qm, other=0.0)
    w = tl.load(Quat_ptr + q * stride_q + 0, mask=qm, other=0.0)
    xx = tl.load(Quat_ptr + q * stride_q + 1, mask=qm, other=0.0)
    yy = tl.load(Quat_ptr + q * stride_q + 2, mask=qm, other=0.0)
    zz = tl.load(Quat_ptr + q * stride_q + 3, mask=qm, other=0.0)
    # y = m @ x_block, m the left-isoclinic matrix of the unit quaternion.
    y0 = w * a - xx * b - yy * c - zz * e
    y1 = xx * a + w * b - zz * c + yy * e
    y2 = yy * a + zz * b + w * c - xx * e
    y3 = zz * a - yy * b + xx * c + w * e
    yb = Y_ptr + n * stride_yn
    tl.store(yb + 4 * q + 0, y0, mask=qm)
    tl.store(yb + 4 * q + 1, y1, mask=qm)
    tl.store(yb + 4 * q + 2, y2, mask=qm)
    tl.store(yb + 4 * q + 3, y3, mask=qm)


def triton_block_rotate(x: torch.Tensor, params) -> torch.Tensor:
    """Fused ``y = x @ PiT`` for a block-diagonal rotation.

    ``x`` is ``[..., D]`` float32 (contiguous in the last dim). ``params`` is a
    ``RotationParams`` (planar or iso). Returns a contiguous float32 tensor of
    the same shape.
    """
    kind = params.kind
    *lead, D = x.shape
    x2d = x.reshape(-1, D).contiguous()
    N = x2d.shape[0]
    y = torch.empty_like(x2d)

    if kind == "planar":
        HALF = D // 2
        BLOCK_H = triton.next_power_of_2(HALF)
        _block_rotate_planar_kernel[(N,)](
            x2d,
            params.cos,
            params.sin,
            y,
            x2d.stride(0),
            y.stride(0),
            D=D,
            HALF=HALF,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )
    elif kind == "iso":
        QUARTER = D // 4
        BLOCK_Q = triton.next_power_of_2(QUARTER)
        _block_rotate_iso_kernel[(N,)](
            x2d,
            params.quat,
            y,
            x2d.stride(0),
            y.stride(0),
            params.quat.stride(0),
            D=D,
            QUARTER=QUARTER,
            BLOCK_Q=BLOCK_Q,
            num_warps=4,
        )
    else:
        raise ValueError(f"triton_block_rotate does not support kind {kind!r}")

    return y.reshape(*lead, D)
