# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx900 (Vega10) int4 W4A16 decode GEMV.

gfx900 has NO MFMA units, so the generic W4A16 path (which uses ``tl.dot`` even
at M=1) runs a padded FP32 GEMM and is *slower* than FP16. For the decode hot
path (small M) a GEMV is the right primitive: it is purely HBM-bandwidth-bound,
needs no matrix units, and int4 weights move 4x fewer bytes than FP16.

Microbench (Qwen3.5-9B MLP shapes, M=1, vs FP16 GEMV):
    gate/up K=4096  N=24576 : 2.82x
    down    K=12288 N=4096  : 4.02x

Packing matches the generic ``triton_w4a16`` kernel: GPTQ-sequential, i.e. column
``j`` of an unpacked int32 uses nibble shift ``(j % 8) * 4``. ``b_q`` is
``[K, N//8] int32``; ``scales`` ``[K//G, N]``; ``qzeros`` ``[K//G, N//8] int32``
or ``None`` for symmetric (uses ``zp_bias``).
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _w4a16_gemv_splitk_kernel(
    a_ptr,            # [M, K] fp16/bf16
    qw_ptr,           # [K, N//8] int32
    scales_ptr,       # [K//G, N]
    qz_ptr,           # [K//G, N//8] int32 (unused if HAS_ZP=0)
    c_ptr,            # [M, N] fp32 accumulator (atomic)
    M, K, N,
    ZP_BIAS,
    G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS_PER_PID: tl.constexpr,
    HAS_ZP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n8 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    mask_n8 = offs_n8 < (N // 8)

    # GPTQ-sequential shifts: column j -> (j % 8) * 4
    j = tl.arange(0, BLOCK_N) % 8
    shifts = j * 4
    shifts2d = shifts[None, :]

    g0 = pid_k * GROUPS_PER_PID
    for gi in range(0, GROUPS_PER_PID):
        g = g0 + gi
        in_range = g < (K // G)
        s_ptrs = scales_ptr + g * N + offs_n
        scales = tl.load(s_ptrs, mask=mask_n & in_range, other=0.0).to(tl.float32)
        if HAS_ZP:
            qz_ptrs = qz_ptr + g * (N // 8) + offs_n8
            qzv = tl.load(qz_ptrs, mask=mask_n8 & in_range, other=0)
            ze = tl.interleave(qzv, qzv); ze = tl.interleave(ze, ze); ze = tl.interleave(ze, ze)
            zf = ((ze >> shifts) & 0xF).to(tl.float32)
        else:
            zf = tl.full((BLOCK_N,), ZP_BIAS, dtype=tl.float32)

        offs_k = g * G + tl.arange(0, G)
        mask_k = offs_k < K
        qw_ptrs = qw_ptr + offs_k[:, None] * (N // 8) + offs_n8[None, :]
        m_w = (mask_k[:, None] & in_range) & mask_n8[None, :]
        qwt = tl.load(qw_ptrs, mask=m_w, other=0)
        bt = tl.interleave(qwt, qwt); bt = tl.interleave(bt, bt); bt = tl.interleave(bt, bt)
        bt = (bt >> shifts2d) & 0xF
        wt = (bt.to(tl.float32) - zf[None, :]) * scales[None, :]   # [G, BLOCK_N]

        # M rows (decode: M is tiny, 1..8). Unrolled small loop.
        for m in range(0, M):
            a = tl.load(a_ptr + m * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            acc = tl.sum(a[:, None] * wt, axis=0)                  # [BLOCK_N]
            tl.atomic_add(c_ptr + m * N + offs_n, acc, mask=mask_n)


# Tuned per-shape configs for gfx900 (from bench_scripts/gemv_bench.py sweeps).
# Key: (N) -> (BLOCK_N, num_warps, split_k, num_stages). split_k=0 => full (n_groups).
_GFX900_GEMV_CONFIGS = {
    24576: (256, 2, 32, 2),   # gate/up
    4096: (128, 1, 0, 3),     # down  (split_k=full)
}
_DEFAULT_CFG = (128, 2, 16, 2)


def w4a16_gemv_gfx900(
    a: torch.Tensor,          # [M, K]
    b_q: torch.Tensor,        # [K, N//8] int32
    scales: torch.Tensor,     # [K//G, N]
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 0,
) -> torch.Tensor:
    M, K = a.shape
    N = b_q.shape[1] * 8
    n_groups = K // group_size
    BLOCK_N, num_warps, split_k, num_stages = _GFX900_GEMV_CONFIGS.get(N, _DEFAULT_CFG)
    if split_k == 0 or split_k > n_groups:
        split_k = n_groups
    gpp = triton.cdiv(n_groups, split_k)
    c = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(n_groups, gpp))
    _w4a16_gemv_splitk_kernel[grid](
        a, b_q, scales, qzeros if qzeros is not None else b_q, c,
        M, K, N, float(zp_bias),
        G=group_size, BLOCK_N=BLOCK_N, GROUPS_PER_PID=gpp,
        HAS_ZP=(qzeros is not None),
        num_warps=num_warps, num_stages=num_stages,
    )
    return c.to(a.dtype)
