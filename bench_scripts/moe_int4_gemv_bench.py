#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 int4 W4A16 MoE-GEMV microbench + correctness (issue #58 loop).

Target: the M=1 (decode) int4 expert GEMM in fused_moe_kernel_gptq_awq, which
uses tl.dot. On gfx908, tl.dot needs BLOCK_M>=16, so 1 real token is padded to
16 (15/16 MFMA lanes wasted) and the GEMM only yields ~80 tiles on 120 CUs.

This script builds a hand GEMV (tl.sum reduction over K, no tl.dot) for ONE
expert's two projections (gate/up: [2N,K], down: [N->H]) at the real
Qwen3-Coder-Next shape, and:
  1. Establishes correctness vs a torch fp16 reference (rel-err <= 1e-3).
  2. Times it vs the reference, reports us AND effective GB/s (distance to the
     ~1.2 TB/s MI100 HBM roofline).

Packing (matches fused_moe_kernel_gptq_awq, use_int4_w4a16, no zero-point):
  B: uint8 [N, K//2]; element (n, k) nibble = (byte >> ((k%2)*4)) & 0xF
     low nibble = even k, high nibble = odd k.
  scale: fp16 [N, K//group_size]; dequant w = (nibble - 8) * scale.

Run:
  PYTHONPATH=. HIP_VISIBLE_DEVICES=2 python bench_scripts/moe_int4_gemv_bench.py
"""

import os
import time

import torch

from vllm.triton_utils import tl, triton

HBM_BW = 1.2e12  # MI100 ~1.2 TB/s


# --------------------------------------------------------------------------- #
# Reference: dequantize int4 -> fp16, then matvec.  Defines ground truth.
# --------------------------------------------------------------------------- #
def dequant_ref(b_packed: torch.Tensor, scale: torch.Tensor, K: int, gs: int):
    """b_packed: uint8 [N, K//2]; scale: fp16 [N, K//gs] -> w fp16 [N, K]."""
    N = b_packed.shape[0]
    low = (b_packed & 0xF).to(torch.int32)  # even k
    high = ((b_packed >> 4) & 0xF).to(torch.int32)  # odd k
    w = torch.empty((N, K), dtype=torch.int32, device=b_packed.device)
    w[:, 0::2] = low
    w[:, 1::2] = high
    w = (w - 8).to(torch.float16)
    # broadcast group scale over K
    scale_full = scale.repeat_interleave(gs, dim=1)[:, :K]
    return w * scale_full


def matvec_ref(x: torch.Tensor, w_fp16: torch.Tensor):
    # x: [K] fp16; w: [N, K] fp16 -> [N] fp32 accumulate
    return w_fp16.to(torch.float32) @ x.to(torch.float32)


# --------------------------------------------------------------------------- #
# Hand GEMV kernel v1: one program per (N-tile), tl.sum reduction over K.
# No tl.dot. Each program loads x[K] once (cached) and reduces.
# --------------------------------------------------------------------------- #
@triton.jit
def int4_gemv_kernel(
    x_ptr,
    b_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N,
    gs: tl.constexpr,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # iterate K in BLOCK_K chunks; B is packed 2-per-byte along K.
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # load x[BLOCK_K]
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        # packed byte index along K = k//2; nibble select = (k%2)*4
        bk = offs_k // 2
        shift = (offs_k % 2) * 4
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + bk[None, :] * stride_bk
        bb = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)
        nib = ((bb >> shift[None, :]) & 0xF).to(tl.float32) - 8.0
        # group scale: scale[n, k//gs]
        sk = offs_k // gs
        s_ptrs = scale_ptr + offs_n[:, None] * stride_sn + sk[None, :] * stride_sk
        sc = tl.load(s_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(
            tl.float32
        )  # noqa: E501
        w = nib * sc  # [BLOCK_N, BLOCK_K] fp32
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(out_ptr + offs_n, acc, mask=n_mask)


def gemv_hand(x, b_packed, scale, K, gs, BLOCK_N=64, BLOCK_K=128):
    N = b_packed.shape[0]
    out = torch.empty((N,), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, BLOCK_N),)
    int4_gemv_kernel[grid](
        x,
        b_packed,
        scale,
        out,
        K,
        N,
        gs,
        b_packed.stride(0),
        b_packed.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


# --------------------------------------------------------------------------- #
# Hand GEMV kernel v2: BATCHED over active experts + split-K across CUs.
# Real M=1 MoE routes ONE token to E_act experts. The right parallelism is
# grid = (E_act * n_n_tiles * SPLIT_K), so we fill all 120 CUs even when each
# expert's N is small (gate_up N=256). split-K programs atomically add into
# the fp32 output. x[K] is shared across all experts (the single token).
# --------------------------------------------------------------------------- #
@triton.jit
def int4_gemv_batched_kernel(
    x_ptr,
    b_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    gs: tl.constexpr,
    E_act: tl.constexpr,
    stride_be,
    stride_bn,
    stride_bk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    # decode pid -> (expert, n_tile, k_split)
    pid_k = pid % SPLIT_K
    pid2 = pid // SPLIT_K
    pid_n = pid2 % n_n_tiles
    pid_e = pid2 // n_n_tiles

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    be = pid_e * stride_be
    se = pid_e * stride_se
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        bk = offs_k // 2
        shift = (offs_k % 2) * 4
        b_ptrs = b_ptr + be + offs_n[:, None] * stride_bn + bk[None, :] * stride_bk
        bb = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)
        nib = ((bb >> shift[None, :]) & 0xF).to(tl.float32) - 8.0
        sk = offs_k // gs
        s_ptrs = scale_ptr + se + offs_n[:, None] * stride_sn + sk[None, :] * stride_sk
        sc = tl.load(s_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(
            tl.float32
        )  # noqa: E501
        acc += tl.sum(nib * sc * x[None, :], axis=1)

    out_ptrs = out_ptr + pid_e * stride_oe + offs_n * stride_on
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=n_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=n_mask)


@triton.jit
def int4_gemv_batched_f16_kernel(
    x_ptr,
    b_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    gs: tl.constexpr,
    E_act: tl.constexpr,
    stride_be,
    stride_bn,
    stride_bk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """fp16 dequant+multiply, fp32 reduce. Half the VGPR/ALU of the fp32 form."""
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_k = pid % SPLIT_K
    pid2 = pid // SPLIT_K
    pid_n = pid2 % n_n_tiles
    pid_e = pid2 // n_n_tiles
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    be = pid_e * stride_be
    se = pid_e * stride_se
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0)  # fp16
        bk = offs_k // 2
        shift = (offs_k % 2) * 4
        b_ptrs = b_ptr + be + offs_n[:, None] * stride_bn + bk[None, :] * stride_bk
        bb = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)
        nib = ((bb >> shift[None, :]) & 0xF).to(tl.float16) - 8.0
        sk = offs_k // gs
        s_ptrs = scale_ptr + se + offs_n[:, None] * stride_sn + sk[None, :] * stride_sk
        sc = tl.load(s_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # fp16
        prod = nib * sc * x[None, :]  # fp16 [BLOCK_N, BLOCK_K]
        acc += tl.sum(prod.to(tl.float32), axis=1)
    out_ptrs = out_ptr + pid_e * stride_oe + offs_n * stride_on
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=n_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=n_mask)


# --------------------------------------------------------------------------- #
# Hand kernel v3: KEEP MFMA (tl.dot) + split-K across CUs.
# This is the design the occupancy diagnosis actually implies: the in-tree
# kernel is MFMA-bound-but-under-occupied (80 WG / 120 CUs). Don't drop MFMA
# (that's ALU-bound, v2's mistake) -- instead split the K contraction into
# SPLIT_K partial GEMMs so grid = E_act * n_n_tiles * SPLIT_K fills all CUs,
# each tl.dot still runs on the matrix units, partials atomic-add into C.
# A is the single decode token padded to BLOCK_M=16 (row 0 real, rest masked),
# exactly like fused_moe_kernel_gptq_awq.
# --------------------------------------------------------------------------- #
@triton.jit
def int4_mfma_splitk_kernel(
    x_ptr,
    b_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    gs: tl.constexpr,
    E_act: tl.constexpr,
    stride_be,
    stride_bn,
    stride_bk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_oe,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_k = pid % SPLIT_K
    pid2 = pid // SPLIT_K
    pid_n = pid2 % n_n_tiles
    pid_e = pid2 // n_n_tiles

    offs_m = tl.arange(0, BLOCK_M)  # only row 0 is the real token
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m == 0
    n_mask = offs_n < N

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    be = pid_e * stride_be
    se = pid_e * stride_se
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < k_end
        # A: [BLOCK_M, BLOCK_K], row 0 = x[kk], rest 0
        a = tl.load(
            x_ptr + kk[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0
        )  # fp16
        # B packed: byte index kk//2, nibble (kk%2)*4 ; layout [N, K/2]
        bk = kk // 2
        shift = (kk % 2) * 4
        b_ptrs = b_ptr + be + bk[:, None] * stride_bk + offs_n[None, :] * stride_bn
        bb = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        nib = ((bb >> shift[:, None]) & 0xF).to(tl.float16) - 8.0  # [BLOCK_K, BLOCK_N]
        sk = kk // gs
        s_ptrs = scale_ptr + se + sk[:, None] * stride_sk + offs_n[None, :] * stride_sn
        sc = tl.load(s_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        b = (nib * sc).to(tl.float16)  # [BLOCK_K, BLOCK_N] dequant fp16
        acc = tl.dot(a, b, acc=acc)  # MFMA

    # write back row 0 only
    out_ptrs = out_ptr + pid_e * stride_oe + offs_n * stride_on
    row0 = tl.sum(tl.where(m_mask[:, None], acc, 0.0), axis=0)  # [BLOCK_N]
    if SPLIT_K == 1:
        tl.store(out_ptrs, row0, mask=n_mask)
    else:
        tl.atomic_add(out_ptrs, row0, mask=n_mask)


def mfma_splitk(x, b_packed, scale, K, gs, BLOCK_N=64, BLOCK_K=64, SPLIT_K=1):
    E_act, N, _ = b_packed.shape
    out = torch.zeros((E_act, N), dtype=torch.float32, device=x.device)
    n_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (E_act * n_n_tiles * SPLIT_K,)
    int4_mfma_splitk_kernel[grid](
        x,
        b_packed,
        scale,
        out,
        K,
        N,
        gs,
        E_act,
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=16,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
    )
    return out


def gemv_hand_batched(
    x, b_packed, scale, K, gs, BLOCK_N=32, BLOCK_K=128, SPLIT_K=1, f16=False
):
    E_act, N, _ = b_packed.shape
    out = torch.zeros((E_act, N), dtype=torch.float32, device=x.device)
    n_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (E_act * n_n_tiles * SPLIT_K,)
    kern = int4_gemv_batched_f16_kernel if f16 else int4_gemv_batched_kernel
    kern[grid](
        x,
        b_packed,
        scale,
        out,
        K,
        N,
        gs,
        E_act,
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
    )
    return out


def bytes_moved(N, K, gs):
    # int4 weights (K/2 bytes/row * N) + fp16 scales (K/gs * 2 * N) + x (K*2) + out (N*4)  # noqa: E501
    return N * (K // 2) + N * (K // gs) * 2 + K * 2 + N * 4


def time_us(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.time() - t0) / iters * 1e6


def time_us_graph(fn, inner=20, iters=50, warmup=5):
    """CUDA-graph timing: captures `inner` calls, removes Python dispatch
    overhead so we measure true GPU kernel time (comparable to rocprof)."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    torch.accelerator.synchronize()
    for _ in range(3):
        g.replay()
    torch.accelerator.synchronize()
    t0 = time.time()
    for _ in range(iters):
        g.replay()
    torch.accelerator.synchronize()
    return (time.time() - t0) / iters / inner * 1e6


def correctness(gs):
    """Single-expert correctness vs torch ref (both kernels)."""
    print("=== correctness (single expert, vs torch fp16 ref) ===")
    for name, N, K in [("gate_up", 256, 2048), ("down", 2048, 128)]:
        x = torch.randn(K, dtype=torch.float16)
        b_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        scale = torch.rand((N, K // gs), dtype=torch.float16) * 0.02 + 0.001
        w_fp16 = dequant_ref(b_packed, scale, K, gs)
        ref = matvec_ref(x, w_fp16)
        h1 = gemv_hand(x, b_packed, scale, K, gs)
        rel1 = ((h1 - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
        # batched form with 1 expert, split-K=4
        hb = gemv_hand_batched(x, b_packed[None], scale[None], K, gs, SPLIT_K=4)[0]
        relb = ((hb - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
        # MFMA + split-K (v3)
        hm = mfma_splitk(
            x, b_packed[None], scale[None], K, gs, BLOCK_N=64, BLOCK_K=64, SPLIT_K=4
        )[0]
        relm = ((hm - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
        ok = max(rel1, relb, relm) < 1e-3
        print(
            f"  {name:8s} N={N:5d} K={K:5d}: v1 {rel1:.2e}  "
            f"v2(splitK=4) {relb:.2e}  v3-MFMA(splitK=4) {relm:.2e}  "
            f"{'OK' if ok else 'FAIL'}"
        )


def sweep_batched(gs, E_act=10):
    """Batched E_act-expert GEMV: sweep BLOCK_N x SPLIT_K, report best vs roofline."""
    print(
        f"\n=== batched {E_act}-expert GEMV (real M=1 MoE decode), sweep "
        f"(CUDA-graph timed = true GPU us) ==="
    )
    results = []
    for name, N, K in [("gate_up", 256, 2048), ("down", 2048, 128)]:
        x = torch.randn(K, dtype=torch.float16)
        b = torch.randint(0, 256, (E_act, N, K // 2), dtype=torch.uint8)
        sc = torch.rand((E_act, N, K // gs), dtype=torch.float16) * 0.02 + 0.001
        # bytes for all E_act experts
        bm = E_act * (N * (K // 2) + N * (K // gs) * 2) + K * 2 + E_act * N * 4
        roof = bm / HBM_BW * 1e6
        # ---- v2: GEMV (no tl.dot) ----
        best2 = (1e9, None)
        for f16 in (False, True):
            for BLOCK_N in (16, 32, 64):
                for SPLIT_K in (1, 2, 4, 8):
                    if K // SPLIT_K < 32:
                        continue
                    try:
                        fn = (
                            lambda x=x,
                            b=b,
                            sc=sc,
                            K=K,
                            BLOCK_N=BLOCK_N,
                            SPLIT_K=SPLIT_K,
                            f16=f16: gemv_hand_batched(  # noqa: E501
                                x,
                                b,
                                sc,
                                K,
                                gs,
                                BLOCK_N=BLOCK_N,
                                SPLIT_K=SPLIT_K,
                                f16=f16,
                            )
                        )
                        t = time_us_graph(fn)
                    except Exception:
                        continue
                    if t < best2[0]:
                        best2 = (t, (BLOCK_N, SPLIT_K, "f16" if f16 else "f32"))
        # ---- v3: MFMA (tl.dot) + split-K ----
        best3 = (1e9, None)
        for BLOCK_N in (32, 64, 128):
            if BLOCK_N > N:
                continue
            for BLOCK_K in (32, 64, 128):
                for SPLIT_K in (1, 2, 4, 8):
                    if K // SPLIT_K < BLOCK_K:
                        continue
                    n_tiles = triton.cdiv(N, BLOCK_N)
                    progs = E_act * n_tiles * SPLIT_K
                    try:
                        fn = (
                            lambda x=x,
                            b=b,
                            sc=sc,
                            K=K,
                            BLOCK_N=BLOCK_N,
                            BLOCK_K=BLOCK_K,
                            SPLIT_K=SPLIT_K: mfma_splitk(  # noqa: E501
                                x,
                                b,
                                sc,
                                K,
                                gs,
                                BLOCK_N=BLOCK_N,
                                BLOCK_K=BLOCK_K,
                                SPLIT_K=SPLIT_K,
                            )
                        )
                        t = time_us_graph(fn)
                    except Exception:
                        continue
                    if t < best3[0]:
                        best3 = (t, (BLOCK_N, BLOCK_K, SPLIT_K, progs))
        t2, c2 = best2
        t3, c3 = best3
        gbs2 = bm / (t2 / 1e6) / 1e9
        gbs3 = bm / (t3 / 1e6) / 1e9
        print(f"  {name:8s} N={N:5d} K={K:5d} roof {roof:.2f}us")
        print(
            f"    v2 GEMV  : {t2:7.1f}us  BLOCK_N={c2[0]} SPLIT_K={c2[1]} "
            f"{c2[2]} | {gbs2:7.1f} GB/s  {t2 / roof:.0f}x"
        )
        print(
            f"    v3 MFMA+K: {t3:7.1f}us  BLOCK_N={c3[0]} BLOCK_K={c3[1]} "
            f"SPLIT_K={c3[2]} progs={c3[3]} | {gbs3:7.1f} GB/s  {t3 / roof:.0f}x"
        )
        results.append((name, min(t2, t3)))
    return results


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    gs = 32
    correctness(gs)
    res = sweep_batched(gs, E_act=10)
    tot = sum(t for _, t in res)
    print(
        f"\n  hand GEMV total (gate_up+down) = {tot:.1f}us/layer  "
        f"vs in-tree gptq_awq GEMM ~57us/layer (rocprof). "
        f"{'WIN' if tot < 57 else 'LOSS'} ({57 / tot:.2f}x)"
    )
    os._exit(0)


if __name__ == "__main__":
    main()
