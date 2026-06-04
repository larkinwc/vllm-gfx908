#!/usr/bin/env python3
"""Apples-to-apples: hand MFMA+split-K kernel vs the REAL in-tree
fused_moe_kernel_gptq_awq, same shapes, same CUDA-graph harness.

This removes the cross-run comparison risk: both the in-tree invoke wrapper and
the hand kernel are timed with identical graph capture, and we check the hand
kernel reproduces the in-tree kernel's output for the full 10-active-expert
M=1 decode (not just a single expert).

Run:
  PYTHONPATH=. HIP_VISIBLE_DEVICES=2 python bench_scripts/moe_int4_headtohead.py
"""
import os
import sys
import time

import torch
import triton

sys.path.insert(0, os.path.dirname(__file__))
from moe_int4_gemv_bench import mfma_splitk, dequant_ref  # noqa: E402

from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    invoke_fused_moe_wna16_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
import triton.language as tl  # noqa: E402

HBM_BW = 1.2e12


def time_us_graph(fn, inner=20, iters=50, warmup=5):
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


def run_intree(A, B, C, Bs, sorted_ids, expert_ids, ntpp, top_k, N, K, gs, bn, bk):
    """Invoke the real in-tree gptq_awq triton kernel for one projection."""
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
              "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 2}
    invoke_fused_moe_wna16_triton_kernel(
        A, B, C, Bs, None, None, sorted_ids, expert_ids, ntpp,
        mul_routed_weight=False, top_k=top_k, config=config,
        compute_type=tl.float16, use_int8_w8a16=False, use_int4_w4a16=True,
        block_shape=[0, gs],
    )
    return C


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    gs = 32
    E = 512          # global experts
    top_k = 10
    M = 1
    # gate_up projection shape (per-shard TP=4): N=256, K=2048
    for name, N, K in [("gate_up", 256, 2048), ("down", 2048, 128)]:
        A = torch.randn(M, K, dtype=torch.float16)
        B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
        Bs = torch.rand((E, N, K // gs), dtype=torch.float16) * 0.02 + 0.001

        # routing: 1 token -> top_k distinct experts
        topk_ids = torch.randperm(E)[:top_k].to(torch.int32).view(1, top_k)
        bn, bk = 32, 64    # in-tree bs=1 default
        sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, 16, E)
        C = torch.zeros(M, top_k, N, dtype=torch.float16)

        # ---- in-tree kernel ----
        run_intree(A, B, C, Bs, sorted_ids, expert_ids, ntpp, top_k, N, K, gs, bn, bk)
        intree_out = C.clone()  # [1, top_k, N]
        t_in = time_us_graph(lambda: run_intree(
            A, B, C, Bs, sorted_ids, expert_ids, ntpp, top_k, N, K, gs, bn, bk))

        # ---- hand v3 on the same top_k active experts ----
        act = topk_ids.view(-1).long()      # [top_k] global expert ids
        b_act = B[act].contiguous()          # [top_k, N, K//2]
        s_act = Bs[act].contiguous()         # [top_k, N, K//gs]
        x = A[0].contiguous()                # [K]

        # correctness: hand vs in-tree, per active expert
        best = (1e9, None)
        relmax = 0.0
        for BLOCK_N in (32, 64, 128):
            if BLOCK_N > N:
                continue
            for BLOCK_K in (32, 64, 128):
                for SPLIT_K in (1, 2, 4, 8):
                    if K // SPLIT_K < BLOCK_K:
                        continue
                    try:
                        out = mfma_splitk(x, b_act, s_act, K, gs,
                                          BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                          SPLIT_K=SPLIT_K)
                        t = time_us_graph(lambda: mfma_splitk(
                            x, b_act, s_act, K, gs, BLOCK_N=BLOCK_N,
                            BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K))
                    except Exception:
                        continue
                    if t < best[0]:
                        best = (t, (BLOCK_N, BLOCK_K, SPLIT_K))
        # verify correctness at best config vs in-tree per-expert
        bn3, bk3, sk3 = best[1]
        hand = mfma_splitk(x, b_act, s_act, K, gs, BLOCK_N=bn3,
                           BLOCK_K=bk3, SPLIT_K=sk3)  # [top_k, N] fp32
        intree_pe = intree_out[0].to(torch.float32)   # [top_k, N], order = topk
        rel = ((hand - intree_pe).abs().max()
               / (intree_pe.abs().max() + 1e-6)).item()

        bm = top_k * (N * (K // 2) + N * (K // gs) * 2) + K * 2 + top_k * N * 4
        roof = bm / HBM_BW * 1e6
        print(f"{name:8s} N={N:5d} K={K:5d} roof {roof:.2f}us")
        print(f"  in-tree gptq_awq : {t_in:7.1f}us")
        print(f"  hand MFMA+splitK : {best[0]:7.1f}us  "
              f"BLOCK_N={bn3} BLOCK_K={bk3} SPLIT_K={sk3}  "
              f"rel-err {rel:.2e}  {'OK' if rel < 1e-3 else 'FAIL'}")
        print(f"  --> {t_in/best[0]:.2f}x {'WIN' if best[0] < t_in else 'LOSS'}")
    os._exit(0)


if __name__ == "__main__":
    main()
