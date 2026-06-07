#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration correctness + timing for the gfx908 split-K wna16 MoE path.

Compares fused_experts output with SPLIT_K forced to 1 (baseline) vs the
gfx908 gate (SPLIT_K=8) at the real Qwen3-Coder-Next decode shape, and times
both. Run:
  PYTHONPATH=. HIP_VISIBLE_DEVICES=2 python bench_scripts/moe_splitk_integration_test.py
"""

import os
import time

import torch

import vllm.model_executor.layers.fused_moe.fused_moe as fm
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def build(E, H, N, GS):
    x = torch.randn(1, H, dtype=torch.float16)
    w1 = torch.randint(0, 255, (E, 2 * N, H // 2), dtype=torch.uint8)
    w2 = torch.randint(0, 255, (E, H, N // 2), dtype=torch.uint8)
    w1s = torch.rand((E, 2 * N, H // GS), dtype=torch.float16) * 0.02 + 0.001
    w2s = torch.rand((E, H, N // GS), dtype=torch.float16) * 0.02 + 0.001
    gate = torch.randn(1, E, dtype=torch.float16)
    return x, w1, w2, w1s, w2s, gate


def run(x, w1, w2, qc, tw, ti):
    return fused_experts(x, w1, w2, tw, ti, inplace=False, quant_config=qc)


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


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    E, H, TK, GS, N = 512, 2048, 10, 32, 128
    x, w1, w2, w1s, w2s, gate = build(E, H, N, GS)
    qc = int4_w4a16_moe_quant_config(
        w1_scale=w1s, w2_scale=w2s, w1_zp=None, w2_zp=None, block_shape=[0, GS]
    )
    tw, ti, _ = fused_topk(x, gate, TK, renormalize=True)

    orig = fm.get_default_config

    def cfg_force_split(M, E_, N_, K_, topk_, dtype_, blk):
        c = orig(M, E_, N_, K_, topk_, dtype_, blk)
        return c

    # baseline: monkeypatch the gate off (force SPLIT_K=1)
    import vllm.model_executor.layers.fused_moe.fused_moe as fmod

    real_on = fmod._on_mi100

    fmod._on_mi100 = lambda: False
    out_base = run(x, w1, w2, qc, tw, ti).clone()
    t_base = time_us_graph(lambda: run(x, w1, w2, qc, tw, ti))

    fmod._on_mi100 = lambda: True
    out_sk = run(x, w1, w2, qc, tw, ti).clone()
    t_sk = time_us_graph(lambda: run(x, w1, w2, qc, tw, ti))

    fmod._on_mi100 = real_on

    rel = ((out_sk - out_base).abs().max() / (out_base.abs().max() + 1e-6)).item()
    print(f"SPLIT_K=1 baseline : {t_base:7.1f} us")
    print(f"SPLIT_K=8 (gfx908) : {t_sk:7.1f} us")
    print(f"speedup            : {t_base / t_sk:.2f}x")
    print(f"output rel-err     : {rel:.2e}  {'OK' if rel < 1e-2 else 'FAIL'}")
    os._exit(0)


if __name__ == "__main__":
    main()
