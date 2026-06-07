#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Flash-Decoding Split-K for decode attention on MI100.

Measures decode attention latency across sequence lengths to demonstrate
the benefit of adaptive KV-dimension splitting. Compares:
  1. 2D kernel (no split-K, single thread block per query)
  2. 3D kernel with fixed 8 splits (current default)
  3. 3D kernel with adaptive split-K (Flash-Decoding)

Usage:
    python benchmarks/kernels/benchmark_flash_decoding.py

Requirements:
    Requires GPU (MI100 or CUDA). Results are most interesting on MI100
    where the 120 CUs benefit from higher split-K at long sequences.
"""

import argparse
import time

import torch

from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.triton_attn import (
    _compute_flash_decoding_splits,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def benchmark_decode_attention(
    num_seqs: int,
    kv_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_splits: int | None,
    max_flash_decoding_splits: int | None,
    num_warmup: int = 10,
    num_iters: int = 100,
    use_2d: bool = False,
) -> float:
    """Benchmark decode attention and return median latency in microseconds."""
    dtype = torch.float16
    num_blocks = 32768
    scale = head_size**-0.5

    query = torch.randn(
        num_seqs, num_query_heads, head_size, dtype=dtype, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)

    query_lens = [1] * num_seqs
    kv_lens = [kv_len] * num_seqs
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )

    output = torch.empty_like(query)

    # Setup segment buffers
    head_size_padded = next_power_of_2(head_size)
    if use_2d:
        seq_threshold_3D = None
        actual_splits = None
        segm_output = None
        segm_max = None
        segm_expsum = None
    else:
        actual_splits = num_splits if num_splits else 8
        seq_threshold_3D = num_seqs
        segm_output = torch.empty(
            (num_seqs, num_query_heads, actual_splits, head_size_padded),
            dtype=torch.float32,
            device="cuda",
        )
        segm_max = torch.empty(
            (num_seqs, num_query_heads, actual_splits),
            dtype=torch.float32,
            device="cuda",
        )
        segm_expsum = torch.empty(
            (num_seqs, num_query_heads, actual_splits),
            dtype=torch.float32,
            device="cuda",
        )

    def run_once():
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens_tensor,
            max_seqlen_q=1,
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=actual_splits,
            max_flash_decoding_splits=max_flash_decoding_splits,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
        )

    # Warmup
    for _ in range(num_warmup):
        run_once()
    torch.accelerator.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_iters):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        run_once()
        torch.accelerator.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # us

    latencies.sort()
    return latencies[len(latencies) // 2]  # Median


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flash-Decoding Split-K")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="KV sequence lengths to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--num-query-heads",
        type=int,
        default=8,
        help="Number of query heads",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=2,
        help="Number of KV heads",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=128,
        help="Head dimension",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    args = parser.parse_args()

    print("=" * 90)
    print("Flash-Decoding Split-K Benchmark")
    print("=" * 90)
    print(
        f"Config: {args.num_query_heads} Q heads, {args.num_kv_heads} KV "
        f"heads, head_size={args.head_size}, block_size={args.block_size}"
    )
    print(f"Iterations: {args.num_iters}")
    print()

    for batch_size in args.batch_sizes:
        print(f"{'─' * 90}")
        print(f"Batch size = {batch_size}")
        print(f"{'─' * 90}")
        print(
            f"{'Seq Len':>10} │ {'2D (us)':>10} │ "
            f"{'Fixed-8 (us)':>12} │ {'Adaptive (us)':>14} │ "
            f"{'Splits':>6} │ {'Speedup vs 2D':>14} │ "
            f"{'Speedup vs F8':>14}"
        )
        print(
            f"{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 12}─┼─{'─' * 14}─┼─"
            f"{'─' * 6}─┼─{'─' * 14}─┼─{'─' * 14}"
        )

        for seq_len in args.seq_lens:
            # Compute adaptive split count
            adaptive_splits = _compute_flash_decoding_splits(
                max_seq_len=seq_len,
                num_seqs=batch_size,
                num_kv_heads=args.num_kv_heads,
                tile_size=32,
            )

            # 1. 2D kernel (no split-K)
            lat_2d = benchmark_decode_attention(
                num_seqs=batch_size,
                kv_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                num_splits=None,
                max_flash_decoding_splits=None,
                num_iters=args.num_iters,
                use_2d=True,
            )

            # 2. Fixed 8 splits (current approach)
            lat_fixed8 = benchmark_decode_attention(
                num_seqs=batch_size,
                kv_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                num_splits=8,
                max_flash_decoding_splits=None,
                num_iters=args.num_iters,
            )

            # 3. Adaptive Flash-Decoding
            lat_adaptive = benchmark_decode_attention(
                num_seqs=batch_size,
                kv_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                num_splits=adaptive_splits,
                max_flash_decoding_splits=adaptive_splits,
                num_iters=args.num_iters,
            )

            speedup_2d = lat_2d / lat_adaptive if lat_adaptive > 0 else 0
            speedup_f8 = lat_fixed8 / lat_adaptive if lat_adaptive > 0 else 0

            print(
                f"{seq_len:>10} │ {lat_2d:>10.1f} │ {lat_fixed8:>12.1f} │ "
                f"{lat_adaptive:>14.1f} │ {adaptive_splits:>6} │ "
                f"{speedup_2d:>13.2f}x │ {speedup_f8:>13.2f}x"
            )

        print()

    # Also print split count schedule for reference
    print(f"\n{'=' * 90}")
    print("Flash-Decoding Split Schedule (batch=1)")
    print(f"{'=' * 90}")
    print(
        f"{'Seq Len':>10} │ {'KV Heads=2':>12} │ {'KV Heads=4':>12} │ "
        f"{'KV Heads=8':>12}"
    )
    print(f"{'─' * 10}─┼─{'─' * 12}─┼─{'─' * 12}─┼─{'─' * 12}")
    for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        splits = []
        for kv_heads in [2, 4, 8]:
            s = _compute_flash_decoding_splits(
                max_seq_len=seq_len,
                num_seqs=1,
                num_kv_heads=kv_heads,
                tile_size=32,
            )
            splits.append(s)
        print(f"{seq_len:>10} │ {splits[0]:>12} │ {splits[1]:>12} │ {splits[2]:>12}")


if __name__ == "__main__":
    main()
