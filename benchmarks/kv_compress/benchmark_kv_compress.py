# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for SnapKV + PyramidKV KV Cache Compression.

Measures:
    1. Compression ratios and memory savings across configurations
    2. Token selection quality (attention score coverage)
    3. Compression latency (one-shot at prefill time)
    4. Throughput impact estimation for MI100

Usage:
    python benchmarks/kv_compress/benchmark_kv_compress.py
    python benchmarks/kv_compress/benchmark_kv_compress.py --seq-lens 512 1024 2048 4096
    python benchmarks/kv_compress/benchmark_kv_compress.py --device cuda
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

import torch

from vllm.attention.kv_compress.combined import SnapPyramidKVCompressor
from vllm.attention.kv_compress.config import KVCompressConfig, PyramidSchedule
from vllm.attention.kv_compress.pyramidkv import PyramidKVBudgetAllocator
from vllm.attention.kv_compress.snapkv import SnapKVSelector


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    seq_len: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    retention_ratio: float
    pyramid_schedule: str
    pyramid_alpha: float
    compression_ratio: float
    memory_savings_pct: float
    avg_tokens_retained_per_layer: float
    min_tokens_retained: int
    max_tokens_retained: int
    compress_latency_ms: float
    attention_coverage_pct: float
    device: str


def compute_attention_coverage(
    query: torch.Tensor,
    key: torch.Tensor,
    selection_mask: torch.Tensor,
    scale: float,
    seq_len: int,
) -> float:
    """Measure what fraction of total attention weight is captured.

    Computes full attention weights and checks what percentage is
    covered by the selected positions. Higher = better quality.
    """
    # Reshape to [1, H, S, D]
    if query.dim() == 3:
        q = query.unsqueeze(0).transpose(1, 2)
        k = key.unsqueeze(0).transpose(1, 2)
    else:
        q, k = query, key

    # Handle GQA: expand KV heads to match query heads
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    if num_heads_q != num_heads_kv:
        num_groups = num_heads_q // num_heads_kv
        k = k.repeat_interleave(num_groups, dim=1)

    # Full attention weights
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
    attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, -1e9)
    weights = torch.softmax(attn, dim=-1)  # [1, H, S, S]

    # Union mask across heads
    union_mask = selection_mask.any(dim=0)  # [S]

    # For each query position, what fraction of attention is on selected keys
    # weights[:, :, :, selected] .sum(-1) / weights.sum(-1)
    selected_weight = weights[:, :, :, union_mask].sum(dim=-1)
    total_weight = weights.sum(dim=-1)

    coverage = (selected_weight / total_weight.clamp(min=1e-9)).mean()
    return coverage.item() * 100.0


def benchmark_compression_ratios(
    seq_lens: list[int],
    device: str = "cpu",
) -> list[BenchmarkResult]:
    """Benchmark compression ratios across configurations."""
    results = []

    configs = [
        ("linear_alpha4", PyramidSchedule.LINEAR, 4.0, 0.12),
        ("linear_alpha2", PyramidSchedule.LINEAR, 2.0, 0.12),
        ("exp_alpha4", PyramidSchedule.EXPONENTIAL, 4.0, 0.12),
        ("step_alpha4", PyramidSchedule.STEP, 4.0, 0.12),
        ("linear_20pct", PyramidSchedule.LINEAR, 4.0, 0.20),
        ("linear_5pct", PyramidSchedule.LINEAR, 4.0, 0.05),
    ]

    num_layers = 32
    num_kv_heads = 8
    head_size = 128
    num_heads_q = 32

    for name, schedule, alpha, ratio in configs:
        for seq_len in seq_lens:
            config = KVCompressConfig(
                enabled=True,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                retention_ratio=ratio,
                observation_window_size=min(32, seq_len),
                kernel_size=7,
                recent_window_size=min(32, seq_len // 4),
                sink_size=4,
                pyramid_schedule=schedule,
                pyramid_alpha=alpha,
                min_seq_len_to_compress=64,
            )
            compressor = SnapPyramidKVCompressor(config)
            stats = compressor.get_compression_stats(seq_len)

            # Measure actual compression with random data
            scale = 1.0 / (head_size ** 0.5)
            q = torch.randn(
                seq_len, num_heads_q, head_size, device=device
            )
            k = torch.randn(
                seq_len, num_kv_heads, head_size, device=device
            )
            v = torch.randn(
                seq_len, num_kv_heads, head_size, device=device
            )

            # Warmup
            if device != "cpu":
                _ = compressor.compress_layer(
                    0, q, k, v, scale, seq_len
                )
                torch.cuda.synchronize()

            # Time compression for one layer
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = compressor.compress_layer(
                0, q, k, v, scale, seq_len
            )
            if device != "cpu":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000

            # Attention coverage (only for manageable sizes)
            if seq_len <= 2048:
                coverage = compute_attention_coverage(
                    q, k, result.selection_mask, scale, seq_len
                )
            else:
                coverage = -1.0  # Skip for very long sequences

            budgets = stats["pyramid_allocation"]["per_layer_budgets"]

            results.append(BenchmarkResult(
                name=name,
                seq_len=seq_len,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                retention_ratio=ratio,
                pyramid_schedule=schedule.value,
                pyramid_alpha=alpha,
                compression_ratio=stats["overall_compression_ratio"],
                memory_savings_pct=stats["memory_savings_pct"],
                avg_tokens_retained_per_layer=sum(budgets) / len(budgets),
                min_tokens_retained=min(budgets),
                max_tokens_retained=max(budgets),
                compress_latency_ms=latency_ms,
                attention_coverage_pct=coverage,
                device=device,
            ))

    return results


def benchmark_latency_scaling(
    device: str = "cpu",
) -> list[dict]:
    """Measure how compression latency scales with sequence length."""
    results = []
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    if device != "cpu":
        seq_lens.extend([8192, 16384])

    num_kv_heads = 8
    num_heads_q = 32
    head_size = 128

    config = KVCompressConfig(
        num_layers=32,
        num_kv_heads=num_kv_heads,
        retention_ratio=0.12,
        min_seq_len_to_compress=64,
    )
    compressor = SnapPyramidKVCompressor(config)
    scale = 1.0 / (head_size ** 0.5)

    for seq_len in seq_lens:
        q = torch.randn(seq_len, num_heads_q, head_size, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_size, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_size, device=device)

        # Warmup
        _ = compressor.compress_layer(0, q, k, v, scale, seq_len)
        if device != "cpu":
            torch.cuda.synchronize()

        # Benchmark
        n_iters = 10
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = compressor.compress_layer(0, q, k, v, scale, seq_len)
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / n_iters * 1000
        results.append({
            "seq_len": seq_len,
            "avg_latency_ms": round(avg_ms, 3),
            "device": device,
        })

    return results


def benchmark_memory_savings() -> list[dict]:
    """Estimate memory savings for MI100 (32GB HBM2) scenarios."""
    results = []

    # 7B model parameters (LLaMA-2-7B-like)
    models = [
        {
            "name": "LLaMA-2-7B",
            "num_layers": 32,
            "num_kv_heads": 32,  # MHA
            "head_size": 128,
            "model_weight_gb": 14.0,
        },
        {
            "name": "LLaMA-3-8B",
            "num_layers": 32,
            "num_kv_heads": 8,  # GQA
            "head_size": 128,
            "model_weight_gb": 16.0,
        },
        {
            "name": "Mistral-7B",
            "num_layers": 32,
            "num_kv_heads": 8,  # GQA
            "head_size": 128,
            "model_weight_gb": 14.5,
        },
    ]

    gpu_memory_gb = 32.0
    gpu_utilization = 0.9
    available_gb = gpu_memory_gb * gpu_utilization

    for model in models:
        kv_available_gb = available_gb - model["model_weight_gb"]
        if kv_available_gb <= 0:
            continue

        # Bytes per token per layer for KV cache (FP16)
        bytes_per_kv_token = (
            2 * model["num_kv_heads"] * model["head_size"] * 2  # 2 for K+V, 2 for FP16
        )
        bytes_per_token_all_layers = (
            bytes_per_kv_token * model["num_layers"]
        )

        # Max tokens without compression
        max_tokens_no_compress = int(
            kv_available_gb * 1e9 / bytes_per_token_all_layers
        )

        for ratio in [0.05, 0.10, 0.12, 0.20, 0.25]:
            config = KVCompressConfig(
                num_layers=model["num_layers"],
                num_kv_heads=model["num_kv_heads"],
                retention_ratio=ratio,
                pyramid_schedule=PyramidSchedule.LINEAR,
                pyramid_alpha=4.0,
            )
            comp = SnapPyramidKVCompressor(config)

            # Effective max tokens with compression
            effective_multiplier = 1.0 / ratio
            max_tokens_compressed = int(
                max_tokens_no_compress * effective_multiplier
            )

            stats = comp.get_compression_stats(
                seq_len=max_tokens_no_compress
            )

            results.append({
                "model": model["name"],
                "retention_ratio": ratio,
                "kv_available_gb": round(kv_available_gb, 1),
                "max_tokens_no_compress": max_tokens_no_compress,
                "max_tokens_compressed": max_tokens_compressed,
                "capacity_multiplier": round(effective_multiplier, 1),
                "max_context_no_compress": max_tokens_no_compress,
                "max_context_compressed": max_tokens_compressed,
                "memory_savings_pct": round(
                    stats["memory_savings_pct"], 1
                ),
            })

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 100)
    print("SnapKV + PyramidKV Compression Benchmark Results")
    print("=" * 100)

    header = (
        f"{'Config':<20} {'SeqLen':>8} {'Compress':>10} "
        f"{'Savings%':>10} {'AvgRetain':>10} {'Min/Max':>12} "
        f"{'Latency':>10} {'Coverage%':>10}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        coverage_str = (
            f"{r.attention_coverage_pct:.1f}"
            if r.attention_coverage_pct >= 0
            else "N/A"
        )
        print(
            f"{r.name:<20} {r.seq_len:>8} "
            f"{r.compression_ratio:>10.1f}x "
            f"{r.memory_savings_pct:>9.1f}% "
            f"{r.avg_tokens_retained_per_layer:>10.0f} "
            f"{r.min_tokens_retained:>5}/{r.max_tokens_retained:<5} "
            f"{r.compress_latency_ms:>8.2f}ms "
            f"{coverage_str:>10}"
        )


def print_latency_results(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Compression Latency Scaling")
    print("=" * 60)
    print(f"{'SeqLen':>10} {'Latency':>15} {'Device':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['seq_len']:>10} "
            f"{r['avg_latency_ms']:>13.3f}ms "
            f"{r['device']:>10}"
        )


def print_memory_results(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("MI100 (32GB) Memory Savings Estimation")
    print("=" * 100)
    print(
        f"{'Model':<16} {'Retention':>10} {'KV Avail':>10} "
        f"{'MaxCtx(raw)':>12} {'MaxCtx(comp)':>13} "
        f"{'Multiplier':>11} {'Savings%':>10}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['model']:<16} {r['retention_ratio']:>9.0%} "
            f"{r['kv_available_gb']:>8.1f}GB "
            f"{r['max_context_no_compress']:>12,} "
            f"{r['max_context_compressed']:>13,} "
            f"{r['capacity_multiplier']:>10.1f}x "
            f"{r['memory_savings_pct']:>9.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SnapKV + PyramidKV KV Cache Compression"
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[256, 512, 1024, 2048],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--skip-latency", action="store_true",
        help="Skip latency scaling benchmark",
    )
    parser.add_argument(
        "--skip-memory", action="store_true",
        help="Skip memory estimation benchmark",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Running benchmarks on device: {args.device}")
    print(f"Sequence lengths: {args.seq_lens}")

    # 1. Compression ratio benchmark
    print("\n[1/3] Running compression ratio benchmark...")
    ratio_results = benchmark_compression_ratios(args.seq_lens, args.device)
    print_results(ratio_results)

    # 2. Latency scaling
    if not args.skip_latency:
        print("\n[2/3] Running latency scaling benchmark...")
        latency_results = benchmark_latency_scaling(args.device)
        print_latency_results(latency_results)
    else:
        latency_results = []

    # 3. Memory estimation
    if not args.skip_memory:
        print("\n[3/3] Running MI100 memory estimation...")
        memory_results = benchmark_memory_savings()
        print_memory_results(memory_results)
    else:
        memory_results = []

    # Save results
    if args.output:
        output = {
            "compression_ratios": [asdict(r) for r in ratio_results],
            "latency_scaling": latency_results,
            "memory_estimation": memory_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()
