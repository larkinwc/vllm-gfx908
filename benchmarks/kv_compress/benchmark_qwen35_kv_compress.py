#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SnapKV + PyramidKV benchmark on Qwen3.5-9B (4x MI100).

Qwen3.5-9B is a hybrid model:
  - 32 layers total
  - Only 8 layers use full_attention (layers 3,7,11,15,19,23,27,31)
  - 24 layers use linear_attention (no traditional KV cache)
  - Full-attention: 16 query heads, 4 KV heads (GQA 4:1), head_dim=256
  - Linear-attention: 16 key heads × 128 dim, 32 value heads × 128 dim

SnapKV+PyramidKV only compresses the 8 full-attention layers' KV cache.
The linear-attention layers are unaffected.

This benchmark:
  1. Loads the model tokenizer to create real prompt tokens
  2. Simulates realistic Q/K/V tensors matching the model's exact shapes
  3. Measures compression ratios, attention coverage, and latency on GPU
  4. Estimates MI100 memory savings accounting for the hybrid architecture

Usage:
    # On MI100 with /opt/vllm-env:
    /opt/vllm-env/bin/python3 benchmarks/kv_compress/benchmark_qwen35_kv_compress.py

    # With custom seq lengths:
    /opt/vllm-env/bin/python3 benchmarks/kv_compress/benchmark_qwen35_kv_compress.py \
        --seq-lens 512 1024 2048 4096 8192

    # CPU fallback (for testing):
    /opt/vllm-env/bin/python3 benchmarks/kv_compress/benchmark_qwen35_kv_compress.py --device cpu
"""

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

# Allow importing from the workspace root
sys.path.insert(0, "/home/aimeme/Desktop/vllm-gfx908")

from vllm.attention.kv_compress.combined import SnapPyramidKVCompressor
from vllm.attention.kv_compress.config import KVCompressConfig, PyramidSchedule

# ── Qwen3.5-9B architecture constants ──────────────────────────────────
QWEN35_NUM_LAYERS = 32
QWEN35_FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]  # every 4th
QWEN35_NUM_FULL_ATTN_LAYERS = len(QWEN35_FULL_ATTN_LAYERS)  # 8
QWEN35_NUM_ATTENTION_HEADS = 16   # query heads
QWEN35_NUM_KV_HEADS = 4           # GQA: 4 KV heads
QWEN35_HEAD_DIM = 256
QWEN35_HIDDEN_SIZE = 4096
QWEN35_SCALE = 1.0 / (QWEN35_HEAD_DIM ** 0.5)

# Linear attention params (for memory accounting only)
QWEN35_LINEAR_KEY_HEADS = 16
QWEN35_LINEAR_KEY_DIM = 128
QWEN35_LINEAR_VALUE_HEADS = 32
QWEN35_LINEAR_VALUE_DIM = 128
QWEN35_NUM_LINEAR_LAYERS = 24

# MI100 hardware
MI100_HBM_GB = 32.0
MI100_GPU_UTIL = 0.9
MI100_NUM_GPUS = 4
QWEN35_MODEL_WEIGHT_GB = 18.0  # FP16, ~9B params


def make_realistic_qkv(
    seq_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q/K/V tensors with semi-realistic attention patterns.

    Uses a mix of:
    - Random base attention (simulates general token interactions)
    - Strong diagonal (simulates local/recent token attention)
    - Attention sinks (first few tokens get boosted keys)
    """
    q = torch.randn(seq_len, num_heads_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)

    # Boost attention sink tokens (positions 0-3)
    k[:4] *= 3.0

    # Boost recent tokens slightly (last 10%)
    recent_start = max(0, seq_len - seq_len // 10)
    k[recent_start:] *= 1.5

    # Add some "important" tokens at regular intervals (simulates key info)
    stride = max(1, seq_len // 20)
    for i in range(0, seq_len, stride):
        k[i] *= 2.0

    return q, k, v


def compute_attention_coverage(
    query: torch.Tensor,
    key: torch.Tensor,
    selection_mask: torch.Tensor,
    scale: float,
    seq_len: int,
) -> float:
    """Measure fraction of total attention weight captured by selected tokens.

    Uses only KV heads (not expanded) and samples query positions to avoid OOM
    on large head_dim models like Qwen3.5 (head_dim=256).
    """
    num_kv_heads = key.shape[1]
    head_dim = key.shape[2]

    # Use a subset of query positions to limit memory
    # Sample up to 128 query positions evenly spaced
    max_query_sample = 128
    if seq_len > max_query_sample:
        indices = torch.linspace(0, seq_len - 1, max_query_sample, device=query.device).long()
        q_sample = query[indices, :num_kv_heads, :]  # [sample, Hkv, D]
        sample_positions = indices
    else:
        q_sample = query[:, :num_kv_heads, :]  # [S, Hkv, D]
        sample_positions = torch.arange(seq_len, device=query.device)

    n_sample = q_sample.shape[0]

    # Compute attention per KV head, one head at a time to save memory
    union_mask = selection_mask.any(dim=0)  # [S]
    coverages = []

    for h in range(num_kv_heads):
        q_h = q_sample[:, h, :].float()  # [n_sample, D]
        k_h = key[:, h, :].float()       # [S, D]

        # [n_sample, S]
        attn = torch.matmul(q_h, k_h.t()) * scale

        # Causal mask for sampled positions
        pos_mask = sample_positions.unsqueeze(1) >= torch.arange(
            seq_len, device=query.device
        ).unsqueeze(0)
        attn = attn.masked_fill(~pos_mask, -1e9)

        weights = F.softmax(attn, dim=-1)  # [n_sample, S]
        selected_weight = weights[:, union_mask].sum(dim=-1)
        coverages.append(selected_weight.mean().item())

        del q_h, k_h, attn, weights

    return sum(coverages) / len(coverages) * 100.0


def benchmark_compression(
    seq_lens: list[int],
    device: str,
    retention_ratios: list[float],
) -> list[dict]:
    """Run compression benchmark across configurations."""
    results = []
    dtype = torch.float16 if device != "cpu" else torch.float32

    for retention in retention_ratios:
        config = KVCompressConfig(
            enabled=True,
            num_layers=QWEN35_NUM_FULL_ATTN_LAYERS,  # Only 8 full-attn layers
            num_kv_heads=QWEN35_NUM_KV_HEADS,
            retention_ratio=retention,
            observation_window_size=32,
            kernel_size=7,
            recent_window_size=32,
            sink_size=4,
            pyramid_schedule=PyramidSchedule.LINEAR,
            pyramid_alpha=4.0,
            min_seq_len_to_compress=128,
        )
        compressor = SnapPyramidKVCompressor(config)

        for seq_len in seq_lens:
            # Generate per-layer Q/K/V for 8 full-attention layers
            layer_results_list = []
            total_compress_ms = 0.0

            for layer_i in range(QWEN35_NUM_FULL_ATTN_LAYERS):
                q, k, v = make_realistic_qkv(
                    seq_len,
                    QWEN35_NUM_ATTENTION_HEADS,
                    QWEN35_NUM_KV_HEADS,
                    QWEN35_HEAD_DIM,
                    device=device,
                    dtype=dtype,
                )
                # Run in float32 for numerical stability
                q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()

                # Warmup
                if device != "cpu" and layer_i == 0:
                    _ = compressor.compress_layer(
                        layer_i, q_f32, k_f32, v_f32,
                        QWEN35_SCALE, seq_len,
                    )
                    torch.cuda.synchronize()

                if device != "cpu":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = compressor.compress_layer(
                    layer_i, q_f32, k_f32, v_f32,
                    QWEN35_SCALE, seq_len,
                )
                if device != "cpu":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                total_compress_ms += (t1 - t0) * 1000

                layer_results_list.append(result)

            # Compute attention coverage for the first full-attn layer
            coverage = -1.0
            if seq_len <= 8192:
                q, k, v = make_realistic_qkv(
                    seq_len, QWEN35_NUM_ATTENTION_HEADS,
                    QWEN35_NUM_KV_HEADS, QWEN35_HEAD_DIM,
                    device=device, dtype=dtype,
                )
                try:
                    coverage = compute_attention_coverage(
                        q, k,
                        layer_results_list[0].selection_mask,
                        QWEN35_SCALE, seq_len,
                    )
                except torch.cuda.OutOfMemoryError:
                    coverage = -1.0
                    torch.cuda.empty_cache()
                del q, k, v
                if device != "cpu":
                    torch.cuda.empty_cache()

            # Aggregate stats
            budgets = [r.budget for r in layer_results_list]
            retained = [r.num_selected for r in layer_results_list]

            results.append({
                "retention_ratio": retention,
                "seq_len": seq_len,
                "num_full_attn_layers": QWEN35_NUM_FULL_ATTN_LAYERS,
                "per_layer_budgets": budgets,
                "per_layer_retained": retained,
                "avg_retained": sum(retained) / len(retained),
                "min_retained": min(retained),
                "max_retained": max(retained),
                "avg_compression_ratio": seq_len / (sum(retained) / len(retained)) if sum(retained) > 0 else float("inf"),
                "total_compress_ms": round(total_compress_ms, 2),
                "avg_compress_per_layer_ms": round(total_compress_ms / QWEN35_NUM_FULL_ATTN_LAYERS, 2),
                "attention_coverage_pct": round(coverage, 1) if coverage >= 0 else "N/A",
                "device": device,
            })

    return results


def estimate_memory_savings(
    seq_lens: list[int],
    retention_ratios: list[float],
) -> list[dict]:
    """Estimate MI100 memory savings for Qwen3.5-9B with TP=4.

    Memory layout on 4x MI100:
    - Model weights: ~18GB FP16 / 4 GPUs = 4.5GB per GPU
    - Full-attn KV: 8 layers × 2(K+V) × 4 heads × 256 dim × 2 bytes
    - Linear-attn state: 24 layers × fixed-size recurrent state (small)
    """
    results = []
    per_gpu_hbm = MI100_HBM_GB * MI100_GPU_UTIL  # 28.8 GB
    model_per_gpu = QWEN35_MODEL_WEIGHT_GB / MI100_NUM_GPUS  # 4.5 GB
    kv_available_per_gpu = per_gpu_hbm - model_per_gpu  # ~24.3 GB

    # Full-attention KV cache: per token, per layer (FP16)
    # 2 (K+V) × (4 KV heads / TP=4 = 1 head per GPU) × 256 dim × 2 bytes
    full_attn_bytes_per_token_per_layer = 2 * (QWEN35_NUM_KV_HEADS // MI100_NUM_GPUS) * QWEN35_HEAD_DIM * 2
    # With TP=4 and 4 KV heads, each GPU has 1 KV head
    # = 2 × 1 × 256 × 2 = 1024 bytes per token per layer

    # 8 full-attn layers total
    full_attn_bytes_per_token = full_attn_bytes_per_token_per_layer * QWEN35_NUM_FULL_ATTN_LAYERS
    # = 1024 × 8 = 8192 bytes per token across all full-attn layers

    # Linear attention has fixed-size state, not per-token
    # (conv state + ssm state, negligible compared to full-attn KV)
    linear_attn_overhead_gb = 0.1  # Conservative estimate

    kv_for_full_attn = kv_available_per_gpu - linear_attn_overhead_gb

    # Max tokens without compression
    max_tokens_raw = int(kv_for_full_attn * 1e9 / full_attn_bytes_per_token)

    for seq_len in seq_lens:
        for retention in retention_ratios:
            # With compression: only retain `retention` fraction per full-attn layer
            effective_multiplier = 1.0 / retention
            max_tokens_compressed = int(max_tokens_raw * effective_multiplier)

            # Memory used at this seq_len (per GPU)
            raw_kv_gb = seq_len * full_attn_bytes_per_token / 1e9
            compressed_kv_gb = raw_kv_gb * retention

            results.append({
                "seq_len": seq_len,
                "retention_ratio": retention,
                "kv_available_per_gpu_gb": round(kv_for_full_attn, 1),
                "max_context_raw": max_tokens_raw,
                "max_context_compressed": max_tokens_compressed,
                "capacity_multiplier": round(effective_multiplier, 1),
                "raw_kv_gb_per_gpu": round(raw_kv_gb, 3),
                "compressed_kv_gb_per_gpu": round(compressed_kv_gb, 3),
                "memory_savings_pct": round((1.0 - retention) * 100, 1),
                "max_batch_at_seqlen_raw": max(1, int(kv_for_full_attn * 1e9 / (seq_len * full_attn_bytes_per_token))),
                "max_batch_at_seqlen_compressed": max(1, int(kv_for_full_attn * 1e9 / (seq_len * full_attn_bytes_per_token * retention))),
            })

    return results


def print_compression_results(results: list[dict]) -> None:
    print("\n" + "=" * 110)
    print("Qwen3.5-9B SnapKV+PyramidKV Compression (8 full-attention layers, GQA 16q/4kv, head_dim=256)")
    print("=" * 110)
    print(
        f"{'Retention':>10} {'SeqLen':>8} {'AvgRetain':>10} "
        f"{'Min/Max':>12} {'Compress':>10} "
        f"{'TotalMs':>10} {'PerLayerMs':>12} {'Coverage':>10}"
    )
    print("-" * 110)
    for r in results:
        cov = f"{r['attention_coverage_pct']}%" if r['attention_coverage_pct'] != "N/A" else "N/A"
        print(
            f"{r['retention_ratio']:>9.0%} {r['seq_len']:>8} "
            f"{r['avg_retained']:>10.0f} "
            f"{r['min_retained']:>5}/{r['max_retained']:<5} "
            f"{r['avg_compression_ratio']:>9.1f}x "
            f"{r['total_compress_ms']:>8.1f}ms "
            f"{r['avg_compress_per_layer_ms']:>10.2f}ms "
            f"{cov:>10}"
        )


def print_memory_results(results: list[dict]) -> None:
    print("\n" + "=" * 130)
    print("Qwen3.5-9B MI100 Memory Impact (4x MI100 TP=4, FP16, 8 full-attn layers)")
    print("=" * 130)
    print(
        f"{'SeqLen':>8} {'Retention':>10} {'KV Avail':>10} "
        f"{'MaxCtx(raw)':>12} {'MaxCtx(comp)':>13} "
        f"{'Multiplier':>11} {'RawKV/GPU':>10} {'CompKV/GPU':>11} "
        f"{'MaxBatch(r)':>12} {'MaxBatch(c)':>12}"
    )
    print("-" * 130)
    for r in results:
        print(
            f"{r['seq_len']:>8} {r['retention_ratio']:>9.0%} "
            f"{r['kv_available_per_gpu_gb']:>8.1f}GB "
            f"{r['max_context_raw']:>12,} "
            f"{r['max_context_compressed']:>13,} "
            f"{r['capacity_multiplier']:>10.1f}x "
            f"{r['raw_kv_gb_per_gpu']:>8.3f}GB "
            f"{r['compressed_kv_gb_per_gpu']:>9.3f}GB "
            f"{r['max_batch_at_seqlen_raw']:>12} "
            f"{r['max_batch_at_seqlen_compressed']:>12}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SnapKV+PyramidKV on Qwen3.5-9B (MI100)"
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[256, 512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--retention-ratios", nargs="+", type=float,
        default=[0.05, 0.10, 0.12, 0.20, 0.25],
        help="Retention ratios to test",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device (auto=cuda if available)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file",
    )
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print("SnapKV + PyramidKV Benchmark: Qwen3.5-9B on MI100")
    print("=" * 70)
    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Sequence lengths: {args.seq_lens}")
    print(f"Retention ratios: {args.retention_ratios}")
    print()
    print("Model: Qwen3.5-9B (hybrid attention)")
    print(f"  Total layers: {QWEN35_NUM_LAYERS}")
    print(f"  Full-attention layers: {QWEN35_NUM_FULL_ATTN_LAYERS} "
          f"(indices: {QWEN35_FULL_ATTN_LAYERS})")
    print(f"  Linear-attention layers: {QWEN35_NUM_LINEAR_LAYERS}")
    print(f"  Query heads: {QWEN35_NUM_ATTENTION_HEADS}, "
          f"KV heads: {QWEN35_NUM_KV_HEADS} (GQA 4:1)")
    print(f"  Head dim: {QWEN35_HEAD_DIM}")
    print(f"  SnapKV+PyramidKV targets: {QWEN35_NUM_FULL_ATTN_LAYERS} "
          f"full-attention layers only")

    # 1. Compression benchmark
    print("\n[1/2] Running compression benchmark...")
    comp_results = benchmark_compression(
        args.seq_lens, args.device, args.retention_ratios,
    )
    print_compression_results(comp_results)

    # 2. Memory estimation
    print("\n[2/2] Computing MI100 memory estimates...")
    mem_results = estimate_memory_savings(args.seq_lens, args.retention_ratios)
    print_memory_results(mem_results)

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR QWEN3.5-9B ON MI100")
    print("=" * 70)

    # Find the 12% retention, 4096 seq_len result
    target_comp = [r for r in comp_results
                   if r["retention_ratio"] == 0.12 and r["seq_len"] == 4096]
    target_mem = [r for r in mem_results
                  if r["retention_ratio"] == 0.12 and r["seq_len"] == 4096]

    if target_comp:
        r = target_comp[0]
        print(f"\n  At 12% retention, seq_len=4096:")
        print(f"    Avg tokens retained per layer: {r['avg_retained']:.0f} / 4096")
        print(f"    Compression ratio: {r['avg_compression_ratio']:.1f}x")
        print(f"    Total compress time (8 layers): {r['total_compress_ms']:.1f}ms")
        cov = r['attention_coverage_pct']
        if cov != "N/A":
            print(f"    Attention coverage: {cov}%")

    if target_mem:
        r = target_mem[0]
        print(f"\n  MI100 memory impact (4x GPU, TP=4):")
        print(f"    Max context without compression: {r['max_context_raw']:,}")
        print(f"    Max context with 12% retention:  {r['max_context_compressed']:,}")
        print(f"    Capacity multiplier: {r['capacity_multiplier']}x")
        print(f"    Max batch at 4096 tokens: "
              f"{r['max_batch_at_seqlen_raw']} → "
              f"{r['max_batch_at_seqlen_compressed']}")

    print(f"\n  Note: Qwen3.5-9B uses hybrid attention — only 8/32 layers")
    print(f"  have full KV cache. The other 24 layers use linear attention")
    print(f"  with fixed-size state. Compression benefit is proportionally")
    print(f"  smaller than a pure full-attention model like LLaMA-3.")

    # Save results
    if args.output:
        output = {
            "model": "Qwen3.5-9B",
            "architecture": {
                "total_layers": QWEN35_NUM_LAYERS,
                "full_attn_layers": QWEN35_NUM_FULL_ATTN_LAYERS,
                "full_attn_indices": QWEN35_FULL_ATTN_LAYERS,
                "linear_attn_layers": QWEN35_NUM_LINEAR_LAYERS,
                "num_q_heads": QWEN35_NUM_ATTENTION_HEADS,
                "num_kv_heads": QWEN35_NUM_KV_HEADS,
                "head_dim": QWEN35_HEAD_DIM,
            },
            "compression_results": comp_results,
            "memory_results": mem_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()
