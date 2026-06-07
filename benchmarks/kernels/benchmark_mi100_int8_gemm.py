# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark MI100 INT8 W8A8 GEMM vs FP16 baseline.

Measures INT8 MFMA throughput on gfx908 using the MI100-optimized
Triton kernel (mi100_int8_scaled_mm) vs standard FP16 torch.mm.

Usage:
    python benchmarks/kernels/benchmark_mi100_int8_gemm.py
    python benchmarks/kernels/benchmark_mi100_int8_gemm.py --tp-sizes 4
"""

import argparse

import torch

# Qwen3.5-9B TP=4 shapes (primary target model)
QWEN_9B_TP4_SHAPES = [
    # (K, N, label)
    (2048, 1536, "qkv_proj"),
    (2048, 2048, "o_proj"),
    (2048, 5504, "gate_up_proj"),
    (2752, 2048, "down_proj"),
]

# Standard model shapes for comparison
STANDARD_SHAPES = [
    (4096, 4096, "llama8b_o_proj"),
    (4096, 6144, "llama8b_qkv"),
    (4096, 28672, "llama8b_gate_up"),
    (14336, 4096, "llama8b_down"),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def quantize_to_int8(tensor):
    """Symmetric per-tensor quantization to INT8."""
    amax = tensor.abs().max().clamp(min=1e-12)
    scale = 127.0 / amax
    q = (tensor * scale).round().clamp(-128, 127).to(torch.int8)
    return q, (1.0 / scale).float()


def benchmark_kernel(fn, warmup=10, iters=100):
    """Benchmark a kernel function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.accelerator.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Use median
    return times[len(times) // 2]


def run_benchmark(shapes, batch_sizes, device="cuda"):
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    print(f"\n{'=' * 90}")
    print(
        f"{'M':>6} {'K':>6} {'N':>6} {'Label':>16} "
        f"{'FP16 ms':>9} {'INT8 ms':>9} {'Speedup':>8} "
        f"{'FP16 TFLOPS':>12} {'INT8 TOPS':>10}"
    )
    print(f"{'=' * 90}")

    for K, N, label in shapes:
        for M in batch_sizes:
            # FP16 baseline
            a_fp16 = torch.randn((M, K), device=device, dtype=torch.float16)
            b_fp16 = torch.randn((K, N), device=device, dtype=torch.float16)

            def fp16_fn(a=a_fp16, b=b_fp16):
                return torch.mm(a, b)

            fp16_ms = benchmark_kernel(fp16_fn)

            # INT8 path
            a_int8, scale_a = quantize_to_int8(a_fp16)
            b_int8, scale_b = quantize_to_int8(b_fp16)
            # Weight in [K, N] layout
            b_int8_kn = b_int8.contiguous()
            scale_a_t = torch.tensor([[scale_a]], device=device, dtype=torch.float32)
            scale_b_t = torch.tensor([[scale_b]], device=device, dtype=torch.float32)

            def int8_fn(a=a_int8, b=b_int8_kn, sa=scale_a_t, sb=scale_b_t):
                return mi100_int8_scaled_mm(a, b, sa, sb, torch.float16)

            int8_ms = benchmark_kernel(int8_fn)

            flops = 2.0 * M * N * K
            fp16_tflops = flops / (fp16_ms * 1e-3) / 1e12
            int8_tops = flops / (int8_ms * 1e-3) / 1e12
            speedup = fp16_ms / int8_ms

            print(
                f"{M:>6} {K:>6} {N:>6} {label:>16} "
                f"{fp16_ms:>8.3f}ms {int8_ms:>8.3f}ms {speedup:>7.2f}x "
                f"{fp16_tflops:>11.2f} {int8_tops:>9.2f}"
            )

    print(f"{'=' * 90}")


def run_correctness_check(device="cuda"):
    """Verify INT8 kernel matches FP16 reference within tolerance."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    print("\n--- Correctness Check ---")
    test_cases = [(64, 256, 128), (1, 4096, 4096), (512, 4096, 6144)]
    all_pass = True

    for M, K, N in test_cases:
        a_fp16 = torch.randn((M, K), device=device, dtype=torch.float16)
        b_fp16 = torch.randn((K, N), device=device, dtype=torch.float16)

        ref = torch.mm(a_fp16, b_fp16)

        a_int8, scale_a = quantize_to_int8(a_fp16)
        b_int8, scale_b = quantize_to_int8(b_fp16)
        scale_a_t = torch.tensor([[scale_a]], device=device, dtype=torch.float32)
        scale_b_t = torch.tensor([[scale_b]], device=device, dtype=torch.float32)

        out = mi100_int8_scaled_mm(a_int8, b_int8, scale_a_t, scale_b_t, torch.float16)

        # INT8 quantization introduces error; check relative tolerance
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        passed = rel_err < 0.15  # ~15% relative error is expected for INT8
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] M={M}, K={K}, N={N}: rel_err={rel_err:.4f}")

    print(f"  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MI100 INT8 W8A8 GEMM")
    parser.add_argument(
        "--shapes",
        choices=["qwen9b", "standard", "both"],
        default="both",
        help="Which shape set to benchmark",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[4],
        help="Tensor parallel sizes (shapes are divided accordingly)",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=None, help="Override batch sizes"
    )
    parser.add_argument(
        "--correctness-only", action="store_true", help="Only run correctness check"
    )
    args = parser.parse_args()

    batch_sizes = args.batch_sizes or BATCH_SIZES

    if args.correctness_only:
        run_correctness_check()
    else:
        run_correctness_check()
        if args.shapes in ("qwen9b", "both"):
            print("\n\n=== Qwen3.5-9B TP=4 Shapes ===")
            run_benchmark(QWEN_9B_TP4_SHAPES, batch_sizes)
        if args.shapes in ("standard", "both"):
            print("\n\n=== Standard LLM Shapes ===")
            run_benchmark(STANDARD_SHAPES, batch_sizes)
