# SPDX-License-Identifier: Apache-2.0
"""Benchmark for LazyLLM dynamic token pruning during prefill.

Simulates the prefill forward pass with progressive token pruning,
measuring compute savings (FLOPs), latency, and compression ratios
for various configurations and sequence lengths.

Target: Qwen3.5-9B on MI100 (32GB HBM2e).

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/lazy_llm/benchmark_lazy_llm.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.attention.lazy_llm.config import LazyLLMConfig, PruningSchedule
from vllm.attention.lazy_llm.pruner import ProgressiveTokenPruner


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int


QWEN35_9B = ModelSpec(
    name="Qwen3.5-9B",
    num_layers=40,
    hidden_size=3584,
    num_heads=28,
    num_kv_heads=4,
    head_dim=128,
    intermediate_size=18944,
)

LLAMA2_7B = ModelSpec(
    name="LLaMA-2-7B",
    num_layers=32,
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=32,
    head_dim=128,
    intermediate_size=11008,
)


# ---------------------------------------------------------------------------
# FLOP estimation
# ---------------------------------------------------------------------------
def estimate_layer_flops(
    seq_len: int, spec: ModelSpec
) -> dict[str, int]:
    """Estimate FLOPs for one transformer layer during prefill."""
    S = seq_len
    H = spec.hidden_size
    D = spec.head_dim
    Nh = spec.num_heads
    Nkv = spec.num_kv_heads
    I = spec.intermediate_size

    # QKV projection: 3 * (2 * S * H * H) for MHA, adjusted for GQA
    qkv_flops = 2 * S * H * (Nh * D + 2 * Nkv * D)
    # Attention: Q*K^T = 2*Nh*S*S*D, softmax ~ 5*Nh*S*S, attn*V = 2*Nh*S*S*D
    attn_flops = 2 * Nh * S * S * D + 5 * Nh * S * S + 2 * Nh * S * S * D
    # Output projection: 2 * S * Nh*D * H
    o_proj_flops = 2 * S * Nh * D * H
    # MLP: gate_up (2 * S * H * 2*I) + down (2 * S * I * H)
    mlp_flops = 2 * S * H * 2 * I + 2 * S * I * H

    return {
        "qkv": qkv_flops,
        "attention": attn_flops,
        "o_proj": o_proj_flops,
        "mlp": mlp_flops,
        "total": qkv_flops + attn_flops + o_proj_flops + mlp_flops,
    }


def estimate_pruned_flops(
    original_seq_len: int,
    per_layer_active: list[int],
    spec: ModelSpec,
) -> tuple[int, int]:
    """Estimate total FLOPs with and without pruning.

    Returns (baseline_flops, pruned_flops).
    """
    baseline_total = 0
    pruned_total = 0
    for layer_idx in range(spec.num_layers):
        baseline = estimate_layer_flops(original_seq_len, spec)
        baseline_total += baseline["total"]

        active = per_layer_active[layer_idx] if layer_idx < len(per_layer_active) else original_seq_len
        pruned = estimate_layer_flops(active, spec)
        pruned_total += pruned["total"]

    return baseline_total, pruned_total


# ---------------------------------------------------------------------------
# Simulated prefill benchmark
# ---------------------------------------------------------------------------
def simulate_prefill_with_pruning(
    seq_len: int,
    spec: ModelSpec,
    config: LazyLLMConfig,
    device: torch.device,
    warmup: int = 3,
    repeat: int = 10,
) -> dict:
    """Simulate prefill with LazyLLM pruning, measuring latency.

    Simulates the key compute-intensive operations:
    1. QKV projection (linear layer)
    2. Q*K^T attention scoring (for importance)
    3. Token selection
    4. MLP (linear layer)

    Returns timing and compression metrics.
    """
    pruner = ProgressiveTokenPruner(config)
    hidden_dim = spec.hidden_size
    num_heads = spec.num_heads
    num_kv_heads = spec.num_kv_heads
    head_dim = spec.head_dim

    # Create simulated weights (small for benchmark, just to get timing)
    # We only simulate the compute-bound ops
    qkv_weight = torch.randn(
        hidden_dim, (num_heads + 2 * num_kv_heads) * head_dim,
        device=device, dtype=torch.float16,
    )
    o_weight = torch.randn(
        num_heads * head_dim, hidden_dim,
        device=device, dtype=torch.float16,
    )
    mlp_gate_up = torch.randn(
        hidden_dim, spec.intermediate_size * 2,
        device=device, dtype=torch.float16,
    )
    mlp_down = torch.randn(
        spec.intermediate_size, hidden_dim,
        device=device, dtype=torch.float16,
    )

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    def run_prefill_baseline():
        """Standard prefill: all tokens through all layers."""
        hidden = torch.randn(seq_len, hidden_dim, device=device, dtype=torch.float16)
        for _ in range(spec.num_layers):
            qkv = hidden @ qkv_weight
            q = qkv[:, :q_size]
            k = qkv[:, q_size:q_size + kv_size]
            v = qkv[:, q_size + kv_size:]
            # Skip full attention (quadratic), just do o_proj + MLP
            attn_out = hidden[:, :num_heads * head_dim]  # placeholder
            hidden = attn_out @ o_weight
            gate_up = hidden @ mlp_gate_up
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = (F.silu(gate) * up) @ mlp_down
        return hidden

    def run_prefill_lazyllm():
        """LazyLLM prefill: progressive token pruning."""
        hidden = torch.randn(seq_len, hidden_dim, device=device, dtype=torch.float16)
        state = pruner.init_prefill_state(seq_len, device)
        per_layer_active = []

        for layer_idx in range(spec.num_layers):
            cur_len = hidden.shape[0]
            per_layer_active.append(cur_len)

            qkv = hidden @ qkv_weight
            q = qkv[:, :q_size]
            k = qkv[:, q_size:q_size + kv_size]
            v = qkv[:, q_size + kv_size:]

            # Reshape for importance scoring
            q_3d = q.view(cur_len, num_heads, head_dim)
            k_3d = k.view(cur_len, num_kv_heads, head_dim)

            attn_out = hidden[:, :num_heads * head_dim]
            hidden = attn_out @ o_weight
            gate_up = hidden @ mlp_gate_up
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = (F.silu(gate) * up) @ mlp_down

            # Prune after layer computation
            hidden, state = pruner.maybe_prune_tokens(
                layer_idx=layer_idx,
                hidden_states=hidden.float(),
                state=state,
                query=q_3d.float(),
                key=k_3d.float(),
                scale=head_dim ** -0.5,
            )
            hidden = hidden.half()

        return hidden, state, per_layer_active

    # Warmup
    if device.type == "cuda":
        for _ in range(warmup):
            run_prefill_baseline()
            torch.cuda.synchronize()
        for _ in range(warmup):
            run_prefill_lazyllm()
            torch.cuda.synchronize()

    # Benchmark baseline
    if device.type == "cuda":
        torch.cuda.synchronize()
    baseline_times = []
    for _ in range(repeat):
        start = time.perf_counter()
        run_prefill_baseline()
        if device.type == "cuda":
            torch.cuda.synchronize()
        baseline_times.append(time.perf_counter() - start)

    # Benchmark LazyLLM
    if device.type == "cuda":
        torch.cuda.synchronize()
    lazyllm_times = []
    final_state = None
    final_per_layer = None
    for i in range(repeat):
        start = time.perf_counter()
        _, state, per_layer = run_prefill_lazyllm()
        if device.type == "cuda":
            torch.cuda.synchronize()
        lazyllm_times.append(time.perf_counter() - start)
        if i == repeat - 1:
            final_state = state
            final_per_layer = per_layer

    baseline_avg = sum(baseline_times) / len(baseline_times)
    lazyllm_avg = sum(lazyllm_times) / len(lazyllm_times)

    # FLOP estimation
    baseline_flops, pruned_flops = estimate_pruned_flops(
        seq_len, final_per_layer, spec
    )

    summary = pruner.get_pruning_summary(final_state)

    return {
        "seq_len": seq_len,
        "baseline_ms": baseline_avg * 1000,
        "lazyllm_ms": lazyllm_avg * 1000,
        "speedup": baseline_avg / lazyllm_avg if lazyllm_avg > 0 else 0,
        "baseline_tflops": baseline_flops / 1e12,
        "pruned_tflops": pruned_flops / 1e12,
        "flop_reduction": 1.0 - (pruned_flops / baseline_flops) if baseline_flops > 0 else 0,
        "final_active_tokens": final_state.num_active,
        "compression_ratio": final_state.compression_ratio,
        "per_layer_active": final_per_layer,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Compute-only analysis (no GPU needed)
# ---------------------------------------------------------------------------
def analyze_flop_savings(
    spec: ModelSpec,
    seq_lengths: list[int],
    configs: list[tuple[str, LazyLLMConfig]],
):
    """Analyze theoretical FLOP savings for various configs and seq lengths."""
    print(f"\n{'='*80}")
    print(f"  FLOP Savings Analysis: {spec.name}")
    print(f"  {spec.num_layers} layers, {spec.num_heads} heads, "
          f"{spec.num_kv_heads} KV heads, hidden={spec.hidden_size}")
    print(f"{'='*80}")

    for name, config in configs:
        print(f"\n--- Config: {name} (drop={config.drop_ratio}, "
              f"schedule={config.schedule.value}, start_layer={config.start_layer}) ---")
        print(f"{'Seq Len':>10} | {'Baseline TF':>12} | {'Pruned TF':>12} | "
              f"{'Saved %':>8} | {'Final Tokens':>12} | {'Compress':>8}")
        print("-" * 80)

        for seq_len in seq_lengths:
            pruner = ProgressiveTokenPruner(config)
            state = pruner.init_prefill_state(seq_len, torch.device("cpu"))

            # Simulate pruning decisions (use random importance)
            per_layer_active = []
            cur_len = seq_len
            dummy_hidden = torch.randn(cur_len, spec.hidden_size)

            for layer_idx in range(spec.num_layers):
                per_layer_active.append(cur_len)
                if config.should_prune(layer_idx, state.seq_len):
                    num_keep = config.get_num_tokens_to_keep(layer_idx, cur_len)
                    if num_keep < cur_len:
                        importance = torch.randn(cur_len)
                        kept = torch.topk(importance, num_keep).indices.sort().values
                        dummy_hidden = dummy_hidden[kept]
                        cur_len = num_keep

                        # Update state manually for tracking
                        state.active_indices = state.active_indices[
                            kept if kept.max() < state.active_indices.shape[0]
                            else torch.arange(num_keep)
                        ]

            baseline_flops, pruned_flops = estimate_pruned_flops(
                seq_len, per_layer_active, spec
            )
            saved_pct = (1.0 - pruned_flops / baseline_flops) * 100 if baseline_flops > 0 else 0

            print(f"{seq_len:>10,} | {baseline_flops/1e12:>11.2f}T | "
                  f"{pruned_flops/1e12:>11.2f}T | {saved_pct:>7.1f}% | "
                  f"{cur_len:>12,} | {cur_len/seq_len:>7.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LazyLLM Benchmark")
    parser.add_argument("--model", choices=["qwen35", "llama2"], default="qwen35")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--mode", choices=["flops", "latency", "both"], default="flops",
                        help="flops=theoretical analysis, latency=simulated timing")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    spec = QWEN35_9B if args.model == "qwen35" else LLAMA2_7B

    # Define configs to compare
    configs = [
        ("conservative", LazyLLMConfig(
            num_layers=spec.num_layers, drop_ratio=0.3,
            schedule=PruningSchedule.LINEAR, start_layer=4,
            protected_tokens=4, min_tokens=64, min_seq_len=256,
        )),
        ("moderate", LazyLLMConfig(
            num_layers=spec.num_layers, drop_ratio=0.5,
            schedule=PruningSchedule.LINEAR, start_layer=2,
            protected_tokens=4, min_tokens=64, min_seq_len=256,
        )),
        ("aggressive", LazyLLMConfig(
            num_layers=spec.num_layers, drop_ratio=0.7,
            schedule=PruningSchedule.LINEAR, start_layer=2,
            protected_tokens=4, min_tokens=64, min_seq_len=256,
        )),
        ("exponential", LazyLLMConfig(
            num_layers=spec.num_layers, drop_ratio=0.5,
            schedule=PruningSchedule.EXPONENTIAL, start_layer=2,
            protected_tokens=4, min_tokens=64, min_seq_len=256,
        )),
    ]

    if args.mode in ("flops", "both"):
        analyze_flop_savings(spec, args.seq_lengths, configs)

    if args.mode in ("latency", "both"):
        device = torch.device(args.device)
        print(f"\n{'='*80}")
        print(f"  Latency Benchmark: {spec.name} on {device}")
        print(f"{'='*80}")

        # Only benchmark a subset of seq lengths for latency (to save time)
        latency_seq_lens = [s for s in args.seq_lengths if s <= 8192]

        for name, config in configs:
            print(f"\n--- Config: {name} ---")
            print(f"{'Seq Len':>10} | {'Baseline ms':>12} | {'LazyLLM ms':>12} | "
                  f"{'Speedup':>8} | {'FLOP Saved':>10} | {'Final Tok':>10}")
            print("-" * 80)

            for seq_len in latency_seq_lens:
                try:
                    result = simulate_prefill_with_pruning(
                        seq_len, spec, config, device,
                        warmup=args.warmup, repeat=args.repeat,
                    )
                    print(
                        f"{seq_len:>10,} | {result['baseline_ms']:>11.2f} | "
                        f"{result['lazyllm_ms']:>11.2f} | "
                        f"{result['speedup']:>7.2f}x | "
                        f"{result['flop_reduction']*100:>9.1f}% | "
                        f"{result['final_active_tokens']:>10,}"
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"{seq_len:>10,} | OOM")
                    break

    print("\nDone.")


if __name__ == "__main__":
    main()
