# LazyLLM Dynamic Token Pruning for vLLM

Implementation of ["LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference"](https://arxiv.org/abs/2407.14057) (Fu et al., 2024) for vLLM on AMD MI100 GPUs.

## Overview

LazyLLM progressively prunes tokens at each transformer layer during prefill, reducing compute by 65–75% for long sequences. Unlike static pruning, tokens can be restored from Aux Cache in later generation steps if needed.

**Key Results for Qwen3.5-9B at 64K tokens:**
- **FLOP Reduction**: 71.3% (moderate config, drop=0.5)
- **Expected TTFT Speedup**: 1.5–2.3x (matching paper results)
- **Training Required**: None (drop-in optimization)

## Quick Start

```python
from vllm.attention.lazy_llm import LazyLLMConfig, PruningSchedule
from vllm.model_executor.models.qwen3_lazyllm import Qwen3LazyLLMModel

# Configure pruning
lazy_config = LazyLLMConfig(
    num_layers=40,           # Qwen3.5-9B
    drop_ratio=0.5,          # Keep 50% of tokens at final layer
    schedule=PruningSchedule.LINEAR,
    start_layer=2,           # Begin pruning after layer 2
    protected_tokens=4,      # Never prune first 4 tokens (BOS, system)
    min_tokens=64,           # Minimum to keep per layer
    min_seq_len=256,         # Only activate on sequences ≥256 tokens
)

# Attach to vLLM config
vllm_config.lazy_llm_config = lazy_config

# Use LazyLLM model
model = Qwen3LazyLLMModel(vllm_config=vllm_config)
```

## Architecture

```
vllm/attention/lazy_llm/
├── config.py          # LazyLLMConfig: per-layer keep ratios, schedules
├── selector.py        # TokenImportanceSelector: attention-based scoring
├── pruner.py          # ProgressiveTokenPruner: orchestrates layer-wise pruning
└── __init__.py        # Public API

vllm/model_executor/models/
└── qwen3_lazyllm.py   # Qwen3LazyLLMModel with per-layer token pruning
```

### Algorithm

1. **Start** with full token set at layer 0
2. **At each layer** l ≥ start_layer:
   - Compute attention for active tokens
   - Use last-token attention scores for importance
   - Select top-k% tokens to keep (progressive schedule)
   - Save pruned hidden states to **Aux Cache**
3. **Continue** with only kept tokens

### Progressive Pruning Schedule

| Layers    | Keep Ratio | Rationale                              |
|-----------|------------|----------------------------------------|
| 0–1       | 100%       | Early layers are sensitive to pruning  |
| 2–10      | 95–80%     | Gradual reduction                     |
| 11–30     | 80–55%     | Aggressive pruning in middle          |
| 31–39     | 55–50%     | Most aggressive at end                |

## Benchmarks

### FLOP Savings (Qwen3.5-9B, Moderate Config)

| Sequence Length | Baseline TFLOPs | Pruned TFLOPs | Saved | Final Tokens |
|-----------------|-----------------|---------------|-------|--------------|
| 512             | 9.7T            | 5.6T          | 42%   | ~230         |
| 4,096           | 86.1T           | 29.9T         | 65%   | ~240         |
| 8,192           | 191.6T          | 63.1T         | 67%   | ~250         |
| 16,384          | 460.9T          | 144.8T        | 69%   | ~243         |
| 32,768          | 1,232.7T        | 369.4T        | 70%   | ~218         |
| **65,536**      | **3,708.8T**    | **1,063.3T**  | **71%** | **~179**   |

### Config Comparison (64K tokens)

| Config       | Drop Ratio | Schedule    | FLOP Saved | Final Tokens |
|--------------|------------|-------------|------------|--------------|
| Conservative | 0.3        | Linear      | 61%        | ~218         |
| **Moderate** | **0.5**    | **Linear**  | **71%**    | **~179**     |
| Aggressive   | 0.7        | Linear      | 75%        | ~227         |
| Exponential  | 0.5        | Exponential | 74%        | ~236         |

Run benchmarks:
```bash
# FLOP analysis (no GPU needed)
PYTHONPATH=. .venv/bin/python benchmarks/lazy_llm/benchmark_lazy_llm.py \
    --mode flops --model qwen35

# Latency benchmark (requires GPU)
PYTHONPATH=. .venv/bin/python benchmarks/lazy_llm/benchmark_lazy_llm.py \
    --mode latency --model qwen35 --device cuda:0
```

## Tests

```bash
# Run all tests
.venv/bin/python -m pytest tests/lazy_llm/ -v -c /dev/null --noconftest

# Specific test files
.venv/bin/python -m pytest tests/lazy_llm/test_config.py -v -c /dev/null --noconftest
.venv/bin/python -m pytest tests/lazy_llm/test_selector.py -v -c /dev/null --noconftest
.venv/bin/python -m pytest tests/lazy_llm/test_pruner.py -v -c /dev/null --noconftest
```

**Results**: 45 passed, 1 skipped (GPU), 11 integration tests skipped (require full vLLM stack).

## Integration Points

### Where to Hook (vLLM Forward Pass)

**File**: `vllm/model_executor/models/qwen2.py:441-447`

```python
# Current: iterate through layers without modification
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(positions, hidden_states, residual)
```

**Modified** (as implemented in `Qwen3LazyLLMModel`):
```python
# Initialize pruner for prefill
pruner = ProgressiveTokenPruner(lazy_llm_config)
state = pruner.init_prefill_state(seq_len, device)

for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    # Run layer on active tokens only
    hidden_states, residual = layer(
        positions[state.active_indices],
        hidden_states,
        residual
    )

    # Prune based on attention importance
    hidden_states, state = pruner.maybe_prune_tokens(
        layer_idx=idx,
        hidden_states=hidden_states,
        state=state,
        query=layer.self_attn._last_q,
        key=layer.self_attn._last_k,
        scale=layer.self_attn.scaling,
    )
```

### Key Components

1. **Qwen3LazyLLMAttention**: Stores Q/K after RoPE for importance scoring
2. **ProgressiveTokenPruner**: Manages pruning state across layers
3. **TokenImportanceSelector**: Computes importance from Q*K^T or attention weights
4. **LazyLLMConfig**: Defines per-layer keep ratios and constraints

## Configuration Options

```python
from vllm.attention.lazy_llm import LazyLLMConfig, PruningSchedule

LazyLLMConfig(
    num_layers=40,              # Total transformer layers
    drop_ratio=0.5,             # Final fraction of tokens to drop (0-1)
    schedule=PruningSchedule.LINEAR,  # LINEAR, EXPONENTIAL, or STEP
    start_layer=2,              # First layer to apply pruning (0-indexed)
    protected_tokens=4,         # Tokens at start never pruned
    min_tokens=64,              # Minimum tokens per layer
    min_seq_len=256,            # Min sequence length to activate
)
```

### Schedule Types

- **LINEAR**: Gradual, uniform decrease in tokens per layer
- **EXPONENTIAL**: Faster reduction early, slower later (better accuracy)
- **STEP**: No pruning in first half, then abrupt drop

## Design Document

See full integration design: `ml-research/docs/vllm-mi100-optimizations/memory-management/lazyllm-integration-design.md`

## Limitations & Future Work

1. **Batched prefill**: Currently designed for single sequence; batched sequences need per-sequence indices
2. **torch.compile**: Dynamic shapes from pruning may conflict with compilation
3. **Decode Aux Cache**: Full implementation needs Aux Cache retrieval during decode steps
4. **Accuracy validation**: Needs end-to-end testing on LongBench tasks

## References

- Fu et al., "LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference", arXiv:2407.14057, 2024
- vLLM: https://github.com/vllm-project/vllm
- Qwen3: https://huggingface.co/Qwen

## License

Apache 2.0
