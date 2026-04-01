# SPDX-License-Identifier: Apache-2.0
"""LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference.

Reference: Fu et al., 2024 — arXiv:2407.14057

This module implements the LazyLLM algorithm which dynamically selects
different subsets of tokens at each transformer layer during prefill,
progressively pruning unimportant tokens to reduce computation.
Key properties:
  - Training-free, model-agnostic
  - Progressive: keeps more tokens in early layers, fewer in later layers
  - Tokens pruned at layer l can return in later generation steps via Aux Cache
  - Never slower than baseline (worst case = full computation)
"""

from vllm.attention.lazy_llm.config import LazyLLMConfig
from vllm.attention.lazy_llm.selector import TokenImportanceSelector
from vllm.attention.lazy_llm.pruner import ProgressiveTokenPruner

__all__ = [
    "LazyLLMConfig",
    "TokenImportanceSelector",
    "ProgressiveTokenPruner",
]
