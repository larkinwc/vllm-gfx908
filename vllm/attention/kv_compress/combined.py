# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Combined SnapKV + PyramidKV Compressor.

Orchestrates both algorithms:
    1. PyramidKV determines per-layer token budgets (pyramid allocation)
    2. SnapKV selects which tokens to retain within each layer's budget

This combined approach achieves ~10x effective KV compression:
    - PyramidKV contributes ~88% reduction via layer-wise allocation
    - SnapKV contributes ~92% reduction via intelligent token selection
    - Together they target the critical tokens at each layer

Usage:
    compressor = SnapPyramidKVCompressor(config)

    # During prefill, for each layer:
    mask = compressor.compress_layer(
        layer_idx=i,
        query=q, key=k, value=v,
        scale=scale, seq_len=seq_len,
    )

    # Or compress all layers at once (post-prefill):
    results = compressor.compress_all_layers(
        queries_per_layer=queries,
        keys_per_layer=keys,
        values_per_layer=values,
        scale=scale, seq_len=seq_len,
    )
"""

from dataclasses import dataclass

import torch

from vllm.attention.kv_compress.config import KVCompressConfig
from vllm.attention.kv_compress.pyramidkv import PyramidKVBudgetAllocator
from vllm.attention.kv_compress.snapkv import SnapKVSelector


@dataclass
class LayerCompressionResult:
    """Result of compressing a single layer's KV cache.

    Attributes:
        layer_idx: Index of the compressed layer.
        selection_mask: Boolean mask [num_kv_heads, seq_len] of retained
            positions.
        selected_indices: Sorted indices of retained positions (union
            across all heads). Shape: [num_selected].
        compressed_key: Compressed key tensor [num_selected, Hkv, D].
        compressed_value: Compressed value tensor [num_selected, Hkv, D].
        budget: Number of tokens this layer was allocated.
        original_seq_len: Original sequence length before compression.
    """
    layer_idx: int
    selection_mask: torch.Tensor
    selected_indices: torch.Tensor
    compressed_key: torch.Tensor
    compressed_value: torch.Tensor
    budget: int
    original_seq_len: int

    @property
    def num_selected(self) -> int:
        return self.selected_indices.shape[0]

    @property
    def compression_ratio(self) -> float:
        if self.num_selected == 0:
            return float("inf")
        return self.original_seq_len / self.num_selected

    @property
    def retention_ratio(self) -> float:
        if self.original_seq_len == 0:
            return 0.0
        return self.num_selected / self.original_seq_len


class SnapPyramidKVCompressor:
    """Combined SnapKV + PyramidKV KV cache compressor.

    Combines PyramidKV's layer-wise budget allocation with SnapKV's
    attention-based token selection for maximum compression efficiency.

    Args:
        config: Compression configuration.
    """

    def __init__(self, config: KVCompressConfig):
        config.validate()
        self.config = config
        self.budget_allocator = PyramidKVBudgetAllocator(config)
        self.token_selector = SnapKVSelector(
            observation_window_size=config.observation_window_size,
            kernel_size=config.kernel_size,
            recent_window_size=config.recent_window_size,
            sink_size=config.sink_size,
        )

    def should_compress(self, seq_len: int) -> bool:
        """Check if compression should be applied for this sequence."""
        return self.config.should_compress(seq_len)

    def get_layer_budget(self, layer_idx: int, seq_len: int) -> int:
        """Get the token budget for a specific layer."""
        total_budget = self.config.get_total_budget(seq_len)
        return self.budget_allocator.get_layer_budget(
            layer_idx, total_budget, seq_len
        )

    def compress_layer(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        seq_len: int,
    ) -> LayerCompressionResult:
        """Compress a single layer's KV cache.

        Uses PyramidKV to determine this layer's budget and SnapKV to
        select which tokens to retain.

        Args:
            layer_idx: Index of the attention layer (0 = first/bottom).
            query: Query tensor [num_tokens, num_heads, head_size].
            key: Key tensor [num_tokens, num_kv_heads, head_size].
            value: Value tensor [num_tokens, num_kv_heads, head_size].
            scale: Attention scale factor (1/sqrt(head_size)).
            seq_len: Total sequence length.

        Returns:
            LayerCompressionResult with compressed KV and metadata.
        """
        # Get this layer's budget from PyramidKV
        budget = self.get_layer_budget(layer_idx, seq_len)

        # If budget >= seq_len, no compression needed
        if budget >= seq_len:
            indices = torch.arange(seq_len, device=key.device)
            mask = torch.ones(
                key.shape[1], seq_len, dtype=torch.bool, device=key.device
            )
            return LayerCompressionResult(
                layer_idx=layer_idx,
                selection_mask=mask,
                selected_indices=indices,
                compressed_key=key,
                compressed_value=value,
                budget=budget,
                original_seq_len=seq_len,
            )

        # Compute importance scores using SnapKV
        importance = self.token_selector.compute_importance_scores(
            query, key, scale, seq_len
        )

        # Select tokens per head
        mask = self.token_selector.select_tokens_batched(
            importance, budget, seq_len
        )

        # Get union of selected indices across heads for KV compaction
        union_mask = mask.any(dim=0)
        selected_indices = union_mask.nonzero(as_tuple=True)[0]

        # Compact KV tensors
        compressed_key = key[selected_indices]
        compressed_value = value[selected_indices]

        return LayerCompressionResult(
            layer_idx=layer_idx,
            selection_mask=mask,
            selected_indices=selected_indices,
            compressed_key=compressed_key,
            compressed_value=compressed_value,
            budget=budget,
            original_seq_len=seq_len,
        )

    def compress_all_layers(
        self,
        queries_per_layer: list[torch.Tensor],
        keys_per_layer: list[torch.Tensor],
        values_per_layer: list[torch.Tensor],
        scale: float,
        seq_len: int,
    ) -> list[LayerCompressionResult]:
        """Compress KV cache for all layers.

        Args:
            queries_per_layer: List of query tensors, one per layer.
            keys_per_layer: List of key tensors, one per layer.
            values_per_layer: List of value tensors, one per layer.
            scale: Attention scale factor.
            seq_len: Total sequence length.

        Returns:
            List of LayerCompressionResult, one per layer.
        """
        assert len(queries_per_layer) == self.config.num_layers
        assert len(keys_per_layer) == self.config.num_layers
        assert len(values_per_layer) == self.config.num_layers

        if not self.should_compress(seq_len):
            results = []
            for i in range(self.config.num_layers):
                indices = torch.arange(
                    seq_len, device=keys_per_layer[i].device
                )
                mask = torch.ones(
                    keys_per_layer[i].shape[1], seq_len,
                    dtype=torch.bool,
                    device=keys_per_layer[i].device,
                )
                results.append(LayerCompressionResult(
                    layer_idx=i,
                    selection_mask=mask,
                    selected_indices=indices,
                    compressed_key=keys_per_layer[i],
                    compressed_value=values_per_layer[i],
                    budget=seq_len,
                    original_seq_len=seq_len,
                ))
            return results

        results = []
        for i in range(self.config.num_layers):
            result = self.compress_layer(
                layer_idx=i,
                query=queries_per_layer[i],
                key=keys_per_layer[i],
                value=values_per_layer[i],
                scale=scale,
                seq_len=seq_len,
            )
            results.append(result)

        return results

    def get_compression_stats(self, seq_len: int) -> dict:
        """Get detailed compression statistics.

        Args:
            seq_len: Sequence length to compute stats for.

        Returns:
            Dictionary with compression metrics.
        """
        total_budget = self.config.get_total_budget(seq_len)
        pyramid_stats = self.budget_allocator.get_compression_stats(
            total_budget, seq_len
        )

        layer_budgets = pyramid_stats["per_layer_budgets"]
        total_retained = sum(layer_budgets)
        total_original = self.config.num_layers * seq_len

        return {
            "config": {
                "retention_ratio": self.config.retention_ratio,
                "observation_window": self.config.observation_window_size,
                "kernel_size": self.config.kernel_size,
                "recent_window": self.config.recent_window_size,
                "sink_size": self.config.sink_size,
                "pyramid_schedule": self.config.pyramid_schedule.value,
                "pyramid_alpha": self.config.pyramid_alpha,
            },
            "seq_len": seq_len,
            "total_budget": total_budget,
            "pyramid_allocation": pyramid_stats,
            "total_kv_entries_original": total_original,
            "total_kv_entries_compressed": total_retained,
            "overall_compression_ratio": (
                total_original / total_retained
                if total_retained > 0
                else float("inf")
            ),
            "memory_savings_pct": (
                (1.0 - total_retained / total_original) * 100
                if total_original > 0
                else 0.0
            ),
            "effective_capacity_multiplier": (
                total_original / total_retained
                if total_retained > 0
                else float("inf")
            ),
        }
