# SPDX-License-Identifier: Apache-2.0
"""Progressive token pruner for LazyLLM."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.attention.lazy_llm.config import LazyLLMConfig
from vllm.attention.lazy_llm.selector import TokenImportanceSelector


@dataclass
class LayerPruningResult:
    """Result of pruning at a single layer."""
    kept_indices: torch.Tensor
    pruned_indices: torch.Tensor
    num_kept: int
    num_total: int
    importance_scores: torch.Tensor | None = None


@dataclass
class PrefillPruningState:
    """Tracks pruning state across layers during a single prefill pass."""
    seq_len: int
    active_indices: torch.Tensor
    aux_hidden_states: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_results: list[LayerPruningResult] = field(default_factory=list)

    @property
    def num_active(self) -> int:
        return self.active_indices.shape[0]

    @property
    def compression_ratio(self) -> float:
        if self.seq_len == 0:
            return 1.0
        return self.num_active / self.seq_len

    @property
    def total_tokens_saved(self) -> int:
        return sum(r.num_total - r.num_kept for r in self.layer_results)


class ProgressiveTokenPruner:
    """Manages progressive token pruning during prefill.

    Usage:
        1. Create PrefillPruningState via init_prefill_state()
        2. After each layer's attention, call maybe_prune_tokens()
        3. Use returned indices to gather hidden states for next layer
        4. Pruned hidden states are saved in state.aux_hidden_states
    """

    def __init__(self, config: LazyLLMConfig):
        self.config = config
        self.selector = TokenImportanceSelector()

    def init_prefill_state(
        self, seq_len: int, device: torch.device,
    ) -> PrefillPruningState:
        return PrefillPruningState(
            seq_len=seq_len,
            active_indices=torch.arange(seq_len, device=device),
        )

    def maybe_prune_tokens(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        state: PrefillPruningState,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        attention_weights: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, PrefillPruningState]:
        """Conditionally prune tokens after a layer's attention.

        Returns:
            Tuple of (pruned_hidden_states, updated_state).
            If no pruning, returns inputs unchanged.
        """
        num_active = hidden_states.shape[0]

        if not self.config.should_prune(layer_idx, state.seq_len):
            return hidden_states, state

        num_keep = self.config.get_num_tokens_to_keep(
            layer_idx, num_active
        )
        if num_keep >= num_active:
            return hidden_states, state

        # Compute importance scores
        importance = self._get_importance(
            query, key, attention_weights, scale, num_active
        )
        if importance is None:
            return hidden_states, state

        # Build protected indices (first N tokens of the active set)
        protected = None
        if self.config.protected_tokens > 0:
            n_prot = min(self.config.protected_tokens, num_active)
            protected = torch.arange(n_prot, device=hidden_states.device)

        # Select tokens to keep (indices into current active set)
        local_kept = self.selector.select_tokens(
            importance, num_keep, protected
        )

        # Compute pruned indices
        all_idx = torch.arange(num_active, device=hidden_states.device)
        mask = torch.ones(num_active, dtype=torch.bool, device=hidden_states.device)
        mask[local_kept] = False
        local_pruned = all_idx[mask]

        # Save pruned hidden states to aux cache
        if local_pruned.numel() > 0:
            state.aux_hidden_states[layer_idx] = hidden_states[local_pruned].detach()

        # Record result
        result = LayerPruningResult(
            kept_indices=state.active_indices[local_kept],
            pruned_indices=state.active_indices[local_pruned],
            num_kept=local_kept.shape[0],
            num_total=num_active,
            importance_scores=importance.detach(),
        )
        state.layer_results.append(result)

        # Update active indices and hidden states
        state.active_indices = state.active_indices[local_kept]
        return hidden_states[local_kept], state

    def _get_importance(
        self,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        attention_weights: torch.Tensor | None,
        scale: float | None,
        num_active: int,
    ) -> torch.Tensor | None:
        """Compute importance from available signals."""
        if attention_weights is not None:
            return self.selector.compute_importance_from_attention(
                attention_weights
            )
        if query is not None and key is not None and scale is not None:
            return self.selector.compute_importance_from_qk(
                query, key, scale
            )
        return None

    def get_pruning_summary(
        self, state: PrefillPruningState
    ) -> dict:
        """Get a summary of pruning across all layers."""
        per_layer = []
        for i, r in enumerate(state.layer_results):
            per_layer.append({
                "layer": i,
                "kept": r.num_kept,
                "total": r.num_total,
                "ratio": r.num_kept / max(r.num_total, 1),
            })
        return {
            "original_seq_len": state.seq_len,
            "final_active": state.num_active,
            "compression_ratio": state.compression_ratio,
            "total_tokens_saved": state.total_tokens_saved,
            "aux_cache_layers": len(state.aux_hidden_states),
            "per_layer": per_layer,
        }
