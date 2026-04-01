# SPDX-License-Identifier: Apache-2.0
"""Configuration for LazyLLM dynamic token pruning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class PruningSchedule(str, Enum):
    """How aggressively to prune tokens across layers."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"


@dataclass
class LazyLLMConfig:
    """Configuration for LazyLLM progressive token pruning.

    Attributes:
        num_layers: Total number of transformer layers.
        drop_ratio: Final fraction of tokens to drop by the last layer.
            E.g., 0.7 means the last layer processes only 30% of tokens.
        schedule: How to distribute pruning across layers.
        start_layer: First layer where pruning begins (0-indexed).
            Earlier layers always see all tokens.
        protected_tokens: Number of tokens at the start of the sequence
            that are never pruned (e.g., system prompt tokens, BOS).
        min_tokens: Minimum number of tokens to keep at any layer.
            Prevents over-pruning on short sequences.
        min_seq_len: Minimum sequence length to activate pruning.
            Short sequences don't benefit from pruning.
    """
    num_layers: int
    drop_ratio: float = 0.5
    schedule: PruningSchedule = PruningSchedule.LINEAR
    start_layer: int = 2
    protected_tokens: int = 4
    min_tokens: int = 64
    min_seq_len: int = 256

    # Derived: per-layer keep ratios (computed once)
    _keep_ratios: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not 0.0 < self.drop_ratio < 1.0:
            raise ValueError(
                f"drop_ratio must be in (0, 1), got {self.drop_ratio}"
            )
        if self.start_layer < 0 or self.start_layer >= self.num_layers:
            raise ValueError(
                f"start_layer must be in [0, {self.num_layers}), "
                f"got {self.start_layer}"
            )
        if self.min_tokens < 1:
            raise ValueError(
                f"min_tokens must be >= 1, got {self.min_tokens}"
            )
        self._keep_ratios = self._compute_keep_ratios()

    def _compute_keep_ratios(self) -> list[float]:
        """Compute per-layer token keep ratios.

        Layers before start_layer keep 100% of tokens.
        From start_layer onwards, keep ratio decreases according to schedule.
        """
        ratios = [1.0] * self.num_layers
        num_pruning_layers = self.num_layers - self.start_layer
        if num_pruning_layers <= 0:
            return ratios

        final_keep = 1.0 - self.drop_ratio

        for i in range(self.start_layer, self.num_layers):
            # Progress from 0.0 (at start_layer) to 1.0 (at last layer)
            progress = (i - self.start_layer) / max(num_pruning_layers - 1, 1)

            if self.schedule == PruningSchedule.LINEAR:
                keep = 1.0 - progress * self.drop_ratio
            elif self.schedule == PruningSchedule.EXPONENTIAL:
                # Exponential decay: keep = final_keep^progress
                keep = math.pow(final_keep, progress)
            elif self.schedule == PruningSchedule.STEP:
                # Step function: keep everything until halfway, then drop
                if progress < 0.5:
                    keep = 1.0
                else:
                    keep = final_keep
            else:
                keep = 1.0 - progress * self.drop_ratio

            ratios[i] = max(keep, 0.01)  # Never go below 1%

        return ratios

    def get_keep_ratio(self, layer_idx: int) -> float:
        """Get the token keep ratio for a specific layer."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return 1.0
        return self._keep_ratios[layer_idx]

    def get_num_tokens_to_keep(
        self, layer_idx: int, seq_len: int
    ) -> int:
        """Get the number of tokens to keep at a given layer.

        Accounts for protected_tokens and min_tokens constraints.
        """
        if seq_len < self.min_seq_len:
            return seq_len

        keep_ratio = self.get_keep_ratio(layer_idx)
        num_keep = max(
            int(seq_len * keep_ratio),
            self.min_tokens,
            self.protected_tokens,
        )
        return min(num_keep, seq_len)

    def should_prune(self, layer_idx: int, seq_len: int) -> bool:
        """Check if pruning should be applied at this layer."""
        if seq_len < self.min_seq_len:
            return False
        if layer_idx < self.start_layer:
            return False
        return self.get_keep_ratio(layer_idx) < 1.0
