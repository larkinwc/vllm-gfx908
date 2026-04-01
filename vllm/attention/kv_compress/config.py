# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for SnapKV + PyramidKV KV cache compression."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PyramidSchedule(str, Enum):
    """Budget allocation schedule for PyramidKV across layers."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"


@dataclass
class KVCompressConfig:
    """Configuration for combined SnapKV + PyramidKV compression.

    Attributes:
        enabled: Whether KV cache compression is enabled.
        total_budget: Total number of KV tokens to retain across all layers.
            If None, computed as retention_ratio * seq_len.
        retention_ratio: Fraction of tokens to retain (0.0 to 1.0).
            Used when total_budget is None. Default 0.12 matches PyramidKV
            paper's 12% retention.
        observation_window_size: Number of tokens at the end of the prompt
            to use as the observation window for SnapKV scoring.
            Default 32 per the SnapKV paper.
        kernel_size: 1D average pooling kernel size for clustering nearby
            tokens in SnapKV attention score aggregation. Default 7.
        recent_window_size: Number of most recent tokens always retained
            (these are typically important for generation). Default 32.
        sink_size: Number of initial "attention sink" tokens always retained.
            Following StreamingLLM observation. Default 4.
        pyramid_schedule: How to distribute budget across layers.
        pyramid_alpha: Controls the steepness of the pyramid allocation.
            Higher values = more budget to lower layers.
            For LINEAR: ratio between max and min layer budget.
            For EXPONENTIAL: decay factor per layer.
        min_seq_len_to_compress: Minimum sequence length before compression
            kicks in. Short sequences don't benefit. Default 256.
        num_layers: Number of transformer layers (set at init time).
        num_kv_heads: Number of KV attention heads (set at init time).
    """
    enabled: bool = True
    total_budget: Optional[int] = None
    retention_ratio: float = 0.12
    observation_window_size: int = 32
    kernel_size: int = 7
    recent_window_size: int = 32
    sink_size: int = 4
    pyramid_schedule: PyramidSchedule = PyramidSchedule.LINEAR
    pyramid_alpha: float = 4.0
    min_seq_len_to_compress: int = 256
    num_layers: int = 32
    num_kv_heads: int = 8

    def get_total_budget(self, seq_len: int) -> int:
        """Compute total budget across all layers.

        The total budget is distributed across layers by PyramidKV.
        retention_ratio is the *average* per-layer retention; the total
        budget is retention_ratio * seq_len * num_layers.
        """
        if self.total_budget is not None:
            return min(self.total_budget, seq_len * self.num_layers)
        min_per_layer = self.sink_size + self.recent_window_size + 1
        per_layer = max(
            int(seq_len * self.retention_ratio),
            min(min_per_layer, seq_len),
        )
        per_layer = min(per_layer, seq_len)
        return per_layer * self.num_layers

    def should_compress(self, seq_len: int) -> bool:
        """Determine if compression should be applied."""
        if not self.enabled:
            return False
        if seq_len < self.min_seq_len_to_compress:
            return False
        total_budget = self.get_total_budget(seq_len)
        # Compare average per-layer budget to seq_len
        avg_per_layer = total_budget // max(self.num_layers, 1)
        return avg_per_layer < seq_len

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert 0.0 < self.retention_ratio <= 1.0, (
            f"retention_ratio must be in (0, 1], got {self.retention_ratio}"
        )
        assert self.observation_window_size > 0, (
            f"observation_window_size must be > 0, "
            f"got {self.observation_window_size}"
        )
        assert self.kernel_size > 0 and self.kernel_size % 2 == 1, (
            f"kernel_size must be odd and > 0, got {self.kernel_size}"
        )
        assert self.recent_window_size >= 0, (
            f"recent_window_size must be >= 0, got {self.recent_window_size}"
        )
        assert self.sink_size >= 0, (
            f"sink_size must be >= 0, got {self.sink_size}"
        )
        assert self.pyramid_alpha > 0, (
            f"pyramid_alpha must be > 0, got {self.pyramid_alpha}"
        )
        assert self.num_layers > 0, (
            f"num_layers must be > 0, got {self.num_layers}"
        )
        assert self.num_kv_heads > 0, (
            f"num_kv_heads must be > 0, got {self.num_kv_heads}"
        )
