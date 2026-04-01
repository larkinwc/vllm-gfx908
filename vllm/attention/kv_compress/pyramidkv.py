# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PyramidKV: Layer-Wise KV Cache Budget Allocation.

Implements the PyramidKV algorithm (2024) which observes that attention
patterns differ across layers: lower layers have diffuse attention (need
more KV tokens), while upper layers have focused attention (need fewer).
Budget is allocated in a pyramid shape across layers.

The key insight ("Pyramidal Information Funneling") is:
    - Lower layers: attention scatters across many tokens → need large cache
    - Middle layers: attention begins to focus → moderate cache
    - Upper layers: attention concentrates on few tokens → small cache

At only 12% total retention, PyramidKV matches full-cache accuracy.

Reference: https://arxiv.org/abs/2406.02069
"""

import math

import torch

from vllm.attention.kv_compress.config import KVCompressConfig, PyramidSchedule


class PyramidKVBudgetAllocator:
    """Allocates per-layer KV cache budgets using pyramid scheduling.

    Given a total budget of tokens to retain, distributes the budget
    across layers so lower layers get more and upper layers get less.

    Args:
        config: KVCompressConfig with pyramid parameters.
    """

    def __init__(self, config: KVCompressConfig):
        self.config = config
        self.num_layers = config.num_layers
        self.schedule = config.pyramid_schedule
        self.alpha = config.pyramid_alpha
        self._cached_allocations: dict[tuple[int, int], list[int]] = {}

    def compute_layer_budgets(
        self,
        total_budget: int,
        seq_len: int,
    ) -> list[int]:
        """Compute per-layer token budgets.

        Args:
            total_budget: Total number of tokens to retain across all layers.
                Each layer gets a fraction of this budget.
            seq_len: Original sequence length before compression.

        Returns:
            List of length num_layers with per-layer token budgets.
            Sum equals total_budget (modulo rounding).
        """
        cache_key = (total_budget, seq_len)
        if cache_key in self._cached_allocations:
            return self._cached_allocations[cache_key]

        # Minimum per-layer budget: sink + recent + at least 1 selectable
        min_per_layer = (
            self.config.sink_size + self.config.recent_window_size + 1
        )
        min_per_layer = min(min_per_layer, seq_len)

        # If total_budget can't satisfy min_per_layer for all layers,
        # reduce min_per_layer to fit within the total budget
        if min_per_layer * self.num_layers > total_budget:
            min_per_layer = max(1, total_budget // self.num_layers)

        if self.schedule == PyramidSchedule.LINEAR:
            budgets = self._linear_schedule(
                total_budget, seq_len, min_per_layer
            )
        elif self.schedule == PyramidSchedule.EXPONENTIAL:
            budgets = self._exponential_schedule(
                total_budget, seq_len, min_per_layer
            )
        elif self.schedule == PyramidSchedule.STEP:
            budgets = self._step_schedule(
                total_budget, seq_len, min_per_layer
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        self._cached_allocations[cache_key] = budgets
        return budgets

    def _linear_schedule(
        self,
        total_budget: int,
        seq_len: int,
        min_per_layer: int,
    ) -> list[int]:
        """Linear pyramid: budget decreases linearly from bottom to top.

        Layer i gets weight: alpha - (alpha - 1) * i / (num_layers - 1)
        So layer 0 gets alpha * average, last layer gets 1 * average.
        """
        n = self.num_layers
        if n == 1:
            return [min(total_budget, seq_len)]

        # Raw weights: linearly decreasing from alpha to 1
        weights = [
            self.alpha - (self.alpha - 1.0) * i / (n - 1)
            for i in range(n)
        ]
        total_weight = sum(weights)

        # Scale to total_budget
        raw_budgets = [total_budget * w / total_weight for w in weights]

        # Round and enforce minimums, then adjust to match total exactly
        budgets = self._round_and_adjust(
            raw_budgets, total_budget, seq_len, min_per_layer
        )
        return budgets

    def _exponential_schedule(
        self,
        total_budget: int,
        seq_len: int,
        min_per_layer: int,
    ) -> list[int]:
        """Exponential pyramid: budget decays exponentially.

        Layer i gets weight: alpha^(-(i / (num_layers - 1)))
        Lower layers get exponentially more budget.
        """
        n = self.num_layers
        if n == 1:
            return [min(total_budget, seq_len)]

        decay = math.log(self.alpha) / (n - 1)
        weights = [math.exp(-decay * i) for i in range(n)]
        total_weight = sum(weights)

        raw_budgets = [total_budget * w / total_weight for w in weights]
        budgets = self._round_and_adjust(
            raw_budgets, total_budget, seq_len, min_per_layer
        )
        return budgets

    def _step_schedule(
        self,
        total_budget: int,
        seq_len: int,
        min_per_layer: int,
    ) -> list[int]:
        """Step pyramid: three tiers (bottom 1/3, middle 1/3, top 1/3).

        Bottom third gets alpha * average, middle gets average,
        top third gets average / alpha.
        """
        n = self.num_layers
        if n == 1:
            return [min(total_budget, seq_len)]

        third = n // 3
        remainder = n - 2 * third

        weights = []
        for i in range(n):
            if i < third:
                weights.append(self.alpha)
            elif i < third + remainder:
                weights.append(1.0)
            else:
                weights.append(1.0 / self.alpha)

        total_weight = sum(weights)
        raw_budgets = [total_budget * w / total_weight for w in weights]
        budgets = self._round_and_adjust(
            raw_budgets, total_budget, seq_len, min_per_layer
        )
        return budgets

    @staticmethod
    def _round_and_adjust(
        raw_budgets: list[float],
        total_budget: int,
        seq_len: int,
        min_per_layer: int,
    ) -> list[int]:
        """Round float budgets to ints summing to total_budget.

        Uses largest-remainder method for fair rounding, then enforces
        min/max constraints while preserving the total sum.
        """
        n = len(raw_budgets)

        # Step 1: Largest-remainder rounding to get exact sum
        floored = [int(math.floor(b)) for b in raw_budgets]
        remainders = [raw_budgets[i] - floored[i] for i in range(n)]
        current_sum = sum(floored)
        deficit = total_budget - current_sum

        if deficit > 0:
            indices = sorted(
                range(n), key=lambda i: remainders[i], reverse=True
            )
            for idx in indices[:deficit]:
                floored[idx] += 1
        elif deficit < 0:
            indices = sorted(range(n), key=lambda i: remainders[i])
            for idx in indices[:(-deficit)]:
                floored[idx] -= 1

        # Step 2: Enforce min/max constraints while preserving total
        # Iteratively fix violations by redistributing
        for _ in range(n * 2):  # bounded iterations
            excess = 0
            shortfall = 0
            for i in range(n):
                if floored[i] < min_per_layer:
                    shortfall += min_per_layer - floored[i]
                    floored[i] = min_per_layer
                elif floored[i] > seq_len:
                    excess += floored[i] - seq_len
                    floored[i] = seq_len

            # Redistribute excess to underfunded layers
            if excess > 0:
                for i in range(n):
                    if excess <= 0:
                        break
                    room = seq_len - floored[i]
                    give = min(room, excess)
                    floored[i] += give
                    excess -= give

            # Reclaim shortfall from overfunded layers (largest first)
            if shortfall > 0:
                indices = sorted(
                    range(n), key=lambda i: floored[i], reverse=True
                )
                for idx in indices:
                    if shortfall <= 0:
                        break
                    take = min(
                        floored[idx] - min_per_layer, shortfall
                    )
                    if take > 0:
                        floored[idx] -= take
                        shortfall -= take

            if excess == 0 and shortfall == 0:
                break

        return floored

    def get_layer_budget(
        self,
        layer_idx: int,
        total_budget: int,
        seq_len: int,
    ) -> int:
        """Get budget for a single layer.

        Args:
            layer_idx: Index of the layer (0 = bottom/first layer).
            total_budget: Total token budget across all layers.
            seq_len: Original sequence length.

        Returns:
            Number of tokens this layer should retain.
        """
        budgets = self.compute_layer_budgets(total_budget, seq_len)
        return budgets[layer_idx]

    def get_compression_stats(
        self,
        total_budget: int,
        seq_len: int,
    ) -> dict:
        """Get compression statistics for analysis.

        Returns:
            Dictionary with compression metrics.
        """
        budgets = self.compute_layer_budgets(total_budget, seq_len)
        return {
            "num_layers": self.num_layers,
            "seq_len": seq_len,
            "total_budget": total_budget,
            "per_layer_budgets": budgets,
            "min_layer_budget": min(budgets),
            "max_layer_budget": max(budgets),
            "avg_layer_budget": sum(budgets) / len(budgets),
            "overall_retention_ratio": sum(budgets) / (
                self.num_layers * seq_len
            ),
            "per_layer_retention": [b / seq_len for b in budgets],
            "effective_compression_ratio": (
                self.num_layers * seq_len
            ) / sum(budgets) if sum(budgets) > 0 else float("inf"),
        }
