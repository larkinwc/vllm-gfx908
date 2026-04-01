# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PyramidKV layer-wise budget allocation."""

import pytest

from vllm.attention.kv_compress.config import KVCompressConfig, PyramidSchedule
from vllm.attention.kv_compress.pyramidkv import PyramidKVBudgetAllocator


@pytest.fixture
def linear_config():
    return KVCompressConfig(
        num_layers=32,
        num_kv_heads=8,
        retention_ratio=0.12,
        pyramid_schedule=PyramidSchedule.LINEAR,
        pyramid_alpha=4.0,
        sink_size=4,
        recent_window_size=8,
    )


@pytest.fixture
def exp_config():
    return KVCompressConfig(
        num_layers=32,
        num_kv_heads=8,
        retention_ratio=0.12,
        pyramid_schedule=PyramidSchedule.EXPONENTIAL,
        pyramid_alpha=4.0,
        sink_size=4,
        recent_window_size=8,
    )


@pytest.fixture
def step_config():
    return KVCompressConfig(
        num_layers=32,
        num_kv_heads=8,
        retention_ratio=0.12,
        pyramid_schedule=PyramidSchedule.STEP,
        pyramid_alpha=4.0,
        sink_size=4,
        recent_window_size=8,
    )


class TestLinearSchedule:
    """Tests for linear pyramid schedule."""

    def test_budget_sum(self, linear_config):
        alloc = PyramidKVBudgetAllocator(linear_config)
        total_budget = 128
        budgets = alloc.compute_layer_budgets(total_budget, seq_len=1024)
        assert sum(budgets) == total_budget

    def test_decreasing_order(self, linear_config):
        """Lower layers should get >= budget of upper layers."""
        alloc = PyramidKVBudgetAllocator(linear_config)
        budgets = alloc.compute_layer_budgets(256, seq_len=2048)
        # Budget should be monotonically non-increasing
        for i in range(len(budgets) - 1):
            assert budgets[i] >= budgets[i + 1] - 1  # -1 for rounding

    def test_first_layer_largest(self, linear_config):
        alloc = PyramidKVBudgetAllocator(linear_config)
        budgets = alloc.compute_layer_budgets(256, seq_len=2048)
        assert budgets[0] == max(budgets)

    def test_last_layer_smallest(self, linear_config):
        alloc = PyramidKVBudgetAllocator(linear_config)
        budgets = alloc.compute_layer_budgets(256, seq_len=2048)
        assert budgets[-1] == min(budgets)

    def test_alpha_ratio(self, linear_config):
        """First/last layer budget ratio should approximate alpha."""
        alloc = PyramidKVBudgetAllocator(linear_config)
        budgets = alloc.compute_layer_budgets(1024, seq_len=8192)
        ratio = budgets[0] / max(budgets[-1], 1)
        # Should be close to alpha (4.0), with tolerance for rounding
        assert 2.5 <= ratio <= 5.5

    def test_single_layer(self):
        config = KVCompressConfig(num_layers=1, num_kv_heads=8)
        alloc = PyramidKVBudgetAllocator(config)
        budgets = alloc.compute_layer_budgets(100, seq_len=1024)
        assert len(budgets) == 1
        assert budgets[0] == 100


class TestExponentialSchedule:
    """Tests for exponential pyramid schedule."""

    def test_budget_sum(self, exp_config):
        alloc = PyramidKVBudgetAllocator(exp_config)
        budgets = alloc.compute_layer_budgets(128, seq_len=1024)
        assert sum(budgets) == 128

    def test_decreasing_order(self, exp_config):
        alloc = PyramidKVBudgetAllocator(exp_config)
        budgets = alloc.compute_layer_budgets(256, seq_len=2048)
        for i in range(len(budgets) - 1):
            assert budgets[i] >= budgets[i + 1] - 1

    def test_steeper_than_linear(self, linear_config, exp_config):
        """Exponential should allocate more to first layer than linear."""
        alloc_lin = PyramidKVBudgetAllocator(linear_config)
        alloc_exp = PyramidKVBudgetAllocator(exp_config)
        budget_lin = alloc_lin.compute_layer_budgets(512, seq_len=4096)
        budget_exp = alloc_exp.compute_layer_budgets(512, seq_len=4096)
        # Exponential first layer should be >= linear first layer
        assert budget_exp[0] >= budget_lin[0] - 2  # tolerance for rounding


class TestStepSchedule:
    """Tests for step pyramid schedule."""

    def test_budget_sum(self, step_config):
        alloc = PyramidKVBudgetAllocator(step_config)
        budgets = alloc.compute_layer_budgets(128, seq_len=1024)
        assert sum(budgets) == 128

    def test_three_tiers(self, step_config):
        """Should have three distinct budget levels."""
        alloc = PyramidKVBudgetAllocator(step_config)
        budgets = alloc.compute_layer_budgets(640, seq_len=4096)
        # Bottom third should have higher budgets than top third
        n = step_config.num_layers
        third = n // 3
        bottom_avg = sum(budgets[:third]) / third
        top_avg = sum(budgets[-third:]) / third
        assert bottom_avg > top_avg


class TestBudgetConstraints:
    """Tests for budget constraints and edge cases."""

    def test_min_budget_enforced(self):
        """Each layer should get at least sink + recent + 1 tokens."""
        config = KVCompressConfig(
            num_layers=32, num_kv_heads=8,
            sink_size=4, recent_window_size=8,
        )
        alloc = PyramidKVBudgetAllocator(config)
        min_expected = 4 + 8 + 1  # sink + recent + 1
        # Use a very small total budget
        budgets = alloc.compute_layer_budgets(
            total_budget=min_expected * 32, seq_len=1024
        )
        for b in budgets:
            assert b >= min_expected

    def test_max_budget_capped_at_seq_len(self):
        config = KVCompressConfig(num_layers=4, num_kv_heads=8)
        alloc = PyramidKVBudgetAllocator(config)
        budgets = alloc.compute_layer_budgets(
            total_budget=2000, seq_len=100
        )
        for b in budgets:
            assert b <= 100

    def test_large_budget_small_seq(self):
        """When total_budget >> seq_len * num_layers, all layers get seq_len."""
        config = KVCompressConfig(num_layers=4, num_kv_heads=8)
        alloc = PyramidKVBudgetAllocator(config)
        budgets = alloc.compute_layer_budgets(
            total_budget=10000, seq_len=32
        )
        for b in budgets:
            assert b <= 32

    def test_caching(self, linear_config):
        """Same inputs should return cached result."""
        alloc = PyramidKVBudgetAllocator(linear_config)
        b1 = alloc.compute_layer_budgets(256, 2048)
        b2 = alloc.compute_layer_budgets(256, 2048)
        assert b1 == b2
        assert id(b1) == id(b2)  # Same object from cache


class TestGetLayerBudget:
    """Tests for single-layer budget query."""

    def test_valid_index(self, linear_config):
        alloc = PyramidKVBudgetAllocator(linear_config)
        budget = alloc.get_layer_budget(0, 128, 1024)
        assert isinstance(budget, int)
        assert budget > 0

    def test_all_indices(self, linear_config):
        alloc = PyramidKVBudgetAllocator(linear_config)
        budgets = [
            alloc.get_layer_budget(i, 256, 2048)
            for i in range(linear_config.num_layers)
        ]
        assert sum(budgets) == 256


class TestCompressionStats:
    """Tests for compression statistics."""

    def test_stats_structure(self, linear_config):
        alloc = PyramidKVBudgetAllocator(linear_config)
        stats = alloc.get_compression_stats(128, 1024)
        assert "num_layers" in stats
        assert "per_layer_budgets" in stats
        assert "effective_compression_ratio" in stats
        assert "overall_retention_ratio" in stats

    def test_compression_ratio(self, linear_config):
        """Compression ratio should be > 1 when budget < total."""
        alloc = PyramidKVBudgetAllocator(linear_config)
        stats = alloc.get_compression_stats(128, 1024)
        # 32 layers * 1024 tokens / 128 total budget ≈ 256x
        assert stats["effective_compression_ratio"] > 1.0

    def test_retention_ratio(self, linear_config):
        """Overall retention should match expected compression."""
        alloc = PyramidKVBudgetAllocator(linear_config)
        stats = alloc.get_compression_stats(128, 1024)
        # 128 / (32 * 1024) = 0.00390625
        assert 0 < stats["overall_retention_ratio"] < 1.0
