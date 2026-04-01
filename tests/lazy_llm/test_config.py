# SPDX-License-Identifier: Apache-2.0
"""Tests for LazyLLM configuration."""

import pytest

from vllm.attention.lazy_llm.config import LazyLLMConfig, PruningSchedule


class TestLazyLLMConfig:
    """Tests for LazyLLMConfig dataclass."""

    def test_default_config(self):
        cfg = LazyLLMConfig(num_layers=32)
        assert cfg.num_layers == 32
        assert cfg.drop_ratio == 0.5
        assert cfg.schedule == PruningSchedule.LINEAR
        assert cfg.start_layer == 2
        assert cfg.protected_tokens == 4
        assert cfg.min_tokens == 64
        assert cfg.min_seq_len == 256

    def test_keep_ratios_length(self):
        cfg = LazyLLMConfig(num_layers=32)
        assert len(cfg._keep_ratios) == 32

    def test_early_layers_keep_all(self):
        cfg = LazyLLMConfig(num_layers=32, start_layer=4)
        for i in range(4):
            assert cfg.get_keep_ratio(i) == 1.0

    def test_last_layer_matches_drop_ratio(self):
        cfg = LazyLLMConfig(num_layers=32, drop_ratio=0.5)
        last_keep = cfg.get_keep_ratio(31)
        assert abs(last_keep - 0.5) < 0.01

    def test_progressive_decrease(self):
        """Keep ratio should decrease (or stay same) across layers."""
        cfg = LazyLLMConfig(num_layers=32, start_layer=2)
        for i in range(2, 31):
            assert cfg.get_keep_ratio(i) >= cfg.get_keep_ratio(i + 1)

    def test_linear_schedule(self):
        cfg = LazyLLMConfig(
            num_layers=10, drop_ratio=0.5,
            schedule=PruningSchedule.LINEAR, start_layer=0,
        )
        # At layer 0 (progress=0): keep=1.0
        assert cfg.get_keep_ratio(0) == 1.0
        # At last layer (progress=1.0): keep=0.5
        assert abs(cfg.get_keep_ratio(9) - 0.5) < 0.01

    def test_exponential_schedule(self):
        cfg = LazyLLMConfig(
            num_layers=10, drop_ratio=0.5,
            schedule=PruningSchedule.EXPONENTIAL, start_layer=0,
        )
        assert cfg.get_keep_ratio(0) == 1.0
        assert abs(cfg.get_keep_ratio(9) - 0.5) < 0.01
        # Midpoint should be higher than linear midpoint (0.5 + 0.5*0.5/2 = 0.722)
        mid = cfg.get_keep_ratio(4)
        assert mid > 0.7  # exponential decays slower initially

    def test_step_schedule(self):
        cfg = LazyLLMConfig(
            num_layers=10, drop_ratio=0.5,
            schedule=PruningSchedule.STEP, start_layer=0,
        )
        # First half keeps all
        assert cfg.get_keep_ratio(0) == 1.0
        assert cfg.get_keep_ratio(4) == 1.0
        # Second half drops
        assert abs(cfg.get_keep_ratio(9) - 0.5) < 0.01

    def test_should_prune_short_sequence(self):
        cfg = LazyLLMConfig(num_layers=32, min_seq_len=256)
        assert not cfg.should_prune(layer_idx=10, seq_len=100)

    def test_should_prune_early_layer(self):
        cfg = LazyLLMConfig(num_layers=32, start_layer=4)
        assert not cfg.should_prune(layer_idx=2, seq_len=1000)

    def test_should_prune_active_layer(self):
        cfg = LazyLLMConfig(num_layers=32, start_layer=2)
        assert cfg.should_prune(layer_idx=10, seq_len=1000)

    def test_get_num_tokens_to_keep(self):
        cfg = LazyLLMConfig(
            num_layers=32, drop_ratio=0.5,
            min_tokens=64, protected_tokens=4,
        )
        # At last layer with 1000 tokens, keep ~50% = 500
        n = cfg.get_num_tokens_to_keep(31, 1000)
        assert 490 <= n <= 510

    def test_get_num_tokens_respects_min(self):
        cfg = LazyLLMConfig(num_layers=32, drop_ratio=0.99, min_tokens=64)
        # Even with 99% drop, should keep at least 64
        n = cfg.get_num_tokens_to_keep(31, 1000)
        assert n >= 64

    def test_get_num_tokens_short_seq(self):
        cfg = LazyLLMConfig(num_layers=32, min_seq_len=256)
        # Short sequence: keep all
        n = cfg.get_num_tokens_to_keep(31, 100)
        assert n == 100

    def test_invalid_drop_ratio(self):
        with pytest.raises(ValueError, match="drop_ratio"):
            LazyLLMConfig(num_layers=32, drop_ratio=0.0)
        with pytest.raises(ValueError, match="drop_ratio"):
            LazyLLMConfig(num_layers=32, drop_ratio=1.0)

    def test_invalid_start_layer(self):
        with pytest.raises(ValueError, match="start_layer"):
            LazyLLMConfig(num_layers=32, start_layer=32)

    def test_invalid_min_tokens(self):
        with pytest.raises(ValueError, match="min_tokens"):
            LazyLLMConfig(num_layers=32, min_tokens=0)

    def test_out_of_range_layer_idx(self):
        cfg = LazyLLMConfig(num_layers=32)
        assert cfg.get_keep_ratio(-1) == 1.0
        assert cfg.get_keep_ratio(100) == 1.0
