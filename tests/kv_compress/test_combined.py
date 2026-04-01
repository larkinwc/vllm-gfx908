# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for combined SnapKV + PyramidKV compressor."""

import pytest
import torch

from vllm.attention.kv_compress.combined import (
    LayerCompressionResult,
    SnapPyramidKVCompressor,
)
from vllm.attention.kv_compress.config import KVCompressConfig, PyramidSchedule


def _make_qkv(seq_len, num_heads_q, num_heads_kv, head_size, device="cpu"):
    q = torch.randn(seq_len, num_heads_q, head_size, device=device)
    k = torch.randn(seq_len, num_heads_kv, head_size, device=device)
    v = torch.randn(seq_len, num_heads_kv, head_size, device=device)
    return q, k, v


@pytest.fixture
def config():
    return KVCompressConfig(
        enabled=True,
        num_layers=4,
        num_kv_heads=8,
        retention_ratio=0.25,
        observation_window_size=16,
        kernel_size=5,
        recent_window_size=8,
        sink_size=4,
        pyramid_schedule=PyramidSchedule.LINEAR,
        pyramid_alpha=3.0,
        min_seq_len_to_compress=64,
    )


@pytest.fixture
def compressor(config):
    return SnapPyramidKVCompressor(config)


class TestSnapPyramidKVCompressor:
    """Tests for the combined compressor."""

    def test_should_compress(self, compressor):
        assert not compressor.should_compress(32)   # too short
        assert compressor.should_compress(256)       # long enough
        assert compressor.should_compress(1024)      # long

    def test_should_not_compress_when_disabled(self):
        config = KVCompressConfig(enabled=False, num_layers=4, num_kv_heads=8)
        comp = SnapPyramidKVCompressor(config)
        assert not comp.should_compress(1024)

    def test_compress_single_layer(self, compressor):
        seq_len = 256
        q, k, v = _make_qkv(seq_len, 8, 8, 64)
        scale = 1.0 / (64 ** 0.5)

        result = compressor.compress_layer(
            layer_idx=0, query=q, key=k, value=v,
            scale=scale, seq_len=seq_len,
        )

        assert isinstance(result, LayerCompressionResult)
        assert result.layer_idx == 0
        assert result.num_selected < seq_len
        assert result.compressed_key.shape[0] == result.num_selected
        assert result.compressed_value.shape[0] == result.num_selected
        assert result.compression_ratio > 1.0
        assert 0 < result.retention_ratio < 1.0

    def test_pyramid_effect(self, compressor):
        """Lower layers should retain more tokens than upper layers."""
        seq_len = 512
        scale = 1.0 / (64 ** 0.5)

        results = []
        for i in range(compressor.config.num_layers):
            q, k, v = _make_qkv(seq_len, 8, 8, 64)
            result = compressor.compress_layer(
                layer_idx=i, query=q, key=k, value=v,
                scale=scale, seq_len=seq_len,
            )
            results.append(result)

        # First layer should have larger budget than last
        assert results[0].budget >= results[-1].budget

    def test_compress_all_layers(self, compressor):
        seq_len = 256
        scale = 1.0 / (64 ** 0.5)
        num_layers = compressor.config.num_layers

        queries = [torch.randn(seq_len, 8, 64) for _ in range(num_layers)]
        keys = [torch.randn(seq_len, 8, 64) for _ in range(num_layers)]
        values = [torch.randn(seq_len, 8, 64) for _ in range(num_layers)]

        results = compressor.compress_all_layers(
            queries, keys, values, scale, seq_len,
        )

        assert len(results) == num_layers
        for r in results:
            assert isinstance(r, LayerCompressionResult)
            assert r.num_selected <= seq_len

    def test_no_compress_short_seq(self, compressor):
        """Short sequences should pass through uncompressed."""
        seq_len = 32  # below min_seq_len_to_compress
        num_layers = compressor.config.num_layers

        queries = [torch.randn(seq_len, 8, 64) for _ in range(num_layers)]
        keys = [torch.randn(seq_len, 8, 64) for _ in range(num_layers)]
        values = [torch.randn(seq_len, 8, 64) for _ in range(num_layers)]

        results = compressor.compress_all_layers(
            queries, keys, values, scale=0.125, seq_len=seq_len,
        )

        for r in results:
            assert r.num_selected == seq_len

    def test_compressed_values_are_exact_copies(self, compressor):
        """Compressed KV should be exact slices of the original."""
        seq_len = 256
        q, k, v = _make_qkv(seq_len, 8, 8, 64)
        scale = 0.125

        result = compressor.compress_layer(
            layer_idx=0, query=q, key=k, value=v,
            scale=scale, seq_len=seq_len,
        )

        # Compressed values should match original at selected positions
        assert torch.equal(result.compressed_key, k[result.selected_indices])
        assert torch.equal(
            result.compressed_value, v[result.selected_indices]
        )


class TestLayerCompressionResult:
    """Tests for the result dataclass."""

    def test_compression_ratio(self):
        result = LayerCompressionResult(
            layer_idx=0,
            selection_mask=torch.ones(8, 100, dtype=torch.bool),
            selected_indices=torch.arange(25),
            compressed_key=torch.randn(25, 8, 64),
            compressed_value=torch.randn(25, 8, 64),
            budget=25,
            original_seq_len=100,
        )
        assert result.compression_ratio == 4.0
        assert result.retention_ratio == 0.25
        assert result.num_selected == 25

    def test_zero_selected(self):
        result = LayerCompressionResult(
            layer_idx=0,
            selection_mask=torch.zeros(8, 100, dtype=torch.bool),
            selected_indices=torch.tensor([], dtype=torch.long),
            compressed_key=torch.randn(0, 8, 64),
            compressed_value=torch.randn(0, 8, 64),
            budget=0,
            original_seq_len=100,
        )
        assert result.compression_ratio == float("inf")
        assert result.num_selected == 0


class TestCompressionStats:
    """Tests for compression statistics."""

    def test_stats_basic(self, compressor):
        stats = compressor.get_compression_stats(seq_len=1024)
        assert "config" in stats
        assert "overall_compression_ratio" in stats
        assert "memory_savings_pct" in stats
        assert "effective_capacity_multiplier" in stats

    def test_stats_values(self, compressor):
        stats = compressor.get_compression_stats(seq_len=1024)
        # With 25% retention, 4 layers, expect ~4x compression per layer
        # and the overall should account for pyramid distribution
        assert stats["overall_compression_ratio"] > 1.0
        assert stats["memory_savings_pct"] > 0.0
        assert stats["memory_savings_pct"] < 100.0

    def test_stats_pyramid_allocation(self, compressor):
        stats = compressor.get_compression_stats(seq_len=2048)
        alloc = stats["pyramid_allocation"]
        budgets = alloc["per_layer_budgets"]
        assert len(budgets) == compressor.config.num_layers
        assert alloc["max_layer_budget"] >= alloc["min_layer_budget"]


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config(self, config):
        config.validate()  # Should not raise

    def test_invalid_retention_ratio(self):
        config = KVCompressConfig(retention_ratio=0.0)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_kernel_size(self):
        config = KVCompressConfig(kernel_size=4)  # even
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_alpha(self):
        config = KVCompressConfig(pyramid_alpha=-1)
        with pytest.raises(AssertionError):
            config.validate()

    def test_get_total_budget(self, config):
        budget = config.get_total_budget(1024)
        # Total budget = per_layer_budget * num_layers
        expected = int(1024 * config.retention_ratio) * config.num_layers
        assert budget == expected

    def test_get_total_budget_override(self):
        config = KVCompressConfig(total_budget=100)
        assert config.get_total_budget(1024) == 100
        # capped at seq_len * num_layers
        assert config.get_total_budget(50) == 100  # 100 < 50*32


class TestGQASupport:
    """Tests with GQA (grouped-query attention) configurations."""

    def test_gqa_compress(self):
        config = KVCompressConfig(
            num_layers=2,
            num_kv_heads=4,  # GQA: fewer KV heads
            retention_ratio=0.3,
            observation_window_size=8,
            kernel_size=3,
            recent_window_size=4,
            sink_size=2,
            min_seq_len_to_compress=32,
        )
        comp = SnapPyramidKVCompressor(config)

        seq_len = 128
        num_heads_q = 32  # Many query heads
        num_heads_kv = 4  # Few KV heads
        head_size = 64

        q = torch.randn(seq_len, num_heads_q, head_size)
        k = torch.randn(seq_len, num_heads_kv, head_size)
        v = torch.randn(seq_len, num_heads_kv, head_size)
        scale = 1.0 / (head_size ** 0.5)

        result = comp.compress_layer(
            layer_idx=0, query=q, key=k, value=v,
            scale=scale, seq_len=seq_len,
        )

        assert result.num_selected < seq_len
        assert result.compressed_key.shape[1] == num_heads_kv
        assert result.selection_mask.shape[0] == num_heads_kv


class TestRealisticScenarios:
    """Tests simulating realistic MI100 scenarios."""

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
    def test_various_seq_lengths(self, seq_len):
        """Test across sequence lengths relevant to MI100."""
        config = KVCompressConfig(
            num_layers=32,
            num_kv_heads=8,
            retention_ratio=0.12,
            min_seq_len_to_compress=256,
        )
        comp = SnapPyramidKVCompressor(config)
        stats = comp.get_compression_stats(seq_len)

        # At 12% retention, should achieve significant compression
        assert stats["memory_savings_pct"] > 80.0
        assert stats["effective_capacity_multiplier"] > 5.0

    def test_32gb_mi100_scenario(self):
        """Simulate a realistic MI100 memory scenario.

        With 32GB HBM2 and a 7B model:
        - Model weights: ~14GB (FP16)
        - Available for KV: ~14GB
        - FP16 KV per token per layer: 2 * 8 * 128 * 2 = 4096 bytes
        - 32 layers: 131072 bytes per token
        - Without compression: ~112K tokens
        - With 12% retention: ~933K effective tokens
        """
        config = KVCompressConfig(
            num_layers=32,
            num_kv_heads=8,
            retention_ratio=0.12,
            pyramid_schedule=PyramidSchedule.LINEAR,
            pyramid_alpha=4.0,
        )
        comp = SnapPyramidKVCompressor(config)

        # Test with 32K context (target for MI100)
        stats = comp.get_compression_stats(seq_len=32768)
        assert stats["memory_savings_pct"] > 85.0

        # The pyramid allocation should give lower layers more budget
        budgets = stats["pyramid_allocation"]["per_layer_budgets"]
        assert budgets[0] > budgets[-1]
