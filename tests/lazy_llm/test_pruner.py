# SPDX-License-Identifier: Apache-2.0
"""Tests for ProgressiveTokenPruner."""

import torch
import pytest

from vllm.attention.lazy_llm.config import LazyLLMConfig, PruningSchedule
from vllm.attention.lazy_llm.pruner import (
    ProgressiveTokenPruner,
    PrefillPruningState,
    LayerPruningResult,
)


class TestPrefillPruningState:
    """Tests for the PrefillPruningState dataclass."""

    def test_init(self):
        state = PrefillPruningState(
            seq_len=1000,
            active_indices=torch.arange(1000),
        )
        assert state.num_active == 1000
        assert state.compression_ratio == 1.0
        assert state.total_tokens_saved == 0

    def test_compression_ratio_after_pruning(self):
        state = PrefillPruningState(
            seq_len=1000,
            active_indices=torch.arange(500),
        )
        assert abs(state.compression_ratio - 0.5) < 0.01

    def test_total_tokens_saved(self):
        state = PrefillPruningState(
            seq_len=1000,
            active_indices=torch.arange(1000),
            layer_results=[
                LayerPruningResult(
                    kept_indices=torch.arange(800),
                    pruned_indices=torch.arange(200),
                    num_kept=800, num_total=1000,
                ),
                LayerPruningResult(
                    kept_indices=torch.arange(600),
                    pruned_indices=torch.arange(200),
                    num_kept=600, num_total=800,
                ),
            ],
        )
        assert state.total_tokens_saved == 400  # 200 + 200


class TestProgressiveTokenPruner:
    """Tests for the main pruner class."""

    @pytest.fixture
    def config(self):
        return LazyLLMConfig(
            num_layers=32, drop_ratio=0.5,
            schedule=PruningSchedule.LINEAR,
            start_layer=2, min_seq_len=256,
            min_tokens=64, protected_tokens=4,
        )

    @pytest.fixture
    def pruner(self, config):
        return ProgressiveTokenPruner(config)

    def test_init_prefill_state(self, pruner):
        state = pruner.init_prefill_state(1000, torch.device("cpu"))
        assert state.seq_len == 1000
        assert state.num_active == 1000
        assert len(state.aux_hidden_states) == 0

    def test_no_prune_short_sequence(self, pruner):
        """Short sequences should not be pruned."""
        hidden = torch.randn(100, 128)
        state = pruner.init_prefill_state(100, torch.device("cpu"))
        q = torch.randn(100, 8, 64)
        k = torch.randn(100, 8, 64)

        out, state = pruner.maybe_prune_tokens(
            layer_idx=10, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        assert out.shape == hidden.shape
        assert state.num_active == 100

    def test_no_prune_early_layer(self, pruner):
        """Layers before start_layer should not prune."""
        hidden = torch.randn(1000, 128)
        state = pruner.init_prefill_state(1000, torch.device("cpu"))
        q = torch.randn(1000, 8, 64)
        k = torch.randn(1000, 8, 64)

        out, state = pruner.maybe_prune_tokens(
            layer_idx=0, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        assert out.shape[0] == 1000

    def test_prune_active_layer(self, pruner):
        """Active pruning layers should reduce token count."""
        seq_len = 1000
        hidden = torch.randn(seq_len, 128)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))
        q = torch.randn(seq_len, 8, 64)
        k = torch.randn(seq_len, 8, 64)

        out, state = pruner.maybe_prune_tokens(
            layer_idx=20, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        assert out.shape[0] < seq_len
        assert state.num_active < seq_len
        assert len(state.layer_results) == 1

    def test_aux_cache_populated(self, pruner):
        """Pruned tokens should be saved in aux cache."""
        seq_len = 1000
        hidden = torch.randn(seq_len, 128)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))
        q = torch.randn(seq_len, 8, 64)
        k = torch.randn(seq_len, 8, 64)

        _, state = pruner.maybe_prune_tokens(
            layer_idx=20, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        assert 20 in state.aux_hidden_states
        aux = state.aux_hidden_states[20]
        # aux should have the pruned tokens
        assert aux.shape[0] + state.num_active == seq_len
        assert aux.shape[1] == 128

    def test_progressive_pruning(self, pruner):
        """Running through multiple layers should progressively reduce tokens."""
        seq_len = 1000
        hidden_dim = 128
        num_heads = 8
        head_dim = 64
        hidden = torch.randn(seq_len, hidden_dim)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))

        prev_active = seq_len
        for layer_idx in range(32):
            q = torch.randn(hidden.shape[0], num_heads, head_dim)
            k = torch.randn(hidden.shape[0], num_heads, head_dim)
            hidden, state = pruner.maybe_prune_tokens(
                layer_idx=layer_idx, hidden_states=hidden, state=state,
                query=q, key=k, scale=head_dim ** -0.5,
            )
            assert hidden.shape[0] <= prev_active
            prev_active = hidden.shape[0]

        # Should have significantly fewer tokens by end
        assert state.num_active < seq_len
        assert state.compression_ratio < 0.8

    def test_no_prune_without_importance(self, pruner):
        """If no Q/K/attention provided, no pruning should happen."""
        hidden = torch.randn(1000, 128)
        state = pruner.init_prefill_state(1000, torch.device("cpu"))
        out, state = pruner.maybe_prune_tokens(
            layer_idx=20, hidden_states=hidden, state=state,
        )
        assert out.shape[0] == 1000

    def test_protected_tokens_kept(self):
        """Protected tokens should always be kept."""
        config = LazyLLMConfig(
            num_layers=32, drop_ratio=0.9,
            start_layer=0, min_seq_len=10,
            protected_tokens=4, min_tokens=10,
        )
        pruner = ProgressiveTokenPruner(config)
        seq_len = 500
        hidden = torch.randn(seq_len, 128)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))

        # Make first 4 tokens have low importance
        q = torch.randn(seq_len, 8, 64)
        k = torch.randn(seq_len, 8, 64)
        # Zero out first 4 keys so they'd normally be pruned
        k[:4] = 0.0

        _, state = pruner.maybe_prune_tokens(
            layer_idx=20, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        # First 4 original indices should still be active
        active = state.active_indices.tolist()
        for i in range(4):
            assert i in active

    def test_pruning_summary(self, pruner):
        """Summary should contain expected fields."""
        seq_len = 1000
        hidden = torch.randn(seq_len, 128)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))
        q = torch.randn(seq_len, 8, 64)
        k = torch.randn(seq_len, 8, 64)

        hidden, state = pruner.maybe_prune_tokens(
            layer_idx=20, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        summary = pruner.get_pruning_summary(state)
        assert "original_seq_len" in summary
        assert "final_active" in summary
        assert "compression_ratio" in summary
        assert "total_tokens_saved" in summary
        assert "per_layer" in summary
        assert len(summary["per_layer"]) == 1

    def test_active_indices_are_subset(self, pruner):
        """Active indices after pruning must be valid original indices."""
        seq_len = 1000
        hidden = torch.randn(seq_len, 128)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))
        q = torch.randn(seq_len, 8, 64)
        k = torch.randn(seq_len, 8, 64)

        _, state = pruner.maybe_prune_tokens(
            layer_idx=20, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        assert (state.active_indices >= 0).all()
        assert (state.active_indices < seq_len).all()
        # Should be sorted
        assert (state.active_indices[1:] > state.active_indices[:-1]).all()


class TestPrunerGPU:
    """GPU tests (skipped if CUDA not available)."""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda:0")

    def test_gpu_pruning(self, device):
        config = LazyLLMConfig(
            num_layers=32, drop_ratio=0.5,
            start_layer=2, min_seq_len=256,
        )
        pruner = ProgressiveTokenPruner(config)
        seq_len = 2000
        hidden = torch.randn(seq_len, 256, device=device)
        state = pruner.init_prefill_state(seq_len, device)
        q = torch.randn(seq_len, 8, 64, device=device)
        k = torch.randn(seq_len, 8, 64, device=device)

        out, state = pruner.maybe_prune_tokens(
            layer_idx=20, hidden_states=hidden, state=state,
            query=q, key=k, scale=0.125,
        )
        assert out.device == device
        assert state.active_indices.device == device
