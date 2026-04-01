# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for LazyLLM model components.

Tests the full pipeline: LazyLLM config → model forward pass → token pruning.
"""

import pytest
import torch
from torch import nn

# Check if we can import vllm model components
try:
    from vllm.model_executor.models.qwen3_lazyllm import (
        Qwen3LazyLLMAttention,
        Qwen3LazyLLMDecoderLayer,
    )
    VLLM_MODELS_AVAILABLE = True
except ImportError:
    VLLM_MODELS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not VLLM_MODELS_AVAILABLE,
    reason="vLLM model components not available (missing dependencies)",
)


class TestQwen3LazyLLMAttention:
    """Tests for the instrumented attention layer."""

    @pytest.fixture
    def mock_config(self):
        """Minimal config for testing."""
        class Config:
            hidden_size = 128
            num_attention_heads = 4
            num_key_value_heads = 2
            max_position_embeddings = 2048
            rms_norm_eps = 1e-6
            rope_parameters = {}
            intermediate_size = 256
            hidden_act = "silu"
        return Config()

    def test_qk_storage(self, mock_config):
        """Test that Q/K are stored after forward pass."""
        attn = Qwen3LazyLLMAttention(
            hidden_size=mock_config.hidden_size,
            num_heads=mock_config.num_attention_heads,
            num_kv_heads=mock_config.num_key_value_heads,
            rope_parameters=mock_config.rope_parameters,
            max_position=mock_config.max_position_embeddings,
            rms_norm_eps=mock_config.rms_norm_eps,
            qkv_bias=False,
        )

        # Check initial state
        assert attn._last_q is None
        assert attn._last_k is None

        # Run forward
        seq_len = 50
        hidden = torch.randn(seq_len, mock_config.hidden_size)
        positions = torch.arange(seq_len)

        # Need to init weights first to avoid uninitialized errors
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
        attn.apply(init_weights)

        with torch.no_grad():
            _ = attn(positions, hidden)

        # Check Q/K are now stored
        assert attn._last_q is not None
        assert attn._last_k is not None

    def test_qk_shapes(self, mock_config):
        """Test that stored Q/K have correct shapes."""
        num_heads = mock_config.num_attention_heads
        head_dim = mock_config.hidden_size // num_heads

        attn = Qwen3LazyLLMAttention(
            hidden_size=mock_config.hidden_size,
            num_heads=num_heads,
            num_kv_heads=mock_config.num_key_value_heads,
            rope_parameters=mock_config.rope_parameters,
            max_position=mock_config.max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=mock_config.rms_norm_eps,
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
        attn.apply(init_weights)

        seq_len = 50
        hidden = torch.randn(seq_len, mock_config.hidden_size)
        positions = torch.arange(seq_len)

        with torch.no_grad():
            _ = attn(positions, hidden)

        assert attn._last_q.shape == (seq_len, num_heads, head_dim)
        assert attn._last_k.shape[0] == seq_len  # KV heads may differ
        assert attn._last_k.shape[2] == head_dim


class TestQwen3LazyLLMDecoderLayer:
    """Tests for the instrumented decoder layer."""

    def test_decoder_runs(self):
        """Test that decoder layer can run forward."""
        class Config:
            hidden_size = 128
            num_attention_heads = 4
            num_key_value_heads = 2
            max_position_embeddings = 2048
            rms_norm_eps = 1e-6
            rope_parameters = {}
            intermediate_size = 256
            hidden_act = "silu"
            is_causal = True

        config = Config()
        layer = Qwen3LazyLLMDecoderLayer(
            config=config,
            prefix="layers.0",
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
        layer.apply(init_weights)

        seq_len = 30
        hidden = torch.randn(seq_len, config.hidden_size)
        positions = torch.arange(seq_len)

        with torch.no_grad():
            out, residual = layer(positions, hidden, None)

        assert out.shape == (seq_len, config.hidden_size)
        assert residual is not None


class TestModelIntegration:
    """Integration tests for full model with pruning."""

    def test_mock_prefill_pruning_simulation(self):
        """Simulate a prefill with token pruning end-to-end."""
        from vllm.attention.lazy_llm import LazyLLMConfig, PruningSchedule
        from vllm.attention.lazy_llm.pruner import (
            PrefillPruningState,
            ProgressiveTokenPruner,
        )

        # Config
        config = LazyLLMConfig(
            num_layers=8,
            drop_ratio=0.5,
            schedule=PruningSchedule.LINEAR,
            start_layer=2,
            min_seq_len=10,
            min_tokens=4,
            protected_tokens=2,
        )

        # Simulate prefill
        seq_len = 50
        pruner = ProgressiveTokenPruner(config)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))

        # Simulate layers
        hidden_size = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden_size // num_heads

        for layer_idx in range(8):
            cur_len = hidden.shape[0] if layer_idx > 0 else seq_len
            hidden = torch.randn(cur_len, hidden_size)

            # Simulate Q/K for importance
            q = torch.randn(cur_len, num_heads, head_dim)
            k = torch.randn(cur_len, num_kv_heads, head_dim)

            # Run pruning after attention
            hidden, state = pruner.maybe_prune_tokens(
                layer_idx=layer_idx,
                hidden_states=hidden,
                state=state,
                query=q,
                key=k,
                scale=head_dim ** -0.5,
            )

        # Verify pruning occurred
        summary = pruner.get_pruning_summary(state)
        assert summary["original_seq_len"] == seq_len
        assert summary["final_active"] < seq_len
        assert summary["compression_ratio"] < 1.0

    def test_progressive_token_reduction(self):
        """Test that token count decreases progressively across layers."""
        from vllm.attention.lazy_llm import LazyLLMConfig, PruningSchedule
        from vllm.attention.lazy_llm.pruner import ProgressiveTokenPruner

        config = LazyLLMConfig(
            num_layers=10,
            drop_ratio=0.6,
            schedule=PruningSchedule.LINEAR,
            start_layer=2,
            min_seq_len=100,
        )

        seq_len = 200
        pruner = ProgressiveTokenPruner(config)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))

        hidden_size = 32
        num_heads = 4
        head_dim = hidden_size // num_heads

        per_layer_counts = []

        for layer_idx in range(10):
            cur_len = state.num_active
            per_layer_counts.append(cur_len)

            hidden = torch.randn(cur_len, hidden_size)
            q = torch.randn(cur_len, num_heads, head_dim)
            k = torch.randn(cur_len, num_heads, head_dim)

            hidden, state = pruner.maybe_prune_tokens(
                layer_idx=layer_idx,
                hidden_states=hidden,
                state=state,
                query=q,
                key=k,
                scale=head_dim ** -0.5,
            )

        # Verify progressive reduction
        assert per_layer_counts[0] == seq_len
        assert per_layer_counts[1] == seq_len  # No pruning in early layers
        # Later layers should have fewer tokens
        assert per_layer_counts[-1] < per_layer_counts[5]


class TestConfigIntegration:
    """Tests for LazyLLM config integration."""

    def test_config_to_pruner_pipeline(self):
        """Full pipeline from config creation to pruning."""
        from vllm.attention.lazy_llm import LazyLLMConfig, ProgressiveTokenPruner

        config = LazyLLMConfig(
            num_layers=16,
            drop_ratio=0.4,
            start_layer=4,
        )

        pruner = ProgressiveTokenPruner(config)

        # Verify pruner has correct config
        assert pruner.config.num_layers == 16
        assert pruner.config.drop_ratio == 0.4

        # Initialize state
        state = pruner.init_prefill_state(1000, torch.device("cpu"))
        assert state.seq_len == 1000
        assert state.num_active == 1000

    def test_per_layer_keep_ratios(self):
        """Test that keep ratios are applied correctly per layer."""
        from vllm.attention.lazy_llm import LazyLLMConfig, PruningSchedule

        config = LazyLLMConfig(
            num_layers=10,
            drop_ratio=0.5,
            schedule=PruningSchedule.LINEAR,
            start_layer=2,
        )

        # Layers 0-1: keep 100%
        assert config.get_keep_ratio(0) == 1.0
        assert config.get_keep_ratio(1) == 1.0

        # Layer 2: still high (start of gradual drop)
        assert config.get_keep_ratio(2) > 0.9

        # Layer 9: closer to final drop ratio
        assert config.get_keep_ratio(9) < 0.6
        assert config.get_keep_ratio(9) >= 0.5


class TestEdgeCases:
    """Edge case tests for integration."""

    def test_short_sequence_no_pruning(self):
        """Short sequences should not be pruned."""
        from vllm.attention.lazy_llm import LazyLLMConfig, ProgressiveTokenPruner

        config = LazyLLMConfig(
            num_layers=10,
            drop_ratio=0.5,
            min_seq_len=256,
        )

        pruner = ProgressiveTokenPruner(config)
        state = pruner.init_prefill_state(100, torch.device("cpu"))

        hidden = torch.randn(100, 64)
        q = torch.randn(100, 4, 16)
        k = torch.randn(100, 4, 16)

        out, state = pruner.maybe_prune_tokens(
            layer_idx=5,  # Past start_layer
            hidden_states=hidden,
            state=state,
            query=q,
            key=k,
            scale=0.25,
        )

        # Should not prune
        assert out.shape[0] == 100
        assert state.num_active == 100

    def test_early_layer_no_pruning(self):
        """Early layers should not prune."""
        from vllm.attention.lazy_llm import LazyLLMConfig, ProgressiveTokenPruner

        config = LazyLLMConfig(
            num_layers=10,
            drop_ratio=0.5,
            start_layer=4,
        )

        pruner = ProgressiveTokenPruner(config)
        state = pruner.init_prefill_state(1000, torch.device("cpu"))

        hidden = torch.randn(1000, 64)
        q = torch.randn(1000, 4, 16)
        k = torch.randn(1000, 4, 16)

        out, state = pruner.maybe_prune_tokens(
            layer_idx=2,  # Before start_layer
            hidden_states=hidden,
            state=state,
            query=q,
            key=k,
            scale=0.25,
        )

        assert out.shape[0] == 1000

    def test_protected_tokens_always_kept(self):
        """Protected tokens should always be retained."""
        from vllm.attention.lazy_llm import LazyLLMConfig, ProgressiveTokenPruner

        config = LazyLLMConfig(
            num_layers=10,
            drop_ratio=0.9,  # Very aggressive
            start_layer=0,
            protected_tokens=5,
            min_tokens=10,
        )

        seq_len = 100
        pruner = ProgressiveTokenPruner(config)
        state = pruner.init_prefill_state(seq_len, torch.device("cpu"))

        hidden = torch.randn(seq_len, 64)
        q = torch.randn(seq_len, 4, 16)
        k = torch.randn(seq_len, 4, 16)
        # Make first 5 tokens have zero importance
        k[:5] = 0.0

        out, state = pruner.maybe_prune_tokens(
            layer_idx=5,
            hidden_states=hidden,
            state=state,
            query=q,
            key=k,
            scale=0.25,
        )

        # First 5 tokens should still be in active indices
        active = state.active_indices.tolist()
        for i in range(5):
            assert i in active
