# SPDX-License-Identifier: Apache-2.0
"""LazyLLM-aware model components for Qwen3.

Provides instrumented attention layers and a modified model forward pass
that supports per-layer dynamic token pruning.

Usage:
    # Enable via model config
    vllm_config.lazy_llm_config = LazyLLMConfig(...)
    model = Qwen3LazyLLMModel(vllm_config=vllm_config)
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.attention import Attention, AttentionType
from vllm.attention.lazy_llm.config import LazyLLMConfig
from vllm.attention.lazy_llm.pruner import (
    PrefillPruningState,
    ProgressiveTokenPruner,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3Model,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    get_tensor_model_parallel_world_size,
    is_pp_missing_parameter,
    maybe_prefix,
    PPMissingLayer,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context


class Qwen3LazyLLMAttention(nn.Module):
    """Qwen3 attention with Q/K instrumentation for LazyLLM pruning.

    Stores Q and K tensors after RoPE so the LazyLLM pruner can compute
    token importance without recomputing attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Storage for Q/K tensors (set by forward, used by pruner)
        self._last_q: torch.Tensor | None = None
        self._last_k: torch.Tensor | None = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        # Store Q/K for pruner (reshaped for importance scoring)
        num_tokens = q.shape[0]
        self._last_q = q.view(num_tokens, self.num_heads, self.head_dim)
        self._last_k = k.view(num_tokens, self.num_kv_heads, self.head_dim)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3LazyLLMDecoderLayer(nn.Module):
    """Decoder layer with LazyLLM instrumentation.

    Exposes the attention's Q/K for token importance computation.
    """

    def __init__(
        self,
        config: Qwen3Config,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        from vllm.model_executor.models.qwen3 import set_default_rope_theta
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )

        self.self_attn = Qwen3LazyLLMAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # Import MLP from qwen3
        from vllm.model_executor.models.qwen3 import Qwen3MLP
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3LazyLLMModel(Qwen3Model):
    """Qwen3 model with LazyLLM dynamic token pruning during prefill.

    Inherits from Qwen3Model but overrides the forward loop to:
    1. Maintain a ProgressiveTokenPruner state across layers
    2. Prune tokens after each layer's attention
    3. Index positions and residual to match active tokens

    Usage:
        from vllm.attention.lazy_llm import LazyLLMConfig
        lazy_config = LazyLLMConfig(num_layers=40, drop_ratio=0.5)
        vllm_config.lazy_llm_config = lazy_config
        model = Qwen3LazyLLMModel(vllm_config=vllm_config)
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        # Import necessary modules
        from vllm.model_executor.models.utils import make_layers
        from vllm.model_executor.model_loader.weight_utils import (
            make_empty_intermediate_tensors_factory,
        )

        # Don't call Qwen3Model.__init__ directly, go to grandparent Qwen2Model
        # but we need to re-implement because we use different layer type
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Check for sliding window constraints
        from vllm.model_executor.models.qwen2 import is_interleaved
        if is_interleaved(vllm_config.model_config.hf_text_config):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported."
            )

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3LazyLLMDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Initialize LazyLLM pruner if config provided
        self.lazy_llm_config: LazyLLMConfig | None = getattr(
            vllm_config, "lazy_llm_config", None
        )
        self._pruner: ProgressiveTokenPruner | None = None
        self._pruning_state: PrefillPruningState | None = None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with per-layer token pruning.

        The key modification is that after each layer, we potentially
        prune tokens based on attention importance scores.
        """
        # Get forward context to check if this is prefill
        forward_context = get_forward_context()
        is_prefill = forward_context is not None and getattr(
            forward_context, "is_prompt", True
        )

        # Initialize hidden states
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        seq_len = hidden_states.shape[0]

        # Initialize LazyLLM pruner for prefill
        if is_prefill and self.lazy_llm_config is not None:
            self._pruner = ProgressiveTokenPruner(self.lazy_llm_config)
            self._pruning_state = self._pruner.init_prefill_state(
                seq_len, hidden_states.device
            )
            active_indices = self._pruning_state.active_indices
        else:
            self._pruner = None
            self._pruning_state = None
            active_indices = torch.arange(seq_len, device=hidden_states.device)

        # Track per-layer token counts for debugging
        per_layer_active = [seq_len]

        from itertools import islice

        # Forward through layers
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            layer_idx = self.start_layer + idx
            cur_seq_len = hidden_states.shape[0]

            # Index positions for active tokens only
            layer_positions = positions[active_indices]

            # Run layer
            hidden_states, residual = layer(
                positions=layer_positions,
                hidden_states=hidden_states,
                residual=residual,
            )

            # After attention, maybe prune for next layer (only during prefill)
            if (
                is_prefill
                and self._pruner is not None
                and self._pruning_state is not None
                and self.lazy_llm_config.should_prune(layer_idx, seq_len)
            ):
                # Get Q/K from attention layer
                q = layer.self_attn._last_q
                k = layer.self_attn._last_k

                if q is not None and k is not None:
                    hidden_states, self._pruning_state = (
                        self._pruner.maybe_prune_tokens(
                            layer_idx=layer_idx,
                            hidden_states=hidden_states,
                            state=self._pruning_state,
                            query=q,
                            key=k,
                            scale=layer.self_attn.scaling,
                        )
                    )
                    # Update positions and residual
                    active_indices = self._pruning_state.active_indices
                    if residual is not None:
                        # Re-index residual to match pruned hidden states
                        local_indices = torch.arange(
                            cur_seq_len, device=residual.device
                        )
                        mask = torch.isin(
                            local_indices, self._pruning_state.active_indices
                        )
                        residual = residual[mask]

            per_layer_active.append(hidden_states.shape[0])

        if not get_pp_group().is_last_rank:
            from vllm.model_executor.model_loader.weight_utils import (
                IntermediateTensors,
            )
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        # Log pruning statistics if in prefill
        if is_prefill and self._pruner is not None and self._pruning_state is not None:
            summary = self._pruner.get_pruning_summary(self._pruning_state)
            # Could log here or store for retrieval
            self._last_pruning_summary = summary

        return hidden_states
