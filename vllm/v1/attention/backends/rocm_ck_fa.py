# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Composable-Kernel (CK) flash-attention backend for gfx9 (MI100).

Uses the CK-built `flash_attn.flash_attn_varlen_func` (shipped with the
ROCm fork of Dao-AILab/flash-attention, `tridao` branch) to replace our
Triton unified-attention kernel. On MI100 this is ~3-4x faster for
prefill.

Constraints specific to the CK build:
- Paged KV requires `block_size >= 128` (divisible by 128).
- No fp8 KV cache support in this function signature — only fp16/bf16.
  Use `--kv-cache-dtype auto` (or float16/bfloat16), NOT fp8.
- No sliding window support in this build. Models that need SWA must
  fall back to TRITON_ATTN.

The backend reuses the TritonAttentionMetadataBuilder so metadata
construction (block tables, cu_seqlens, reshape_and_cache) stays
identical to the Triton path. Only the forward() attention compute
is swapped.
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


_flash_attn_varlen_func = None


def _resolve_flash_attn():
    """Import flash_attn lazily so vllm startup doesn't fail on boxes
    without it; the backend's validate_configuration will refuse to run
    if the import fails."""
    global _flash_attn_varlen_func
    if _flash_attn_varlen_func is not None:
        return _flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    _flash_attn_varlen_func = flash_attn_varlen_func
    return _flash_attn_varlen_func


class RocmCKFlashAttentionBackend(AttentionBackend):
    # CK FA in this build does not support fp8 KV cache via descale args.
    # Quantized KV requires fallback to TRITON_ATTN.
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # CK paged KV requires block_size % 128 == 0.
        return [MultipleOf(128)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size >= 128 and block_size % 128 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_CK_FA"

    @staticmethod
    def get_impl_cls() -> type["RocmCKFlashAttentionImpl"]:
        return RocmCKFlashAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 128 != 0:
            raise ValueError(
                f"ROCM_CK_FA requires block_size % 128 == 0, got {block_size}"
            )
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # CK FA supports head_size up to 256; we gate on the common set.
        return head_size in (64, 128, 192, 256)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # Only decoder self-attention for now; encoder paths keep using
        # TRITON_ATTN.
        return attn_type in (AttentionType.DECODER,)

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # gfx9 family: MI100 (9.0), MI200 (9.0), MI300 (9.4). We gate
        # specifically on MI100 since that's the only arch we've built
        # the CK flash_attn wheel for in this project.
        from vllm.platforms.rocm import on_mi100

        return on_mi100()


class RocmCKFlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "RocmCKFlashAttentionImpl only supports decoder self-"
                f"attention, got attn_type={attn_type!r}."
            )
        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError(
                "RocmCKFlashAttentionImpl does not support fp8/int8 KV "
                "cache. Use --kv-cache-dtype auto, float16 or bfloat16, "
                "or switch to --attention-backend TRITON_ATTN."
            )
        if sliding_window is not None:
            logger.warning_once(
                "RocmCKFlashAttentionImpl sliding_window=%d not yet "
                "validated against CK FA build; using full attention.",
                sliding_window,
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = (
            torch.tensor(alibi_slopes, dtype=torch.float32)
            if alibi_slopes is not None
            else None
        )
        self.sliding_window = (-1, -1) if sliding_window is None else (
            sliding_window - 1,
            0,
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using CK flash_attn_varlen_func.

        Args:
            query: [num_tokens, num_heads, head_size]
            key:   [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quant is not supported by RocmCKFlashAttentionImpl"
            )
        if attn_metadata is None:
            # Profile run — vllm calls forward with None to size activations.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(1)

        flash_attn_varlen_func = _resolve_flash_attn()

        # CK flash_attn_varlen_func with block_table does prefill + decode
        # + extend in one unified call; cu_seqlens_q spans the flat token
        # batch and seq_lens are looked up via block_table + cu_seqlens_k.
        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]

        # TritonAttentionMetadata only carries `seq_lens` (per-seq kv length).
        # flash_attn_varlen_func needs `cu_seqlens_k` — the cumulative prefix.
        # Must be constructed entirely on GPU so CUDA-graph capture (used on
        # decode steps) doesn't see any CPU->GPU copies.
        seq_lens = attn_metadata.seq_lens
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )

        # Note: ROCm/flash-attention tridao build doesn't support an `out=`
        # kwarg — the function allocates its own return tensor. Copy into
        # vllm's pre-allocated `output` buffer so the rest of the runtime
        # sees the expected in-place write semantics.
        attn_out = flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            block_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            alibi_slopes=self.alibi_slopes,
        )
        out.copy_(attn_out)
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write new K,V tokens into the paged cache.

        vllm's newer scheduler invokes kv-cache write as a separate hook
        so the attention kernel can overlap with it. We forward to the
        same Triton reshape_and_cache used by the Triton backend — CK FA
        itself reads the cache, it does not own it.
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = kv_cache.unbind(1)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )


# Re-export for convenience so the registry can import from this module.
AttentionCGSupport  # noqa: F401 (re-export sentinel for type stubs)
AttentionLayer  # noqa: F401
AttentionSpec  # noqa: F401
