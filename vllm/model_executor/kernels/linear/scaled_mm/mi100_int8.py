# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MI100 (gfx908) INT8 W8A8 optimized kernel.
# Uses Triton INT8 dot product which maps to gfx908's MFMA instructions
# (v_mfma_i32_32x32x4i8, v_mfma_i32_16x16x16i8) at 185 TOPS peak.
# Tile sizes tuned for 120 CUs, wavefront-64, 64KB LDS per CU.

import logging

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = logging.getLogger(__name__)
_mi100_int8_logged = False


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@triton.jit
def mi100_int8_scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # INT8 dot -> INT32 accumulator (maps to v_mfma_i32_*_i8 on gfx908)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # INT8 x INT8 -> INT32 accumulate (MFMA on gfx908)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Dequantize: convert INT32 accumulator to FP32, apply scales
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    result = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    result = scale_b.T * result

    c = result.to(c_ptr.type.element_ty)

    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def mi100_int8_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """MI100-optimized INT8 scaled matmul.

    input:  [M, K] int8
    weight: [K, N] int8
    scale_a: per-tensor [1,1] or per-token [M,1] float32
    scale_b: per-tensor [1,1] or per-channel [N,1] float32
    """
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == torch.int8 and weight.dtype == torch.int8

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    # MI100-tuned tile sizes for gfx908 (120 CUs, wavefront-64, 64KB LDS)
    # INT8 elements are 1 byte each, so we can fit larger tiles in LDS:
    # A tile: BLOCK_M * BLOCK_K * 1B, B tile: BLOCK_K * BLOCK_N * 1B
    # With 64x64x128: A=8KB + B=8KB = 16KB << 64KB LDS
    # With 128x128x64: A=8KB + B=8KB = 16KB, good CU occupancy
    is_small_N = N < 8192
    next_power_of_2_M = max(32, triton.next_power_of_2(M))

    if next_power_of_2_M <= 32:
        # Decode-like: small M, use wider K tiles for bandwidth
        tile_shape = (32, 64, 128) if is_small_N else (32, 128, 128)
    elif next_power_of_2_M <= 64:
        tile_shape = (64, 64, 128) if is_small_N else (64, 128, 128)
    elif next_power_of_2_M <= 128:
        tile_shape = (64, 128, 64)
    else:
        # Prefill-like: large M, balance M/N tiles
        tile_shape = (128, 128, 64)

    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    # L2 cache swizzle: group M-tiles for better reuse
    # MI100 has 8MB L2, grouping helps with weight reuse
    num_pid_n = triton.cdiv(N, block_size_n)
    group_size_m = max(1, min(8, 120 // num_pid_n)) if num_pid_n > 0 else 1

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    mi100_int8_scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        GROUP_SIZE_M=group_size_m,
    )

    return result


class MI100Int8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """MI100-optimized INT8 W8A8 kernel using Triton INT8 MFMA instructions.

    On gfx908, INT8 MFMA provides 185 TOPS (4x the 46 TFLOPS FP16).
    This kernel uses INT8 dot products with INT32 accumulation, which
    Triton maps to v_mfma_i32_32x32x4i8 / v_mfma_i32_16x16x16i8.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        from vllm.platforms.rocm import on_mi100

        if not on_mi100():
            return False, "requires MI100 (gfx908)."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric input only."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        global _mi100_int8_logged
        if not _mi100_int8_logged:
            logger.info(
                "[MI100_INT8] Using Triton INT8 MFMA kernel (185 TOPS INT8 on gfx908)"
            )
            _mi100_int8_logged = True

        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        # Transpose weight to [K, N] for the kernel
        w_q = getattr(layer, w_q_name)
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(w_q.t().data, requires_grad=False),
        )

        # Handle fused module scales
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # Input scale
        if self.config.is_static_input_scheme:
            i_s = getattr(layer, i_s_name)
            if i_s is not None:
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(i_s.max(), requires_grad=False),
                )
            setattr(layer, i_zp_name, None)
        else:
            setattr(layer, i_s_name, None)
            setattr(layer, i_zp_name, None)

        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, _ = self._get_layer_params(layer)

        # Quantize activations to INT8 (dynamic per-token or static)
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=True
        )

        assert x_zp is None, "MI100 INT8 kernel only supports symmetric quantization"

        return mi100_int8_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )
