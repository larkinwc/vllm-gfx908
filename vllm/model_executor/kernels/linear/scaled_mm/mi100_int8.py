# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MI100 (gfx908) INT8 W8A8 optimized kernel.
# Uses Triton INT8 dot product which maps to gfx908's MFMA instructions
# (v_mfma_i32_32x32x4i8, v_mfma_i32_16x16x16i8) at 185 TOPS peak.
# Tile sizes tuned for 120 CUs, wavefront-64, 64KB LDS per CU.

import contextlib
import logging
import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.configs.gfx908.config_loader import (
    load_config as _load_mi100_autotune_config,
)
from vllm.model_executor.kernels.quantization.fused_silu_quant_int8 import (
    is_fused_silu_quant_int8_enabled,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .mi100_hipblaslt import (
    mi100_hipblaslt_scaled_mm,
    mi100_hipblaslt_supports,
)
from .mi100_int8_dispatch import choose_backend
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = logging.getLogger(__name__)
_mi100_int8_logged = False

# Mission-wide env probe — recorded once per import so the engine log shows
# which silu→quant code path the W8A8 MI100 dispatcher will use. The probe
# only inspects the env state; the actual per-call dispatch additionally
# requires an upstream ``SiluAndMul`` to have stashed a fused cache on the
# layer (see ``MI100Int8ScaledMMLinearKernel.apply_weights``).
logger.info(
    "[MI100_FUSED_ACT_QUANT] W8A8 dispatcher fused_silu_quant_int8 default: "
    "enabled=%s (VLLM_MI100_DISABLE_FUSED_ACT_QUANT=%r, "
    "VLLM_DISABLE_FUSED_ACT_QUANT=%r)",
    is_fused_silu_quant_int8_enabled(),
    os.environ.get("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", "(unset)"),
    os.environ.get("VLLM_DISABLE_FUSED_ACT_QUANT", "(unset)"),
)

# M2 GEMM-shape logging instrumentation (env-flag gated).
# Set VLLM_LOG_GEMM_SHAPES=1 to dump (M,N,K) tuples of every W8A8
# linear call into VLLM_GEMM_SHAPES_OUT (default
# /root/bench-int8-w4a16/tensilelite/gemm_shapes_w8a8.csv). Used by
# scripts/mi100/dump_gemm_shapes.py to gather the input for TensileLite
# tuning.
_gemm_shape_log_path: str | None = None
_gemm_shape_log_handle = None


def _maybe_log_gemm_shape(M: int, N: int, K: int) -> None:
    global _gemm_shape_log_path, _gemm_shape_log_handle
    if os.environ.get("VLLM_LOG_GEMM_SHAPES") != "1":
        return
    if _gemm_shape_log_handle is None:
        path = os.environ.get(
            "VLLM_GEMM_SHAPES_OUT",
            "/root/bench-int8-w4a16/tensilelite/gemm_shapes_w8a8.csv",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.exists(path)
        # Append-mode so multi-rank workers can share the same file. The
        # handle is held for the lifetime of the process — a context
        # manager would defeat the purpose of streaming-append logging.
        _gemm_shape_log_handle = open(path, "a", buffering=1)  # noqa: SIM115
        _gemm_shape_log_path = path
        if new_file:
            _gemm_shape_log_handle.write("M,N,K,dtype,layout,pid\n")
    # Logging must never break the forward path.
    with contextlib.suppress(Exception):
        _gemm_shape_log_handle.write(f"{M},{N},{K},int8,row_col,{os.getpid()}\n")


def _get_tp_rank() -> int:
    """Return the tensor-parallel-size key for the CK instance registry.

    The CK dispatch table is keyed by *world size* (``tp_rank=1`` for
    single-process / TP=1, ``tp_rank=4`` for TP=4 column-parallel, etc.)
    rather than the in-group rank — see
    ``csrc/quantization/w8a8/int8/ck/ck_int8_gemm.h``: "with TP=1 set 0,
    with TP=4 set 4 etc. Only used to pick the right registered
    instance."  Despite the header comment, the M4 instances are
    registered as ``tp_rank=1`` for TP=1 and ``tp_rank=4`` for the
    sharded TP=4 entry, so this helper returns ``world_size`` as the
    consistent registry key. A wrong value just means "no CK" rather
    than a correctness issue (the hipBLASLt → Triton fall-through
    handles every shape).
    """
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        ws = int(get_tensor_model_parallel_world_size())
        if ws >= 1:
            return ws
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except (TypeError, ValueError):
        return 1


def _ck_int8_dispatch(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Route a W8A8 INT8 GEMM through ``torch.ops._rocm_C.ck_int8_gemm``.

    ``mi100_int8_scaled_mm`` carries ``weight`` as ``[K, N]`` row-major
    int8 (transposed in ``process_weights_after_loading``), while the
    CK op expects ``b`` as ``[N, K]`` int8 contiguous. Per-token and
    per-channel scales are flattened to 1-D fp32, expanding scalar
    (per-tensor) scales out to the row/column count so the CK
    ``apply_scales_kernel`` epilogue (which indexes ``scale_a[m]`` /
    ``scale_b[n]``) sees a vector of the right length.
    """
    M, K = input.shape
    N = weight.shape[1]

    a = input.contiguous() if not input.is_contiguous() else input
    # weight is [K, N] row-major; CK wants [N, K] row-major contiguous.
    b_nk = weight.t().contiguous()

    sa = scale_a.to(torch.float32).reshape(-1)
    sb = scale_b.to(torch.float32).reshape(-1)
    if sa.numel() == 1 and M > 1:
        sa = sa.expand(M).contiguous()
    if sb.numel() == 1 and N > 1:
        sb = sb.expand(N).contiguous()

    ck_bias: torch.Tensor | None = None
    if bias is not None:
        # apply_scales_kernel expects __half bias [N]. Cast/flatten here
        # so the call site does not need to know the kernel ABI.
        ck_bias = bias.reshape(-1).to(torch.float16)

    tp_rank = _get_tp_rank()
    out = torch.ops._rocm_C.ck_int8_gemm(a, b_nk, sa, sb, ck_bias, tp_rank)
    # CK always returns fp16; cast if caller asked for a different dtype.
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


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

    # === Fused dequant + bias + cast epilogue ============================
    # 1. Convert INT32 accumulator -> FP32.
    # 2. Apply per-token activation scale (scale_a, [M] when not scalar).
    # 3. Apply per-channel weight scale (scale_b, [N] when not scalar).
    # 4. Optionally add bias (FP32 add for fidelity).
    # 5. Cast to output dtype and store. No separate dequant launch.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    result = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    result = scale_b.T * result

    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias_vec = tl.load(bias_ptrs, bias_mask).to(tl.float32)
        result = result + bias_vec[None, :]

    c = result.to(c_ptr.type.element_ty)

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

    _maybe_log_gemm_shape(M, N, K)

    # M4 dispatch: choose_backend() picks CK > hipBLASLt > Triton. CK is
    # selected only when an instance is registered for the (M, N, K,
    # tp_rank) tuple; the runtime ``ck_int8_gemm_supports`` re-check is
    # defence-in-depth against the registry shifting between dispatch and
    # invocation. ``VLLM_DISABLE_CK=1`` short-circuits this branch and
    # falls through to the legacy hipBLASLt/Triton path.
    tp_rank = _get_tp_rank()
    backend = choose_backend(M, N, K, tp_rank)
    if backend == "ck":
        ck_supports = getattr(
            getattr(torch.ops, "_rocm_C", None),
            "ck_int8_gemm_supports",
            None,
        )
        if ck_supports is not None and bool(ck_supports(M, N, K, tp_rank)):
            return _ck_int8_dispatch(input, weight, scale_a, scale_b, out_dtype, bias)

    # M2 dispatch: prefer hipBLASLt for tuned shapes; otherwise fall through
    # to the existing Triton kernel below. VLLM_DISABLE_HIPBLASLT=1 forces
    # 100% Triton dispatch (validated by VAL-TENSILE-008).
    if mi100_hipblaslt_supports(M, N, K):
        return mi100_hipblaslt_scaled_mm(
            input, weight, scale_a, scale_b, out_dtype, bias
        )

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
    # INT8 elements are 1 byte each, so we can fit larger tiles in LDS.
    # M3: prefer per-shape autotune-selected configs from
    # vllm/model_executor/kernels/configs/gfx908/mi100_int8_M*_N*_K*.json
    # falling back to the static heuristic below for shapes that have not
    # been autotuned yet.
    cfg = _load_mi100_autotune_config("mi100_int8", M=M, N=N, K=K)
    extra_launch: dict = {}

    if cfg is not None:
        block_size_m = int(cfg["BLOCK_M"])
        block_size_n = int(cfg["BLOCK_N"])
        block_size_k = int(cfg["BLOCK_K"])
        group_size_m = int(cfg.get("GROUP_SIZE_M", 8))
        if "num_warps" in cfg:
            extra_launch["num_warps"] = int(cfg["num_warps"])
        if "num_stages" in cfg:
            extra_launch["num_stages"] = int(cfg["num_stages"])
        if "matrix_instr_nonkdim" in cfg:
            extra_launch["matrix_instr_nonkdim"] = int(cfg["matrix_instr_nonkdim"])
        if "kpack" in cfg:
            extra_launch["kpack"] = int(cfg["kpack"])
        if "waves_per_eu" in cfg:
            extra_launch["waves_per_eu"] = int(cfg["waves_per_eu"])
    else:
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

        # L2 cache swizzle: group M-tiles for better reuse
        # MI100 has 8MB L2, grouping helps with weight reuse
        num_pid_n_h = triton.cdiv(N, block_size_n)
        group_size_m = max(1, min(8, 120 // num_pid_n_h)) if num_pid_n_h > 0 else 1

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

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
        **extra_launch,
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
        """W8A8 INT8 linear forward (gfx908).

        When the env-gated ``fused_silu_quant_int8`` path is active (the
        default; disabled by ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`` or the
        legacy alias ``VLLM_DISABLE_FUSED_ACT_QUANT=1``) AND an upstream
        ``SiluAndMul`` has stashed a fused (int8, scale) cache on this layer
        via the ``_mi100_fused_silu_cache`` attribute, the activation
        quantization step is skipped — the cached per-token int8 + fp32 scale
        flow straight into the W8A8 GEMM, eliminating one HBM round-trip on
        the MLP path (issue #33).

        When the cache attribute is absent (no upstream wire-in) OR the
        env-gated flag is set, behavior is byte-identical to the legacy
        ``scaled_int8_quant`` + ``mi100_int8_scaled_mm`` composition.
        """
        w_q, w_s, i_s, i_zp, _ = self._get_layer_params(layer)

        # Fused-path opt-in: the upstream MLP forward
        # (Qwen2MLP / Qwen2MoeMLP via
        # ``try_stash_fused_silu_quant_int8``) may stash the result of
        # ``fused_silu_quant_int8`` on this consuming layer just before
        # invoking the GEMM. The cache shape is ``(int8 [M, K], fp32
        # [M, 1])`` matching ``ops.scaled_int8_quant`` semantics. The
        # producer is the single point that gates on
        # ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT`` — once the cache exists
        # the consumer MUST honor it (the producer skipped the legacy
        # silu_and_mul call so falling through here would feed the GEMM
        # a 0-element placeholder and crash). Disable-path semantics
        # are preserved end-to-end because env-disabled producers never
        # stash in the first place.
        fused_cache = getattr(layer, "_mi100_fused_silu_cache", None)
        if fused_cache is not None:
            # Always clear the cache so a stale stash from a prior layer does
            # not leak into a later forward. Done first so an exception in
            # the GEMM does not orphan the attribute on the layer.
            layer._mi100_fused_silu_cache = None
            x_q, x_s = fused_cache
            assert x_q.dtype == torch.int8, (
                "fused_silu_quant_int8 cache must carry int8 activations; "
                f"got {x_q.dtype}."
            )
            return mi100_int8_scaled_mm(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                bias=bias,
            )

        # Quantize activations to INT8 (dynamic per-token or static)
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=True
        )

        assert x_zp is None, "MI100 INT8 kernel only supports symmetric quantization"

        return mi100_int8_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )
