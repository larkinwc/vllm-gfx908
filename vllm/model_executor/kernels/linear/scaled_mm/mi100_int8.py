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
    is_fused_silu_quant_int8_disabled,
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
from .mi100_int8_dispatch import (
    choose_backend,
    choose_emit_int8_next,
)
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
    out_int8_ptr,
    out_scale_ptr,
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
    EMIT_INT8_NEXT: tl.constexpr = False,
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

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

    if EMIT_INT8_NEXT:
        # M2 INT8-emit store epilogue (issue #26). The fp16 output tile is
        # converted to per-token-scaled int8 in registers/LDS *before* the
        # HBM store, so the next consuming W8A8 GEMM can read the int8 +
        # fp32 scale tensors directly without a fp16→int8 dequant/requant
        # round-trip through HBM. The per-token absmax is reduced across
        # the BLOCK_SIZE_N tile dimension; the dispatcher
        # (mi100_int8_dispatch.py) is responsible for only activating this
        # path when BLOCK_SIZE_N covers the full row N (i.e. a single
        # column tile per (pid_m, ...)) — otherwise the per-tile reductions
        # would disagree across pid_n shards for the same M-row.
        # Mask out-of-bounds columns to zero so they cannot inflate the
        # absmax above the valid-data range.
        col_mask = offs_cn[None, :] < N
        result_masked = tl.where(col_mask, result, 0.0)
        # Per-token absmax over the N tile (axis=-1).
        row_absmax = tl.max(tl.abs(result_masked), axis=-1)
        # Guard against zero rows (all-zero tile) — keep scale finite so
        # the divide-and-round below stays well-defined; the resulting
        # int8 row is still all-zero, which is the correct degenerate
        # output for a zero input row.
        eps = 1e-12
        row_scale = tl.maximum(row_absmax, eps) / 127.0
        # Quantize: int8 = round(result / scale * 127) ≡
        # round(result / row_scale)  (since row_scale = absmax/127)
        out_int8 = result_masked / row_scale[:, None]
        # Symmetric round-half-to-even semantics via libdevice are not
        # available across ROCm/CUDA in Triton 3.5.1; tl.extra.libdevice
        # rint is fine on both. Fall back to round-to-nearest-even via
        # the standard cast chain (float→int8 in Triton truncates toward
        # zero, so add a 0.5 magnitude before casting).
        out_int8 = tl.where(out_int8 >= 0, out_int8 + 0.5, out_int8 - 0.5)
        # Clamp to int8 range before cast (defensive — should already be
        # ≤ |127| after the scale).
        out_int8 = tl.maximum(tl.minimum(out_int8, 127.0), -128.0)
        out_int8 = out_int8.to(tl.int8)

        out_int8_ptrs = (
            out_int8_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        )
        out_int8_mask = (offs_cm[:, None] < M) & col_mask
        tl.store(out_int8_ptrs, out_int8, mask=out_int8_mask)

        # Per-token fp32 scale, shape [M, 1]. Store only when this program
        # owns column tile 0 to avoid redundant writes across pid_n shards
        # (dispatcher should guarantee a single pid_n in EMIT_INT8_NEXT
        # mode, but be safe — the value is identical across shards in that
        # configuration).
        if pid_n == 0:
            out_scale_ptrs = out_scale_ptr + offs_cm
            out_scale_mask = offs_cm < M
            tl.store(out_scale_ptrs, row_scale, mask=out_scale_mask)
    else:
        c = result.to(c_ptr.type.element_ty)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        tl.store(c_ptrs, c, mask=c_mask)


def mi100_int8_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype] = torch.float16,
    bias: torch.Tensor | None = None,
    emit_int8_next: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MI100-optimized INT8 scaled matmul.

    input:  [M, K] int8
    weight: [K, N] int8
    scale_a: per-tensor [1,1] or per-token [M,1] float32
    scale_b: per-tensor [1,1] or per-channel [N,1] float32

    When ``emit_int8_next=False`` (the default), behavior is byte-identical to
    the legacy fp16 path: returns a single ``[M, N]`` tensor in ``out_dtype``.

    When ``emit_int8_next=True``, the Triton kernel runs with
    ``EMIT_INT8_NEXT=True``, fusing a per-token absmax→int8 quantization into
    the store epilogue (issue #26). Returns ``(out_int8 [M, N] int8,
    out_scale [M, 1] fp32)`` ready to feed the next W8A8 GEMM without an
    intermediate fp16 HBM round-trip. The CK/hipBLASLt fast paths are skipped
    in this mode — they emit fp16 only.
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
    # CK / hipBLASLt fast paths emit fp16 only; bypass them when the caller
    # asks for the fused int8-emit epilogue (M2 issue #26). The Triton
    # kernel below is the sole owner of EMIT_INT8_NEXT=True.
    if not emit_int8_next:
        tp_rank = _get_tp_rank()
        backend = choose_backend(M, N, K, tp_rank)
        if backend == "ck":
            ck_supports = getattr(
                getattr(torch.ops, "_rocm_C", None),
                "ck_int8_gemm_supports",
                None,
            )
            if ck_supports is not None and bool(ck_supports(M, N, K, tp_rank)):
                return _ck_int8_dispatch(
                    input, weight, scale_a, scale_b, out_dtype, bias
                )

        # M2 dispatch: prefer hipBLASLt for tuned shapes; otherwise fall
        # through to the existing Triton kernel below.
        # VLLM_DISABLE_HIPBLASLT=1 forces 100% Triton dispatch (validated by
        # VAL-TENSILE-008).
        if mi100_hipblaslt_supports(M, N, K):
            return mi100_hipblaslt_scaled_mm(
                input, weight, scale_a, scale_b, out_dtype, bias
            )

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b
    # Per-channel weight scales sometimes arrive as ``[1, N]`` (the call
    # site that mirrors ``torch.matmul`` broadcasting); normalize to the
    # ``[N, 1]`` layout the kernel expects.
    if scale_b.dim() == 2 and scale_b.shape[0] == 1 and scale_b.shape[1] == N:
        scale_b = scale_b.reshape(N, 1).contiguous()

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)
    if emit_int8_next:
        out_int8 = torch.empty((M, N), dtype=torch.int8, device=input.device)
        out_scale = torch.empty((M, 1), dtype=torch.float32, device=input.device)
    else:
        # Tensors must be valid Triton args even when the constexpr branch
        # is dead-code. Reuse ``result`` so the kernel sees a real buffer
        # — the EMIT_INT8_NEXT=False branch never dereferences these
        # pointers.
        out_int8 = result
        out_scale = result

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    # MI100-tuned tile sizes for gfx908 (120 CUs, wavefront-64, 64KB LDS)
    # INT8 elements are 1 byte each, so we can fit larger tiles in LDS.
    # M3: prefer per-shape autotune-selected configs from
    # vllm/model_executor/kernels/configs/gfx908/mi100_int8_M*_N*_K*.json
    # falling back to the static heuristic below for shapes that have not
    # been autotuned yet.
    cfg = _load_mi100_autotune_config(
        "mi100_int8", M=M, N=N, K=K, emit_int8_next=emit_int8_next
    )
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

    # --- M2 wire-in correctness guard ---------------------------------------
    # Discovered during m2-epilogue-fusion (commit a64742222) +
    # m2-numerics-extension (commit 3f397bd95) + this m2-wire-in fix:
    # the EMIT_INT8_NEXT store epilogue reduces the per-token absmax over a
    # single BLOCK_SIZE_N tile rather than the full row ``N``. Only
    # ``pid_n == 0`` writes ``out_scale[m]``. If a caller fires
    # EMIT_INT8_NEXT=True with ``BLOCK_SIZE_N < N``, the int8 rows from
    # ``pid_n > 0`` carry their own tile's absmax but the stored scale is
    # tile-0's — so ``out_int8 * out_scale`` only reconstructs tile-0 values
    # and silently corrupts tile-1+ outputs. The default heuristic above
    # picks BLOCK_SIZE_N ∈ {64, 128} which span NONE of Qwen3.5-9B's N
    # values {3584, 5120, 12544, 18944} in a single tile.
    #
    # The fix lives in the Python wrapper (NOT in the dispatcher's
    # whitelist) so EMIT_INT8_NEXT=True is impossible to misuse from any
    # caller — the dispatcher, the wrapper-level smoke, and the
    # m2-numerics-extension tests all benefit. We force BLOCK_SIZE_N to
    # ``next_power_of_2(N)`` and assert ``block_size_n >= N`` as
    # defense-in-depth.
    if emit_int8_next:
        # Clamp at 32768: forcing BLOCK_SIZE_N > 32768 would push the
        # per-tile numel past Triton's 2**20 cap on realistic BLOCK_SIZE_M
        # values. Qwen3.5-9B's largest N is 18944, well under this cap.
        _EMIT_INT8_NEXT_BLOCK_N_CAP = 32768
        forced_block_n = triton.next_power_of_2(int(N))
        assert forced_block_n <= _EMIT_INT8_NEXT_BLOCK_N_CAP, (
            f"EMIT_INT8_NEXT requires next_power_of_2(N) "
            f"<= {_EMIT_INT8_NEXT_BLOCK_N_CAP}, got N={N} "
            f"-> {forced_block_n}; route to emit_int8_next=False at dispatch."
        )
        block_size_n = forced_block_n
        # gfx908 has 64 KiB LDS per CU; the kernel needs LDS for the
        # int8 input tile (BLOCK_M × BLOCK_K) + the int8 weight tile
        # (BLOCK_K × BLOCK_N) per pipeline stage, plus a fp32 accumulator
        # spill for the EMIT_INT8_NEXT row-absmax reduction. With the
        # default heuristic above, BLOCK_M=64..128 paired with the forced
        # BLOCK_N≥256 quickly overflows the 64 KiB LDS budget. Cap
        # BLOCK_M / BLOCK_K so the (BLOCK_M+BLOCK_N) * BLOCK_K * 1B tile
        # stays under ~48 KiB, leaving headroom for the int32 accumulator
        # spill and the row-absmax reduction.
        # Rule of thumb that matches the m2-numerics-extension tests
        # (which work for the full Qwen3.5-9B shape grid at small
        # BLOCK_N): if forced BLOCK_N > 256, halve BLOCK_M to 32 and
        # BLOCK_K to 64. This keeps LDS comfortably within the 64 KiB
        # limit while still mapping to MFMA tile sizes (16×16, 32×32 on
        # gfx908).
        # When a pinned EMIT_INT8_NEXT=True config has been loaded for
        # this shape (mi100_int8_M*_N*_K*_e1.json), the autotuned
        # BLOCK_M / BLOCK_K already encode the LDS-budget-safe choice
        # and the default halving heuristic below would only push the
        # tile farther under the limit (or, worse, force a smaller-than-
        # autotuned tile that the autotune sweep already disqualified).
        # Skip the halving clamp in that case so the pinned config wins.
        if cfg is None and block_size_n > 256:
            block_size_m = min(block_size_m, 32)
            block_size_k = min(block_size_k, 64)
        # Per-channel weight-scale broadcast tile must match the forced
        # BLOCK_SIZE_N so the masked load still covers the column range.
        block_size_sb = 1 if has_scalar(scale_b) else block_size_n
        # Per-token activation-scale broadcast tile follows the (possibly
        # reduced) BLOCK_SIZE_M.
        block_size_sa = 1 if has_scalar(scale_a) else block_size_m
        # Re-derive the L2-swizzle group factor for the new BLOCK_SIZE_N
        # so the launch metadata stays internally consistent. ``cdiv``
        # returns >= 1 for any positive N, so the ``num_pid_n_h > 0``
        # branch is the production path; the ternary fallback to ``1``
        # keeps the expression exhaustive on the pathological ``N == 0``
        # case (already caught by the early ``assert M > 0 and N > 0
        # and K > 0`` above).
        num_pid_n_h = triton.cdiv(N, block_size_n)
        group_size_m = max(1, min(8, 120 // num_pid_n_h)) if num_pid_n_h > 0 else 1
        assert block_size_n >= N, (
            f"EMIT_INT8_NEXT requires BLOCK_SIZE_N >= N, got {block_size_n} < {N}"
        )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # When emit_int8_next is True the int8 store uses ``stride_cm`` /
    # ``stride_cn`` against ``out_int8_ptr``; both ``result`` and
    # ``out_int8`` are freshly-allocated [M, N] row-major contiguous so
    # their element strides agree. We deliberately pass the int8 strides
    # in that mode to keep the kernel's stride math element-correct for
    # the active store target.
    if emit_int8_next:
        stride_cm = out_int8.stride(0)
        stride_cn = out_int8.stride(1)
    else:
        stride_cm = result.stride(0)
        stride_cn = result.stride(1)

    mi100_int8_scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        out_int8,
        out_scale,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        GROUP_SIZE_M=group_size_m,
        EMIT_INT8_NEXT=emit_int8_next,
        **extra_launch,
    )

    if emit_int8_next:
        return out_int8, out_scale
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

        # M2 dispatcher wire-in (issue #26): probe ``choose_emit_int8_next``
        # once per layer so the engine log records the per-layer fusion
        # decision (mission step 5 of m2-wire-in). The dispatcher already
        # logs every call at ``DEBUG`` level; we additionally raise the
        # first per-layer firing to INFO so the launcher's default log
        # level captures one grep-friendly line per layer for the M2
        # smoke evidence (mission step 8). The decision is informational
        # only — no consumer plumbing exists yet for the int8/scale
        # outputs (a future feature stitches that across the MLP /
        # attention boundary), so the actual ``mi100_int8_scaled_mm``
        # call below stays on the byte-identical fp16-emit path.
        if not getattr(layer, "_mi100_emit_int8_next_logged", False):
            M_dim = int(x.shape[0]) if x.dim() >= 1 else 1
            N_dim = int(w_q.shape[1])
            K_dim = int(w_q.shape[0])
            layer_name = getattr(layer, "prefix", None) or type(layer).__name__
            decision = choose_emit_int8_next(M_dim, N_dim, K_dim, layer_name=layer_name)
            logger.info(
                "[MI100_INT8] emit_int8_next=%s dispatch on layer=%s "
                "shape=(M=%d,N=%d,K=%d)",
                decision,
                layer_name,
                M_dim,
                N_dim,
                K_dim,
            )
            layer._mi100_emit_int8_next_logged = True

        # M2 consumer wire-in (issue #26): the upstream W8A8 producer
        # (e.g. ``qkv_proj`` for the attn block) may have stashed the
        # EMIT_INT8_NEXT kernel's ``(int8 [M, N], fp32 [M, 1])`` output
        # on this consuming layer via the ``_mi100_fused_mm_cache``
        # attribute. The cache shape matches ``ops.scaled_int8_quant``
        # semantics, so it flows straight into the dispatcher's W8A8
        # GEMM call below — no intermediate fp16 HBM round-trip, no
        # redundant per-token requantization. The producer is the
        # single point that gates on
        # ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT``; the consumer
        # additionally re-checks here as defense-in-depth in case the
        # env flips between producer-stash and consumer-consume (e.g.
        # an `os.environ` mutation between forwards). Per anti-pattern
        # #14 the cache is single-shot: ``del`` it immediately so a
        # stale stash from a prior forward (whose M dimension differs
        # from the current batch) cannot leak into the next call. This
        # branch runs BEFORE the M1 silu-fusion cache check so that —
        # although the two caches correspond to different layer pairs
        # (qkv→o_proj for attn vs gate_up→down_proj for MLP) and are
        # mutually exclusive in practice — the M2 cache always takes
        # priority if both happen to be present on the same layer.
        mm_cache = getattr(layer, "_mi100_fused_mm_cache", None)
        if mm_cache is not None and not is_fused_silu_quant_int8_disabled():
            x_q, x_s = mm_cache
            del layer._mi100_fused_mm_cache  # one-shot — anti-pattern #14
            assert x_q.dtype == torch.int8, (
                f"_mi100_fused_mm_cache must carry int8 activations; got {x_q.dtype}."
            )
            return self._mi100_dispatch_scaled_mm(
                layer, x_q, w_q, x_s, w_s, x.dtype, bias
            )

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
            return self._mi100_dispatch_scaled_mm(
                layer, x_q, w_q, x_s, w_s, x.dtype, bias
            )

        # Quantize activations to INT8 (dynamic per-token or static)
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=True
        )

        assert x_zp is None, "MI100 INT8 kernel only supports symmetric quantization"

        return self._mi100_dispatch_scaled_mm(layer, x_q, w_q, x_s, w_s, x.dtype, bias)

    def _mi100_dispatch_scaled_mm(
        self,
        layer: torch.nn.Module,
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        x_s: torch.Tensor,
        w_s: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Invoke ``mi100_int8_scaled_mm`` with optional M2 producer wire-in.

        When ``choose_emit_int8_next(layer)`` returns True AND the env flag
        is not disabled AND ``layer._mi100_next_w8a8_linear`` points at a
        downstream consumer linear, this branch invokes the kernel with
        ``emit_int8_next=True``, stashes the resulting ``(int8, scale)``
        tuple on ``next._mi100_fused_mm_cache`` for the consumer to pick
        up at its own ``apply_weights`` entry, and returns a fp16
        reconstruction of the GEMM output (``int8 * scale``) so the
        producer's caller — e.g. ``QKVParallelLinear`` whose output is
        split and fed into RoPE — still observes the legacy fp16
        contract.

        Per ADDENDUM (mission feature description): if the EMIT_INT8_NEXT
        kernel raises ``RuntimeError("out of resource: shared memory")``
        for this shape (gfx908 LDS budget overflow at BLOCK_N ≥ 16384),
        we set ``layer._mi100_emit_int8_next_lds_disabled = True`` so
        every subsequent forward on the same layer bypasses the producer
        branch without paying the exception cost.

        On every negative branch (env-disabled, dispatcher said False,
        no next-linear wired, LDS OOR sticky-disable already set, or any
        other ``RuntimeError`` from the producer kernel) the call falls
        through to the byte-identical legacy ``emit_int8_next=False``
        path with NO cache stashed.
        """
        next_linear = getattr(layer, "_mi100_next_w8a8_linear", None)
        producer_enabled = (
            next_linear is not None
            and not is_fused_silu_quant_int8_disabled()
            and not getattr(layer, "_mi100_emit_int8_next_lds_disabled", False)
        )

        if producer_enabled:
            M_dim = int(x_q.shape[0])
            N_dim = int(w_q.shape[1])
            K_dim = int(w_q.shape[0])
            layer_name = getattr(layer, "prefix", None) or type(layer).__name__
            if choose_emit_int8_next(M_dim, N_dim, K_dim, layer_name=layer_name):
                try:
                    int8_out, scale_out = mi100_int8_scaled_mm(
                        x_q,
                        w_q,
                        scale_a=x_s,
                        scale_b=w_s,
                        out_dtype=out_dtype,
                        bias=bias,
                        emit_int8_next=True,
                    )
                except RuntimeError as e:
                    # gfx908 LDS budget: the EMIT_INT8_NEXT kernel forces
                    # BLOCK_SIZE_N = next_power_of_2(N); for N >= 8192 the
                    # forced BLOCK_N >= 16384 + the int8 (BLOCK_M+BLOCK_N)
                    # tile overflows the 64 KiB LDS budget at every legal
                    # BLOCK_K. Per the mission addendum the dispatcher
                    # decision logic is NOT modified; only this call site
                    # catches the runtime exception and persists a
                    # per-layer sticky-disable so subsequent forwards skip
                    # the producer branch outright (no exception cost,
                    # no cache stash).
                    if "out of resource" in str(e) and "shared memory" in str(e):
                        layer._mi100_emit_int8_next_lds_disabled = True
                        logger.info(
                            "[MI100_INT8] emit_int8_next=True disabled on "
                            "layer=%s shape=(M=%d,N=%d,K=%d) — gfx908 LDS "
                            "budget exceeded; falling back to fp16 store "
                            "for this layer.",
                            layer_name,
                            M_dim,
                            N_dim,
                            K_dim,
                        )
                    else:
                        raise
                else:
                    # Stash for the downstream consumer to pick up at its
                    # own apply_weights entry. Per anti-pattern #14 the
                    # cache is single-shot — the consumer MUST ``del`` it
                    # on consume so a stale stash cannot leak across
                    # forward steps (different batch's M dimension).
                    # ``next_linear`` is guaranteed non-None here by the
                    # ``producer_enabled`` guard above; the assert is a
                    # narrowing hint for mypy.
                    assert next_linear is not None
                    next_linear._mi100_fused_mm_cache = (int8_out, scale_out)
                    # Reconstruct fp16 output for the producer's caller.
                    # ``int8_out * scale_out`` is the per-token dequant
                    # inverse of the EMIT_INT8_NEXT epilogue
                    # (``out_int8 = round(result / row_scale)``,
                    # ``scale_out[m] = row_absmax[m] / 127.0``). The
                    # producer's downstream consumer reads from the
                    # cache, NOT from this fp16 reconstruction — this is
                    # only needed for callers that further consume the
                    # producer's own output as fp16 (e.g. QKV → RoPE).
                    # The EMIT_INT8_NEXT kernel epilogue already folded
                    # ``bias`` into ``result`` before quantizing, so the
                    # reconstructed fp16 carries bias too — do NOT add
                    # it again here.
                    return (int8_out.to(torch.float32) * scale_out).to(out_dtype)

        # Legacy fallback: byte-identical to pre-PR-#38 wire-in behavior.
        return mi100_int8_scaled_mm(
            x_q,
            w_q,
            scale_a=x_s,
            scale_b=w_s,
            out_dtype=out_dtype,
            bias=bias,
        )
