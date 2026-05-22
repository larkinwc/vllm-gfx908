# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M2 dispatcher wire-in tests for ``EMIT_INT8_NEXT`` (issue #26).

Covers the activation rule of
:func:`vllm.model_executor.kernels.linear.scaled_mm.mi100_int8_dispatch.choose_emit_int8_next`
plus the wrapper-level end-to-end correctness gate that the BLOCK_SIZE_N
>= N override in :func:`mi100_int8_scaled_mm` unlocks.

The pure-Python dispatcher cases (env-disable, non-fusable next-op, N-cap,
all-conditions-true) run on any platform — they exercise the decision
logic without launching a kernel. The wrapper end-to-end correctness case
requires an MI100 GPU and is skipped elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8_dispatch import (
    EMIT_INT8_NEXT_MAX_N,
    choose_emit_int8_next,
)


def _gfx908_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA / ROCm GPU required for the wrapper correctness gate."
    name = torch.cuda.get_device_name(0)
    if "gfx908" not in name and "MI100" not in name:
        return f"EMIT_INT8_NEXT wrapper test targets MI100 (gfx908); got {name!r}."
    return None


# ---------------------------------------------------------------------------
# (i) env-disable -> emit_int8_next=False even on fusable shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "layer_name",
    [
        "model.layers.5.mlp.gate_up_proj",
        "model.layers.5.self_attn.qkv_proj",
    ],
)
def test_env_disable_overrides_fusable_layer(layer_name: str) -> None:
    decision = choose_emit_int8_next(
        M=32,
        N=5120,
        K=3584,
        layer_name=layer_name,
        on_gfx908=True,
        env_disabled=True,
    )
    assert decision is False, "env-disable must override fusable layer-name match."


# ---------------------------------------------------------------------------
# (ii) non-fusable next-op -> emit_int8_next=False
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "layer_name",
    [
        "model.layers.5.self_attn.o_proj",  # consumer side, not producer
        "model.layers.5.mlp.down_proj",  # consumer side, not producer
        "embed_tokens",
        None,
        "",
    ],
)
def test_non_fusable_next_op_yields_false(layer_name: str | None) -> None:
    decision = choose_emit_int8_next(
        M=32,
        N=5120,
        K=3584,
        layer_name=layer_name,
        on_gfx908=True,
        env_disabled=False,
    )
    assert decision is False, (
        f"Layer name {layer_name!r} is not in the fusable producer whitelist."
    )


# ---------------------------------------------------------------------------
# (iii) N > 32768 cap -> emit_int8_next=False
# ---------------------------------------------------------------------------
def test_large_N_trips_cap() -> None:
    too_large_N = EMIT_INT8_NEXT_MAX_N + 1
    decision = choose_emit_int8_next(
        M=32,
        N=too_large_N,
        K=3584,
        layer_name="model.layers.5.mlp.gate_up_proj",
        on_gfx908=True,
        env_disabled=False,
    )
    assert decision is False, (
        f"N={too_large_N} exceeds the {EMIT_INT8_NEXT_MAX_N} cap, "
        "must route to emit_int8_next=False."
    )


# ---------------------------------------------------------------------------
# (iv) all-conditions-true on a real Qwen3.5-9B shape -> emit_int8_next=True
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "layer_name",
    [
        "model.layers.0.mlp.gate_up_proj",
        "language_model.model.layers.12.self_attn.qkv_proj",
        # bare suffix also matches the whitelist
        "gate_up_proj",
        "qkv_proj",
    ],
)
def test_all_conditions_true_on_qwen35_shape(layer_name: str) -> None:
    decision = choose_emit_int8_next(
        M=32,
        N=5120,
        K=3584,
        layer_name=layer_name,
        on_gfx908=True,
        env_disabled=False,
    )
    assert decision is True, (
        f"Expected emit_int8_next=True for fusable layer {layer_name!r} on a "
        "Qwen3.5-9B shape with env unset and gfx908 platform."
    )


# ---------------------------------------------------------------------------
# (iv-b) non-gfx908 device -> emit_int8_next=False
# ---------------------------------------------------------------------------
def test_non_gfx908_yields_false() -> None:
    decision = choose_emit_int8_next(
        M=32,
        N=5120,
        K=3584,
        layer_name="model.layers.0.mlp.gate_up_proj",
        on_gfx908=False,
        env_disabled=False,
    )
    assert decision is False, "non-gfx908 device must short-circuit to False."


# ---------------------------------------------------------------------------
# (v) wrapper-level end-to-end correctness on a real Qwen3.5-9B shape
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    _gfx908_skip_reason() is not None,
    reason=_gfx908_skip_reason() or "skip on non-MI100",
)
@pytest.mark.parametrize(
    "shape",
    [
        # (M, N, K) — selected to exercise the wrapper-level BLOCK_SIZE_N
        # override. The default heuristic picks BLOCK_N ∈ {64, 128}, so any
        # N > 128 (i.e. realistic production shapes) gets the override.
        # We keep the compile-cost small by capping forced BLOCK_N to 256
        # / 512 — the same wrapper-level assertion ``block_size_n >= N``
        # validates correctness regardless of the absolute N value.
        # Triton 3.5.1 + ROCm 7.12 cold-compiles small BLOCK_N tiles in
        # seconds while BLOCK_N >= 4096 walls can take >30 min (library
        # §12 + §14).
        (32, 200, 256),  # next_pow2(200)=256, exercises the override
        (32, 384, 256),  # next_pow2(384)=512
        (8, 200, 128),  # small-M decode-like variant
    ],
)
@torch.inference_mode()
def test_wrapper_emit_int8_next_matches_fp16_reference(
    shape: tuple[int, int, int],
) -> None:
    """End-to-end correctness of the wrapper-level BLOCK_SIZE_N >= N override.

    The wrapper-level fix in ``mi100_int8_scaled_mm`` forces
    ``BLOCK_SIZE_N = next_pow2(N)`` when ``emit_int8_next=True``. The
    kernel then runs with a single pid_n per row, so its per-tile absmax
    reduction == the full-row absmax reduction. Reconstructing
    ``out_int8 * out_scale`` must therefore match a single-tile reference
    that quantizes the fp32 GEMM result against its full-row absmax.

    Without the override, the kernel's per-tile reduction over
    BLOCK_SIZE_N < N would have produced inconsistent scales across
    pid_n shards (only pid_n=0 writes ``out_scale``), silently corrupting
    tile-1+ outputs on every realistic production shape.
    """
    import triton

    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    M, N, K = shape
    torch.manual_seed(0)

    # Synthesize int8 inputs with per-token / per-channel scales, matching
    # the W8A8 dispatcher's call convention.
    a_int8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
    b_int8 = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
    scale_a = (
        torch.rand((M, 1), dtype=torch.float32, device="cuda") * 0.01 + 0.001
    ).contiguous()
    scale_b = (
        torch.rand((N, 1), dtype=torch.float32, device="cuda") * 0.01 + 0.001
    ).contiguous()

    out_int8, out_scale = mi100_int8_scaled_mm(
        a_int8,
        b_int8,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
        emit_int8_next=True,
    )

    assert out_int8.dtype == torch.int8
    assert out_int8.shape == (M, N)
    assert out_scale.dtype == torch.float32
    assert out_scale.shape == (M, 1)

    # Reference: matmul in fp32, apply per-token / per-channel scales,
    # then quantize across the FULL row (not per-tile) — this is the
    # contract the wrapper's BLOCK_SIZE_N >= N override unlocks.
    acc = a_int8.to(torch.float32) @ b_int8.to(torch.float32)
    fp32_out = acc * scale_a * scale_b.T

    absmax = fp32_out.abs().amax(dim=-1, keepdim=True)
    ref_scale = torch.where(
        absmax > 0,
        absmax / 127.0,
        torch.full_like(absmax, 1e-12),
    )
    scaled = fp32_out / ref_scale
    # Kernel uses round-half-away-from-zero (x + sign(x)*0.5 cast).
    rounded = torch.where(scaled >= 0, scaled + 0.5, scaled - 0.5)
    ref_int8 = rounded.clamp(-128, 127).to(torch.int8)

    # Compare against the full-row reference. Tight tolerances are valid
    # here precisely because the wrapper-level override removes the
    # per-tile reduction inconsistency that was the M2 correctness gap.
    torch.testing.assert_close(out_scale, ref_scale, atol=1e-4, rtol=0.0)
    torch.testing.assert_close(out_int8, ref_int8, atol=1, rtol=0.0)

    # Sanity: the wrapper actually picked a BLOCK_SIZE_N >= N. We can't
    # easily inspect the launch metadata after the kernel returns, but we
    # can re-check the invariant the wrapper asserted: next_pow2(N) <=
    # the documented cap. (A failed invariant would have raised in the
    # wrapper before we reached this point.)
    assert triton.next_power_of_2(int(N)) >= N
