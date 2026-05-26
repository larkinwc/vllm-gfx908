# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M2 producer/consumer cache unit test (issue #26, VAL-M2-003).

Covers the producer-side wire-in landed in
:func:`MI100Int8ScaledMMLinearKernel.apply_weights` /
:func:`MI100Int8ScaledMMLinearKernel._mi100_dispatch_scaled_mm` plus the
consumer-side cache consumption that mirrors the existing M1
``_mi100_fused_silu_cache`` path. Two test functions:

    * ``test_qkv_to_o_proj_cache_hit`` — exercises the M2
      ``_mi100_fused_mm_cache`` attribute populated by the producer wire-in:
      a fusable ``qkv_proj``-named producer layer stashes ``(int8, scale)`` on
      the downstream ``o_proj`` consumer, and the consumer's next
      ``apply_weights`` invocation picks the cache up, deletes it (anti-pattern
      #14: single-shot), and skips :func:`scaled_int8_quant` entirely.

    * ``test_gate_up_to_down_proj_cache_hit`` — exercises the existing M1
      ``_mi100_fused_silu_cache`` consumer path: pre-stash the silu+quant cache
      on a ``down_proj``-named consumer and assert the next ``apply_weights``
      call consumes it without calling :func:`scaled_int8_quant`.

CHOSEN APPROACH (per ADDENDUM in the feature description)
---------------------------------------------------------
The simplest path is to construct hand-rolled producer + consumer layers as
plain :class:`torch.nn.Module` instances carrying the same attributes the
production code keys off (``weight``, ``weight_scale``, ``input_scale``,
``input_zero_point``, ``azp_adj``, ``logical_widths``, ``prefix``,
``_mi100_next_w8a8_linear``). The production
:func:`MI100Int8ScaledMMLinearKernel.apply_weights` reads these attributes
directly — no monkeypatch on the kernel is required. This bypasses the heavy
Qwen2 model-loading path while still exercising the **real** producer/consumer
cache code path in ``mi100_int8.py`` end-to-end:

    producer.apply_weights(x_fp16)
        → choose_emit_int8_next(...)  -> True (qkv_proj on gfx908, env unset)
        → mi100_int8_scaled_mm(..., emit_int8_next=True)
        → consumer._mi100_fused_mm_cache = (int8_out, scale_out)

    consumer.apply_weights(x_fp16_placeholder)
        → reads layer._mi100_fused_mm_cache
        → del layer._mi100_fused_mm_cache  (anti-pattern #14)
        → skips scaled_int8_quant entirely
        → dispatches mi100_int8_scaled_mm on the cached (int8, scale)

Shape choice
------------
Per the ADDENDUM, Qwen3.5-9B's qkv_proj N=10240 overflows the gfx908 64 KiB LDS
budget in EMIT_INT8_NEXT mode (the wire-in correctly falls back via the sticky
``_mi100_emit_int8_next_lds_disabled`` flag, but that means the producer-stash
branch is NOT exercised on that shape). The unit test therefore uses a small
N <= 1024 with K=256: ``next_pow2(N=512) = 512`` → BLOCK_N=512, and BLOCK_K=64
keeps the int8 tile well under 64 KiB. This compiles in seconds on Triton
3.5.1 / ROCm 7.12 and exercises the producer-stash branch (NOT the
LDS-disable fallback).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from vllm._custom_ops import scaled_int8_quant as _real_scaled_int8_quant
from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
    MI100Int8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig,
)


def _gfx908_skip_reason() -> str | None:
    """Return a skip reason string when this host is not an MI100 (gfx908)."""
    if not torch.cuda.is_available():
        return "CUDA / ROCm GPU required for the producer/consumer cache test."
    name = torch.cuda.get_device_name(0)
    if "gfx908" not in name and "MI100" not in name:
        return f"producer/consumer cache test targets MI100 (gfx908); got {name!r}."
    return None


pytestmark = pytest.mark.skipif(
    _gfx908_skip_reason() is not None,
    reason=_gfx908_skip_reason() or "skip on non-MI100",
)


# Test-only shape. See module docstring "Shape choice" for the rationale.
# These values are intentionally small so that the EMIT_INT8_NEXT producer
# kernel compiles in seconds AND fits comfortably under the gfx908 64 KiB
# LDS budget (BLOCK_N forced to next_pow2(N)=512; BLOCK_K capped at 64).
_M = 16
_N = 512
_K = 256


def _make_kernel() -> MI100Int8ScaledMMLinearKernel:
    """Instantiate the production kernel under the same config wiring vLLM
    uses for a per-token / per-channel symmetric W8A8 INT8 layer."""
    cfg = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    return MI100Int8ScaledMMLinearKernel(
        cfg,
        layer_param_names=[
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ],
    )


def _make_layer(prefix: str, N: int, K: int) -> torch.nn.Module:
    """Build a hand-rolled torch.nn.Module that carries the attributes the
    MI100 kernel's apply_weights reads off ``layer``.

    The weight is stored in the post-``process_weights_after_loading`` layout
    (``[K, N]`` int8 row-major) so we can skip the production
    ``process_weights_after_loading`` path entirely — the test only needs the
    forward to see a kernel-shaped tensor on the layer.
    """
    layer = torch.nn.Module()
    weight = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
    weight_scale = torch.rand((N, 1), dtype=torch.float32, device="cuda") * 0.01 + 0.001
    layer.weight = weight  # type: ignore[assignment]
    layer.weight_scale = weight_scale  # type: ignore[assignment]
    layer.input_scale = None  # type: ignore[assignment]
    layer.input_zero_point = None  # type: ignore[assignment]
    layer.azp_adj = None  # type: ignore[assignment]
    # logical_widths is read by process_weights_after_loading; not exercised
    # here but present so any defensive getattr() lookup downstream returns a
    # sensible default.
    layer.logical_widths = [N]  # type: ignore[assignment]
    layer.prefix = prefix
    return layer


@pytest.fixture
def kernel() -> MI100Int8ScaledMMLinearKernel:
    return _make_kernel()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts from the default (fused-on) env state."""
    monkeypatch.delenv("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", raising=False)
    monkeypatch.delenv("VLLM_DISABLE_FUSED_ACT_QUANT", raising=False)


@torch.inference_mode()
def test_qkv_to_o_proj_cache_hit(
    kernel: MI100Int8ScaledMMLinearKernel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2 producer/consumer cache (qkv_proj → o_proj).

    Exercises three assertions:
      1. The producer's apply_weights stashes ``_mi100_fused_mm_cache`` on the
         consumer (o_proj) and the consumer's next apply_weights call consumes
         it without calling ``scaled_int8_quant``.
      2. The cache attribute is absent on the consumer after consume
         (anti-pattern #14: one-shot delete).
      3. With ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`` the producer does NOT
         stash on the consumer, and a subsequent consumer call falls through
         to the legacy ``scaled_int8_quant`` path.
    """
    producer = _make_layer("model.layers.0.self_attn.qkv_proj", N=_N, K=_K)
    consumer = _make_layer("model.layers.0.self_attn.o_proj", N=_N, K=_K)
    # Producer/consumer wire — set by the model-graph hook in production.
    producer._mi100_next_w8a8_linear = consumer  # type: ignore[assignment]

    x_fp16 = torch.randn((_M, _K), dtype=torch.float16, device="cuda")

    # --- (1) Producer stashes on consumer ---------------------------------
    _ = kernel.apply_weights(producer, x_fp16)
    assert getattr(consumer, "_mi100_fused_mm_cache", None) is not None, (
        "Producer apply_weights did not stash _mi100_fused_mm_cache on "
        "consumer; check choose_emit_int8_next() decision and "
        "_mi100_next_w8a8_linear wiring."
    )
    cached_int8, cached_scale = consumer._mi100_fused_mm_cache
    assert cached_int8.dtype == torch.int8
    assert cached_int8.shape == (_M, _N)
    assert cached_scale.dtype == torch.float32
    assert cached_scale.shape == (_M, 1)

    # --- (2) Consumer consumes cache without calling scaled_int8_quant ----
    # ``o_proj`` consumes the [M, N] producer output. The consumer's K equals
    # the producer's N here (the per-layer K dim of o_proj). Build a fresh
    # o_proj layer with the matching K=_N so apply_weights sees a valid shape.
    o_proj = _make_layer("model.layers.0.self_attn.o_proj", N=_K, K=_N)
    # Re-stash the cache on this fresh layer with the right K.
    o_proj._mi100_fused_mm_cache = (cached_int8, cached_scale)
    x_consumer_placeholder = torch.randn((_M, _N), dtype=torch.float16, device="cuda")

    quant_call_count = {"n": 0}

    def _spy(*args, **kwargs):
        quant_call_count["n"] += 1
        return _real_scaled_int8_quant(*args, **kwargs)

    with patch(
        "vllm.model_executor.kernels.linear.scaled_mm.mi100_int8.ops.scaled_int8_quant",
        side_effect=_spy,
    ):
        out = kernel.apply_weights(o_proj, x_consumer_placeholder)

    assert quant_call_count["n"] == 0, (
        "Consumer fell through to scaled_int8_quant even though "
        "_mi100_fused_mm_cache was populated — the M2 consumer wire-in did "
        "not consume the cache."
    )
    assert getattr(o_proj, "_mi100_fused_mm_cache", None) is None, (
        "Consumer did not delete _mi100_fused_mm_cache after consume; "
        "violates anti-pattern #14 (single-shot)."
    )
    assert out.shape == (_M, _K)

    # --- (3) Env-disabled: producer does NOT stash; consumer falls through ---
    monkeypatch.setenv("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", "1")

    producer_disabled = _make_layer("model.layers.1.self_attn.qkv_proj", N=_N, K=_K)
    consumer_disabled = _make_layer("model.layers.1.self_attn.o_proj", N=_K, K=_N)
    producer_disabled._mi100_next_w8a8_linear = consumer_disabled

    _ = kernel.apply_weights(producer_disabled, x_fp16)
    assert getattr(consumer_disabled, "_mi100_fused_mm_cache", None) is None, (
        "Producer stashed _mi100_fused_mm_cache despite "
        "VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1 — producer wire-in must honor "
        "the env flag."
    )

    quant_call_count["n"] = 0
    with patch(
        "vllm.model_executor.kernels.linear.scaled_mm.mi100_int8.ops.scaled_int8_quant",
        side_effect=_spy,
    ):
        _ = kernel.apply_weights(consumer_disabled, x_consumer_placeholder)
    assert quant_call_count["n"] == 1, (
        "Consumer did NOT call scaled_int8_quant on env-disabled path with no "
        "cache present; expected legacy fallback to take exactly one quant "
        f"call, got {quant_call_count['n']}."
    )


@torch.inference_mode()
def test_gate_up_to_down_proj_cache_hit(
    kernel: MI100Int8ScaledMMLinearKernel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1 silu-cache consumer path (gate_up_proj → down_proj).

    The M1 ``_mi100_fused_silu_cache`` is populated by the upstream MLP
    forward (``try_stash_fused_silu_quant_int8`` in
    ``fused_silu_quant_int8.py``), NOT by the kernel's own apply_weights —
    that helper runs between ``gate_up_proj`` and ``down_proj``. The
    consumer-side path under test here is the branch in
    ``MI100Int8ScaledMMLinearKernel.apply_weights`` that reads
    ``layer._mi100_fused_silu_cache``, clears it (sets to ``None`` per the
    M1 contract), and dispatches the GEMM on the cached ``(int8, scale)``.

    Three assertions:
      1. With the cache present, the consumer's apply_weights consumes it
         without calling ``scaled_int8_quant``.
      2. ``_mi100_fused_silu_cache`` is cleared to ``None`` after consume
         (M1 contract — distinct from the M2 ``del`` semantics).
      3. With ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1``, the upstream producer
         helper ``try_stash_fused_silu_quant_int8`` does NOT stash, and a
         consumer call with no cache falls through to ``scaled_int8_quant``.
    """
    from vllm.model_executor.kernels.quantization.fused_silu_quant_int8 import (
        try_stash_fused_silu_quant_int8,
    )

    down_proj = _make_layer("model.layers.0.mlp.down_proj", N=_N, K=_K)

    # Synthesize the (int8, scale) cache the M1 producer would have stashed.
    cached_int8 = torch.randint(-127, 128, (_M, _K), dtype=torch.int8, device="cuda")
    cached_scale = (
        torch.rand((_M, 1), dtype=torch.float32, device="cuda") * 0.01 + 0.001
    )
    down_proj._mi100_fused_silu_cache = (cached_int8, cached_scale)

    x_placeholder = torch.randn((_M, _K), dtype=torch.float16, device="cuda")

    quant_call_count = {"n": 0}

    def _spy(*args, **kwargs):
        quant_call_count["n"] += 1
        return _real_scaled_int8_quant(*args, **kwargs)

    # --- (1) Consumer consumes silu cache without scaled_int8_quant -------
    with patch(
        "vllm.model_executor.kernels.linear.scaled_mm.mi100_int8.ops.scaled_int8_quant",
        side_effect=_spy,
    ):
        out = kernel.apply_weights(down_proj, x_placeholder)

    assert quant_call_count["n"] == 0, (
        "down_proj consumer fell through to scaled_int8_quant even though "
        "_mi100_fused_silu_cache was populated — the M1 consumer wire-in did "
        "not consume the cache."
    )
    assert out.shape == (_M, _N)

    # --- (2) Cache cleared after consume (M1 contract: set to None) -------
    assert down_proj._mi100_fused_silu_cache is None, (
        "Consumer did not clear _mi100_fused_silu_cache after consume; "
        "stale stash would leak into the next forward (different batch's M)."
    )

    # --- (3) Env-disabled: M1 producer helper does NOT stash --------------
    monkeypatch.setenv("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", "1")

    down_proj_disabled = _make_layer("model.layers.1.mlp.down_proj", N=_N, K=_K)

    # Build a plausible gate_up fp16 [M, 2*K] tensor and call the M1 producer
    # helper. With env=1, it MUST short-circuit before stashing.
    gate_up_fp16 = torch.randn((_M, 2 * _K), dtype=torch.float16, device="cuda")
    # Make _down_proj_is_mi100_w8a8_int8 return True so the only remaining
    # negative branch is the env flag.
    fake_scheme = type("FakeScheme", (), {"kernel": kernel})()
    down_proj_disabled.scheme = fake_scheme  # type: ignore[assignment]

    stashed = try_stash_fused_silu_quant_int8(gate_up_fp16, down_proj_disabled)
    assert stashed is False, (
        "try_stash_fused_silu_quant_int8 returned True despite "
        "VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1 — M1 producer must honor "
        "the env flag."
    )
    assert getattr(down_proj_disabled, "_mi100_fused_silu_cache", None) is None

    # And a subsequent consumer call with no cache falls through to the
    # legacy quant path.
    quant_call_count["n"] = 0
    with patch(
        "vllm.model_executor.kernels.linear.scaled_mm.mi100_int8.ops.scaled_int8_quant",
        side_effect=_spy,
    ):
        _ = kernel.apply_weights(down_proj_disabled, x_placeholder)
    assert quant_call_count["n"] == 1, (
        "down_proj consumer did NOT call scaled_int8_quant on env-disabled "
        "path with no cache present; expected legacy fallback to take "
        f"exactly one quant call, got {quant_call_count['n']}."
    )
