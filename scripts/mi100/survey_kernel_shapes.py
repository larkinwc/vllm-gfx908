#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5-9B w8a8 shape survey for the three MI100 fused kernels.

Records every ``(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)``
tuple invoked on the three fused Triton kernels during a prefill + 1 decode
step of a short Qwen3.5-9B w8a8 generation:

* ``mi100_int8_scaled_mm_kernel`` — W8A8 INT8 scaled GEMM (both
  ``EMIT_INT8_NEXT=False`` and ``=True`` variants).
* ``_fused_silu_quant_int8_kernel`` — fused SiLU-and-mul + INT8 quant.
* ``_fused_int8_quant_kernel`` — fused per-token INT8 quant.

The recorder is implemented entirely in the *Python launcher wrappers* —
i.e. it monkey-patches ``triton.runtime.JITFunction.add_pre_run_hook`` on
each kernel object. The ``@triton.jit`` kernel bodies are NOT touched
(mission constraint).

To exercise the ``EMIT_INT8_NEXT=True`` variant of
``mi100_int8_scaled_mm_kernel`` *without* landing the full M2 producer-
side wire-in (which is a separate feature, ``m2-producer-wire-in``), this
survey:

1. Tags one representative ``qkv_proj`` layer with
   ``_mi100_next_w8a8_linear = <its o_proj sibling>``. This attribute is
   inert in the current code (the consumer wire-in does not yet honor
   it), but documents the producer→consumer pairing the survey is
   intentionally exercising.
2. Manually invokes ``mi100_int8_scaled_mm(..., emit_int8_next=True)``
   once per qkv-shape × representative-M for the prefill + 1-decode M
   values observed during the short generation. The pre-run hook on
   ``mi100_int8_scaled_mm_kernel`` records the True-variant tile
   parameters alongside the False-variant ones.

Output: ``/root/bench-int8-w4a16-m2-producer/m1-shapes/qwen3p5_9b_w8a8_shapes.json``
with three top-level keys (one per kernel), each carrying a ``shapes``
array of dicts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger("survey_kernel_shapes")


# ---------------------------------------------------------------------------
# Per-kernel positional-arg → (M, N, K) extractors.
#
# Each kernel's positional arg list (matching the @triton.jit signature) is
# encoded as a tuple of (index, label) pairs that lets us pull M/N/K out of
# the pre_run_hook's *args without poking the kernel body. K may be None
# when the kernel has no K dim (e.g. per-token quant); the survey records
# K=N (i.e. the hidden dim) in that case so downstream autotune sweeps can
# still index by a single shape-key triple.
# ---------------------------------------------------------------------------

# mi100_int8_scaled_mm_kernel(a_ptr, b_ptr, scale_a_ptr, scale_b_ptr, c_ptr,
#                             bias_ptr, out_int8_ptr, out_scale_ptr,
#                             M, N, K, stride_am, ...)
_MI100_INT8_SCALED_MM_M_IDX = 8
_MI100_INT8_SCALED_MM_N_IDX = 9
_MI100_INT8_SCALED_MM_K_IDX = 10

# _fused_silu_quant_int8_kernel(x_ptr, q_ptr, s_ptr, M, H, stride_xm, ...)
# Treat H as both "N" and "K" for downstream shape-key consistency (the
# fused silu kernel inputs are [M, 2H] and outputs [M, H]; the H dim is
# the only non-M shape parameter the kernel branches on).
_FUSED_SILU_M_IDX = 3
_FUSED_SILU_H_IDX = 4

# _fused_int8_quant_kernel(x_ptr, xq_ptr, scale_ptr, M, N, stride_xm, ...)
_FUSED_QUANT_M_IDX = 3
_FUSED_QUANT_N_IDX = 4


def _to_int(maybe_tensor: Any) -> int | None:
    """Best-effort extraction of a Python int from a kernel positional arg.

    Triton may pass ``int`` directly or wrap shape ints in a 0-d tensor /
    a tl.constexpr. Returns ``None`` if extraction fails so the recorder
    never crashes the forward path.
    """
    try:
        if isinstance(maybe_tensor, int):
            return int(maybe_tensor)
        if hasattr(maybe_tensor, "item"):
            return int(maybe_tensor.item())
        return int(maybe_tensor)
    except Exception:
        return None


# Thread-local storage of (shape, BLOCK params) seen per kernel. Keyed on a
# tuple of all recorded params to dedupe; insertion-ordered so the JSON
# output is stable across runs.
class ShapeRecorder:
    """Thread-safe ordered recorder of ``(shape, block, launch)`` tuples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tables: dict[str, OrderedDict[tuple, dict]] = {}

    def record(self, kernel_name: str, entry: dict) -> None:
        # Dedupe key spans the full tuple (everything the autotune sweep
        # cares about). We canonicalize to ints/strings so the dict is
        # hashable and JSON-friendly.
        key = (
            entry.get("M"),
            entry.get("N"),
            entry.get("K"),
            entry.get("BLOCK_M"),
            entry.get("BLOCK_N"),
            entry.get("BLOCK_K"),
            entry.get("num_warps"),
            entry.get("num_stages"),
            entry.get("EMIT_INT8_NEXT"),
        )
        with self._lock:
            table = self._tables.setdefault(kernel_name, OrderedDict())
            if key not in table:
                table[key] = entry

    def snapshot(self) -> dict[str, list[dict]]:
        with self._lock:
            return {
                kname: list(table.values()) for kname, table in self._tables.items()
            }


_recorder = ShapeRecorder()


def _make_mi100_int8_scaled_mm_hook(kernel_name: str):
    """Pre-run hook for mi100_int8_scaled_mm_kernel.

    Pulls M, N, K from positional args (indices 8/9/10) and the BLOCK_*
    constexprs from kwargs. The constexprs are passed by the Python
    wrapper as ``BLOCK_SIZE_M``, ``BLOCK_SIZE_N``, ``BLOCK_SIZE_K``,
    ``EMIT_INT8_NEXT``; ``num_warps`` and ``num_stages`` are launch
    options also passed via kwargs.
    """

    def _hook(*args, **kwargs):  # noqa: ANN002, ANN003
        try:
            M = _to_int(args[_MI100_INT8_SCALED_MM_M_IDX])
            N = _to_int(args[_MI100_INT8_SCALED_MM_N_IDX])
            K = _to_int(args[_MI100_INT8_SCALED_MM_K_IDX])
            entry: dict[str, Any] = {
                "M": M,
                "N": N,
                "K": K,
                "BLOCK_M": kwargs.get("BLOCK_SIZE_M"),
                "BLOCK_N": kwargs.get("BLOCK_SIZE_N"),
                "BLOCK_K": kwargs.get("BLOCK_SIZE_K"),
                "GROUP_SIZE_M": kwargs.get("GROUP_SIZE_M"),
                "num_warps": kwargs.get("num_warps"),
                "num_stages": kwargs.get("num_stages"),
                "EMIT_INT8_NEXT": bool(kwargs.get("EMIT_INT8_NEXT", False)),
            }
            _recorder.record(kernel_name, entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[survey] %s hook error: %s", kernel_name, exc)

    return _hook


def _make_fused_silu_hook(kernel_name: str):
    """Pre-run hook for _fused_silu_quant_int8_kernel."""

    def _hook(*args, **kwargs):  # noqa: ANN002, ANN003
        try:
            M = _to_int(args[_FUSED_SILU_M_IDX])
            H = _to_int(args[_FUSED_SILU_H_IDX])
            entry: dict[str, Any] = {
                "M": M,
                "N": H,
                "K": 2 * H if H is not None else None,  # input is [M, 2H]
                "BLOCK_M": kwargs.get("BLOCK_M"),
                "BLOCK_N": kwargs.get("BLOCK_H"),
                "BLOCK_K": None,
                "num_warps": kwargs.get("num_warps"),
                "num_stages": kwargs.get("num_stages"),
                "EMIT_INT8_NEXT": None,
            }
            _recorder.record(kernel_name, entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[survey] %s hook error: %s", kernel_name, exc)

    return _hook


def _make_fused_int8_quant_hook(kernel_name: str):
    """Pre-run hook for _fused_int8_quant_kernel."""

    def _hook(*args, **kwargs):  # noqa: ANN002, ANN003
        try:
            M = _to_int(args[_FUSED_QUANT_M_IDX])
            N = _to_int(args[_FUSED_QUANT_N_IDX])
            entry: dict[str, Any] = {
                "M": M,
                "N": N,
                "K": N,  # no separate K dim; mirror N for shape-key parity.
                "BLOCK_M": kwargs.get("BLOCK_M"),
                "BLOCK_N": kwargs.get("BLOCK_N"),
                "BLOCK_K": None,
                "num_warps": kwargs.get("num_warps"),
                "num_stages": kwargs.get("num_stages"),
                "EMIT_INT8_NEXT": None,
            }
            _recorder.record(kernel_name, entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[survey] %s hook error: %s", kernel_name, exc)

    return _hook


def install_hooks() -> None:
    """Attach pre-run hooks to the three fused-kernel JITFunctions.

    Hooks are added via ``JITFunction.add_pre_run_hook`` (triton 3.5.1
    API). Each hook receives the same ``*args, **kwargs`` that the
    Python launcher passed to ``kernel[grid](...)``, so we see both
    positional shape ints and constexpr ``BLOCK_*`` / ``num_warps`` /
    ``num_stages`` kwargs without recompiling anything.
    """
    from vllm.model_executor.kernels.linear.mixed_precision.fused_int8_quant import (
        _fused_int8_quant_kernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm_kernel,
    )
    from vllm.model_executor.kernels.quantization.fused_silu_quant_int8 import (
        _fused_silu_quant_int8_kernel,
    )

    mi100_int8_scaled_mm_kernel.add_pre_run_hook(
        _make_mi100_int8_scaled_mm_hook("mi100_int8_scaled_mm_kernel")
    )
    _fused_silu_quant_int8_kernel.add_pre_run_hook(
        _make_fused_silu_hook("_fused_silu_quant_int8_kernel")
    )
    _fused_int8_quant_kernel.add_pre_run_hook(
        _make_fused_int8_quant_hook("fused_int8_quant")
    )
    logger.info("[survey] installed pre-run hooks on all three fused kernels")


def _resolve_model(llm: Any) -> Any | None:
    """Walk LLM → engine → executor → worker → model_runner → model.

    The exact attribute path varies between V0 and V1 engines and between
    multiprocess / in-process executors. We try a small set of known
    paths and return the first ``nn.Module`` we find.
    """
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    if engine is None:
        return None
    candidate_paths = [
        # V1 in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
        ("model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("model_executor", "driver_worker", "model_runner", "model"),
        # V1 multiprocess (won't work — child has model — but kept for safety)
        (
            "engine_core",
            "engine_core",
            "model_executor",
            "driver_worker",
            "worker",
            "model_runner",
            "model",
        ),
        # V0 fallback
        ("model_executor", "driver_worker", "model"),
    ]
    for path in candidate_paths:
        obj: Any = engine
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _find_qkv_o_pair(llm: Any) -> tuple[Any, Any] | None:
    """Locate one ``(qkv_proj, o_proj)`` Linear pair on a Qwen2 attention layer.

    Returns ``(qkv_proj, o_proj)`` for layer 0 if both are reachable
    through the engine's model runner. Returns ``None`` if the model
    layout does not expose the expected attribute path. Used purely as
    a survey-only fixture for the EMIT_INT8_NEXT=True variant — no
    in-band wiring is performed.
    """
    try:
        model = _resolve_model(llm)
        if model is None:
            logger.warning("[survey] _resolve_model returned None")
            return None
        # Qwen3.5 layout: ``language_model.model.layers[i].self_attn.{qkv,o}_proj``.
        # Qwen2 layout: ``model.layers[i].self_attn.{qkv,o}_proj``.
        # Walk both candidate paths to the layer list.
        layers = None
        for path in (
            ("language_model", "model", "layers"),
            ("model", "layers"),
        ):
            obj: Any = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                layers = obj
                break
        if layers is None:
            logger.warning(
                "[survey] could not locate layers list; model type=%s",
                type(model).__name__,
            )
            return None
        # Find the first layer that exposes both ``self_attn.qkv_proj`` and
        # ``self_attn.o_proj`` — Qwen3.5 alternates SSM ("linear_attn")
        # blocks with attention blocks, so layer 0 is not necessarily an
        # attention layer.
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            qkv = getattr(attn, "qkv_proj", None)
            o = getattr(attn, "o_proj", None)
            if qkv is not None and o is not None:
                return qkv, o
        logger.warning("[survey] no layer carried both qkv_proj and o_proj")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[survey] _find_qkv_o_pair failed: %s", exc)
        return None


def _exercise_fused_int8_quant_fixture(
    hidden_sizes: list[int], m_values: list[int]
) -> int:
    """Directly invoke ``fused_per_token_quant_int8`` for typical Qwen3.5-9B
    activation shapes.

    ``fused_int8_quant`` is a candidate Triton kernel (alternative to the
    C++ ``ops.scaled_int8_quant`` used by the W8A8 wrapper). It is NOT
    invoked on the production hot path today, so the model load + short
    generation never exercises it. To still capture its shape set for the
    M1-F2 autotune sweep we directly invoke the wrapper here as a
    survey-only fixture using the prefill/decode M values observed during
    the generation and the model's two characteristic hidden sizes.
    """
    import torch

    from vllm.model_executor.kernels.linear.mixed_precision.fused_int8_quant import (
        fused_per_token_quant_int8,
    )

    fired = 0
    device = torch.device("cuda:0")
    for M in m_values:
        if M <= 0:
            continue
        for N in hidden_sizes:
            try:
                x = torch.zeros((M, N), dtype=torch.float16, device=device)
                _q, _s = fused_per_token_quant_int8(x)
                fired += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[survey] fused_per_token_quant_int8 fixture failed "
                    "for (M=%d, N=%d): %s",
                    M,
                    N,
                    exc,
                )
    return fired


def _exercise_emit_int8_next_variant(qkv: Any, o_proj: Any, m_values: list[int]) -> int:
    """Manually invoke ``mi100_int8_scaled_mm(..., emit_int8_next=True)``.

    The current ``apply_weights`` code path always passes
    ``emit_int8_next=False`` (the producer-side wire-in is a separate
    feature, ``m2-producer-wire-in``). To make the survey hook see the
    True-variant tile parameters too, we directly invoke the wrapper
    function for the qkv_proj's resolved (K, N) weight shape across a
    representative set of M values seen during the short generation.

    Tags ``qkv_proj._mi100_next_w8a8_linear = o_proj`` for diagnostic
    parity with the future M2 wire-in (the attribute is inert in the
    current code base).
    """
    import torch

    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    # Document the producer→consumer pair the survey is exercising.
    qkv._mi100_next_w8a8_linear = o_proj  # type: ignore[attr-defined]

    # Resolve qkv_proj weight: process_weights_after_loading transposes
    # the weight to [K, N] int8.
    w_q_name = "weight"
    w_s_name = "weight_scale"
    # CompressedTensors W8A8 layers expose Linear.weight as the int8 w_q
    # after process_weights_after_loading; weight_scale is the per-channel
    # fp32 scale.
    w_q = getattr(qkv, w_q_name, None)
    w_s = getattr(qkv, w_s_name, None)
    if w_q is None or w_s is None or w_q.dtype != torch.int8:
        logger.warning(
            "[survey] qkv_proj weight not in expected int8 [K, N] form; "
            "skipping EMIT_INT8_NEXT=True fixture (dtype=%s)",
            None if w_q is None else w_q.dtype,
        )
        return 0
    K, N = int(w_q.shape[0]), int(w_q.shape[1])
    device = w_q.device

    # Per-token scale shape [M, 1] fp32; per-channel scale already [N, 1].
    fired = 0
    for M in m_values:
        if M <= 0:
            continue
        try:
            x = torch.empty((M, K), dtype=torch.int8, device=device)
            x.zero_()  # content irrelevant for shape survey; avoids NaNs.
            sa = torch.full((M, 1), 1e-3, dtype=torch.float32, device=device)
            # Reshape weight_scale to [N, 1] if it arrived as [N] or [1, N].
            w_scale = w_s.reshape(-1, 1) if w_s.dim() == 1 else w_s
            out = mi100_int8_scaled_mm(
                x,
                w_q,
                scale_a=sa,
                scale_b=w_scale,
                out_dtype=torch.float16,
                bias=None,
                emit_int8_next=True,
            )
            assert isinstance(out, tuple), "emit_int8_next=True returns (int8, scale)"
            fired += 1
        except AssertionError:
            # The wrapper guards against dispatch-incompatible (M, N, K)
            # combos by re-routing through the False path. We do not
            # want a single rejected M to abort the survey.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[survey] EMIT_INT8_NEXT=True invocation failed for M=%d "
                "(K=%d, N=%d): %s",
                M,
                K,
                N,
                exc,
            )
    return fired


def run_survey(
    model_path: str,
    out_path: Path,
    prompt: str,
    max_tokens: int,
) -> dict[str, list[dict]]:
    """Load the model, install hooks, run one short generation, snapshot."""
    # Force in-process EngineCore. The default V1 multiprocessing path
    # forks a child engine-core process that imports our kernel modules
    # in its own interpreter, where the parent's pre-run hooks are
    # invisible. Setting ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` keeps the
    # EngineCore in the survey process so the hooks fire on every kernel
    # launch. This is the same toggle ``benchmarks/startup.py`` uses for
    # in-process measurement.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Hooks MUST be installed BEFORE the engine loads, so warmup
    # invocations are captured too.
    install_hooks()

    # Import here so the hooks-installation above already targets the
    # same module instances vLLM will use.
    from vllm import LLM, SamplingParams

    logger.info("[survey] loading model: %s", model_path)
    # ``Qwen3.5-9B-w8a8`` ships with ``Qwen3_5ForConditionalGeneration``
    # in ``config.json::architectures``, which forces vLLM to spin up the
    # multimodal input processor. On the Transformers 4.x pin used in
    # this worktree, the MM image processor fails to construct because
    # the tokenizer is a ``CachedPreTrainedTokenizerFast`` not the
    # expected ``Qwen2Tokenizer``. Setting ``limit_mm_per_prompt`` to
    # zero across every supported modality short-circuits
    # ``MultiModalRegistry.supports_multimodal_inputs`` to ``False`` and
    # skips MM initialization entirely (see registry.py: "All limits of
    # multimodal modalities supported by the model are set to 0, running
    # in text-only mode."). The W8A8 Linear weights load identically;
    # the survey only cares about Linear shapes.
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=2048,
        block_size=32,
        enforce_eager=True,  # skip cudagraph capture to make the survey faster
        gpu_memory_utilization=0.80,
        enable_chunked_prefill=False,
        # Trust the local model directory; it is the same one used by the
        # production launch scripts.
        trust_remote_code=False,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    logger.info(
        "[survey] running short generation: prompt=%r max_tokens=%d", prompt, max_tokens
    )
    outputs = llm.generate([prompt], sampling)
    for out in outputs:
        logger.info(
            "[survey] generation output: %s",
            out.outputs[0].text.replace("\n", " ")[:128],
        )

    # Exercise the EMIT_INT8_NEXT=True variant. We pull representative M
    # values from the False-variant invocations of
    # ``mi100_int8_scaled_mm_kernel`` already recorded during the
    # generation (prefill + 1 decode).
    snap_pre = _recorder.snapshot()
    int8_table = snap_pre.get("mi100_int8_scaled_mm_kernel", [])
    seen_m = sorted({e["M"] for e in int8_table if e.get("M") is not None})
    # Keep one prefill-like M (largest) and one decode-like M (smallest)
    # — that minimum surface still covers both BLOCK_M heuristics. Falls
    # back to [1, 4] if the recorder somehow saw nothing (defensive — the
    # generation above always produces at least the prefill M).
    m_values = sorted({seen_m[0], seen_m[-1]}) if seen_m else [1, 4]

    qkv_o = _find_qkv_o_pair(llm)
    if qkv_o is not None:
        qkv, o = qkv_o
        fired = _exercise_emit_int8_next_variant(qkv, o, m_values)
        logger.info(
            "[survey] EMIT_INT8_NEXT=True fixture fired %d times (m_values=%s)",
            fired,
            m_values,
        )
    else:
        logger.warning(
            "[survey] could not locate qkv_proj/o_proj pair; "
            "EMIT_INT8_NEXT=True variant will be absent from the survey"
        )

    # Fire ``fused_per_token_quant_int8`` directly for the model's
    # characteristic hidden sizes. The kernel is a candidate alternative
    # to ``ops.scaled_int8_quant`` and is not on today's production hot
    # path, so it never fires during the generation above. Survey-only
    # invocation lets us still cover its (M, N) grid for the M1-F2
    # autotune sweep. Qwen3.5-9B w8a8 hidden_size=4096, intermediate=12288.
    fused_quant_fired = _exercise_fused_int8_quant_fixture(
        hidden_sizes=[4096, 12288],
        m_values=m_values if m_values else [1, 4],
    )
    logger.info(
        "[survey] fused_int8_quant fixture fired %d times (m_values=%s)",
        fused_quant_fired,
        m_values,
    )

    snap = _recorder.snapshot()

    # Emit JSON with three top-level keys, mirroring the validation
    # contract (VAL-M1-001). Wrap each kernel's table under a "shapes"
    # array so future versions can attach kernel-level metadata.
    payload: dict[str, dict[str, list[dict]]] = {}
    for kname in (
        "mi100_int8_scaled_mm_kernel",
        "_fused_silu_quant_int8_kernel",
        "fused_int8_quant",
    ):
        payload[kname] = {"shapes": snap.get(kname, [])}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    logger.info("[survey] wrote shapes to %s", out_path)
    # Re-snapshot in the original (un-wrapped) format for the caller
    # (test harness reuse / smoke).
    return snap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B w8a8 shape survey for the 3 MI100 fused kernels."
    )
    parser.add_argument(
        "--model",
        default="/models/Qwen3.5-9B-w8a8",
        help="Path to the Qwen3.5-9B w8a8 model directory.",
    )
    parser.add_argument(
        "--out",
        default=(
            "/root/bench-int8-w4a16-m2-producer/m1-shapes/qwen3p5_9b_w8a8_shapes.json"
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--prompt",
        default="hi",
        help="Generation prompt (kept short to bound runtime).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Max generation tokens (prefill + N-1 decode steps).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    out_path = Path(args.out)
    snap = run_survey(
        model_path=args.model,
        out_path=out_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )

    # Summary print to stdout for the wrapping shell harness.
    print("[survey] shapes recorded per kernel:")
    for kname, table in snap.items():
        emit_true = sum(1 for e in table if e.get("EMIT_INT8_NEXT") is True)
        print(f"  {kname}: {len(table)} distinct (EMIT_INT8_NEXT=True: {emit_true})")
    print(f"[survey] wrote {out_path}")
    return 0


if __name__ == "__main__":
    # PYTHONPATH gotcha (AGENTS.md §13): editable install only picks up
    # this worktree when PYTHONPATH is set. We do NOT export it from
    # within the script because that would mutate the parent shell; the
    # caller is responsible. Surface a friendly error if the import
    # cannot resolve to a worktree-local vllm.
    try:
        import vllm  # noqa: F401  (early validation; intentional)
    except ImportError as exc:
        sys.stderr.write(
            "[survey] could not import vllm — did you set "
            "PYTHONPATH to the worktree root before invoking?\n"
            f"{exc}\n"
        )
        sys.exit(2)
    sys.exit(main())
