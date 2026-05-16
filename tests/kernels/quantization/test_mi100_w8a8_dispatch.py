# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the M2 W8A8 dispatcher: hipBLASLt path for tuned shapes,
Triton fallback for everything else, ``VLLM_DISABLE_HIPBLASLT=1`` override.

Focuses on dispatch behaviour rather than absolute speed. Numerical
correctness of the hipBLASLt path is covered separately by
``test_hipblaslt_int8_correctness.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest
import torch

pytest.importorskip(
    "vllm.model_executor.kernels.linear.scaled_mm.mi100_int8"
)

from vllm.model_executor.kernels.linear.scaled_mm import (  # noqa: E402
    mi100_hipblaslt,
    mi100_int8,
)


def _on_gfx908() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return arch.startswith("gfx908")


pytestmark = pytest.mark.skipif(
    not _on_gfx908(),
    reason="MI100 gfx908 W8A8 dispatcher tests require a gfx908 GPU.",
)


@pytest.fixture
def tmp_tuned_shapes(tmp_path: Path) -> Path:
    manifest = {
        "schema_version": 1,
        "rocm_version": "7.12",
        "arch": "gfx908",
        "shapes": [
            {"M": 32, "N": 64, "K": 64},
            {"M": 64, "N": 128, "K": 64},
        ],
    }
    p = tmp_path / "tuned.json"
    p.write_text(json.dumps(manifest))
    mi100_hipblaslt.reset_tuned_shape_cache()
    with mock.patch.dict(os.environ, {
        "VLLM_HIPBLASLT_TUNED_SHAPES": str(p),
    }):
        yield p
    mi100_hipblaslt.reset_tuned_shape_cache()


def _make_w8a8_tensors(M: int, N: int, K: int, device: str = "cuda"):
    a = torch.randint(-32, 31, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-32, 31, (K, N), dtype=torch.int8, device=device)
    sa = torch.full((M, 1), 0.01, dtype=torch.float32, device=device)
    sb = torch.full((N, 1), 0.02, dtype=torch.float32, device=device)
    return a, b, sa, sb


def test_supports_returns_true_for_listed_shape(tmp_tuned_shapes):
    assert mi100_hipblaslt.mi100_hipblaslt_supports(32, 64, 64) is True
    assert mi100_hipblaslt.mi100_hipblaslt_supports(64, 128, 64) is True


def test_supports_returns_false_for_unknown_shape(tmp_tuned_shapes):
    assert mi100_hipblaslt.mi100_hipblaslt_supports(33, 64, 64) is False
    assert mi100_hipblaslt.mi100_hipblaslt_supports(64, 64, 64) is False


def test_disable_env_forces_triton_path(tmp_tuned_shapes):
    with mock.patch.dict(os.environ, {"VLLM_DISABLE_HIPBLASLT": "1"}):
        assert mi100_hipblaslt.is_hipblaslt_disabled() is True
        # Even a tuned shape must report unsupported when the override is set.
        assert mi100_hipblaslt.mi100_hipblaslt_supports(32, 64, 64) is False


def test_dispatcher_prefers_hipblaslt_for_tuned_shape(tmp_tuned_shapes):
    """A tuned (M, N, K) must hit the hipBLASLt branch."""
    M, N, K = 32, 64, 64
    a, b, sa, sb = _make_w8a8_tensors(M, N, K)

    called = {"hipblaslt": 0, "triton": 0}
    real_hipblaslt = mi100_int8.mi100_hipblaslt_scaled_mm

    def hipblaslt_spy(*args, **kwargs):
        called["hipblaslt"] += 1
        return real_hipblaslt(*args, **kwargs)

    def kernel_spy(*args, **kwargs):
        called["triton"] += 1

    with mock.patch.object(mi100_int8, "mi100_hipblaslt_scaled_mm",
                           hipblaslt_spy), \
         mock.patch.object(mi100_int8, "mi100_int8_scaled_mm_kernel",
                           kernel_spy):
        out = mi100_int8.mi100_int8_scaled_mm(
            a, b, sa, sb, out_dtype=torch.float16,
        )
    assert called["hipblaslt"] == 1
    assert called["triton"] == 0
    assert out.shape == (M, N)
    assert out.dtype == torch.float16


def test_dispatcher_falls_back_to_triton_for_untuned(tmp_tuned_shapes):
    """An off-list (M, N, K) must hit the Triton kernel branch."""
    M, N, K = 33, 64, 64  # not in the manifest
    a, b, sa, sb = _make_w8a8_tensors(M, N, K)

    called_hipblaslt = []

    def hipblaslt_spy(*args, **kwargs):
        called_hipblaslt.append(1)

    with mock.patch.object(mi100_int8, "mi100_hipblaslt_scaled_mm",
                           hipblaslt_spy):
        out = mi100_int8.mi100_int8_scaled_mm(
            a, b, sa, sb, out_dtype=torch.float16,
        )
    assert called_hipblaslt == []
    assert out.shape == (M, N)


def test_disable_env_forces_triton_even_for_tuned_shape(tmp_tuned_shapes):
    """VLLM_DISABLE_HIPBLASLT=1 must route a tuned shape to Triton."""
    M, N, K = 32, 64, 64
    a, b, sa, sb = _make_w8a8_tensors(M, N, K)

    hits = []

    def hipblaslt_spy(*args, **kwargs):
        hits.append(1)

    with mock.patch.dict(os.environ, {"VLLM_DISABLE_HIPBLASLT": "1"}), \
         mock.patch.object(mi100_int8, "mi100_hipblaslt_scaled_mm",
                           hipblaslt_spy):
        out = mi100_int8.mi100_int8_scaled_mm(
            a, b, sa, sb, out_dtype=torch.float16,
        )
    assert hits == []
    assert out.shape == (M, N)


def test_disable_env_output_matches_default_triton(tmp_tuned_shapes):
    """VLLM_DISABLE_HIPBLASLT=1 must produce the same output as plain Triton
    for an off-list shape (i.e. fallback path is bit-identical)."""
    M, N, K = 33, 64, 64
    a, b, sa, sb = _make_w8a8_tensors(M, N, K)

    out_default = mi100_int8.mi100_int8_scaled_mm(
        a, b, sa, sb, out_dtype=torch.float16,
    )
    with mock.patch.dict(os.environ, {"VLLM_DISABLE_HIPBLASLT": "1"}):
        out_disabled = mi100_int8.mi100_int8_scaled_mm(
            a, b, sa, sb, out_dtype=torch.float16,
        )
    assert torch.equal(out_default, out_disabled)


def test_committed_manifest_is_loadable():
    """The committed manifest at configs/gfx908/hipblaslt_tuned_shapes.json
    must parse and contain at least one entry so VAL-TENSILE-004 evidence
    is reproducible from the repo alone."""
    mi100_hipblaslt.reset_tuned_shape_cache()
    tuned, manifest = mi100_hipblaslt._load_tuned_shapes_set()
    assert len(tuned) >= 1, manifest
    assert manifest.get("arch") == "gfx908"
    assert "rocm_version" in manifest
    mi100_hipblaslt.reset_tuned_shape_cache()
