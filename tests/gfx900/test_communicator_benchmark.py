# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

resolve_tensor_elements = import_module(
    "benchmarks.kernels.benchmark_device_communicators"
).resolve_tensor_elements


def _args(**overrides):
    values = {
        "dtype": "float16",
        "hidden_size": 4096,
        "sequence_lengths": [2],
        "num_elements": None,
        "message_size_bytes": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolve_tensor_elements_preserves_legacy_shape_mode() -> None:
    assert resolve_tensor_elements(_args()) == ([8192], "sequence_lengths")


def test_resolve_tensor_elements_accepts_repeatable_element_groups() -> None:
    assert resolve_tensor_elements(_args(num_elements=[[1, 2], [4]])) == (
        [1, 2, 4],
        "elements",
    )


def test_resolve_tensor_elements_converts_exact_bytes() -> None:
    assert resolve_tensor_elements(_args(message_size_bytes=[[8, 16]])) == (
        [4, 8],
        "bytes",
    )


@pytest.mark.parametrize(
    "args",
    [
        _args(num_elements=[[1]], message_size_bytes=[[2]]),
        _args(message_size_bytes=[[3]]),
    ],
)
def test_resolve_tensor_elements_rejects_ambiguous_or_misaligned_sizes(args) -> None:
    with pytest.raises(ValueError):
        resolve_tensor_elements(args)


def test_dtype_selection_is_fp16_for_gfx900_matrix() -> None:
    assert torch.float16.itemsize == 2
