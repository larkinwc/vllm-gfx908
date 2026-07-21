# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

benchmark_module = import_module("benchmarks.kernels.benchmark_device_communicators")
resolve_tensor_elements = benchmark_module.resolve_tensor_elements


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


def test_resolve_tensor_elements_matches_roadmap_kib_values() -> None:
    """hidden_size=4096, fp16 batch sizes 1/8/32/64/96 -> 8/64/256/512/768 KiB."""
    args = _args(hidden_size=4096, sequence_lengths=[1, 8, 32, 64, 96])
    elements, mode = resolve_tensor_elements(args)
    assert mode == "sequence_lengths"
    byte_sizes_kib = [n * torch.float16.itemsize / 1024 for n in elements]
    assert byte_sizes_kib == [8.0, 64.0, 256.0, 512.0, 768.0]


def test_unsupported_communicators_reported_with_reasons(monkeypatch) -> None:
    """Unavailable communicators must be recorded with an explicit reason,
    never silently dropped or substituted."""

    class _Disabled:
        def __init__(self, *args, **kwargs) -> None:
            self.disabled = True

    class _Raises:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("missing library")

    monkeypatch.setattr(benchmark_module, "CustomAllreduce", _Disabled)
    monkeypatch.setattr(benchmark_module, "PyNcclCommunicator", _Raises)
    monkeypatch.setattr(benchmark_module, "SymmMemCommunicator", _Disabled)
    monkeypatch.setattr(benchmark_module, "FlashInferAllReduce", _Disabled)

    benchmark = benchmark_module.CommunicatorBenchmark(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        cpu_group=None,
        tensor_elements=[8],
        dtype=torch.float16,
    )

    assert benchmark.custom_allreduce is None
    assert benchmark.pynccl_comm is None
    assert benchmark.symm_mem_comm_multimem is None
    assert benchmark.symm_mem_comm_two_shot is None
    assert benchmark.fi_ar_comm is None

    expected_names = {
        "ca_1stage",
        "ca_2stage",
        "pynccl",
        "pynccl-symm",
        "symm_mem_multimem",
        "symm_mem_two_shot",
        "flashinfer_trtllm",
        "flashinfer_mnnvl",
    }
    assert set(benchmark.unsupported) == expected_names
    assert all(benchmark.unsupported[name] for name in expected_names)
    assert "RuntimeError" in benchmark.unsupported["pynccl"]
    assert "missing library" in benchmark.unsupported["pynccl"]
    assert benchmark.unsupported["pynccl"] == benchmark.unsupported["pynccl-symm"]
