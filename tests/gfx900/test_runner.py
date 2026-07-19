# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from scripts.gfx900.common import contains_error_signature, redact_environment
from scripts.gfx900.runner import (
    TERMINAL_STATUSES,
    _missing_required_metrics,
    _run_benchmark,
    resolve_cell,
)


def _profile() -> dict:
    return {
        "baseline_environment": {"NCCL_ALGO": "Ring", "HOME": "/tmp"},
        "models": {"model": {"id": "model", "environment": {"VLLM_USE_V1": "1"}}},
        "device_groups": {"group": {"id": "group", "visible_indices": [3, 5]}},
    }


def _matrix(environment: dict[str, str]) -> dict:
    return {
        "cells": {
            "cell": {
                "model_alias": "model",
                "device_group": "group",
                "environment": environment,
                "variant_keys": list(environment),
                "phase": "smoke",
                "workload": "short",
            }
        }
    }


def test_resolve_cell_applies_precedence() -> None:
    value = resolve_cell(_profile(), _matrix({"NCCL_ALGO": "Tree"}), "cell")
    assert value["environment"]["NCCL_ALGO"] == "Tree"
    assert value["environment"]["VLLM_USE_V1"] == "1"
    assert value["environment"]["HIP_VISIBLE_DEVICES"] == "3,5"


def test_resolve_cell_rejects_device_group_override() -> None:
    matrix = _matrix({"HIP_VISIBLE_DEVICES": "0,1"})
    with pytest.raises(ValueError, match="controlled"):
        resolve_cell(_profile(), matrix, "cell")


def test_resolve_cell_rejects_undeclared_override() -> None:
    matrix = _matrix({"NCCL_ALGO": "Tree"})
    matrix["cells"]["cell"]["variant_keys"] = []
    with pytest.raises(ValueError, match="undeclared"):
        resolve_cell(_profile(), matrix, "cell")


def test_secret_redaction_and_fatal_signatures() -> None:
    assert (
        redact_environment({"HF_TOKEN": "secret", "VLLM_USE_V1": "1"})["HF_TOKEN"]
        == "<redacted>"
    )
    assert not contains_error_signature("assertion passed in room zoom")
    assert contains_error_signature("RuntimeError: HIP out of memory")
    assert contains_error_signature("AssertionError: bad invariant")


def test_missing_required_metrics_fails_closed() -> None:
    assert _missing_required_metrics(
        ["preemptions", "kv_cache_usage_perc"],
        [{"metrics": {"vllm:num_preemptions_total": 0.0}}],
    ) == ["kv_cache_usage_perc"]


def test_benchmark_uses_saved_result_file(tmp_path, monkeypatch) -> None:
    result_path = tmp_path / "result.json"
    class FakeProcess:
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(argv, **kwargs):
        result_path.write_text(json.dumps({"output_throughput": 1.0}))
        return FakeProcess()

    monkeypatch.setattr("scripts.gfx900.runner.subprocess.Popen", fake_popen)
    result = _run_benchmark(["vllm", "bench", "serve"], {}, 1, result_path)
    assert result["result"]["output_throughput"] == 1.0
    assert "--save-result" in result["argv"]
    assert "--result-filename" in result["argv"]


def test_terminal_statuses_are_complete() -> None:
    assert {
        "PASS",
        "IMPROVEMENT",
        "REGRESSION",
        "INCOMPARABLE",
        "UNSUPPORTED",
        "FAILED",
    } == TERMINAL_STATUSES
