# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from scripts.gfx900.cli import _validate
from scripts.gfx900.common import contains_error_signature, redact_environment
from scripts.gfx900.runner import (
    TERMINAL_STATUSES,
    _missing_required_metrics,
    _run_benchmark,
    resolve_cell,
    run_cell,
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


def _confirm_profile() -> dict:
    profile = _profile()
    profile["executables"] = {"vllm": "/usr/bin/vllm"}
    return profile


def _confirm_matrix() -> dict:
    return {
        "cells": {
            "cell": {
                "model_alias": "model",
                "device_group": "group",
                "environment": {},
                "variant_keys": [],
                "phase": "confirm",
                "workload": "short",
                "trial_policy": "confirm",
                "confirm_launches": 3,
                "server_argv": [],
                "benchmark_argv": [],
                "required_metrics": [],
                "timeout_seconds": 1,
                "port": 8123,
                "warmup_runs": 1,
                "recorded_runs": 3,
                "promotion_policy": "confirm",
            }
        }
    }


def _manifest() -> dict:
    return {"manifest_sha256": "manifest", "platform_sha256": "platform"}


class _FakeServer:
    pid = 123


def _successful_trial(result_path) -> dict:
    launch_index = int(result_path.parent.parent.name.split("-")[1])
    values = {
        0: [1.0, 1.0, 1.0],
        1: [2.0, 100.0, 100.0],
        2: [3.0, 101.0, 101.0],
    }
    if result_path.name.startswith("warmup"):
        throughput = 0.0
    else:
        trial_index = int(result_path.stem.rsplit("-", 1)[1])
        throughput = values[launch_index][trial_index]
    return {
        "result": {"output_throughput": throughput, "failed": 0},
        "peak_metrics": {},
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


@pytest.mark.parametrize("trial_policy", ["smoke", "screen"])
def test_nonconfirm_policies_keep_single_server_artifacts(
    tmp_path, monkeypatch, trial_policy
) -> None:
    server_starts = []
    terminated = []
    matrix = _confirm_matrix()
    cell = matrix["cells"]["cell"]
    cell["trial_policy"] = trial_policy
    cell.pop("confirm_launches")
    cell["recorded_runs"] = 1

    def fake_popen(argv, **kwargs):
        server_starts.append(argv)
        return _FakeServer()

    def fake_benchmark(argv, environment, timeout, result_path, base_url=None):
        return {
            "result": {"output_throughput": 1.0, "failed": 0},
            "peak_metrics": {},
        }

    monkeypatch.setattr("scripts.gfx900.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.gfx900.runner._terminate", terminated.append)
    monkeypatch.setattr("scripts.gfx900.runner._is_port_free", lambda port: True)
    monkeypatch.setattr("scripts.gfx900.runner._wait_for_server", lambda *args: None)
    monkeypatch.setattr("scripts.gfx900.runner._scrape_metrics", lambda base_url: {})
    monkeypatch.setattr("scripts.gfx900.runner._run_benchmark", fake_benchmark)

    artifact = run_cell(
        profile=_confirm_profile(),
        matrix=matrix,
        cell_id="cell",
        output_dir=tmp_path,
        manifest=_manifest(),
    )

    assert len(server_starts) == 1
    assert len(terminated) == 1
    assert "confirm" not in artifact
    assert len(artifact["trials"]) == 1


def test_confirm_runs_independent_launches_and_reduces_launch_medians(
    tmp_path, monkeypatch
) -> None:
    server_starts = []
    terminated = []

    def fake_popen(argv, **kwargs):
        server_starts.append(argv)
        return _FakeServer()

    monkeypatch.setattr("scripts.gfx900.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.gfx900.runner._terminate", terminated.append)
    monkeypatch.setattr("scripts.gfx900.runner._is_port_free", lambda port: True)
    monkeypatch.setattr("scripts.gfx900.runner._wait_for_server", lambda *args: None)
    monkeypatch.setattr("scripts.gfx900.runner._scrape_metrics", lambda base_url: {})
    benchmark_paths = []

    def fake_benchmark(argv, environment, timeout, result_path, base_url=None):
        benchmark_paths.append(result_path)
        return _successful_trial(result_path)

    monkeypatch.setattr("scripts.gfx900.runner._run_benchmark", fake_benchmark)

    artifact = run_cell(
        profile=_confirm_profile(),
        matrix=_confirm_matrix(),
        cell_id="cell",
        output_dir=tmp_path,
        manifest=_manifest(),
    )

    assert len(server_starts) == 3
    assert len(terminated) == 3
    assert sum(path.name.startswith("warmup") for path in benchmark_paths) == 3
    assert sum(path.name.startswith("trial") for path in benchmark_paths) == 9
    assert [
        launch["aggregate"]["output_throughput"]
        for launch in artifact["confirm"]["launches"]
    ] == [1.0, 100.0, 101.0]
    assert artifact["aggregate"]["output_throughput"] == 100.0
    assert artifact["confirm"]["reduction"]["method"] == "median_of_launch_medians"
    assert not _validate(artifact, "result.schema.json")


def test_confirm_cleans_up_and_records_a_failed_launch(tmp_path, monkeypatch) -> None:
    server_starts = []
    terminated = []

    def fake_popen(argv, **kwargs):
        server_starts.append(argv)
        return _FakeServer()

    def failing_benchmark(argv, environment, timeout, result_path, base_url=None):
        raise RuntimeError("benchmark failure")

    monkeypatch.setattr("scripts.gfx900.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.gfx900.runner._terminate", terminated.append)
    monkeypatch.setattr("scripts.gfx900.runner._is_port_free", lambda port: True)
    monkeypatch.setattr("scripts.gfx900.runner._wait_for_server", lambda *args: None)
    monkeypatch.setattr("scripts.gfx900.runner._run_benchmark", failing_benchmark)

    artifact = run_cell(
        profile=_confirm_profile(),
        matrix=_confirm_matrix(),
        cell_id="cell",
        output_dir=tmp_path,
        manifest=_manifest(),
    )

    assert len(server_starts) == 1
    assert len(terminated) == 1
    assert artifact["verdict"]["status"] == "FAILED"
    assert artifact["confirm"]["launches"][0]["status"] == "FAILED"
    assert artifact["confirm"]["launches"][0]["failure_reason"] == "benchmark failure"


def test_confirm_interrupt_persists_evidence_after_cleanup(
    tmp_path, monkeypatch
) -> None:
    server_starts = []
    terminated = []

    def fake_popen(argv, **kwargs):
        server_starts.append(argv)
        return _FakeServer()

    def interrupted_benchmark(argv, environment, timeout, result_path, base_url=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("scripts.gfx900.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.gfx900.runner._terminate", terminated.append)
    monkeypatch.setattr("scripts.gfx900.runner._is_port_free", lambda port: True)
    monkeypatch.setattr("scripts.gfx900.runner._wait_for_server", lambda *args: None)
    monkeypatch.setattr("scripts.gfx900.runner._run_benchmark", interrupted_benchmark)

    with pytest.raises(KeyboardInterrupt):
        run_cell(
            profile=_confirm_profile(),
            matrix=_confirm_matrix(),
            cell_id="cell",
            output_dir=tmp_path,
            manifest=_manifest(),
        )

    artifact = json.loads((tmp_path / "cells/cell/cell.json").read_text())
    assert len(server_starts) == 1
    assert len(terminated) == 1
    assert artifact["confirm"]["launches"][0]["failure_reason"] == "KeyboardInterrupt"


def test_confirm_resume_preserves_completed_launches(tmp_path, monkeypatch) -> None:
    server_starts = []
    terminated = []
    fail_first_attempt = True

    def fake_popen(argv, **kwargs):
        server_starts.append(argv)
        return _FakeServer()

    def benchmark_with_one_failure(
        argv, environment, timeout, result_path, base_url=None
    ):
        if (
            fail_first_attempt
            and result_path.parent.parent.name == "launch-1-attempt-1"
        ):
            raise RuntimeError("benchmark failure")
        return _successful_trial(result_path)

    monkeypatch.setattr("scripts.gfx900.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.gfx900.runner._terminate", terminated.append)
    monkeypatch.setattr("scripts.gfx900.runner._is_port_free", lambda port: True)
    monkeypatch.setattr("scripts.gfx900.runner._wait_for_server", lambda *args: None)
    monkeypatch.setattr("scripts.gfx900.runner._scrape_metrics", lambda base_url: {})
    monkeypatch.setattr(
        "scripts.gfx900.runner._run_benchmark", benchmark_with_one_failure
    )

    first = run_cell(
        profile=_confirm_profile(),
        matrix=_confirm_matrix(),
        cell_id="cell",
        output_dir=tmp_path,
        manifest=_manifest(),
    )
    completed_launch = json.loads(json.dumps(first["confirm"]["launches"][0]))
    fail_first_attempt = False

    resumed = run_cell(
        profile=_confirm_profile(),
        matrix=_confirm_matrix(),
        cell_id="cell",
        output_dir=tmp_path,
        manifest=_manifest(),
    )

    assert len(server_starts) == 4
    assert len(terminated) == 4
    assert resumed["verdict"]["status"] == "PASS"
    assert resumed["confirm"]["launches"][0] == completed_launch
    assert [
        (launch["launch_index"], launch["attempt"], launch["status"])
        for launch in resumed["confirm"]["launches"]
    ] == [(0, 1, "PASS"), (1, 1, "FAILED"), (1, 2, "PASS"), (2, 1, "PASS")]


def test_confirm_resume_rejects_changed_manifest_and_preserves_evidence(
    tmp_path, monkeypatch
) -> None:
    server_starts = []
    terminated = []

    def fake_popen(argv, **kwargs):
        server_starts.append(argv)
        return _FakeServer()

    def failing_benchmark(argv, environment, timeout, result_path, base_url=None):
        raise RuntimeError("benchmark failure")

    monkeypatch.setattr("scripts.gfx900.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.gfx900.runner._terminate", terminated.append)
    monkeypatch.setattr("scripts.gfx900.runner._is_port_free", lambda port: True)
    monkeypatch.setattr("scripts.gfx900.runner._wait_for_server", lambda *args: None)
    monkeypatch.setattr("scripts.gfx900.runner._run_benchmark", failing_benchmark)

    first = run_cell(
        profile=_confirm_profile(),
        matrix=_confirm_matrix(),
        cell_id="cell",
        output_dir=tmp_path,
        manifest=_manifest(),
    )
    result_path = tmp_path / "cells/cell/cell.json"
    evidence = result_path.read_bytes()
    changed_manifest = {**_manifest(), "manifest_sha256": "other-manifest"}

    with pytest.raises(RuntimeError, match="manifest SHA"):
        run_cell(
            profile=_confirm_profile(),
            matrix=_confirm_matrix(),
            cell_id="cell",
            output_dir=tmp_path,
            manifest=changed_manifest,
        )

    assert first["verdict"]["status"] == "FAILED"
    assert result_path.read_bytes() == evidence
    assert len(server_starts) == 1
    assert len(terminated) == 1
