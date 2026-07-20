# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

from scripts.gfx900.cli import (
    EXIT_INCOMPARABLE,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REGRESSION,
    _semantic_matrix,
    _validate,
    command_compare_cells,
)
from scripts.gfx900.common import read_json


def test_confirm_schema_requires_declared_three_launch_policy() -> None:
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/gfx900/matrices/reference.json"
    )
    matrix = json.loads(matrix_path.read_text())
    cell = matrix["cells"]["smoke-fp16-tp8"]
    cell["trial_policy"] = "confirm"
    assert "'confirm_launches' is a required property" in _validate(
        matrix, "matrix.schema.json"
    )

    cell["confirm_launches"] = 3
    cell["recorded_runs"] = 3
    assert not _validate(matrix, "matrix.schema.json")


def test_reference_matrix_declares_all_clean_source_confirm_cells() -> None:
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads(
        (root / "scripts/gfx900/matrices/reference.json").read_text()
    )
    profile = json.loads(
        (root / "scripts/gfx900/profiles/c4130-2.json").read_text()
    )
    expected = {
        "confirm-capacity-auto-c32": {
            "phase": "profile",
            "port": 8010,
            "predecessor": "capacity-auto",
            "variant_keys": ["benchmark_argv"],
            "kv_cache_dtype": "auto",
        },
        "confirm-capacity-turboquant-c32": {
            "phase": "profile",
            "port": 8011,
            "predecessor": "capacity-turboquant-c32",
            "variant_keys": [],
            "kv_cache_dtype": "turboquant_k8v4",
        },
        "confirm-dense-eager-c1": {
            "phase": "profile",
            "port": 8012,
            "predecessor": "topology-fp16-tp8",
            "variant_keys": ["benchmark_argv", "server_argv"],
        },
        "confirm-graphs-full-decode-c1": {
            "phase": "confirm",
            "port": 8014,
            "predecessor": "confirm-dense-eager-c1",
            "variant_keys": ["server_argv"],
        },
    }

    assert not _validate(matrix, "matrix.schema.json")
    assert not _validate(profile, "profile.schema.json")
    assert not _semantic_matrix(profile, matrix)
    assert set(expected) <= matrix["cells"].keys()
    assert len({cell["port"] for cell in matrix["cells"].values()}) == len(
        matrix["cells"]
    )

    assert matrix["cells"]["smoke-fp16-tp8"]["timeout_seconds"] == 1200
    assert {
        cell_id
        for cell_id, cell in matrix["cells"].items()
        if cell["timeout_seconds"] == 1200
    } == {"smoke-fp16-tp8"}

    revision = profile["models"]["qwen35_fp16"]["revision"]
    for cell_id, details in expected.items():
        cell = matrix["cells"][cell_id]
        assert cell["trial_policy"] == "confirm"
        assert cell["confirm_launches"] == 3
        assert cell["warmup_runs"] == 1
        assert cell["recorded_runs"] == 3
        assert cell["model_alias"] == "qwen35_fp16"
        assert cell["device_group"] == "single_socket_8"
        assert cell["phase"] == details["phase"]
        assert cell["port"] == details["port"]
        assert cell["predecessor"] == details["predecessor"]
        assert cell["variant_keys"] == details["variant_keys"]
        assert cell["server_argv"][
            cell["server_argv"].index("--revision") + 1
        ] == revision

    for cell_id in (
        "confirm-capacity-auto-c32",
        "confirm-capacity-turboquant-c32",
    ):
        cell = matrix["cells"][cell_id]
        assert cell["benchmark_argv"][-2:] == ["--max-concurrency", "32"]
        assert (
            cell["server_argv"][cell["server_argv"].index("--kv-cache-dtype") + 1]
            == expected[cell_id]["kv_cache_dtype"]
        )
        assert "--enforce-eager" in cell["server_argv"]

    dense = matrix["cells"]["confirm-dense-eager-c1"]
    graph = matrix["cells"]["confirm-graphs-full-decode-c1"]
    assert dense["benchmark_argv"][3:10] == [
        "512",
        "--random-output-len",
        "512",
        "--num-prompts",
        "128",
        "--max-concurrency",
        "1",
    ]
    assert dense["benchmark_argv"] == graph["benchmark_argv"]
    assert "--enforce-eager" in dense["server_argv"]
    assert "--enforce-eager" not in graph["server_argv"]
    assert "FULL_DECODE_ONLY" in graph["server_argv"][
        graph["server_argv"].index("--compilation-config") + 1
    ]

    assert matrix["cells"]["capacity-auto"]["trial_policy"] == "screen"
    assert matrix["cells"]["capacity-turboquant-c32"]["trial_policy"] == "screen"


def _profile() -> dict:
    return {"models": {"model": {}}, "device_groups": {"group": {}}}


def _matrix() -> dict:
    return {
        "phases": ["smoke", "graphs"],
        "policies": {"smoke": {}},
        "workloads": {"short": {}},
        "cells": {
            "smoke": {
                "phase": "smoke",
                "model_alias": "model",
                "device_group": "group",
                "workload": "short",
                "promotion_policy": "smoke",
            },
            "graphs": {
                "phase": "graphs",
                "predecessor": "smoke",
                "model_alias": "model",
                "device_group": "group",
                "workload": "short",
                "promotion_policy": "smoke",
            },
        },
    }


def test_semantic_matrix_accepts_ordered_predecessor() -> None:
    assert not _semantic_matrix(_profile(), _matrix())


def test_semantic_matrix_rejects_unknown_predecessor() -> None:
    matrix = _matrix()
    matrix["cells"]["graphs"]["predecessor"] = "missing"
    assert "unknown predecessor missing" in _semantic_matrix(_profile(), matrix)[0]


def _cell(
    cell_id: str,
    *,
    output_throughput: float,
    p99_tpot_ms: float,
    p99_ttft_ms: float,
    platform_sha256: str = "platform-a",
) -> dict:
    return {
        "schema_version": 1,
        "identity": {
            "cell_id": cell_id,
            "phase": "confirm",
            "started_at": "2026-07-20T00:00:00+00:00",
        },
        "provenance": {
            "manifest_sha256": "manifest-sha",
            "platform_sha256": platform_sha256,
        },
        "configuration": {
            "resolved_digest": "digest",
            "launch_argv": [],
            "benchmark_argv": [],
            "environment": {},
        },
        "workload": "workload",
        "trials": [],
        "aggregate": {
            "output_throughput": output_throughput,
            "p99_tpot_ms": p99_tpot_ms,
            "p99_ttft_ms": p99_ttft_ms,
        },
        "verdict": {"status": "PASS", "failure_reason": None},
    }


def _write_cell(path: Path, cell: dict) -> str:
    path.write_text(json.dumps(cell))
    return str(path)


def test_compare_cells_command_reports_improvement(tmp_path: Path) -> None:
    baseline = _write_cell(
        tmp_path / "baseline.json",
        _cell("baseline", output_throughput=100, p99_tpot_ms=10, p99_ttft_ms=20),
    )
    candidate = _write_cell(
        tmp_path / "candidate.json",
        _cell("candidate", output_throughput=110, p99_tpot_ms=10, p99_ttft_ms=20),
    )
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        baseline=baseline, candidate=candidate, output=str(output)
    )

    assert command_compare_cells(args) == EXIT_OK

    result = read_json(output)
    assert result["verdict"] == "IMPROVEMENT"
    assert result["baseline_cell_id"] == "baseline"
    assert result["candidate_cell_id"] == "candidate"
    assert result["baseline_path"] == baseline
    assert result["candidate_path"] == candidate


def test_compare_cells_command_reports_pass(tmp_path: Path) -> None:
    baseline = _write_cell(
        tmp_path / "baseline.json",
        _cell("baseline", output_throughput=100, p99_tpot_ms=10, p99_ttft_ms=20),
    )
    candidate = _write_cell(
        tmp_path / "candidate.json",
        _cell("candidate", output_throughput=100.5, p99_tpot_ms=10, p99_ttft_ms=20),
    )
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        baseline=baseline, candidate=candidate, output=str(output)
    )

    assert command_compare_cells(args) == EXIT_OK
    assert read_json(output)["verdict"] == "PASS"


def test_compare_cells_command_reports_regression(tmp_path: Path) -> None:
    baseline = _write_cell(
        tmp_path / "baseline.json",
        _cell("baseline", output_throughput=100, p99_tpot_ms=10, p99_ttft_ms=20),
    )
    candidate = _write_cell(
        tmp_path / "candidate.json",
        _cell("candidate", output_throughput=90, p99_tpot_ms=10, p99_ttft_ms=20),
    )
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        baseline=baseline, candidate=candidate, output=str(output)
    )

    assert command_compare_cells(args) == EXIT_REGRESSION
    assert read_json(output)["verdict"] == "REGRESSION"


def test_compare_cells_command_rejects_platform_mismatch(tmp_path: Path) -> None:
    baseline = _write_cell(
        tmp_path / "baseline.json",
        _cell(
            "baseline",
            output_throughput=100,
            p99_tpot_ms=10,
            p99_ttft_ms=20,
            platform_sha256="platform-a",
        ),
    )
    candidate = _write_cell(
        tmp_path / "candidate.json",
        _cell(
            "candidate",
            output_throughput=110,
            p99_tpot_ms=10,
            p99_ttft_ms=20,
            platform_sha256="platform-b",
        ),
    )
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        baseline=baseline, candidate=candidate, output=str(output)
    )

    assert command_compare_cells(args) == EXIT_INCOMPARABLE

    result = read_json(output)
    assert result["verdict"] == "INCOMPARABLE"
    assert result["baseline_platform_sha256"] == "platform-a"
    assert result["candidate_platform_sha256"] == "platform-b"


def test_compare_cells_command_rejects_schema_invalid_input(tmp_path: Path) -> None:
    baseline_cell = _cell(
        "baseline", output_throughput=100, p99_tpot_ms=10, p99_ttft_ms=20
    )
    del baseline_cell["provenance"]
    baseline = _write_cell(tmp_path / "baseline.json", baseline_cell)
    candidate = _write_cell(
        tmp_path / "candidate.json",
        _cell("candidate", output_throughput=110, p99_tpot_ms=10, p99_ttft_ms=20),
    )
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        baseline=baseline, candidate=candidate, output=str(output)
    )

    assert command_compare_cells(args) == EXIT_INVALID
    assert not output.exists()
