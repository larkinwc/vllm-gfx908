# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resumable server/benchmark orchestration for declared gfx900 matrix cells."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from scripts.gfx900.analysis import median, resolved_configuration_digest
from scripts.gfx900.common import (
    contains_error_signature,
    redact_environment,
    utc_now,
    write_json,
)

TERMINAL_STATUSES = {
    "PASS",
    "IMPROVEMENT",
    "REGRESSION",
    "INCOMPARABLE",
    "UNSUPPORTED",
    "FAILED",
}

REQUIRED_METRIC_NAMES = {
    "preemptions": "vllm:num_preemptions_total",
    "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
}


def _merge_environment(*layers: Mapping[str, str]) -> dict[str, str]:
    allowed_parent = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VLLM_", "NCCL_", "RCCL_", "HIP_", "HSA_", "ROCR_"))
    }
    for layer in layers:
        allowed_parent.update(layer)
    return allowed_parent


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_server(base_url: str, model: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{base_url.rstrip('/')}/health"
    models_url = f"{base_url.rstrip('/')}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:  # noqa: S310
                healthy = response.status == 200
            with urlopen(models_url, timeout=2) as response:  # noqa: S310
                models = json.loads(response.read()).get("data", [])
            model_is_served = any(
                entry.get("id") == model or entry.get("root") == model
                for entry in models
            )
            if healthy and (model_is_served or len(models) == 1):
                return
        except (URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready at {base_url}")


def _scrape_metrics(base_url: str) -> dict[str, float]:
    """Read numeric Prometheus samples needed by declared promotion gates."""
    with urlopen(f"{base_url.rstrip('/')}/metrics", timeout=5) as response:  # noqa: S310
        payload = response.read().decode("utf-8")
    metrics: dict[str, float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if not value:
            continue
        try:
            metric_name = name.split("{", 1)[0]
            numeric_value = float(value)
            if "cache_usage" in metric_name:
                metrics[metric_name] = max(
                    metrics.get(metric_name, numeric_value), numeric_value
                )
            else:
                metrics[metric_name] = metrics.get(metric_name, 0.0) + numeric_value
        except ValueError:
            continue
    return metrics


def _metric_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float]:
    """Preserve gauges and reduce counters to a per-trial delta."""
    metrics: dict[str, float] = {}
    for name, value in after.items():
        metrics[name] = (
            value - before[name]
            if name in before and "cache_usage" not in name
            else value
        )
    return metrics


def _missing_required_metrics(
    required_metrics: list[str], trials: list[Mapping[str, Any]]
) -> list[str]:
    """Return declared gate metrics absent from any recorded trial."""
    return [
        required
        for required in required_metrics
        if required in REQUIRED_METRIC_NAMES
        and any(
            REQUIRED_METRIC_NAMES[required] not in trial.get("metrics", {})
            for trial in trials
        )
    ]


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def _run_benchmark(
    argv: list[str],
    environment: Mapping[str, str],
    timeout: float,
    result_path: Path,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run ``vllm bench serve`` and load its authoritative saved JSON result."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [*argv, "--save-result", "--result-dir", str(result_path.parent),
               "--result-filename", result_path.name]
    peak_metrics: dict[str, float] = {}
    with (
        tempfile.TemporaryFile(mode="w+t") as stdout_file,
        tempfile.TemporaryFile(mode="w+t") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env=dict(environment),
        )
        started = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - started > timeout:
                _terminate(process)
                raise TimeoutError(f"benchmark exceeded {timeout} seconds")
            if base_url:
                try:
                    samples = _scrape_metrics(base_url)
                except (OSError, TimeoutError, URLError):
                    samples = {}
                for name, value in samples.items():
                    if "cache_usage" in name:
                        peak_metrics[name] = max(peak_metrics.get(name, value), value)
            time.sleep(1)
        stdout_file.seek(0)
        stderr_file.seek(0)
        result = subprocess.CompletedProcess(
            command, process.returncode, stdout_file.read(), stderr_file.read()
        )
    raw: dict[str, Any] = {
        "argv": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    raw["peak_metrics"] = peak_metrics
    if result.returncode:
        raise RuntimeError(f"benchmark exited {result.returncode}: {result.stderr}")
    if not result_path.is_file():
        raise RuntimeError(f"benchmark did not write result: {result_path}")
    try:
        raw["result"] = json.loads(result_path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"benchmark result is invalid JSON: {error}") from error
    return raw


def resolve_cell(
    profile: Mapping[str, Any], matrix: Mapping[str, Any], cell_id: str
) -> dict[str, Any]:
    """Resolve profile aliases and reject undeclared environment overrides."""
    cell = matrix["cells"][cell_id]
    model = profile["models"][cell["model_alias"]]
    group = profile["device_groups"][cell["device_group"]]
    environment = _merge_environment(
        profile["baseline_environment"],
        model.get("environment", {}),
        cell["environment"],
    )
    if "HIP_VISIBLE_DEVICES" in environment:
        raise ValueError(
            "HIP_VISIBLE_DEVICES is controlled by the selected profile device group"
        )
    environment["HIP_VISIBLE_DEVICES"] = ",".join(
        str(index) for index in group["visible_indices"]
    )
    variant_keys = set(cell["variant_keys"])
    prohibited = set(cell["environment"]) - variant_keys
    if prohibited:
        raise ValueError(
            f"cell {cell_id} overrides undeclared keys: {sorted(prohibited)}"
        )
    return {
        "cell": cell,
        "model": model,
        "device_group": group,
        "environment": environment,
        "configuration_digest": resolved_configuration_digest(
            {
                "cell": cell,
                "model": model,
                "device_group": group,
                "environment": environment,
            }
        ),
    }


def run_cell(
    *,
    profile: Mapping[str, Any],
    matrix: Mapping[str, Any],
    cell_id: str,
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one declared cell. This function is deliberately injectable in tests."""
    resolved = resolve_cell(profile, matrix, cell_id)
    cell = resolved["cell"]
    cell_dir = output_dir / "cells" / cell_id
    result_path = cell_dir / "cell.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("verdict", {}).get("status") in TERMINAL_STATUSES:
            return existing
    cell_dir.mkdir(parents=True, exist_ok=True)
    port = int(cell["port"])
    if not _is_port_free(port):
        raise RuntimeError(f"port {port} is already in use")
    base_url = f"http://127.0.0.1:{port}"
    launch_argv = [
        profile["executables"]["vllm"],
        "serve",
        resolved["model"]["id"],
        *cell["server_argv"],
        "--port",
        str(port),
    ]
    benchmark_argv = [
        profile["executables"]["vllm"],
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--model",
        resolved["model"]["id"],
        *cell["benchmark_argv"],
    ]
    process: subprocess.Popen[str] | None = None
    trials: list[dict[str, Any]] = []
    status = "FAILED"
    failure_reason: str | None = None
    try:
        process = subprocess.Popen(
            launch_argv,
            stdout=(cell_dir / "server.stdout.log").open("w"),
            stderr=(cell_dir / "server.stderr.log").open("w"),
            text=True,
            env=resolved["environment"],
            start_new_session=True,
        )
        _wait_for_server(
            base_url, resolved["model"]["id"], float(cell["timeout_seconds"])
        )
        benchmark_results = cell_dir / "benchmark_results"
        for index in range(cell["warmup_runs"]):
            _run_benchmark(
                benchmark_argv,
                resolved["environment"],
                float(cell["timeout_seconds"]),
                benchmark_results / f"warmup-{index}.json",
            )
        for index in range(cell["recorded_runs"]):
            before_metrics = _scrape_metrics(base_url)
            trial = _run_benchmark(
                benchmark_argv,
                resolved["environment"],
                float(cell["timeout_seconds"]),
                benchmark_results / f"trial-{index}.json",
                base_url=base_url,
            )
            trial["metrics"] = _metric_delta(before_metrics, _scrape_metrics(base_url))
            trial["metrics"].update(trial.pop("peak_metrics"))
            trials.append(trial)
        output = [trial["result"] for trial in trials]
        status = (
            "PASS"
            if all(not result.get("failed", 0) for result in output)
            else "FAILED"
        )
        if status == "FAILED":
            failure_reason = "benchmark reported failed requests"
        missing_metrics = _missing_required_metrics(
            cell["required_metrics"], trials
        )
        if missing_metrics:
            status = "FAILED"
            failure_reason = (
                f"required metrics unavailable: {', '.join(missing_metrics)}"
            )
    except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        failure_reason = str(error)
    finally:
        if process is not None:
            _terminate(process)
    logs = ""
    for log in (cell_dir / "server.stdout.log", cell_dir / "server.stderr.log"):
        if log.exists():
            logs += log.read_text(errors="replace")
    if contains_error_signature(logs):
        status = "FAILED"
        failure_reason = failure_reason or "fatal server error signature"
    aggregate: dict[str, Any] = {}
    if trials:
        results = [trial["result"] for trial in trials]
        for key in (
            "output_throughput",
            "p99_tpot_ms",
            "p99_ttft_ms",
            "mean_ttft_ms",
            "median_tpot_ms",
        ):
            values = [
                result[key]
                for result in results
                if isinstance(result.get(key), (float, int))
            ]
            if values:
                aggregate[key] = median(values)
        aggregate["failed_requests"] = sum(
            int(result.get("failed", 0)) for result in results
        )
    artifact = {
        "schema_version": 1,
        "identity": {
            "cell_id": cell_id,
            "phase": cell["phase"],
            "started_at": utc_now(),
        },
        "provenance": {
            "manifest_sha256": manifest["manifest_sha256"],
            "platform_sha256": manifest["platform_sha256"],
        },
        "configuration": {
            "resolved_digest": resolved["configuration_digest"],
            "launch_argv": launch_argv,
            "benchmark_argv": benchmark_argv,
            "environment": redact_environment(resolved["environment"]),
        },
        "workload": cell["workload"],
        "trials": trials,
        "aggregate": aggregate,
        "verdict": {"status": status, "failure_reason": failure_reason},
    }
    write_json(result_path, artifact)
    return artifact


def check_disk_headroom(path: Path, required_gib: int) -> None:
    free = shutil.disk_usage(path).free
    if free < required_gib * 1024**3:
        free_gib = free / 1024**3
        raise RuntimeError(
            f"output filesystem has {free_gib:.1f} GiB free; need {required_gib} GiB"
        )
