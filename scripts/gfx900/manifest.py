# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collection and compatibility checks for immutable gfx900 platform facts."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

from scripts.gfx900.common import (
    read_json,
    run_command,
    selected_environment,
    sha256_file,
    sha256_json,
    utc_now,
)


def _first_line(command: list[str]) -> str | None:
    result = run_command(command)
    if result.get("returncode") != 0:
        return None
    text = str(result.get("stdout", "")).strip()
    return text.splitlines()[0] if text else None


def _git_source(root: Path) -> dict[str, Any]:
    commit = _first_line(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = run_command(["git", "-C", str(root), "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(str(status.get("stdout", "")).strip())}


def _python_runtime() -> dict[str, Any]:
    probe = (
        "import json,sys; "
        "import torch; "
        "import triton; "
        "import vllm; "
        "devices=[]; "
        "\nfor i in range(torch.cuda.device_count()):\n"
        " p=torch.cuda.get_device_properties(i); "
        " devices.append({'visible_index':i,'name':torch.cuda.get_device_name(i),"
        "'arch':str(getattr(p,'gcnArchName','')).split(':')[0],"
        "'cu_count':getattr(p,'multi_processor_count',None),"
        "'total_memory':getattr(p,'total_memory',None)});"
        "\nprint(json.dumps({'python':sys.version, 'torch':torch.__version__, "
        "'hip':getattr(torch.version,'hip',None), 'triton':triton.__version__, "
        "'triton_file':getattr(triton,'__file__',None), 'vllm':vllm.__version__, "
        "'vllm_commit':getattr(vllm,'__commit__',None), 'devices':devices}))"
    )
    result = run_command([sys.executable, "-c", probe])
    if result.get("returncode") != 0:
        return {"python": sys.version, "probe_error": result.get("stderr", "")}
    import json

    return json.loads(str(result["stdout"]))


def _library(path: str | None) -> dict[str, str | None]:
    if path is None:
        return {"path": None, "sha256": None}
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def _resolve_library(
    name: str, rocm_root: Path, override: str | None = None
) -> str | None:
    """Resolve an override or ROCm shared library.

    Shared objects are not executables and cannot be resolved with ``which``.
    """
    candidates = [Path(override)] if override else []
    candidates.extend((rocm_root / "lib" / name, rocm_root / "lib64" / name))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _library_paths(
    rocm_root: Path, runtime: dict[str, Any]
) -> dict[str, dict[str, str | None]]:
    return {
        "rccl": _library(
            _resolve_library(
                "librccl.so", rocm_root, os.environ.get("VLLM_NCCL_SO_PATH")
            )
        ),
        "rocblas": _library(_resolve_library("librocblas.so", rocm_root)),
        "hipblaslt": _library(_resolve_library("libhipblaslt.so", rocm_root)),
        "triton": _library(runtime.get("triton_file")),
    }


def _profile_group(profile: dict[str, Any]) -> dict[str, Any]:
    groups = profile["device_groups"]
    return groups[profile.get("default_device_group", next(iter(groups)))]


def build_manifest(profile: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Collect portable facts; platform-specific enrichers update this JSON."""
    source = _git_source(root)
    runtime = _python_runtime()
    group = _profile_group(profile)
    selected_indices = set(group["visible_indices"])
    live_devices = [
        device
        for device in runtime.pop("devices", [])
        if device["visible_index"] in selected_indices
    ]
    reset_method = _first_line(["cat", "/sys/module/amdgpu/parameters/reset_method"])
    topology = run_command(["rocm-smi", "--showtopo"])
    rocm_root = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "host": {
            "hostname": platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "source": source,
        "runtime": {
            "executable": sys.executable,
            **runtime,
            "rocm_root": str(rocm_root),
        },
        "libraries": _library_paths(rocm_root, runtime),
        "devices": live_devices,
        "topology": {
            "raw": topology.get("stdout", ""),
            "returncode": topology.get("returncode"),
        },
        "numa": profile["numa_bindings"],
        "power": profile["expected_platform"].get("power", {}),
        "kernel": {
            "reset_method": int(reset_method)
            if reset_method and reset_method.isdigit()
            else reset_method
        },
        "environment": selected_environment(),
        "profile": {"host_id": profile["host_id"], "device_group": group["id"]},
    }
    compatibility = {
        "runtime": {
            key: manifest["runtime"].get(key)
            for key in ("executable", "python", "torch", "hip", "triton", "rocm_root")
        },
        "libraries": manifest["libraries"],
        "devices": manifest["devices"],
        "topology": manifest["topology"],
        "numa": manifest["numa"],
        "power": manifest["power"],
        "kernel": manifest["kernel"],
        "device_group": group,
    }
    manifest["platform_compatibility_keys"] = compatibility
    manifest["platform_sha256"] = sha256_json(compatibility)
    manifest["manifest_sha256"] = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"captured_at", "manifest_sha256"}
        }
    )
    return manifest


def validate_manifest(
    manifest: dict[str, Any], profile: dict[str, Any], *, require_clean: bool = False
) -> list[str]:
    """Return validation errors rather than concealing failed preconditions."""
    errors: list[str] = []
    source = manifest.get("source", {})
    if require_clean and source.get("dirty"):
        errors.append("reference capture requires a clean source tree")
    expected = profile["expected_platform"]
    if expected.get("reset_method") != 2:
        errors.append("profile must require amdgpu.reset_method=2")
    if manifest.get("kernel", {}).get("reset_method") != 2:
        errors.append("amdgpu.reset_method is not 2")
    devices = manifest.get("devices", [])
    if not devices:
        errors.append("no selected devices were recorded")
    for device in devices:
        if device.get("arch") != "gfx900":
            errors.append("selected device is not gfx900")
    if not manifest.get("runtime", {}).get("vllm"):
        errors.append("unable to import vLLM from the active interpreter")
    for library_name, library in manifest.get("libraries", {}).items():
        if not library.get("sha256"):
            errors.append(f"unable to resolve required {library_name} library hash")
    if not manifest.get("platform_sha256"):
        errors.append("missing platform compatibility digest")
    return errors


def compare_platforms(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare only the digest intended to remain fixed during an experiment."""
    equal = baseline.get("platform_sha256") == candidate.get("platform_sha256")
    return {
        "verdict": "PASS" if equal else "INCOMPARABLE",
        "baseline_platform_sha256": baseline.get("platform_sha256"),
        "candidate_platform_sha256": candidate.get("platform_sha256"),
    }


def load_profile(path: Path) -> dict[str, Any]:
    return read_json(path)
