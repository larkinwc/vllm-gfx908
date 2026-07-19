# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, dependency-light primitives shared by the gfx900 benchmark harness."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ERROR_SIGNATURES = (
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"hipErrorOutOfMemory", re.IGNORECASE),
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"\bAssertionError\b", re.IGNORECASE),
    re.compile(r"\billegal instruction\b", re.IGNORECASE),
)
_SECRET_NAME = re.compile(r"(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


def utc_now() -> str:
    """Return an RFC 3339 timestamp with an explicit UTC offset."""
    return datetime.now(UTC).isoformat()


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for hashes and reproducible artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return a sorted environment view without secrets."""
    return {
        key: "<redacted>" if _SECRET_NAME.search(key) else value
        for key, value in sorted(environment.items())
    }


def selected_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Select performance-relevant variables, redacting any accidental secrets."""
    source = os.environ if environment is None else environment
    prefixes = ("VLLM_", "NCCL_", "RCCL_", "HIP_", "HSA_", "ROCR_")
    return redact_environment(
        {key: value for key, value in source.items() if key.startswith(prefixes)}
    )


def run_command(
    argv: Sequence[str],
    *,
    timeout: float = 30,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run an argv command and retain stdout/stderr without a shell."""
    try:
        result = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=None if environment is None else dict(environment),
        )
        return {
            "argv": list(argv),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except FileNotFoundError as error:
        return {
            "argv": list(argv),
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "argv": list(argv),
            "returncode": None,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "timed_out": True,
        }


def contains_error_signature(text: str) -> bool:
    """Detect unambiguous fatal signatures without flagging ordinary log prose."""
    return any(signature.search(text) for signature in ERROR_SIGNATURES)
