# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure reduction and promotion helpers for gfx900 benchmark artifacts."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from scripts.gfx900.common import sha256_json

THROUGHPUT_REGRESSION_PCT = -3.0
LATENCY_REGRESSION_PCT = 2.0
HBM_REGRESSION_PCT = 2.0
IMPROVEMENT_PCT = 3.0


def percent_delta(candidate: float, baseline: float) -> float | None:
    """Return candidate-versus-baseline percent delta, or None for zero baseline."""
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def median(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        raise ValueError("cannot calculate a median from no values")
    return float(statistics.median(numbers))


def geometric_mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers or any(value <= 0 for value in numbers):
        raise ValueError("geometric mean requires non-empty positive values")
    return math.exp(sum(math.log(value) for value in numbers) / len(numbers))


def select_graph_buckets(
    rows: Iterable[Mapping[str, Any]],
    *,
    coverage: float = 0.95,
    max_bucket: int | None = None,
) -> dict[str, Any]:
    """Choose capture sizes from aggregated graph statistics.

    ``count`` is the number of identical scheduler observations. Capture sizes
    are padded-token sizes, while padding cost is counted in token occurrences.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    counts: dict[int, int] = defaultdict(int)
    padding_costs: dict[int, int] = defaultdict(int)
    complete = True
    for row in rows:
        try:
            actual = int(row["num_unpadded_tokens"])
            padded = int(row["num_padded_tokens"])
            count = int(row["count"])
            mode = str(row["runtime_mode"])
        except (KeyError, TypeError, ValueError):
            complete = False
            continue
        if actual <= 0 or padded < actual or count <= 0:
            complete = False
            continue
        if mode.lower() == "none":
            continue
        if max_bucket is not None and padded > max_bucket:
            continue
        counts[padded] += count
        padding_costs[padded] += count * (padded - actual)
    if not counts:
        fallback = [
            bucket
            for bucket in (1, 8, 32)
            if max_bucket is None or bucket <= max_bucket
        ]
        return {
            "status": "FALLBACK",
            "buckets": fallback,
            "coverage": 0.0,
            "padding": None,
        }

    total = sum(counts.values())
    target = math.ceil(total * coverage)
    chosen: list[int] = []
    covered = 0
    for bucket in sorted(
        counts, key=lambda value: (-counts[value], padding_costs[value], value)
    ):
        if covered >= target:
            break
        chosen.append(bucket)
        covered += counts[bucket]
    if 1 in counts and 1 not in chosen:
        chosen.append(1)
    chosen.sort()
    return {
        "status": "DERIVED" if complete else "PARTIAL",
        "buckets": chosen,
        "coverage": covered / total,
        "padding": sum(padding_costs[bucket] for bucket in chosen),
    }


def capacity_frontier(cells: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the largest concurrency meeting the preemption-free capacity gate."""
    accepted: list[Mapping[str, Any]] = []
    for cell in cells:
        aggregate = cell.get("aggregate", {})
        metrics = aggregate.get("metrics", {})
        concurrency = cell.get("workload", {}).get("concurrency")
        if not isinstance(concurrency, int):
            continue
        if (
            aggregate.get("failed_requests", 0) == 0
            and metrics.get("preemptions", 0) == 0
            and metrics.get("kv_cache_usage_perc", 0) < 100
            and aggregate.get("emitted_output_tokens", 0)
            >= aggregate.get("expected_output_tokens", 0)
        ):
            accepted.append(cell)
    if not accepted:
        return {"status": "NO_FRONTIER", "concurrency": None}
    winner = max(accepted, key=lambda cell: cell["workload"]["concurrency"])
    return {
        "status": "PASS",
        "concurrency": winner["workload"]["concurrency"],
        "cell_id": winner.get("identity", {}).get("cell_id"),
    }


def compare_cells(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply generic throughput/latency/HBM promotion gates to aggregate data."""
    base = baseline["aggregate"]
    current = candidate["aggregate"]
    deltas = {
        "output_throughput": percent_delta(
            current["output_throughput"], base["output_throughput"]
        ),
        "p99_tpot_ms": percent_delta(current["p99_tpot_ms"], base["p99_tpot_ms"]),
        "p99_ttft_ms": percent_delta(current["p99_ttft_ms"], base["p99_ttft_ms"]),
    }
    if "hbm_bytes_per_output_token" in base and "hbm_bytes_per_output_token" in current:
        deltas["hbm_bytes_per_output_token"] = percent_delta(
            current["hbm_bytes_per_output_token"], base["hbm_bytes_per_output_token"]
        )
    regressions = [
        metric
        for metric, delta in deltas.items()
        if delta is not None
        and (
            (metric == "output_throughput" and delta < THROUGHPUT_REGRESSION_PCT)
            or (
                metric != "output_throughput"
                and delta
                > (
                    HBM_REGRESSION_PCT
                    if metric.startswith("hbm_")
                    else LATENCY_REGRESSION_PCT
                )
            )
        )
    ]
    improvement = (
        deltas["output_throughput"] is not None
        and deltas["output_throughput"] >= IMPROVEMENT_PCT
    )
    return {
        "deltas": deltas,
        "verdict": "REGRESSION"
        if regressions
        else "IMPROVEMENT"
        if improvement
        else "PASS",
        "regressions": regressions,
    }


def resolved_configuration_digest(configuration: Mapping[str, Any]) -> str:
    return sha256_json(configuration)
