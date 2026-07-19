# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from scripts.gfx900.analysis import (
    capacity_frontier,
    compare_cells,
    percent_delta,
    select_graph_buckets,
)


def test_percent_delta_zero_baseline_is_not_a_number() -> None:
    assert percent_delta(1, 0) is None


def test_graph_buckets_use_count_not_padding() -> None:
    result = select_graph_buckets(
        [
            {
                "num_unpadded_tokens": 1,
                "num_padded_tokens": 1,
                "num_paddings": 0,
                "count": 100,
                "runtime_mode": "FULL",
            },
            {
                "num_unpadded_tokens": 7,
                "num_padded_tokens": 8,
                "num_paddings": 1,
                "count": 1,
                "runtime_mode": "FULL",
            },
        ],
        coverage=0.95,
    )
    assert result == {
        "status": "DERIVED",
        "buckets": [1],
        "coverage": 100 / 101,
        "padding": 0,
    }


def test_graph_buckets_fall_back_for_incomplete_rows() -> None:
    assert select_graph_buckets([{"runtime_mode": "FULL"}], max_bucket=16)[
        "buckets"
    ] == [1, 8]


def test_capacity_frontier_requires_all_capacity_conditions() -> None:
    cells = [
        {
            "identity": {"cell_id": "c4"},
            "workload": {"concurrency": 4},
            "aggregate": {
                "failed_requests": 0,
                "emitted_output_tokens": 40,
                "expected_output_tokens": 40,
                "metrics": {"preemptions": 0, "kv_cache_usage_perc": 80},
            },
        },
        {
            "identity": {"cell_id": "c8"},
            "workload": {"concurrency": 8},
            "aggregate": {
                "failed_requests": 0,
                "emitted_output_tokens": 80,
                "expected_output_tokens": 80,
                "metrics": {"preemptions": 1, "kv_cache_usage_perc": 95},
            },
        },
    ]
    assert capacity_frontier(cells) == {
        "status": "PASS",
        "concurrency": 4,
        "cell_id": "c4",
    }


def test_compare_cells_reports_latency_regression() -> None:
    baseline = {
        "aggregate": {"output_throughput": 100, "p99_tpot_ms": 10, "p99_ttft_ms": 20}
    }
    candidate = {
        "aggregate": {"output_throughput": 102, "p99_tpot_ms": 10.3, "p99_ttft_ms": 20}
    }
    assert compare_cells(baseline, candidate)["verdict"] == "REGRESSION"
