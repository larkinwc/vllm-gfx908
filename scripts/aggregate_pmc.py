#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Aggregate rocprofv3 counter_collection.csv into omniperf-equivalent
roofline numbers for the given hot kernel.

rocprofv3 counter CSV layout:
    "Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id",
    "Thread_Id","Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size",
    "LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count",
    "SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"

So one row PER counter PER kernel call. We aggregate by Counter_Name
across the matching Kernel_Name.

Peaks (gfx908 / MI100):
  * FP16/INT8 MFMA peak: 184.6 TFLOPS / 184.6 TOPS
  * HBM2 peak: 1228.8 GB/s
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PEAK_FP16_INT8_TFLOPS = 184.6
PEAK_HBM_GB_S = 1228.8


def aggregate(csv_path: Path, kernel_substring: str) -> dict:
    counters: dict[str, float] = defaultdict(float)
    durations: dict[int, int] = {}  # dispatch_id -> dur_ns
    n_calls_seen: set[int] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kn = row.get("Kernel_Name", "")
            if kernel_substring not in kn:
                continue
            try:
                dispatch_id = int(row["Dispatch_Id"])
            except (ValueError, KeyError):
                continue
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                durations[dispatch_id] = end - start
            except (ValueError, KeyError):
                pass
            n_calls_seen.add(dispatch_id)
            cname = row.get("Counter_Name", "")
            try:
                cval = float(row.get("Counter_Value", 0))
            except ValueError:
                continue
            counters[cname] += cval
    return {
        "n_calls": len(n_calls_seen),
        "total_dur_ns": sum(durations.values()),
        "counters": dict(counters),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counter-csv", action="append", required=True,
                    help="path to counter_collection.csv (may repeat)")
    ap.add_argument("--kernel-substring", required=True,
                    help="substring of kernel name to filter on, e.g. scaled_mm_kernel")
    ap.add_argument("--quant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    merged_counters: dict[str, float] = defaultdict(float)
    n_calls = 0
    total_dur_ns = 0
    for p in args.counter_csv:
        agg = aggregate(Path(p), args.kernel_substring)
        for k, v in agg["counters"].items():
            merged_counters[k] += v
        n_calls += agg["n_calls"]
        total_dur_ns += agg["total_dur_ns"]

    print(f"matched {n_calls} kernel-call dispatch_ids, total_dur_ns={total_dur_ns}")
    print(json.dumps(dict(merged_counters), indent=2))

    ach_tflops = None
    ach_hbm = None
    if total_dur_ns > 0:
        # SQ_INSTS_VALU is one instruction per wavefront. Each
        # wavefront is 64 lanes -> "ops" measured here is VALU ops.
        # NOTE: this is a coarse compute proxy; MFMA-specific counters
        # (SQ_INSTS_MFMA_*) are the precise numerator for INT8 TFLOPS,
        # but rocprofv3 1.2 on gfx908 only exposes SQ_INSTS_VALU as a
        # cross-platform compute counter.
        sq_valu = merged_counters.get("SQ_INSTS_VALU", 0)
        if sq_valu > 0:
            ach_tflops = (sq_valu * 64) / (total_dur_ns / 1e9) / 1e12

        # HBM bytes via TCC requests (each request = 64 B cache line).
        tcc_rd = merged_counters.get("TCP_TCC_READ_REQ_sum", 0)
        tcc_wr = merged_counters.get("TCP_TCC_WRITE_REQ_sum", 0)
        if tcc_rd or tcc_wr:
            bytes_total = (tcc_rd + tcc_wr) * 64
            ach_hbm = bytes_total / (total_dur_ns / 1e9) / 1e9

    summary = {
        "status": "ok" if (ach_tflops or ach_hbm) else "counters_empty",
        "hot_kernel": args.kernel_substring,
        "quant": args.quant,
        "n_calls": n_calls,
        "total_kernel_ns": total_dur_ns,
        "achieved_TFLOPs_VALU_proxy": ach_tflops,
        "peak_TFLOPs_gfx908_fp16_int8": PEAK_FP16_INT8_TFLOPS,
        "percent_of_peak_compute": (
            100.0 * ach_tflops / PEAK_FP16_INT8_TFLOPS if ach_tflops else None
        ),
        "achieved_HBM_GB_s": ach_hbm,
        "peak_HBM_GB_s_gfx908": PEAK_HBM_GB_S,
        "percent_of_peak_hbm": (
            100.0 * ach_hbm / PEAK_HBM_GB_S if ach_hbm else None
        ),
        "raw_counters": dict(merged_counters),
        "input_files": list(args.counter_csv),
        "note": (
            "VALU proxy underestimates MFMA throughput because gfx908 "
            "exposes wave-level SQ_INSTS_VALU only. Use this %peak as a "
            "lower bound; actual int8/fp16 MFMA utilization is computed "
            "from kernel-trace timing in BENCH_INT8_W4A16_BASELINE.md. "
            "TCP_TCC_*_REQ_sum gives a tight HBM bandwidth measurement."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "raw_counters"}, indent=2))


if __name__ == "__main__":
    main()
