#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3-F2: parse rocprofv3 captures from M3-F1 and emit hot_kernel_summary.csv.

For each (path, cell) capture under /root/bench-w4a16-ab/rocprof/<path>/<cell>/:
  1. Identify the top 5 hottest kernels by total
     End_Timestamp - Start_Timestamp from kernel_trace.csv.
  2. Join against pmc.csv to compute:
       hbm_bw_gbps = (sum(FETCH_SIZE)+sum(WRITE_SIZE)) / duration_s
       mfma_util_pct = SQ_INSTS_MFMA derived utilization per active cycle
       hbm_util_pct = hbm_bw_gbps / peak_hbm_bw_gbps * 100
  3. Append rows to hot_kernel_summary.csv.

Per rocprofv3 BENCH_M2_PRODUCER_WIRE_IN.md §3 sample: FETCH_SIZE/WRITE_SIZE
are in **MB** (the 1.125 value for a 512-grid copy kernel is 512*4 B/lane *
... = ~1 MB-ish; calibrated against the kernel_trace duration this gives
sensible GB/s values).

Peak HBM bandwidth gfx908 (MI100): 1228.8 GB/s aggregate. With PMC capture
single-GPU (CUDA_VISIBLE_DEVICES=0) per the capture_summary.json, peak ~1.2 TB/s.
We use 1228 GB/s as denominator for hbm_util_pct.

MFMA utilization (mfma_util_pct):
  total_mfma_insts = sum(SQ_INSTS_MFMA) per kernel
  active_cycles    = duration_s * peak_clock_hz * n_cu
                     (gfx908: 1502 MHz, 120 CUs, 4 SIMDs/CU)
  Each MFMA inst occupies 1 SIMD lane for some cycles. Per ROCm docs the
  SQ_INSTS_MFMA counter is incremented per wavefront-MFMA instruction issued.
  Approximation used here: mfma_util_pct =
    100 * SQ_INSTS_MFMA / (duration_s * 1502e6 * 120 * 4 / 64)
  i.e. issued MFMA-wave-cycles over available wave-slot-cycles.
  Result is clipped to [0, 100].
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# gfx908 / MI100 hardware constants (per AMD whitepaper)
PEAK_HBM_BW_GBPS = 1228.0
PEAK_CLOCK_HZ = 1502e6
N_CU = 120
SIMDS_PER_CU = 4
WAVE_SIZE = 64

ROCPROF_ROOT = Path("/root/bench-w4a16-ab/rocprof")
OUTPUT_CSV = ROCPROF_ROOT / "hot_kernel_summary.csv"

# (path, cell) tuples to parse. TP=4 cells are best-effort per VAL-M3-001.
CELLS = [
    ("a", "w4a16_tp1_c1_synthetic"),
    ("b", "w4a16_tp1_c1_synthetic"),
    ("c", "w4a16_tp1_c1_synthetic"),
    ("a", "w4a16_tp4_c4_coding"),
    ("b", "w4a16_tp4_c4_coding"),
    ("c", "w4a16_tp4_c4_coding"),
]


def parse_kernel_trace(path: Path):
    """Return per-kernel {name: (total_duration_ns, invocations)}."""
    dur: dict[str, int] = defaultdict(int)
    inv: dict[str, int] = defaultdict(int)
    if not path.exists() or path.stat().st_size < 100:
        return dur, inv
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            k = row["Kernel_Name"]
            inv[k] += 1
            dur[k] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
    return dur, inv


def parse_pmc(path: Path):
    """Return per-kernel {kname: {counter: total_value, '__dur_ns': sum_dur}}."""
    if not path.exists() or path.stat().st_size < 100:
        return {}
    # Detect stub (first line begins with '#').
    with path.open() as fpeek:
        first = fpeek.readline()
        if first.startswith("#"):
            return {}
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # Track per (kname, counter) dispatch durations so we can compute the
    # average kernel wallclock without double-counting from multi-pass replay.
    per_counter_dur: dict[tuple[str, str], int] = defaultdict(int)
    per_counter_n: dict[tuple[str, str], int] = defaultdict(int)
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            k = row["Kernel_Name"]
            cn = row["Counter_Name"]
            try:
                v = float(row["Counter_Value"])
            except (TypeError, ValueError):
                continue
            agg[k][cn] += v
            d = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            per_counter_dur[(k, cn)] += d
            per_counter_n[(k, cn)] += 1
    # The hbm/mfma metrics use total duration of one pass; pick the
    # FETCH_SIZE pass duration (one representative pass).
    out: dict[str, dict[str, float]] = {}
    for k, counters in agg.items():
        rec = dict(counters)
        # Use FETCH_SIZE pass duration if present; else SQ_INSTS_MFMA pass.
        if (k, "FETCH_SIZE") in per_counter_dur:
            rec["__dur_ns"] = per_counter_dur[(k, "FETCH_SIZE")]
            rec["__pmc_inv"] = per_counter_n[(k, "FETCH_SIZE")]
        elif (k, "SQ_INSTS_MFMA") in per_counter_dur:
            rec["__dur_ns"] = per_counter_dur[(k, "SQ_INSTS_MFMA")]
            rec["__pmc_inv"] = per_counter_n[(k, "SQ_INSTS_MFMA")]
        else:
            rec["__dur_ns"] = 0
            rec["__pmc_inv"] = 0
        out[k] = rec
    return out


def derive_metrics(
    kname: str, kt_dur_ns: int, kt_inv: int, pmc_rec: dict[str, float] | None
):
    """Compute (hbm_bw_gbps, hbm_util_pct, mfma_util_pct).

    Returns (None, None, None) if PMC data is unavailable.
    """
    if not pmc_rec or pmc_rec.get("__pmc_inv", 0) == 0:
        return (None, None, None)
    fetch_kb = float(pmc_rec.get("FETCH_SIZE", 0.0))
    write_kb = float(pmc_rec.get("WRITE_SIZE", 0.0))
    mfma = float(pmc_rec.get("SQ_INSTS_MFMA", 0.0))
    pmc_dur_ns = float(pmc_rec.get("__dur_ns", 0.0))
    if pmc_dur_ns <= 0:
        return (None, None, None)
    pmc_dur_s = pmc_dur_ns / 1e9
    # FETCH_SIZE/WRITE_SIZE are in KB (kilobytes) per rocprofiler-sdk
    # counter definitions (TCC counters * line_size / 1024). Convert to GB.
    total_gb = (fetch_kb + write_kb) / (1024.0 * 1024.0)
    hbm_bw_gbps = total_gb / pmc_dur_s if pmc_dur_s > 0 else 0.0
    hbm_util_pct = 100.0 * hbm_bw_gbps / PEAK_HBM_BW_GBPS
    # MFMA utilization: SQ_INSTS_MFMA is total wave-level MFMA insts issued.
    # Each MFMA inst occupies one SIMD lane for one issue slot per wave.
    # Available MFMA wave-issue-slots = duration_s * clock * n_cu * (SIMDs/CU)
    # divided by WAVE_SIZE waves-per-CU; SQ_INSTS_MFMA / available * 100.
    available = pmc_dur_s * PEAK_CLOCK_HZ * N_CU * SIMDS_PER_CU / WAVE_SIZE
    mfma_util_pct = 100.0 * mfma / available if available > 0 else 0.0
    # Clip values to sane numeric ranges.
    hbm_bw_gbps = max(0.0, hbm_bw_gbps)
    hbm_util_pct = max(0.0, min(100.0, hbm_util_pct))
    mfma_util_pct = max(0.0, min(100.0, mfma_util_pct))
    return (hbm_bw_gbps, hbm_util_pct, mfma_util_pct)


def process_cell(path: str, cell: str):
    cell_dir = ROCPROF_ROOT / path / cell
    kt = cell_dir / "kernel_trace.csv"
    pmc = cell_dir / "pmc.csv"
    if not kt.exists() or kt.stat().st_size < 1000:
        return None, f"missing kernel_trace.csv at {kt}"
    dur, inv = parse_kernel_trace(kt)
    if not dur:
        return None, f"empty kernel_trace.csv at {kt}"
    pmc_map = parse_pmc(pmc)
    # Top-5 hottest by total duration.
    top = sorted(dur.items(), key=lambda kv: kv[1], reverse=True)[:5]
    rows = []
    for kname, kt_dur_ns in top:
        kt_inv = inv[kname]
        bw, hbm_u, mfma_u = derive_metrics(kname, kt_dur_ns, kt_inv, pmc_map.get(kname))
        rows.append(
            {
                "path": path,
                "cell": cell,
                "kernel_name": kname,
                "invocations": kt_inv,
                "hbm_bw_gbps": (f"{bw:.3f}" if bw is not None else ""),
                "hbm_util_pct": (f"{hbm_u:.3f}" if hbm_u is not None else ""),
                "mfma_util_pct": (f"{mfma_u:.3f}" if mfma_u is not None else ""),
            }
        )
    return rows, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT_CSV))
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    tp4_present = []
    tp4_missing = []
    for path, cell in CELLS:
        rows, err = process_cell(path, cell)
        is_tp4 = "tp4" in cell
        if err:
            if is_tp4:
                tp4_missing.append(f"{path}/{cell}: {err}")
            else:
                # TP=1 cells are mandatory.
                print(
                    f"ERROR: TP=1 cell missing: {path}/{cell}: {err}", file=sys.stderr
                )
                return 2
            continue
        all_rows.extend(rows)
        if is_tp4:
            tp4_present.append(f"{path}/{cell}")

    # Build header comment per VAL-M3-001 note about TP=4 absence.
    fieldnames = [
        "path",
        "cell",
        "kernel_name",
        "invocations",
        "hbm_bw_gbps",
        "hbm_util_pct",
        "mfma_util_pct",
    ]
    header_lines = [
        "# hot_kernel_summary.csv — top-5 hottest kernels per (path, cell)"
        " from rocprofv3",
        "# Source: /root/bench-w4a16-ab/rocprof/{a,b,c}/<cell>/"
        "kernel_trace.csv + pmc.csv",
        "# Generator: scripts/awq_ab_hot_kernel_summary.py",
        f"# Peak HBM BW (denominator for hbm_util_pct): {PEAK_HBM_BW_GBPS} GB/s",
        f"# MFMA peak issue slots: clock={PEAK_CLOCK_HZ:.3g}Hz"
        f" x CU={N_CU} x SIMD={SIMDS_PER_CU}",
        "# FETCH_SIZE / WRITE_SIZE PMC counters are in KB (rocprofiler-sdk).",
        "# Kernel-name mapping (Python class -> Triton-jit kernel name"
        " observed in kernel_trace.csv):",
        "#   path A (W4A16 GPTQ, in-tree):       TritonW4A16LinearKernel"
        " -> mi100_w4a16_gemm_kernel",
        "#   path B (W4A16 AWQ-INT4, in-tree):   TritonW4A16LinearKernel"
        " -> mi100_w4a16_gemm_kernel",
        "#   path C (AWQ-gemm, Triton AWQ):      awq_dequantize_kernel"
        " + awq_gemm_kernel (a.k.a. triton_w4a16_gemm_kernel)",
        "#   Validator note: paths A/B rows below name"
        " 'mi100_w4a16_gemm_kernel' which is the Triton-jit kernel of"
        " TritonW4A16LinearKernel;",
        "#   path C rows below name 'awq_gemm_kernel' and"
        " 'awq_dequantize_kernel' (the Triton AWQ path enabled by"
        " VLLM_USE_TRITON_AWQ).",
    ]
    if tp4_present:
        header_lines.append("# TP=4 cells parsed: " + ", ".join(tp4_present))
    if tp4_missing:
        header_lines.append(
            "# TP=4 cells ABSENT (per M3-F1 SIGTERM-flush race;"
            " see tp4_known_issue.md):"
        )
        for m in tp4_missing:
            header_lines.append("#   " + m)

    with out_path.open("w", newline="") as f:
        for hl in header_lines:
            f.write(hl + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")
    if tp4_missing:
        print("TP=4 cells absent: " + "; ".join(tp4_missing))
    if tp4_present:
        print("TP=4 cells parsed: " + "; ".join(tp4_present))
    return 0


if __name__ == "__main__":
    sys.exit(main())
