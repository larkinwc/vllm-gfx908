#!/opt/vllm-env/bin/python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GEMM Kernel Profiling for MI100 (gfx908) using rocprofv3

Profiles vLLM inference to identify GEMM bottlenecks and measure MFMA utilization.
Implements the agentic profiling loop from
ml-research/kernel-tuning/agentic-profiling-loop.md.

Phases:
  1. TRIAGE:  kernel-trace summary to identify hottest kernels
  2. DEEP:    PMC counter profiling on top-N GEMM kernels
  3. ANALYZE: Compute BW utilization, MFMA utilization, classify bottlenecks

Usage:
    # Full profiling pipeline
    python profile_gemm_kernels.py --output-dir /root/gemm-profile-results

    # Triage only (fast, low overhead)
    python profile_gemm_kernels.py --mode triage --output-dir /root/gemm-profile-results

    # Deep profile specific kernel
    python profile_gemm_kernels.py --mode deep --kernel-regex ".*gemm.*" \\
        --output-dir /root/gemm-profile-results

Environment:
    Requires rocprofv3 in PATH
    Requires vLLM server running on localhost:8000
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import regex as re

VLLM_PYTHON = "/opt/vllm-env/bin/python3"
ROCPROFV3 = "/opt/rocm/bin/rocprofv3"
DEFAULT_OUTPUT_DIR = "/root/gemm-profile-results"
SERVER_URL = "http://localhost:8000"
DEFAULT_MODEL = "/models/Qwen3.5-9B"

MI100_PEAK_BW_BYTES_PER_SEC = 1.23e12  # 1.23 TB/s HBM2
MI100_PEAK_FLOPS_FP16 = 184.6e12  # 184.6 TFLOPS FP16 (with MFMA)
MI100_NUM_CUS = 120


def check_server_health():
    try:
        import urllib.request

        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def send_inference_request(
    prompt="Write a Python function to sort a list.", max_tokens=128
):
    import urllib.request

    payload = json.dumps(
        {
            "model": os.path.basename(DEFAULT_MODEL),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
    ).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def create_profiling_workload_script(output_dir, num_requests=5):
    """Create a small Python script that sends requests to exercise GEMM paths."""
    script = f"""#!/opt/vllm-env/bin/python3
import json, urllib.request, time, sys

SERVER_URL = "{SERVER_URL}"
MODEL = "{os.path.basename(DEFAULT_MODEL)}"

prompts = [
    "Write a function to reverse a linked list in Python.",
    "Explain the time complexity of merge sort.",
    "Create a REST API endpoint for user authentication.",
    "Implement a binary search algorithm in C++.",
    "Write a SQL query with multiple joins and aggregation.",
]

print(f"Sending {{len(prompts[:{{num_requests}}])}} requests to exercise GEMM paths...")
for i, prompt in enumerate(prompts[:{num_requests}]):
    payload = json.dumps({{
        "model": MODEL,
        "messages": [{{"role": "user", "content": prompt}}],
        "max_tokens": 128,
        "temperature": 0.7,
    }}).encode()
    req = urllib.request.Request(
        f"{{SERVER_URL}}/v1/chat/completions",
        data=payload,
        headers={{"Content-Type": "application/json"}},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
            tokens = data.get("usage", {{}}).get("completion_tokens", 0)
            print(f"  [{{i+1}}] {{tokens}} tokens generated")
    except Exception as e:
        print(f"  [{{i+1}}] ERROR: {{e}}", file=sys.stderr)
print("Workload complete.")
"""
    script_path = os.path.join(output_dir, "_profiling_workload.py")
    os.makedirs(output_dir, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def run_kernel_trace_triage(output_dir):
    """
    Phase 1: Fast kernel trace to identify hottest kernels.
    Low overhead (~1-5%). Captures kernel names and durations.
    """
    print("=" * 60)
    print("Phase 1: TRIAGE - Kernel Trace Summary")
    print("=" * 60)
    print()

    if not check_server_health():
        print("ERROR: Server not running. Start vLLM first.")
        sys.exit(1)

    workload_script = create_profiling_workload_script(output_dir)
    trace_dir = os.path.join(output_dir, "kernel_trace")
    os.makedirs(trace_dir, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/rocm/core-7.12/lib"

    cmd = [
        ROCPROFV3,
        "--kernel-trace",
        "--output-format",
        "csv",
        "--output-directory",
        trace_dir,
        "--",
        VLLM_PYTHON,
        workload_script,
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])

    # Parse kernel trace CSV
    return parse_kernel_trace(trace_dir)


def parse_kernel_trace(trace_dir):
    """Parse rocprofv3 kernel trace CSV and rank kernels by duration."""
    csv_files = list(Path(trace_dir).glob("**/*kernel_trace*.csv"))
    if not csv_files:
        csv_files = list(Path(trace_dir).glob("**/*.csv"))

    if not csv_files:
        print("WARNING: No kernel trace CSV files found in", trace_dir)
        return []

    kernels = {}
    for csv_file in csv_files:
        print(f"Parsing: {csv_file}")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get(
                    "Kernel_Name", row.get("kernel_name", row.get("Name", ""))
                )
                duration_ns = int(
                    row.get(
                        "Duration_ns", row.get("duration", row.get("DurationNs", 0))
                    )
                )
                if name:
                    if name not in kernels:
                        kernels[name] = {
                            "name": name,
                            "count": 0,
                            "total_ns": 0,
                            "durations": [],
                        }
                    kernels[name]["count"] += 1
                    kernels[name]["total_ns"] += duration_ns
                    kernels[name]["durations"].append(duration_ns)

    # Sort by total time
    ranked = sorted(kernels.values(), key=lambda k: k["total_ns"], reverse=True)

    # Compute stats
    total_time = sum(k["total_ns"] for k in ranked)
    print(f"\nTotal kernel time: {total_time / 1e6:.1f} ms")
    print(f"Total unique kernels: {len(ranked)}")
    print()

    print(
        f"{'Rank':<5} {'%Time':>6} {'Count':>7}"
        f" {'Avg(us)':>10} {'Total(ms)':>10} {'Kernel Name'}"
    )
    print("-" * 100)

    gemm_kernels = []
    for i, k in enumerate(ranked[:30]):
        pct = 100.0 * k["total_ns"] / total_time if total_time > 0 else 0
        avg_us = (k["total_ns"] / k["count"]) / 1000.0
        total_ms = k["total_ns"] / 1e6
        short_name = k["name"][:60]
        is_gemm = any(
            g in k["name"].lower()
            for g in [
                "gemm",
                "gemv",
                "rocblas",
                "hipblas",
                "mfma",
                "matmul",
                "linear",
                "sgemm",
                "hgemm",
                "batched",
            ]
        )

        marker = " [GEMM]" if is_gemm else ""
        print(
            f"{i + 1:<5} {pct:>5.1f}% {k['count']:>7}"
            f" {avg_us:>10.1f} {total_ms:>10.2f} {short_name}{marker}"
        )

        if is_gemm:
            k["pct_time"] = pct
            k["avg_us"] = avg_us
            gemm_kernels.append(k)

    print()
    print(f"GEMM kernels found: {len(gemm_kernels)}")
    gemm_total_pct = sum(k["pct_time"] for k in gemm_kernels)
    print(f"GEMM % of total kernel time: {gemm_total_pct:.1f}%")

    return ranked, gemm_kernels


def run_deep_profile(output_dir, kernel_regex=None, top_n=3, ranked_kernels=None):
    """
    Phase 2: Deep PMC counter profiling on top GEMM kernels.
    Measures MFMA utilization, memory bandwidth, L2 hit rate.
    """
    print()
    print("=" * 60)
    print("Phase 2: DEEP PROFILE - PMC Counters")
    print("=" * 60)
    print()

    if not check_server_health():
        print("ERROR: Server not running.")
        sys.exit(1)

    workload_script = create_profiling_workload_script(output_dir, num_requests=3)
    deep_dir = os.path.join(output_dir, "deep_profile")
    os.makedirs(deep_dir, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/rocm/core-7.12/lib"

    # Build kernel filter
    if kernel_regex:
        kernel_filter = ["--kernel-include-regex", kernel_regex]
    elif ranked_kernels:
        # Filter to top-N kernels
        names = [k["name"] for k in ranked_kernels[:top_n]]
        regex = "|".join(re.escape(n) for n in names)
        kernel_filter = ["--kernel-include-regex", regex]
    else:
        kernel_filter = [
            "--kernel-include-regex",
            ".*gemm.*|.*Gemm.*|.*GEMM.*|.*rocblas.*|.*hipblas.*|.*matmul.*",
        ]

    # Memory counters pass
    print("Pass 1: Memory counters (FETCH_SIZE, WRITE_SIZE, TCC_HIT, TCC_MISS)...")
    mem_dir = os.path.join(deep_dir, "memory")
    os.makedirs(mem_dir, exist_ok=True)

    cmd_mem = [
        ROCPROFV3,
        "--pmc",
        "FETCH_SIZE,WRITE_SIZE,TCC_HIT_sum,TCC_MISS_sum",
        *kernel_filter,
        "--output-format",
        "csv",
        "--output-directory",
        mem_dir,
        "--",
        VLLM_PYTHON,
        workload_script,
    ]

    result = subprocess.run(
        cmd_mem, env=env, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"  Memory counters pass failed: {result.stderr[:500]}")
        # Try without TCC counters (may not be available on all configurations)
        print("  Retrying with FETCH_SIZE,WRITE_SIZE only...")
        cmd_mem_fallback = [
            ROCPROFV3,
            "--pmc",
            "FETCH_SIZE,WRITE_SIZE",
            *kernel_filter,
            "--output-format",
            "csv",
            "--output-directory",
            mem_dir,
            "--",
            VLLM_PYTHON,
            workload_script,
        ]
        result = subprocess.run(
            cmd_mem_fallback, env=env, capture_output=True, text=True, timeout=600
        )
    print("  Done." if result.returncode == 0 else f"  Failed: {result.stderr[:300]}")

    # Compute counters pass
    print("Pass 2: Compute counters (SQ_WAVES, MFMA cycles, VALU)...")
    compute_dir = os.path.join(deep_dir, "compute")
    os.makedirs(compute_dir, exist_ok=True)

    cmd_compute = [
        ROCPROFV3,
        "--pmc",
        "SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_MFMA",
        *kernel_filter,
        "--output-format",
        "csv",
        "--output-directory",
        compute_dir,
        "--",
        VLLM_PYTHON,
        workload_script,
    ]

    result = subprocess.run(
        cmd_compute, env=env, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"  Compute counters pass failed: {result.stderr[:500]}")
        # Fallback to just wave count
        print("  Retrying with SQ_WAVES only...")
        cmd_waves = [
            ROCPROFV3,
            "--pmc",
            "SQ_WAVES",
            *kernel_filter,
            "--output-format",
            "csv",
            "--output-directory",
            compute_dir,
            "--",
            VLLM_PYTHON,
            workload_script,
        ]
        result = subprocess.run(
            cmd_waves, env=env, capture_output=True, text=True, timeout=600
        )
    print("  Done." if result.returncode == 0 else f"  Failed: {result.stderr[:300]}")

    return parse_deep_profile(deep_dir)


def parse_deep_profile(deep_dir):
    """Parse PMC counter results and compute utilization metrics."""
    results = {}

    for subdir in ["memory", "compute"]:
        path = os.path.join(deep_dir, subdir)
        csv_files = list(Path(path).glob("**/*.csv"))
        for csv_file in csv_files:
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("Kernel_Name", row.get("kernel_name", ""))
                        if not name:
                            continue
                        if name not in results:
                            results[name] = {"name": name}
                        results[name].update(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ("Kernel_Name", "kernel_name")
                            }
                        )
            except Exception as e:
                print(f"  Warning: Could not parse {csv_file}: {e}")

    if not results:
        print("WARNING: No PMC counter data found.")
        return {}

    print()
    print("GEMM Kernel Analysis:")
    print("=" * 80)

    for name, data in results.items():
        print(f"\nKernel: {name[:80]}")

        # Memory bandwidth
        fetch = float(data.get("FETCH_SIZE", 0))
        write = float(data.get("WRITE_SIZE", 0))
        duration_ns = float(data.get("Duration_ns", data.get("duration", 1)))
        duration_s = duration_ns / 1e9

        if duration_s > 0 and (fetch + write) > 0:
            # 32 bytes per fetch unit
            bw_bytes_per_sec = (fetch + write) * 32 / duration_s
            bw_util = bw_bytes_per_sec / MI100_PEAK_BW_BYTES_PER_SEC * 100
            print(
                f"  Memory BW: {bw_bytes_per_sec / 1e9:.1f} GB/s"
                f" ({bw_util:.1f}% of peak)"
            )

        # L2 cache
        l2_hit = float(data.get("TCC_HIT_sum", 0))
        l2_miss = float(data.get("TCC_MISS_sum", 0))
        if l2_hit + l2_miss > 0:
            l2_rate = l2_hit / (l2_hit + l2_miss) * 100
            print(f"  L2 Hit Rate: {l2_rate:.1f}%")

        # MFMA utilization
        sq_waves = float(data.get("SQ_WAVES", 0))
        mfma_insts = float(data.get("SQ_INSTS_MFMA", 0))
        valu_insts = float(data.get("SQ_INSTS_VALU", 0))

        if mfma_insts > 0 or valu_insts > 0:
            total_insts = mfma_insts + valu_insts
            mfma_pct = mfma_insts / total_insts * 100 if total_insts > 0 else 0
            print(
                f"  MFMA Instructions: {mfma_insts:.0f} ({mfma_pct:.1f}% of VALU+MFMA)"
            )
            print(f"  VALU Instructions: {valu_insts:.0f}")
            print(f"  Wavefronts: {sq_waves:.0f}")

        # Bottleneck classification
        if duration_s > 0:
            classify_bottleneck(data)

    return results


def classify_bottleneck(data):
    """Classify kernel bottleneck based on PMC counters."""
    fetch = float(data.get("FETCH_SIZE", 0))
    write = float(data.get("WRITE_SIZE", 0))
    duration_ns = float(data.get("Duration_ns", data.get("duration", 1)))
    mfma_insts = float(data.get("SQ_INSTS_MFMA", 0))
    valu_insts = float(data.get("SQ_INSTS_VALU", 0))

    duration_s = duration_ns / 1e9
    bw_bytes_per_sec = (fetch + write) * 32 / duration_s if duration_s > 0 else 0
    bw_util = (
        bw_bytes_per_sec / MI100_PEAK_BW_BYTES_PER_SEC
        if MI100_PEAK_BW_BYTES_PER_SEC > 0
        else 0
    )

    total_insts = mfma_insts + valu_insts
    mfma_pct = mfma_insts / total_insts if total_insts > 0 else 0

    if bw_util > 0.6 and mfma_pct < 0.3:
        print(
            f"  Bottleneck: MEMORY-BOUND"
            f" (BW {bw_util * 100:.0f}%, MFMA {mfma_pct * 100:.0f}%)"
        )
        print(
            "  Recommendation: Already memory-bound."
            " Consider quantization or KV compression."
        )
    elif mfma_pct > 0.5 and bw_util < 0.3:
        print(
            f"  Bottleneck: COMPUTE-BOUND"
            f" (BW {bw_util * 100:.0f}%, MFMA {mfma_pct * 100:.0f}%)"
        )
        print("  Recommendation: Adjust tile sizes or use larger MFMA variants.")
    elif mfma_pct < 0.3 and bw_util < 0.3:
        print(
            f"  Bottleneck: LATENCY-BOUND"
            f" (BW {bw_util * 100:.0f}%, MFMA {mfma_pct * 100:.0f}%)"
        )
        print(
            "  Recommendation: Low utilization."
            " Check occupancy, launch overhead, sync stalls."
        )
    else:
        print(
            f"  Bottleneck: BALANCED"
            f" (BW {bw_util * 100:.0f}%, MFMA {mfma_pct * 100:.0f}%)"
        )


def generate_report(output_dir, triage_results=None, deep_results=None):
    """Generate a summary report of profiling findings."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "hardware": "4x AMD Instinct MI100 (gfx908)",
        "peak_memory_bw": "1.23 TB/s",
        "peak_fp16_flops": "184.6 TFLOPS",
        "num_cus": MI100_NUM_CUS,
    }

    if triage_results:
        ranked, gemm_kernels = triage_results
        report["triage"] = {
            "total_kernels": len(ranked),
            "total_gemm_kernels": len(gemm_kernels),
            "top_10_kernels": [
                {
                    "name": k["name"][:100],
                    "count": k["count"],
                    "total_ms": k["total_ns"] / 1e6,
                    "avg_us": k.get("avg_us", 0),
                }
                for k in ranked[:10]
            ],
            "gemm_kernels": [
                {
                    "name": k["name"][:100],
                    "pct_time": k.get("pct_time", 0),
                    "count": k["count"],
                    "avg_us": k.get("avg_us", 0),
                }
                for k in gemm_kernels
            ],
        }

    if deep_results:
        report["deep_profile"] = {
            name: {k: str(v) for k, v in data.items()}
            for name, data in deep_results.items()
        }

    report_file = os.path.join(output_dir, "gemm_profile_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_file}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Kernel Profiling for MI100 (gfx908)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "triage", "deep"],
        default="full",
        help="Profiling mode (default: full)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--kernel-regex",
        help="Regex filter for kernel names (deep mode)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top kernels to deep-profile (default: 3)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model path (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()
    global DEFAULT_MODEL
    DEFAULT_MODEL = args.model

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "triage":
        triage = run_kernel_trace_triage(args.output_dir)
        generate_report(args.output_dir, triage_results=triage)

    elif args.mode == "deep":
        deep = run_deep_profile(args.output_dir, kernel_regex=args.kernel_regex)
        generate_report(args.output_dir, deep_results=deep)

    elif args.mode == "full":
        triage = run_kernel_trace_triage(args.output_dir)
        if triage:
            ranked, gemm_kernels = triage
            deep = run_deep_profile(
                args.output_dir,
                top_n=args.top_n,
                ranked_kernels=gemm_kernels if gemm_kernels else ranked[: args.top_n],
            )
            generate_report(args.output_dir, triage_results=triage, deep_results=deep)
        else:
            print("No triage results, skipping deep profile.")


if __name__ == "__main__":
    main()
