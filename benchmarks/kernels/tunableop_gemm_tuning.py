#!/opt/vllm-env/bin/python3
"""
TunableOp GEMM Autotuning for MI100 (gfx908)

Automates PyTorch's TunableOp to find optimal rocBLAS/hipBLASLt GEMM kernels
for the exact matrix shapes encountered during vLLM inference.

Workflow:
  1. Record: Run inference with TunableOp recording to capture GEMM shapes
  2. Tune:   Offline-tune all captured shapes (tries all rocBLAS algorithms)
  3. Replay: Run inference with tuned results loaded (zero overhead)

Usage:
    # Full pipeline: record shapes, tune, validate
    python tunableop_gemm_tuning.py --mode full --output-dir /root/tunableop-results

    # Record GEMM shapes only (server must be running)
    python tunableop_gemm_tuning.py --mode record --output-dir /root/tunableop-results

    # Tune previously recorded shapes offline
    python tunableop_gemm_tuning.py --mode tune --input-file /root/tunableop-results/untuned_gemms.csv

    # Generate launch script with TunableOp env vars
    python tunableop_gemm_tuning.py --mode generate-launch

Environment:
    Requires vLLM server running on localhost:8000
    Requires 4x MI100 GPUs with ROCm 7.x and PyTorch 2.11+rocm7.2
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


VLLM_PYTHON = "/opt/vllm-env/bin/python3"
DEFAULT_OUTPUT_DIR = "/root/tunableop-results"
DEFAULT_MODEL = "/models/Qwen3.5-9B"
SERVER_URL = "http://localhost:8000"

TUNABLEOP_ENV = {
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "PYTORCH_TUNABLEOP_TUNING": "1",
    "PYTORCH_TUNABLEOP_VERBOSE": "1",
    "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS": "30",
    "PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS": "100",
}

WARMUP_PROMPTS = [
    "Write a Python function to calculate fibonacci numbers efficiently.",
    "Explain the difference between a stack and a queue in computer science.",
    "Create a bash script that monitors disk usage and sends alerts.",
    "Implement a binary search tree in C++ with insert and delete operations.",
    "Write a SQL query to find the top 10 customers by revenue with joins.",
    "Design a REST API for a todo list application with CRUD operations.",
    "Explain how garbage collection works in modern programming languages.",
    "Write a recursive function to traverse a directory tree in Python.",
]


def check_server_health():
    """Check if vLLM server is running and healthy."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def send_warmup_request(prompt, max_tokens=256):
    """Send a single request to the vLLM server to exercise GEMM kernels."""
    import urllib.request
    payload = json.dumps({
        "model": os.path.basename(DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def record_gemm_shapes(output_dir):
    """
    Run warmup requests against a TunableOp-enabled server to record GEMM shapes.
    The server must have been started with PYTORCH_TUNABLEOP_ENABLED=1 and
    PYTORCH_TUNABLEOP_RECORD_UNTUNED=1.
    """
    print("Phase 1: Recording GEMM shapes via warmup requests...")
    print(f"Sending {len(WARMUP_PROMPTS)} warmup prompts to exercise all GEMM paths")
    print()

    if not check_server_health():
        print("ERROR: vLLM server not reachable at", SERVER_URL)
        print("Start the server with TunableOp env vars first.")
        print("See: python tunableop_gemm_tuning.py --mode generate-launch")
        sys.exit(1)

    results = []
    for i, prompt in enumerate(WARMUP_PROMPTS):
        print(f"  [{i+1}/{len(WARMUP_PROMPTS)}] Sending: {prompt[:60]}...")
        try:
            resp = send_warmup_request(prompt)
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            print(f"    Generated {tokens} tokens")
            results.append({"prompt": prompt[:60], "tokens": tokens, "success": True})
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"prompt": prompt[:60], "error": str(e), "success": False})

    # Also send a batch of concurrent requests to exercise batched GEMM shapes
    print()
    print("  Sending batch of 4 concurrent requests for batched GEMM shapes...")
    import concurrent.futures
    batch_prompts = WARMUP_PROMPTS[:4]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_warmup_request, p) for p in batch_prompts]
        for f in concurrent.futures.as_completed(futures):
            try:
                resp = f.result()
                tokens = resp.get("usage", {}).get("completion_tokens", 0)
                print(f"    Batch request: {tokens} tokens")
            except Exception as e:
                print(f"    Batch request ERROR: {e}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_prompts": len(WARMUP_PROMPTS),
        "successful": sum(1 for r in results if r["success"]),
        "results": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "warmup_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWarmup summary saved to {summary_file}")
    return summary


def tune_gemm_offline(untuned_file, output_dir):
    """
    Tune GEMM shapes offline using PyTorch's TunableOp offline tuner.
    Reads untuned shapes from file and finds optimal algorithms.
    """
    print(f"Phase 2: Offline tuning GEMM shapes from {untuned_file}...")

    if not os.path.exists(untuned_file):
        print(f"ERROR: Untuned file not found: {untuned_file}")
        print("Run with --mode record first, or check PYTORCH_TUNABLEOP_FILENAME path.")
        sys.exit(1)

    # Use PyTorch's built-in offline tuner
    tune_script = f"""
import torch
import torch.cuda.tunable as tunable
import os
import time

tunable.enable(True)
tunable.tuning_enable(True)
tunable.set_max_tuning_duration(60)  # 60ms per shape
tunable.set_max_tuning_iterations(200)  # up to 200 algo candidates

input_file = "{untuned_file}"
output_file = os.path.join("{output_dir}", "tunableop_results.csv")
tunable.set_filename(output_file)

print(f"Reading untuned GEMMs from: {{input_file}}")
print(f"Output will be saved to: {{output_file}}")

# For multi-GPU tuning
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {{num_gpus}}")

start = time.time()
if num_gpus > 1:
    pattern = input_file.replace('.csv', '_gpu*.csv')
    # Check if per-GPU files exist
    import glob
    gpu_files = glob.glob(pattern)
    if gpu_files:
        print(f"Found {{len(gpu_files)}} per-GPU untuned files")
        tunable.mgpu_tune_gemm_in_file(input_file.replace('.csv', '_gpu*.csv'), num_gpus)
    else:
        print("No per-GPU files, tuning single file across all GPUs")
        tunable.tune_gemm_in_file(input_file)
else:
    tunable.tune_gemm_in_file(input_file)

elapsed = time.time() - start
print(f"Tuning completed in {{elapsed:.1f}}s")

# Get results
results = tunable.get_results()
print(f"Total tuned entries: {{len(results)}}")
for op, entries in results.items():
    print(f"  {{op}}: {{len(entries)}} shapes tuned")
"""
    os.makedirs(output_dir, exist_ok=True)
    script_file = os.path.join(output_dir, "_tune_offline.py")
    with open(script_file, "w") as f:
        f.write(tune_script)

    env = os.environ.copy()
    env.update({
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "1",
    })

    result = subprocess.run(
        [VLLM_PYTHON, script_file],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
    return result.returncode


def generate_launch_script(output_dir, tunableop_results_file=None):
    """Generate a launch script with TunableOp environment variables."""
    if tunableop_results_file is None:
        tunableop_results_file = os.path.join(output_dir, "tunableop_results.csv")

    script = f"""#!/bin/bash
# launch-vllm-tunableop.sh - vLLM with TunableOp-tuned GEMM kernels for MI100
#
# Generated by tunableop_gemm_tuning.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# TunableOp replays pre-tuned rocBLAS/hipBLASLt algorithm selections
# for the exact GEMM shapes encountered during Qwen3.5-9B inference with TP=4.
# This is zero-overhead at runtime (just a CSV lookup on first GEMM call).

set -e

# ========================================
# MI100 Required Environment Variables
# ========================================
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_SKINNY_GEMM=1
export VLLM_ROCM_USE_AITER=1
export TORCH_COMPILE_DISABLE=1

# ========================================
# TunableOp Configuration
# ========================================
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0          # Replay only, no tuning overhead
export PYTORCH_TUNABLEOP_FILENAME={tunableop_results_file}

# ========================================
# Performance Tuning
# ========================================
export HIP_FORCE_DEV_KERNARG=1             # Faster kernel launches

# ========================================
# Launch Server
# ========================================
MODEL="${{1:-/models/Qwen3.5-9B}}"
PORT="${{2:-8000}}"
MODEL_NAME=$(basename "$MODEL")

COMPILATION_CONFIG='{{"cudagraph_mode":"FULL_DECODE_ONLY"}}'

echo "Starting vLLM with TunableOp-tuned GEMM kernels..."
echo "  Model: $MODEL"
echo "  TunableOp results: {tunableop_results_file}"
echo ""

LANGUAGE_MODEL_ONLY=""
if [[ "$MODEL_NAME" == "Qwen3.5-9B" ]]; then
    LANGUAGE_MODEL_ONLY="--language-model-only"
fi

exec {VLLM_PYTHON} -m vllm.entrypoints.openai.api_server \\
    --port $PORT \\
    --model "$MODEL" \\
    --dtype float16 \\
    --trust-remote-code \\
    --tensor-parallel-size 4 \\
    --max-model-len 32768 \\
    --max-num-batched-tokens 8192 \\
    --block-size 32 \\
    --enable-prefix-caching \\
    --compilation-config "$COMPILATION_CONFIG" \\
    $LANGUAGE_MODEL_ONLY
"""

    os.makedirs(output_dir, exist_ok=True)
    script_file = os.path.join(output_dir, "launch-vllm-tunableop.sh")
    with open(script_file, "w") as f:
        f.write(script)
    os.chmod(script_file, 0o755)
    print(f"Launch script written to: {script_file}")

    # Also generate a tuning launch script (for the recording phase)
    tuning_script = f"""#!/bin/bash
# launch-vllm-tunableop-tuning.sh - Run vLLM with TunableOp RECORDING enabled
#
# This starts the server in tuning mode. Every GEMM operation will be auto-tuned
# on first encounter. This adds latency to the first inference pass but finds
# optimal rocBLAS algorithms for each unique GEMM shape.
#
# After warmup, stop the server. The tuned results are saved to the CSV file.
# Then use launch-vllm-tunableop.sh for production (replay mode, zero overhead).

set -e

export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_SKINNY_GEMM=1
export VLLM_ROCM_USE_AITER=1
export TORCH_COMPILE_DISABLE=1

# TunableOp in TUNING mode
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_VERBOSE=1
export PYTORCH_TUNABLEOP_FILENAME={tunableop_results_file}
export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=60
export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=200
export PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1048576

# Performance
export HIP_FORCE_DEV_KERNARG=1

MODEL="${{1:-/models/Qwen3.5-9B}}"
PORT="${{2:-8000}}"
MODEL_NAME=$(basename "$MODEL")

COMPILATION_CONFIG='{{"cudagraph_mode":"FULL_DECODE_ONLY"}}'

echo "Starting vLLM in TunableOp TUNING mode..."
echo "  Model: $MODEL"
echo "  TunableOp results will be saved to: {tunableop_results_file}"
echo ""
echo "IMPORTANT: Send warmup requests to exercise all GEMM paths."
echo "  python tunableop_gemm_tuning.py --mode record --output-dir {output_dir}"
echo ""

LANGUAGE_MODEL_ONLY=""
if [[ "$MODEL_NAME" == "Qwen3.5-9B" ]]; then
    LANGUAGE_MODEL_ONLY="--language-model-only"
fi

exec {VLLM_PYTHON} -m vllm.entrypoints.openai.api_server \\
    --port $PORT \\
    --model "$MODEL" \\
    --dtype float16 \\
    --trust-remote-code \\
    --tensor-parallel-size 4 \\
    --max-model-len 32768 \\
    --max-num-batched-tokens 8192 \\
    --block-size 32 \\
    --enable-prefix-caching \\
    --compilation-config "$COMPILATION_CONFIG" \\
    $LANGUAGE_MODEL_ONLY
"""

    tuning_script_file = os.path.join(output_dir, "launch-vllm-tunableop-tuning.sh")
    with open(tuning_script_file, "w") as f:
        f.write(tuning_script)
    os.chmod(tuning_script_file, 0o755)
    print(f"Tuning launch script written to: {tuning_script_file}")

    return script_file, tuning_script_file


def run_full_pipeline(output_dir, skip_server_restart=False):
    """
    Full TunableOp autotuning pipeline:
    1. Stop any existing server
    2. Start server with TunableOp tuning enabled
    3. Send warmup requests to exercise GEMM paths
    4. Stop server (saves tuned results)
    5. Start server with tuned results (replay mode)
    6. Run benchmarks to measure improvement
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tunableop_file = os.path.join(output_dir, "tunableop_results.csv")

    print("=" * 60)
    print("TunableOp GEMM Autotuning Pipeline for MI100")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Output dir: {output_dir}")
    print(f"TunableOp results: {tunableop_file}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Stop existing server
    if not skip_server_restart:
        print("Step 1: Stopping existing vLLM server...")
        subprocess.run(
            "lsof -ti :8000 | xargs kill -9 2>/dev/null; sleep 3; "
            "pkill -9 -f 'VLLM::' 2>/dev/null; sleep 2",
            shell=True, capture_output=True,
        )
        print("  Done.")
        print()

        # Step 2: Start server with TunableOp tuning enabled
        print("Step 2: Starting vLLM with TunableOp TUNING mode...")
        generate_launch_script(output_dir, tunableop_file)
        tuning_launch = os.path.join(output_dir, "launch-vllm-tunableop-tuning.sh")

        log_file = os.path.join(output_dir, f"server_tuning_{timestamp}.log")
        server_proc = subprocess.Popen(
            ["bash", tuning_launch],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
        )
        print(f"  Server PID: {server_proc.pid}")
        print(f"  Log: {log_file}")

        # Wait for health
        print("  Waiting for server health (up to 180s for tuning mode)...")
        for i in range(90):
            if check_server_health():
                print(f"  Server healthy after {(i+1)*2}s")
                break
            time.sleep(2)
        else:
            print("  ERROR: Server did not become healthy in 180s")
            print(f"  Check log: {log_file}")
            server_proc.kill()
            sys.exit(1)
        print()

    # Step 3: Send warmup requests
    print("Step 3: Sending warmup requests to exercise GEMM paths...")
    record_gemm_shapes(output_dir)
    print()

    # Send additional varied-length requests for comprehensive shape coverage
    print("Step 3b: Additional warmup with varied output lengths...")
    for max_tok in [64, 128, 512]:
        try:
            send_warmup_request(
                "Write a detailed explanation of how hash tables work.",
                max_tokens=max_tok,
            )
            print(f"  max_tokens={max_tok}: OK")
        except Exception as e:
            print(f"  max_tokens={max_tok}: {e}")
    print()

    if not skip_server_restart:
        # Step 4: Stop server (tuned results are saved)
        print("Step 4: Stopping tuning-mode server (results saved)...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        subprocess.run(
            "pkill -9 -f 'VLLM::' 2>/dev/null; sleep 3",
            shell=True, capture_output=True,
        )
        print("  Done.")
        print()

        # Check if tuned results were saved
        if os.path.exists(tunableop_file):
            with open(tunableop_file) as f:
                lines = f.readlines()
            print(f"  TunableOp results: {len(lines)} entries in {tunableop_file}")
        else:
            print(f"  WARNING: TunableOp results file not found at {tunableop_file}")
            print("  The tuning may have saved to a default location.")
            # Check default locations
            for candidate in [
                "/tmp/tunableop_results.csv",
                os.path.expanduser("~/.cache/pytorch/tunableop_results.csv"),
            ]:
                if os.path.exists(candidate):
                    print(f"  Found at: {candidate}")
                    break
        print()

        # Step 5: Start production server with tuned results
        print("Step 5: Starting vLLM with tuned GEMM kernels (replay mode)...")
        replay_launch = os.path.join(output_dir, "launch-vllm-tunableop.sh")

        log_file = os.path.join(output_dir, f"server_replay_{timestamp}.log")
        server_proc = subprocess.Popen(
            ["bash", replay_launch],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
        )
        print(f"  Server PID: {server_proc.pid}")
        print(f"  Log: {log_file}")

        print("  Waiting for server health...")
        for i in range(60):
            if check_server_health():
                print(f"  Server healthy after {(i+1)*2}s")
                break
            time.sleep(2)
        else:
            print("  ERROR: Server did not become healthy in 120s")
            server_proc.kill()
            sys.exit(1)
        print()

    print("=" * 60)
    print("TunableOp tuning complete. Server running with tuned GEMM kernels.")
    print(f"TunableOp results: {tunableop_file}")
    print()
    print("Run benchmarks to measure improvement:")
    print(f"  {VLLM_PYTHON} /root/benchmark-scripts/coding_agent_bench.py \\")
    print("    --concurrency 1 --requests 20 --model /models/Qwen3.5-9B")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="TunableOp GEMM Autotuning for MI100 (gfx908)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "record", "tune", "generate-launch"],
        default="full",
        help="Operation mode (default: full)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        help="Input file for tune mode (untuned GEMMs CSV)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--skip-server-restart",
        action="store_true",
        help="Skip server restart (assume server is already running with TunableOp)",
    )

    args = parser.parse_args()
    global DEFAULT_MODEL
    DEFAULT_MODEL = args.model

    if args.mode == "full":
        run_full_pipeline(args.output_dir, args.skip_server_restart)
    elif args.mode == "record":
        record_gemm_shapes(args.output_dir)
    elif args.mode == "tune":
        if not args.input_file:
            print("ERROR: --input-file required for tune mode")
            sys.exit(1)
        tune_gemm_offline(args.input_file, args.output_dir)
    elif args.mode == "generate-launch":
        generate_launch_script(args.output_dir)


if __name__ == "__main__":
    main()
