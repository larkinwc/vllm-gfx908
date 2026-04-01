#!/bin/bash
# run_gemm_optimization.sh - GEMM Kernel Optimization Pipeline for MI100 (gfx908)
#
# Implements TunableOp autotuning + rocprofv3 profiling as described in
# ml-research/kernel-tuning/agentic-profiling-loop.md
#
# Expected uplift: 10-25% throughput across the board via optimal GEMM algorithm selection.
#
# Usage:
#   ./run_gemm_optimization.sh              # Full pipeline
#   ./run_gemm_optimization.sh tune-only    # TunableOp tuning only
#   ./run_gemm_optimization.sh profile-only # rocprofv3 profiling only
#   ./run_gemm_optimization.sh bench-only   # Benchmark comparison only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_PYTHON="/opt/vllm-env/bin/python3"
TUNABLEOP_DIR="/root/tunableop-results"
PROFILE_DIR="/root/gemm-profile-results"
BENCH_RESULTS="/root/benchmark-results"
BENCHMARK_SCRIPTS="/root/benchmark-scripts"
TUNABLEOP_CSV="${TUNABLEOP_DIR}/tunableop_results.csv"
MODEL="/models/Qwen3.5-9B"
PORT=8000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# MI100 environment
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_SKINNY_GEMM=1
export VLLM_ROCM_USE_AITER=1
export TORCH_COMPILE_DISABLE=1
export HIP_FORCE_DEV_KERNARG=1

stop_server() {
    echo "Stopping any existing vLLM server..."
    lsof -ti :${PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 3
    pkill -9 -f 'VLLM::' 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    local max_wait=${1:-180}
    echo "Waiting for server health (up to ${max_wait}s)..."
    for i in $(seq 1 $((max_wait / 2))); do
        if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
            echo "Server healthy after $((i * 2))s"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not become healthy in ${max_wait}s"
    return 1
}

start_server_tuning() {
    echo ""
    echo "========================================="
    echo "Starting vLLM in TunableOp TUNING mode"
    echo "========================================="
    mkdir -p "${TUNABLEOP_DIR}"

    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_TUNING=1
    export PYTORCH_TUNABLEOP_VERBOSE=1
    export PYTORCH_TUNABLEOP_FILENAME="${TUNABLEOP_CSV}"
    export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=60
    export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=200
    export PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1048576

    local LOG="${TUNABLEOP_DIR}/server_tuning_${TIMESTAMP}.log"
    COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

    ${VLLM_PYTHON} -m vllm.entrypoints.openai.api_server \
        --port ${PORT} \
        --model "${MODEL}" \
        --dtype float16 \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --block-size 32 \
        --enable-prefix-caching \
        --compilation-config "${COMPILATION_CONFIG}" \
        --language-model-only \
        2>&1 | tee "${LOG}" &

    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}, Log: ${LOG}"
}

start_server_replay() {
    echo ""
    echo "========================================="
    echo "Starting vLLM with tuned GEMM kernels"
    echo "========================================="

    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_TUNING=0
    export PYTORCH_TUNABLEOP_FILENAME="${TUNABLEOP_CSV}"
    unset PYTORCH_TUNABLEOP_VERBOSE

    local LOG="${TUNABLEOP_DIR}/server_replay_${TIMESTAMP}.log"
    COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

    ${VLLM_PYTHON} -m vllm.entrypoints.openai.api_server \
        --port ${PORT} \
        --model "${MODEL}" \
        --dtype float16 \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --block-size 32 \
        --enable-prefix-caching \
        --compilation-config "${COMPILATION_CONFIG}" \
        --language-model-only \
        2>&1 | tee "${LOG}" &

    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}, Log: ${LOG}"
}

run_warmup() {
    echo ""
    echo "========================================="
    echo "Sending warmup requests for GEMM tuning"
    echo "========================================="

    ${VLLM_PYTHON} "${SCRIPT_DIR}/tunableop_gemm_tuning.py" \
        --mode record \
        --output-dir "${TUNABLEOP_DIR}" \
        --model "${MODEL}"
}

run_benchmarks() {
    local label="${1:-tunableop}"

    echo ""
    echo "========================================="
    echo "Running benchmarks: ${label}"
    echo "========================================="

    for concurrency in 1 2 4; do
        echo ""
        echo "--- Concurrency ${concurrency} ---"
        ${VLLM_PYTHON} "${BENCHMARK_SCRIPTS}/coding_agent_bench.py" \
            --concurrency ${concurrency} \
            --requests 20 \
            --model "${MODEL}" \
            --output-dir /models \
            2>&1 || echo "Benchmark c=${concurrency} failed"
    done
}

compare_results() {
    echo ""
    echo "========================================="
    echo "Comparing Results: Baseline vs TunableOp"
    echo "========================================="

    ${VLLM_PYTHON} - <<'PYTHON_SCRIPT'
import json, glob, os
from datetime import datetime

def get_latest_results(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f), os.path.basename(files[-1])

print(f"{'Metric':<30} {'Baseline':>12} {'TunableOp':>12} {'Delta':>10} {'Change':>8}")
print("-" * 75)

# We compare the 2 most recent results at each concurrency
# Assuming baseline was run before TunableOp
for c in [1, 2, 4]:
    pattern = f"/models/Qwen3.5-9B_coding_agent_concurrency{c}_*.json"
    files = sorted(glob.glob(pattern))
    if len(files) < 2:
        print(f"  c={c}: Need at least 2 result files for comparison")
        continue

    # Second-to-last = baseline, last = tunableop
    with open(files[-2]) as f:
        baseline = json.load(f)
    with open(files[-1]) as f:
        tunableop = json.load(f)

    bs = baseline["summary"]
    ts = tunableop["summary"]

    label = f"c={c}"
    metrics = [
        (f"{label} throughput (tok/s)", bs["aggregate_decode_tok_per_s"], ts["aggregate_decode_tok_per_s"]),
        (f"{label} TPOT (ms)", bs["avg_tpot_ms"], ts["avg_tpot_ms"]),
        (f"{label} TTFT (ms)", bs["avg_ttft_ms"], ts["avg_ttft_ms"]),
    ]

    for name, bv, tv in metrics:
        delta = tv - bv
        if "TPOT" in name or "TTFT" in name:
            pct = (delta / bv * 100) if bv != 0 else 0
            better = "BETTER" if delta < 0 else "WORSE" if delta > 0 else "SAME"
        else:
            pct = (delta / bv * 100) if bv != 0 else 0
            better = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print(f"  {name:<28} {bv:>12.2f} {tv:>12.2f} {delta:>+10.2f} {pct:>+6.1f}% {better}")
    print()

PYTHON_SCRIPT
}

# ========================================
# Main logic
# ========================================
MODE="${1:-full}"

case "$MODE" in
    full)
        echo "============================================="
        echo "GEMM Kernel Optimization Pipeline for MI100"
        echo "============================================="
        echo "Timestamp: ${TIMESTAMP}"
        echo "Model: ${MODEL}"
        echo ""

        # Step 1: Stop server, start in tuning mode
        stop_server
        start_server_tuning
        wait_for_server 180 || exit 1

        # Step 2: Warmup to exercise all GEMM paths
        run_warmup

        # Step 3: Stop tuning server
        stop_server

        # Check tuned results
        if [ -f "${TUNABLEOP_CSV}" ]; then
            ENTRIES=$(wc -l < "${TUNABLEOP_CSV}")
            echo ""
            echo "TunableOp results: ${ENTRIES} entries in ${TUNABLEOP_CSV}"
        else
            echo ""
            echo "WARNING: No TunableOp results file found!"
            echo "Continuing with replay mode anyway (PyTorch may use default path)."
        fi

        # Step 4: Start replay server
        start_server_replay
        wait_for_server 120 || exit 1

        # Step 5: Run benchmarks
        run_benchmarks "tunableop"

        # Step 6: Compare
        compare_results

        echo ""
        echo "Pipeline complete. Server still running on port ${PORT}."
        echo "TunableOp results: ${TUNABLEOP_CSV}"
        ;;

    tune-only)
        stop_server
        start_server_tuning
        wait_for_server 180 || exit 1
        run_warmup
        stop_server
        echo ""
        echo "Tuning complete. Results: ${TUNABLEOP_CSV}"
        ;;

    profile-only)
        echo "Running GEMM profiling with rocprofv3..."
        ${VLLM_PYTHON} "${SCRIPT_DIR}/profile_gemm_kernels.py" \
            --mode full \
            --output-dir "${PROFILE_DIR}" \
            --model "${MODEL}"
        ;;

    bench-only)
        run_benchmarks "current"
        compare_results
        ;;

    *)
        echo "Usage: $0 [full|tune-only|profile-only|bench-only]"
        exit 1
        ;;
esac
