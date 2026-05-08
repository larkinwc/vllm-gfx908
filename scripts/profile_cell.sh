#!/usr/bin/env bash
# =============================================================================
# scripts/profile_cell.sh -- rocprofv3 kernel + HIP trace for one bench cell.
#
# Captures >= 60 s of steady-state for a target cell from the M1 grid.
# Wraps the *client* with rocprofv3 -- this profiles the GPU kernels
# launched while the benchmark client streams requests at the running
# vLLM server.  Equivalent to rocprofv2 --kernel-trace --hip-trace
# from the contract.
#
# We instead rocprofv3-attach to the *server* PID so the trace contains
# every GPU op the server issues during the steady-state window.  This
# gives a clean kernel-trace CSV per cell.
#
# Usage:
#   profile_cell.sh <cell_id> <quant> <tp> <concurrency> <workload>
#
# Cells profiled (per VAL-BASE-005):
#   w8a8  tp=1 c=1 synthetic | tp=4 c=4 synthetic
#   w4a16 tp=1 c=1 synthetic | tp=4 c=4 synthetic
#
# Outputs under /root/bench-int8-w4a16/baseline/profile/<cell_id>/.
# =============================================================================
set -uo pipefail

cell_id=${1:?cell_id required}
quant=${2:?quant required}      # w8a8 | w4a16
tp=${3:?tp required}             # 1 | 4
conc=${4:?concurrency required}  # 1 | 4
workload=${5:?workload required} # synthetic | coding

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
BASELINE_ROOT=/root/bench-int8-w4a16/baseline
DATASET=/root/bench-int8-w4a16/datasets/coding_agent.jsonl
PY=/opt/vllm-env/bin/python3
PROF_DIR=$BASELINE_ROOT/profile/${cell_id}
mkdir -p "$PROF_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts) profile $cell_id] $*"; }

if [[ "$quant" == w8a8 ]]; then model=/models/Qwen3.5-9B-w8a8
else model=/models/Qwen3.5-9B-w4a16; fi

# Common env (mirror run_baseline.sh)
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1

if [[ "$tp" == "4" ]]; then
  export VLLM_MI100_DISABLE_CUSTOM_AR=1
  unset CUDA_VISIBLE_DEVICES || true
  if [[ "$quant" == w4a16 ]]; then
    export TRITON_CACHE_DIR=/root/bench-int8-w4a16/baseline/triton_cache_w4a16
    export NCCL_TIMEOUT_HOURS=2
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
  fi
  EXTRA="--disable-custom-all-reduce"
else
  export VLLM_MI100_DISABLE_CUSTOM_AR=0
  export CUDA_VISIBLE_DEVICES=0
  EXTRA=""
fi

# 1. Wait for any orphans to clear (manifest stop semantics).
sleep 2
PIDS_BEFORE=$(pgrep -f 'vllm.entrypoints' || true)
if [[ -n "$PIDS_BEFORE" ]]; then
  log "warning: existing vllm.entrypoints PIDs $PIDS_BEFORE -- aborting (run baseline first)"
  exit 2
fi

SRV_LOG=$PROF_DIR/server.log

# 2. Start server under rocprofv3 (kernel + HIP trace, CSV output).
#    rocprofv3 wraps the python child; the trace covers everything the
#    server does -- profile_run, graph capture, AND the steady-state
#    benchmark window.  We slice to steady-state during analysis by
#    discarding the first 30 s of records via timestamp filter.
RPROF_OUT=$PROF_DIR/rocprof_${cell_id}
log "starting vLLM-under-rocprofv3 (model=$model tp=$tp) -> $SRV_LOG"
cd "$REPO"
nohup rocprofv3 \
  --kernel-trace \
  --hip-trace \
  -d "$PROF_DIR" \
  -o "rocprof_${cell_id}" \
  -f csv \
  -- \
  $PY -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --dtype float16 \
    --tensor-parallel-size "$tp" \
    --max-model-len 32768 \
    --block-size 32 \
    --enable-prefix-caching \
    --language-model-only \
    --trust-remote-code \
    --gpu-memory-utilization 0.93 \
    --port 8000 \
    $EXTRA > "$SRV_LOG" 2>&1 &
SRVPID=$!
echo "$SRVPID" > "$PROF_DIR/server.pid"
log "rocprofv3 server PID=$SRVPID"

# 3. Wait health
ok=0
for i in $(seq 1 480); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "  healthy after ${i}*5s"
    ok=1; break
  fi
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    log "  server died early -- last 30 lines of $SRV_LOG:"
    tail -n 30 "$SRV_LOG"
    exit 1
  fi
  sleep 5
done
if [[ $ok -eq 0 ]]; then
  log "  healthcheck timeout"; tail -n 30 "$SRV_LOG"
  kill -TERM "$SRVPID" 2>/dev/null || true
  sleep 3
  kill -KILL "$SRVPID" 2>/dev/null || true
  exit 1
fi

# 4. Drive a steady-state benchmark for >= 60 s.
#    For TP=1 c=1 we expect ~6.5 s/req; 200 prompts ~22 min steady-state.
#    For TP=4 c=4 ~150 s. Either way the rocprof trace easily exceeds
#    1000 GPU records.
BENCH_RAW=$PROF_DIR/bench_raw.json
log "  driving benchmark (workload=$workload, conc=$conc)"
if [[ "$workload" == synthetic ]]; then
  BENCH_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos)
else
  BENCH_ARGS=(--dataset-name custom --dataset-path "$DATASET" --custom-output-len 256 --skip-chat-template)
fi
# Profiling needs >= 60s steady-state and >= 1000 kernel records.
# At 6.5s/prompt for c=1 TP=1 we hit both bars in ~10 prompts.
# 50 prompts is conservative and keeps the rocprofv3 trace small enough
# to disk (typical 100-300 MB).
PROFILE_PROMPTS=${PROFILE_PROMPTS:-50}
$PY -m vllm.entrypoints.cli.main bench serve \
  --model "$model" \
  --base-url "http://127.0.0.1:8000" \
  --num-prompts "$PROFILE_PROMPTS" \
  --request-rate inf \
  --max-concurrency "$conc" \
  --seed 42 \
  --save-result \
  --result-dir "$PROF_DIR" \
  --result-filename "bench_raw.json" \
  --trust-remote-code \
  "${BENCH_ARGS[@]}" \
  > "$PROF_DIR/bench_client.log" 2>&1
log "  benchmark client done"

# 5. Stop server cleanly so rocprofv3 flushes the CSVs.
log "  stopping server PID=$SRVPID (SIGINT for clean rocprof flush)"
kill -INT "$SRVPID" 2>/dev/null || true
sleep 5
kill -TERM "$SRVPID" 2>/dev/null || true
sleep 5
kill -KILL "$SRVPID" 2>/dev/null || true
sleep 2

# Also kill any worker children rocprof spawned.
pgrep -P "$SRVPID" 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true

# 6. List captured files and emit a summary.
log "  rocprof outputs:"
find "$PROF_DIR" -maxdepth 3 -name "*.csv" -printf "    %p (%s bytes)\n" 2>/dev/null
N_KERNEL_RECORDS=0
KFILE=$(find "$PROF_DIR" -name "*kernel_trace*.csv" | head -1)
if [[ -n "$KFILE" && -s "$KFILE" ]]; then
  N_KERNEL_RECORDS=$(($(wc -l < "$KFILE") - 1))
  log "  kernel-trace records: $N_KERNEL_RECORDS"
fi

cat > "$PROF_DIR/profile_summary.json" <<EOF
{
  "cell_id": "${cell_id}",
  "quant": "${quant}",
  "tp": ${tp},
  "concurrency": ${conc},
  "workload": "${workload}",
  "kernel_trace_csv": "${KFILE}",
  "n_kernel_records": ${N_KERNEL_RECORDS},
  "server_log": "${SRV_LOG}",
  "bench_log": "${PROF_DIR}/bench_client.log",
  "raw_bench_json": "${BENCH_RAW}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
log "  summary -> $PROF_DIR/profile_summary.json"
log "done."
exit 0
