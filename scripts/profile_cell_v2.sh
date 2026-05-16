#!/usr/bin/env bash
# =============================================================================
# scripts/profile_cell_v2.sh -- rocprofv3 capture using attach-to-running-PID.
#
# Strategy:
#   1. Start vLLM server normally; wait for /health.
#   2. Use rocprofv3 -P (start_delay):(duration) to record 60 s of GPU
#      activity in a child process that drives the benchmark.  rocprofv3
#      attaches to the wrapped python via HSA tools API; after the
#      collection window it cleanly flushes CSVs and exits.
#   3. Wait for that wrapped process to finalize, then stop the server.
#
# This avoids the SIGINT/SIGTERM races that cause rocprofv3 to drop CSVs.
#
# Usage:
#   profile_cell_v2.sh <cell_id> <quant> <tp> <concurrency> <workload>
# =============================================================================
set -uo pipefail

cell_id=${1:?cell_id required}
quant=${2:?quant required}
tp=${3:?tp required}
conc=${4:?concurrency required}
workload=${5:?workload required}

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

# 1. Start vLLM server (no rocprof wrapper).
SRV_LOG=$PROF_DIR/server.log
log "starting vLLM (model=$model tp=$tp) -> $SRV_LOG"
cd "$REPO"
nohup $PY -m vllm.entrypoints.openai.api_server \
  --model "$model" --dtype float16 --tensor-parallel-size "$tp" \
  --max-model-len 32768 --block-size 32 --enable-prefix-caching \
  --language-model-only --trust-remote-code --gpu-memory-utilization 0.93 \
  --port 8000 $EXTRA > "$SRV_LOG" 2>&1 &
SRVPID=$!
echo "$SRVPID" > "$PROF_DIR/server.pid"

ok=0
for i in $(seq 1 480); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "  healthy after ${i}*5s"
    ok=1; break
  fi
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    log "  server died"; tail -n 30 "$SRV_LOG"; exit 1
  fi
  sleep 5
done
[[ $ok -eq 1 ]] || { log "healthcheck timeout"; exit 1; }

# 2. Drive a short pre-bench warmup (5 prompts) so we capture
#    steady-state, not graph-capture artifacts.
log "  warmup 5 prompts"
if [[ "$workload" == synthetic ]]; then
  WARM_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos)
else
  WARM_ARGS=(--dataset-name custom --dataset-path "$DATASET" --custom-output-len 256 --skip-chat-template)
fi
$PY -m vllm.entrypoints.cli.main bench serve \
  --model "$model" --base-url "http://127.0.0.1:8000" \
  --num-prompts 5 --request-rate inf --max-concurrency "$conc" --seed 42 \
  --trust-remote-code "${WARM_ARGS[@]}" \
  > "$PROF_DIR/warmup.log" 2>&1 || true

# 3. Run rocprofv3-wrapped benchmark client. rocprofv3 traces the child
#    Python process which is the bench-serve client. The CLIENT does
#    *not* run kernels itself -- but it sends HTTP requests that cause
#    the server (separate process) to run kernels. So we cannot trace
#    the client.  Instead, we re-architect: stop the server and restart
#    it under rocprofv3 with --collection-period bounded to match the
#    benchmark window.
log "  stopping server cleanly to relaunch under rocprofv3"
kill -INT "$SRVPID" 2>/dev/null || true
for i in $(seq 1 30); do
  if ! kill -0 "$SRVPID" 2>/dev/null; then break; fi
  sleep 1
done
kill -TERM "$SRVPID" 2>/dev/null || true
sleep 5
kill -KILL "$SRVPID" 2>/dev/null || true
sleep 2
pgrep -f 'vllm.entrypoints' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true

# 4. Relaunch server under rocprofv3 with --collection-period.
#    Format: -P start_delay:duration:repeat. We use 60 s start delay
#    (covers profile_run + graph capture + warmup) and 120 s collection
#    (well over the 60 s gate), no repeat.
RPROF_OUT="rocprof_${cell_id}"
log "starting rocprofv3 + vLLM (collection 60s start_delay, 120s record)"
nohup rocprofv3 \
  --kernel-trace \
  --hip-trace \
  -d "$PROF_DIR" \
  -o "$RPROF_OUT" \
  -f csv \
  --collection-period 60:120:1 \
  --collection-period-unit sec \
  --disable-signal-handlers \
  -- \
  $PY -m vllm.entrypoints.openai.api_server \
    --model "$model" --dtype float16 --tensor-parallel-size "$tp" \
    --max-model-len 32768 --block-size 32 --enable-prefix-caching \
    --language-model-only --trust-remote-code --gpu-memory-utilization 0.93 \
    --port 8000 $EXTRA > "$SRV_LOG.prof" 2>&1 &
SRVPID=$!
echo "$SRVPID" > "$PROF_DIR/server_prof.pid"

ok=0
for i in $(seq 1 600); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "  rocprof server healthy after ${i}*5s"
    ok=1; break
  fi
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    log "  rocprof server died"; tail -n 30 "$SRV_LOG.prof"; exit 1
  fi
  sleep 5
done
[[ $ok -eq 1 ]] || { log "rocprof healthcheck timeout"; exit 1; }

# 5. Drive a fixed-duration benchmark (50 prompts) to keep the GPU
#    busy throughout the 120 s collection window.
PROFILE_PROMPTS=${PROFILE_PROMPTS:-50}
log "  driving rocprof bench (workload=$workload, conc=$conc, n=$PROFILE_PROMPTS)"
$PY -m vllm.entrypoints.cli.main bench serve \
  --model "$model" --base-url "http://127.0.0.1:8000" \
  --num-prompts "$PROFILE_PROMPTS" \
  --request-rate inf --max-concurrency "$conc" --seed 42 \
  --save-result --result-dir "$PROF_DIR" --result-filename "bench_raw.json" \
  --trust-remote-code "${BENCH_ARGS[@]:-${WARM_ARGS[@]}}" \
  > "$PROF_DIR/bench_client.log" 2>&1 || true
log "  benchmark client done"

# 6. Wait until rocprofv3 has finished its 120 s collection period.
#    Then send SIGINT (rocprofv3 will flush CSVs).
log "  waiting for rocprofv3 collection window to finalize"
sleep 30
log "  sending SIGINT to wrapped python to trigger flush"
WRAP_CHILDREN=$(pgrep -P "$SRVPID" 2>/dev/null || echo "")
if [[ -n "$WRAP_CHILDREN" ]]; then
  for pid in $WRAP_CHILDREN; do
    kill -INT "$pid" 2>/dev/null || true
  done
fi
kill -INT "$SRVPID" 2>/dev/null || true
# Give rocprofv3 plenty of time to finalize.
for i in $(seq 1 60); do
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    log "  rocprofv3 wrapper exited cleanly after ${i}s"; break
  fi
  sleep 1
done
kill -TERM "$SRVPID" 2>/dev/null || true
sleep 5
kill -KILL "$SRVPID" 2>/dev/null || true
sleep 2
pgrep -f 'vllm.entrypoints' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true

# 7. Find the kernel-trace CSV.
KFILE=$(find "$PROF_DIR" -name "*kernel_trace*.csv" 2>/dev/null | head -1)
N_RECORDS=0
if [[ -n "$KFILE" && -s "$KFILE" ]]; then
  N_RECORDS=$(($(wc -l < "$KFILE") - 1))
  log "  kernel-trace records: $N_RECORDS  ($KFILE)"
else
  log "  WARNING no kernel-trace CSV found"
  find "$PROF_DIR" -maxdepth 3 -name "*.csv"
fi

cat > "$PROF_DIR/profile_summary.json" <<EOF
{
  "cell_id": "${cell_id}",
  "quant": "${quant}",
  "tp": ${tp},
  "concurrency": ${conc},
  "workload": "${workload}",
  "kernel_trace_csv": "${KFILE}",
  "n_kernel_records": ${N_RECORDS},
  "server_log": "${SRV_LOG}.prof",
  "bench_log": "${PROF_DIR}/bench_client.log",
  "raw_bench_json": "${PROF_DIR}/bench_raw.json",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
log "  summary -> $PROF_DIR/profile_summary.json"
log "done."
exit 0
