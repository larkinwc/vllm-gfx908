#!/usr/bin/env bash
# =============================================================================
# scripts/profile_cell_v3.sh -- single-pass rocprofv3 capture using
# --collection-period to bound the trace window cleanly.
#
# rocprofv3 launches vLLM, waits until the server is healthy, drives a
# 50-prompt benchmark.  rocprofv3 itself is told to record only seconds
# 90-180 of process wall-clock (90 s start delay covers profile_run +
# graph capture + warmup; 90 s collection covers >=15 c=1 prompts at
# 6.5 s/prompt = >1500 kernel records easily).  After the collection
# window closes rocprofv3 stops recording but the process continues;
# the bench client finishes; the script kills the server cleanly.
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

if [[ "$workload" == synthetic ]]; then
  BENCH_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos)
else
  BENCH_ARGS=(--dataset-name custom --dataset-path "$DATASET" --custom-output-len 256 --skip-chat-template)
fi

# Pre-clean any orphans
sleep 2
pgrep -f vllm.entrypoints 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 2

# IMPORTANT: the editable install of vllm in /opt/vllm-env points at a
# *different* worktree (upstream-sync-2026-05-03) which does not have
# the M0 tokenizer-registry patch. Set PYTHONPATH so this worktree's
# vllm wins on import resolution.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"

# Start rocprofv3-wrapped server in background.
SRV_LOG=$PROF_DIR/server.log
log "starting rocprofv3 + vLLM (model=$model tp=$tp, PYTHONPATH=$REPO)"
log "  collection-period: 90s start delay, 90s record"
nohup rocprofv3 \
  --kernel-trace \
  --hip-trace \
  -d "$PROF_DIR" \
  -o "rocprof_${cell_id}" \
  -f csv \
  --collection-period 90:90:1 \
  --collection-period-unit sec \
  -- \
  $PY -m vllm.entrypoints.openai.api_server \
    --model "$model" --dtype float16 --tensor-parallel-size "$tp" \
    --max-model-len 32768 --block-size 32 --enable-prefix-caching \
    --language-model-only --trust-remote-code --gpu-memory-utilization 0.93 \
    --port 8000 $EXTRA > "$SRV_LOG" 2>&1 &
SRVPID=$!
echo "$SRVPID" > "$PROF_DIR/server.pid"

# Wait for healthy (allow up to 600 s).
ok=0
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "  healthy after ${i}*5s"; ok=1; break
  fi
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    log "  server died"; tail -n 30 "$SRV_LOG"; exit 1
  fi
  sleep 5
done
[[ $ok -eq 1 ]] || { log "healthcheck timeout"; tail -n 30 "$SRV_LOG"; exit 1; }

# Drive bench: 50 prompts at the target concurrency. This will overlap
# rocprofv3's 90-180s collection window. We use --num-prompts 60 so the
# bench runs >=300s (more than enough to overlap collection).
PROFILE_PROMPTS=${PROFILE_PROMPTS:-60}
log "  driving bench (workload=$workload, conc=$conc, n=$PROFILE_PROMPTS)"
$PY -m vllm.entrypoints.cli.main bench serve \
  --model "$model" --base-url "http://127.0.0.1:8000" \
  --num-prompts "$PROFILE_PROMPTS" \
  --request-rate inf --max-concurrency "$conc" --seed 42 \
  --save-result --result-dir "$PROF_DIR" --result-filename "bench_raw.json" \
  --trust-remote-code "${BENCH_ARGS[@]}" \
  > "$PROF_DIR/bench_client.log" 2>&1 || true
log "  benchmark client done"

# rocprofv3 only finalizes CSVs once the wrapped process exits cleanly.
# Strategy:
#   * Wait until the 90 s collection window has fully closed (we know
#     it ended at start_delay + duration = 90 + 90 = 180 s after
#     rocprofv3 launch). The bench client took >180 s so collection is
#     done by now.
#   * Send SIGTERM to the rocprofv3 wrapper itself; it forwards to the
#     wrapped python and waits for child exit before flushing.
log "  waiting 30 s for rocprofv3 collection window to close"
sleep 30

log "  sending SIGTERM to rocprofv3 wrapper PID=$SRVPID for clean flush"
kill -TERM "$SRVPID" 2>/dev/null || true

# Give rocprofv3 a generous 5 minutes to flush. CSVs for ~30k kernels
# can take 30-90 s on a busy host.
for i in $(seq 1 300); do
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    log "  rocprofv3 wrapper exited cleanly after ${i}s"; break
  fi
  sleep 1
done
# Last-resort cleanup if rocprofv3 didn't exit:
if kill -0 "$SRVPID" 2>/dev/null; then
  log "  rocprofv3 still running after 300 s -- escalating to SIGKILL"
  kill -KILL "$SRVPID" 2>/dev/null || true
  sleep 2
fi
pgrep -f vllm.entrypoints 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 5

# Locate the kernel-trace CSV.
KFILE=$(find "$PROF_DIR" -name "*kernel_trace*.csv" 2>/dev/null | head -1)
N_RECORDS=0
if [[ -n "$KFILE" && -s "$KFILE" ]]; then
  N_RECORDS=$(($(wc -l < "$KFILE") - 1))
  log "  kernel-trace records: $N_RECORDS"
else
  log "  WARN no kernel-trace CSV; checking for alternates"
  find "$PROF_DIR" -maxdepth 4 -name "*.csv" -ls 2>&1 | head -10
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
  "server_log": "${SRV_LOG}",
  "bench_log": "${PROF_DIR}/bench_client.log",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
log "  summary -> $PROF_DIR/profile_summary.json"
log "done."
exit 0
