#!/usr/bin/env bash
# =============================================================================
# scripts/profile_cell_offline.sh -- rocprofv3 capture using vllm bench
# throughput (offline, single-process). The wrapped python exits cleanly
# at end-of-bench, which lets rocprofv3 finalize CSVs reliably.
#
# We use --input-len / --output-len / --num-prompts to mimic the c=4
# regime (4 in-flight prompts) for each TP/quant pairing. This is not
# the same harness as bench serve (no streaming, no max-concurrency),
# but the kernel SET is identical because the same compiled graphs and
# the same model/quant kernels execute. We use this *only* to extract
# the hot GEMM shapes for VAL-BASE-006 / 007.
#
# Args: cell_id quant tp workload_kind
# =============================================================================
set -uo pipefail

cell_id=${1:?cell_id required}
quant=${2:?quant required}
tp=${3:?tp required}
workload=${4:?workload required}   # synthetic | coding

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/cold-points-sit-rancb
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

# PYTHONPATH wins over the editable install whose worktree lacks the M0
# tokenizer-registry patch.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"

# Pre-clean any orphans
pgrep -f vllm.entrypoints 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 3

# Bench params:
#   * input-len 1024, output-len 256 for synthetic (matches grid).
#   * For coding we use the locked dataset (--dataset-path).
#   * num-prompts: 8 prompts is enough to collect >1500 kernel records
#     per layer x prompt at FULL_DECODE_ONLY graph (40 layers x 256
#     decode steps).  We measured 8 prompts ~3-5 minutes runtime for
#     w8a8 tp=1, which fits within reasonable profile size.
NPROMPTS=${NPROMPTS:-8}
if [[ "$workload" == synthetic ]]; then
  BENCH_ARGS=(
    --backend vllm
    --dataset-name random
    --random-input-len 1024
    --random-output-len 256
    --num-prompts "$NPROMPTS"
  )
else
  BENCH_ARGS=(
    --backend vllm
    --dataset-name sharegpt
    --dataset-path "$DATASET"
    --num-prompts "$NPROMPTS"
    --output-len 256
  )
fi

CMD_LOG=$PROF_DIR/bench.log
log "starting rocprofv3 + offline vllm bench throughput (model=$model tp=$tp, n=$NPROMPTS)"
# --process-sync makes rocprofv3 wait for sub-processes to flush their
# CSVs before the wrapper returns. Critical for vLLM TP>=2 because
# each TP worker is a separate Python process.
# Use %pid% template so each TP worker writes its own file.
nohup rocprofv3 \
  --kernel-trace \
  --process-sync \
  -d "$PROF_DIR" \
  -o "rocprof_${cell_id}_pid%pid%" \
  -f csv \
  -- \
  $PY -m vllm.entrypoints.cli.main bench throughput \
    --model "$model" \
    --dtype float16 \
    --tensor-parallel-size "$tp" \
    --max-model-len 32768 \
    --block-size 32 \
    --enable-prefix-caching \
    --language-model-only \
    --gpu-memory-utilization 0.93 \
    --trust-remote-code \
    --seed 42 \
    $EXTRA "${BENCH_ARGS[@]}" \
    > "$CMD_LOG" 2>&1 &
WPID=$!
echo "$WPID" > "$PROF_DIR/wrapper.pid"
log "  rocprofv3 PID=$WPID"

# Just wait for the wrapper to exit (offline bench will exit cleanly).
# Allow up to 30 minutes (covers slow w4a16 tp=1 cold-start + 8 prompts).
for i in $(seq 1 1800); do
  if ! kill -0 "$WPID" 2>/dev/null; then
    log "  wrapper exited cleanly after ${i}s"; break
  fi
  sleep 1
done
if kill -0 "$WPID" 2>/dev/null; then
  log "  wrapper still alive after 1800s -- killing"
  kill -KILL "$WPID" 2>/dev/null || true
  sleep 2
fi
# Give rocprofv3 a moment to finish writing CSVs; do NOT kill any
# residual rocprofv3 processes here -- their finalize handlers must
# run.  Only orphan vllm/VLLM workers from a prior crash should be
# cleaned up.  Wait 30 s for in-flight finalization.
sleep 30
# Now we can safely reap orphan workers.
pgrep -af 'VLLM::' 2>/dev/null
pgrep -f 'VLLM::' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 3

# Locate the kernel-trace CSV.
KFILE=$(find "$PROF_DIR" -name "*kernel_trace*.csv" 2>/dev/null | head -1)
N_RECORDS=0
if [[ -n "$KFILE" && -s "$KFILE" ]]; then
  N_RECORDS=$(($(wc -l < "$KFILE") - 1))
  log "  kernel-trace records: $N_RECORDS  ($KFILE)"
else
  log "  WARN no kernel-trace CSV; checking for alternates"
  find "$PROF_DIR" -maxdepth 4 -name "*.csv" 2>&1 | head -10
fi

cat > "$PROF_DIR/profile_summary.json" <<EOF
{
  "cell_id": "${cell_id}",
  "quant": "${quant}",
  "tp": ${tp},
  "workload": "${workload}",
  "tracer": "rocprofv3 1.2.0 (kernel-trace)",
  "harness": "vllm bench throughput (offline) -- substitute for bench serve under profiler",
  "kernel_trace_csv": "${KFILE}",
  "n_kernel_records": ${N_RECORDS},
  "bench_log": "${CMD_LOG}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
log "  summary -> $PROF_DIR/profile_summary.json"
log "done."
exit 0
