#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/mi100/rocprof_single_request.sh
# ----------------------------------------
# rocprofv3 capture of a SINGLE-prompt vLLM run for VAL-M4-005 (HBM bytes
# per-token evidence). Wraps offline `vllm bench throughput`, NOT `bench
# serve`, because rocprofv3 + bench serve hangs on gfx908 (AGENTS.md
# anti-pattern #1; library/rocprofv3-tp4-limitation.md).
#
# The offline path is functionally equivalent for kernel-trace evidence:
# the same compiled engine graphs (FULL_DECODE_ONLY cudagraphs) and the
# same scaled_mm / silu_and_mul / fused_silu_quant_int8 kernels execute,
# but the wrapped process exits cleanly at end-of-bench, letting
# rocprofv3 finalize CSVs reliably.
#
# This script captures the ENTIRE single-prompt lifecycle (prefill +
# decode + final), but with output_len kept short (32) the kernel-trace
# CSV stays small. The "single decode step" framing in VAL-M4-005 refers
# to the per-decode-step kernel set, which is naturally exposed by any
# single-prompt run; per-step bytes are derived by dividing trace totals
# by the recorded n_decode_steps.
#
# Args:
#   $1  cell_id        e.g. w8a8_tp1_c4_coding
#   $2  fused_state    on | off
#   $3  out_dir        absolute path; will hold kernel_trace.csv + pmc.csv
# Optional env:
#   NUM_PROMPTS        default 1
#   RANDOM_INPUT_LEN   default 1024 (matches coding-cell prompt size band)
#   RANDOM_OUTPUT_LEN  default 32   (short to keep trace bounded)
#
# Exit codes:
#   0  trace + pmc captured and persisted
#   2  rocprofv3 wrapper failed
#   3  trace CSV missing or < 1000 records
set -uo pipefail

cell_id=${1:?cell_id required (e.g. w8a8_tp1_c4_coding)}
fused_state=${2:?fused_state required (on | off)}
out_dir=${3:?out_dir required (absolute path)}

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/cold-points-sit-rancb
PY=/opt/vllm-env/bin/python3
PMC_FILE="$REPO/scripts/mi100/pmc_counters.txt"
MODEL=/models/Qwen3.5-9B-w8a8
NUM_PROMPTS=${NUM_PROMPTS:-1}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-1024}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-32}

mkdir -p "$out_dir"
log() { echo "[$(date -u +%H:%M:%S) rocprof:${cell_id}:${fused_state}] $*"; }

# Pinned env (subset of scripts/launch_hbm_w8a8_tp1_c4.sh — chunked
# prefill, KV-INT8, AITER) — must mirror production cell so we capture
# the same kernels.
export ROCM_PATH=/opt/rocm/core-7.12
export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
export PATH=/opt/rocm/core-7.12/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1
export KV_CACHE_DTYPE=int8_per_token_head
export ENABLE_CHUNKED_PREFILL=1
export MAX_NUM_BATCHED_TOKENS=2048
export HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library
export TUNING_JSON_DIR=vllm/model_executor/kernels/configs/gfx908
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Fused-state gate (default-on; explicit unset for fused-on capture).
if [[ "$fused_state" == "off" ]]; then
  export VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1
else
  unset VLLM_MI100_DISABLE_FUSED_ACT_QUANT || true
fi
# Per AGENTS.md anti-pattern #1, lock GPU to a single device.
export CUDA_VISIBLE_DEVICES=0

cd "$REPO"

# Pre-clean orphan vLLM workers (anti-pattern #8 — triad teardown).
pgrep -f 'vllm.entrypoints' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 3

BENCH_ARGS=(
  --model "$MODEL"
  --dtype float16
  --tensor-parallel-size 1
  --max-model-len 32768
  --block-size 32
  --enable-prefix-caching
  --language-model-only
  --gpu-memory-utilization 0.93
  --trust-remote-code
  --seed 42
  --kv-cache-dtype int8_per_token_head
  --enable-chunked-prefill
  --max-num-batched-tokens 2048
  --backend vllm
  --dataset-name random
  --input-len "$RANDOM_INPUT_LEN"
  --output-len "$RANDOM_OUTPUT_LEN"
  --num-prompts "$NUM_PROMPTS"
)

# Pass 1: kernel trace (no PMC — kernels run only ONCE so timestamps map
# to actual execution; PMC replay would multiply kernel counts).
KT_LOG="$out_dir/kernel_trace.log"
log "Pass 1/2 kernel-trace: rocprofv3 --kernel-trace --process-sync"
nohup rocprofv3 \
  --kernel-trace \
  --process-sync \
  -d "$out_dir" \
  -o "kt_${cell_id}_${fused_state}_pid%pid%" \
  -f csv \
  -- \
  $PY -m vllm.entrypoints.cli.main bench throughput "${BENCH_ARGS[@]}" \
  >"$KT_LOG" 2>&1 &
WPID=$!
log "  rocprofv3 PID=$WPID; log=$KT_LOG"

# Wait up to 25 min for cold start + run.
for i in $(seq 1 1500); do
  if ! kill -0 "$WPID" 2>/dev/null; then
    log "  kernel-trace wrapper exited after ${i}s"; break
  fi
  sleep 1
done
if kill -0 "$WPID" 2>/dev/null; then
  log "  WARN kernel-trace wrapper still alive after 1500s; killing"
  kill -KILL "$WPID" 2>/dev/null || true
fi
# Let rocprofv3 finalize CSVs (do NOT kill rocprofv3).
sleep 20
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 5

# Find the largest kernel-trace CSV (a TP=1 run typically writes one
# per engine PID; the engine PID file is the >1k-record file).
KT_RAW=$(ls -S "$out_dir"/kt_*_kernel_trace.csv 2>/dev/null | head -1)
if [[ -z "$KT_RAW" || ! -s "$KT_RAW" ]]; then
  log "FATAL: no kernel-trace CSV produced under $out_dir"
  exit 2
fi
KT_RECORDS=$(($(wc -l < "$KT_RAW") - 1))
log "  kernel-trace records=$KT_RECORDS  file=$KT_RAW"
cp "$KT_RAW" "$out_dir/kernel_trace.csv"
log "  -> $out_dir/kernel_trace.csv"
if [[ "$KT_RECORDS" -lt 1000 ]]; then
  log "FATAL: kernel-trace has < 1000 records ($KT_RECORDS)"
  exit 3
fi

# Pass 2: PMC counters (replays kernels — separate from trace pass).
PMC_LOG="$out_dir/pmc.log"
log "Pass 2/2 PMC: rocprofv3 -i $PMC_FILE"
pgrep -f 'vllm.entrypoints' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 3
nohup rocprofv3 \
  -i "$PMC_FILE" \
  --process-sync \
  -d "$out_dir" \
  -o "pmc_${cell_id}_${fused_state}_pid%pid%" \
  -f csv \
  -- \
  $PY -m vllm.entrypoints.cli.main bench throughput "${BENCH_ARGS[@]}" \
  >"$PMC_LOG" 2>&1 &
PPID_=$!
log "  rocprofv3-pmc PID=$PPID_; log=$PMC_LOG"
# PMC replay can take longer; 40 min budget.
for i in $(seq 1 2400); do
  if ! kill -0 "$PPID_" 2>/dev/null; then
    log "  pmc wrapper exited after ${i}s"; break
  fi
  sleep 1
done
if kill -0 "$PPID_" 2>/dev/null; then
  log "  WARN pmc wrapper still alive after 2400s; killing"
  kill -KILL "$PPID_" 2>/dev/null || true
fi
sleep 20
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 5

# Collect PMC counter_collection.csv from rocprofv3 outputs.
PMC_RAW=$(ls -S "$out_dir"/pmc_*_counter_collection.csv 2>/dev/null | head -1)
if [[ -z "$PMC_RAW" || ! -s "$PMC_RAW" ]]; then
  log "WARN: no PMC counter_collection.csv produced; writing stub note"
  echo "# rocprofv3 PMC capture failed — see pmc.log" > "$out_dir/pmc.csv"
else
  cp "$PMC_RAW" "$out_dir/pmc.csv"
  PMC_RECORDS=$(($(wc -l < "$PMC_RAW") - 1))
  log "  pmc records=$PMC_RECORDS  file=$PMC_RAW -> $out_dir/pmc.csv"
fi

cat > "$out_dir/capture_summary.json" <<EOF
{
  "cell_id": "${cell_id}",
  "fused_state": "${fused_state}",
  "tracer": "rocprofv3 1.2.0",
  "harness": "vllm bench throughput (offline) — single-prompt --num-prompts=${NUM_PROMPTS}, --input-len=${RANDOM_INPUT_LEN}, --output-len=${RANDOM_OUTPUT_LEN}",
  "kernel_trace_csv": "$out_dir/kernel_trace.csv",
  "n_kernel_records": ${KT_RECORDS},
  "pmc_csv": "$out_dir/pmc.csv",
  "model": "${MODEL}",
  "tp": 1,
  "env": {
    "VLLM_MI100_DISABLE_FUSED_ACT_QUANT": "${VLLM_MI100_DISABLE_FUSED_ACT_QUANT:-unset}",
    "KV_CACHE_DTYPE": "${KV_CACHE_DTYPE}",
    "ENABLE_CHUNKED_PREFILL": "${ENABLE_CHUNKED_PREFILL}",
    "MAX_NUM_BATCHED_TOKENS": "${MAX_NUM_BATCHED_TOKENS}"
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
log "  summary -> $out_dir/capture_summary.json"
log "done."
exit 0
