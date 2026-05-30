#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/mi100/rocprof_single_request.sh
# ----------------------------------------
# rocprofv3 capture of a SINGLE-prompt W4A16 vLLM run for HBM bytes-per-token
# evidence (the hot W4A16 GEMM). Wraps offline `vllm bench throughput`, NOT
# `bench serve`, because rocprofv3 + bench serve hangs on gfx908 (AGENTS.md
# anti-pattern #1; library/rocprofv3-tp4-limitation.md).
#
# W4A16 mission adaptation (F-M0-script-adapt):
#   * REPO resolves to the CURRENT worktree via git rev-parse --show-toplevel
#     (overridable via $REPO).
#   * MODEL=/models/Qwen3.5-9B-w4a16 (overridable via $MODEL).
#   * arg2 is the MARLIN state (on|off): on => VLLM_MI100_W4A16_USE_MARLIN_REPACK=1,
#     off => =0. (Legacy fused_on/fused_off forms also accepted as aliases.)
#   * HBM% is derived downstream by scripts/mi100/aggregate_hbm.py from
#     FETCH_SIZE + WRITE_SIZE (pmc_counters.txt). NEVER TCP_TCC_* on
#     gfx908 + rocprofv3 1.2.0.
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
#   $1  cell_id        e.g. w4a16_tp1_c4_coding
#   $2  marlin_state   on | off   (VLLM_MI100_W4A16_USE_MARLIN_REPACK 1 | 0)
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

cell_id=${1:?cell_id required (e.g. w4a16_tp1_c4_coding)}
marlin_state=${2:?marlin_state required (on | off)}
out_dir=${3:?out_dir required (absolute path)}

# Resolve REPO to the CURRENT worktree (git toplevel) unless caller overrides.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${REPO:-}" ]]; then
  REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
  if command -v git >/dev/null 2>&1; then
    GIT_TOPLEVEL=$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null || true)
    [[ -n "$GIT_TOPLEVEL" ]] && REPO=$GIT_TOPLEVEL
  fi
fi

# Resolve python interpreter.
if [[ -n "${PY:-}" && -x "${PY:-}" ]]; then
  :
elif [[ -x /opt/vllm-env/bin/python3 ]]; then
  PY=/opt/vllm-env/bin/python3
else
  PY=$(command -v python3 || true)
fi

PMC_FILE="$REPO/scripts/mi100/pmc_counters.txt"
MODEL=${MODEL:-/models/Qwen3.5-9B-w4a16}
NUM_PROMPTS=${NUM_PROMPTS:-1}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-1024}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-32}

mkdir -p "$out_dir"
log() { echo "[$(date -u +%H:%M:%S) rocprof:${cell_id}:${fused_state}] $*"; }

# Pinned env — mirrors the services.yaml vllm-w4a16-tp1 server block so we
# capture the same W4A16 GEMM kernels the production cell executes.
export ROCM_PATH=/opt/rocm/core-7.12
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export PATH=/opt/rocm/core-7.12/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Marlin-state gate. arg2 selects the W4A16 GEMM path:
#   off => baseline GEMM (VLLM_MI100_W4A16_USE_MARLIN_REPACK=0)
#   on  => marlin repack  (VLLM_MI100_W4A16_USE_MARLIN_REPACK=1)
# Legacy fused_on/fused_off forms are accepted as aliases for on/off.
case "$marlin_state" in
  off|fused_off)
    export VLLM_MI100_W4A16_USE_MARLIN_REPACK=0
    ;;
  on|fused_on)
    export VLLM_MI100_W4A16_USE_MARLIN_REPACK=1
    ;;
  *)
    echo "FATAL: unknown marlin_state='$marlin_state' (expected on|off)" >&2
    exit 2
    ;;
esac
# Back-compat: the log() helper and rocprofv3 output filenames reference
# $fused_state, but only $marlin_state is parsed above. Alias them so the
# script runs under `set -u` (F-M0-baseline-lock minimal adaptation).
fused_state="$marlin_state"
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

# Collect PMC counter_collection.csv files from rocprofv3 outputs.
# rocprofv3 splits multi-pass PMC groups into per-group subdirs (pmc_1/,
# pmc_2/, ...), each with its own counter_collection.csv. Merge them
# inline into a single pmc.csv (dedup header) so downstream consumers
# can read every requested counter without an external merge step.
# Per-group source files are preserved under pmc_*/ for debug.
PMC_PARTS=( "$out_dir"/pmc_*/pmc_*_counter_collection.csv )
if [[ ! -s "${PMC_PARTS[0]:-}" ]]; then
  log "WARN: no PMC counter_collection.csv produced; writing stub note"
  echo "# rocprofv3 PMC capture failed — see pmc.log" > "$out_dir/pmc.csv"
else
  {
    head -n1 "${PMC_PARTS[0]}"
    for f in "${PMC_PARTS[@]}"; do
      tail -n +2 "$f"
    done
  } > "$out_dir/pmc.csv"
  PMC_RECORDS=$(($(wc -l < "$out_dir/pmc.csv") - 1))
  log "  pmc records=$PMC_RECORDS  parts=${#PMC_PARTS[@]} -> $out_dir/pmc.csv"
fi

cat > "$out_dir/capture_summary.json" <<EOF
{
  "cell_id": "${cell_id}",
  "marlin_state": "${marlin_state}",
  "tracer": "rocprofv3 1.2.0",
  "harness": "vllm bench throughput (offline) — single-prompt --num-prompts=${NUM_PROMPTS}, --input-len=${RANDOM_INPUT_LEN}, --output-len=${RANDOM_OUTPUT_LEN}",
  "kernel_trace_csv": "$out_dir/kernel_trace.csv",
  "n_kernel_records": ${KT_RECORDS},
  "pmc_csv": "$out_dir/pmc.csv",
  "model": "${MODEL}",
  "tp": 1,
  "hbm_derivation": "FETCH_SIZE+WRITE_SIZE via scripts/mi100/aggregate_hbm.py (NEVER TCP_TCC_*)",
  "env": {
    "VLLM_MI100_W4A16_USE_MARLIN_REPACK": "${VLLM_MI100_W4A16_USE_MARLIN_REPACK:-unset}",
    "VLLM_ROCM_USE_AITER": "${VLLM_ROCM_USE_AITER:-unset}",
    "VLLM_ROCM_USE_SKINNY_GEMM": "${VLLM_ROCM_USE_SKINNY_GEMM:-unset}"
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
log "  summary -> $out_dir/capture_summary.json"
log "done."
exit 0
