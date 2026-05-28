#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/awq_ab_rocprof.sh — M3-F1 rocprofv3 capture wrapper for the
# AWQ vs GPTQ A/B mission.
#
# Captures rocprofv3 1.2.0 traces for one (path, cell) tuple. Wraps the
# OFFLINE `vllm bench throughput` invocation (NOT `bench serve`) per
# AGENTS.md anti-pattern #1 / library/rocprofv3-tp4-limitation.md: the
# `bench serve` path hangs on gfx908 under rocprofv3 instrumentation.
#
# Two rocprofv3 passes per cell:
#   1) kernel-trace only (no PMC) — kernels run once, timestamps map to
#      actual execution; CSV is the "kernel rows" the validator checks.
#   2) PMC (multi-pass: FETCH_SIZE+WRITE_SIZE, then VALUUtilization+
#      SQ_INSTS_MFMA) — kernel replay multiplies kernel counts; only
#      used for HBM/VALU/MFMA derived metrics.
#
# Args:
#   $1  path      a | b | c
#   $2  cell      w4a16_tp1_c1_synthetic | w4a16_tp4_c4_coding
#
# Output dir: /root/bench-w4a16-ab/rocprof/<path>/<cell>/
#   - kt_*_kernel_trace.csv      (kernel-trace pass)
#   - pmc_*/pmc_*_counter_collection.csv   (per-PMC-group files)
#   - pmc.csv                    (concatenated PMC)
#   - kernel_trace.log, pmc.log, capture_summary.json
#
# Per the feature description: use --num-prompts 20 to keep trace size
# manageable. Each capture dir must contain ≥1 CSV with non-empty kernel
# rows after the run.
set -uo pipefail

path=${1:?path required (a|b|c)}
cell=${2:?cell required (w4a16_tp1_c1_synthetic|w4a16_tp4_c4_coding)}

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/puny-animals-go-wp0to
PY=/opt/vllm-env/bin/python3
PMC_FILE="$REPO/scripts/awq_ab_pmc_counters.txt"
OUT_DIR=/root/bench-w4a16-ab/rocprof/${path}/${cell}
mkdir -p "$OUT_DIR"

log() { echo "[$(date -u +%H:%M:%S) rocprof:${path}:${cell}] $*"; }

# --- Model + TP per path + cell -------------------------------------------
case "$path" in
  a) MODEL=/models/Qwen3.5-9B-w4a16; QUANT_FLAG=() ;;
  b) MODEL=/models/Qwen3.5-9B-AWQ-INT4; QUANT_FLAG=() ;;
  c) MODEL=/models/Qwen3.5-9B-AWQ-gemm; QUANT_FLAG=(--quantization awq) ;;
  *) echo "FATAL: unknown path=$path"; exit 2 ;;
esac

case "$cell" in
  w4a16_tp1_c1_synthetic)
    TP=1
    INPUT_LEN=1024
    OUTPUT_LEN=256
    ;;
  w4a16_tp4_c4_coding)
    TP=4
    INPUT_LEN=1024
    OUTPUT_LEN=32  # coding-like short outputs; keeps PMC replay tractable
    ;;
  *) echo "FATAL: unknown cell=$cell"; exit 2 ;;
esac

# --- Pinned env (mirrors scripts/launch_ab_<path>_w4a16_*.sh M4 stack) ----
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
export MAX_NUM_BATCHED_TOKENS=4096
export HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library
export TUNING_JSON_DIR=vllm/model_executor/kernels/configs/gfx908
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$TP" == "4" ]]; then
  export NCCL_ALGO=Ring
  export VLLM_MI100_DISABLE_CUSTOM_AR=1
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES=0
  unset NCCL_ALGO || true
fi

cd "$REPO"

# Pre-clean orphan vLLM workers
pgrep -f 'vllm.entrypoints' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 3

HARNESS=(
  $PY -m vllm.entrypoints.cli.main bench throughput
  --model "$MODEL"
  --dtype float16
  --tensor-parallel-size "$TP"
  --max-model-len 32768
  --block-size 32
  --enable-prefix-caching
  --language-model-only
  --gpu-memory-utilization 0.93
  --trust-remote-code
  --seed 42
  --kv-cache-dtype int8_per_token_head
  --enable-chunked-prefill
  --max-num-batched-tokens 4096
  "${QUANT_FLAG[@]}"
  --backend vllm
  --dataset-name random
  --input-len "$INPUT_LEN"
  --output-len "$OUTPUT_LEN"
  --num-prompts 20
)
if [[ "$TP" == "4" ]]; then
  HARNESS+=(--disable-custom-all-reduce)
  # Skip cudagraph capture for TP=4 — reduces sources of subprocess SIGTERM
  # during teardown that prevent rocprofv3 from flushing worker CSVs.
  HARNESS+=(--enforce-eager)
fi

# --- Pass 1: kernel-trace -------------------------------------------------
KT_LOG="$OUT_DIR/kernel_trace.log"
log "Pass 1/2 kernel-trace: model=$MODEL TP=$TP in=$INPUT_LEN out=$OUTPUT_LEN"
setsid nohup /opt/rocm/core-7.12/bin/rocprofv3 \
  --kernel-trace \
  --process-sync \
  -d "$OUT_DIR" \
  -o "kt_${path}_${cell}_pid%pid%" \
  --output-format csv \
  -- \
  setsid -w \
  "${HARNESS[@]}" \
  >"$KT_LOG" 2>&1 &
WPID=$!
log "  rocprofv3 PID=$WPID; log=$KT_LOG"

# Up to 35 min for cold-start + bench (path B group_size=32 + path C dequant slow).
TIMEOUT_S=2100
for i in $(seq 1 "$TIMEOUT_S"); do
  if ! kill -0 "$WPID" 2>/dev/null; then
    log "  kernel-trace wrapper exited after ${i}s"; break
  fi
  sleep 1
done
if kill -0 "$WPID" 2>/dev/null; then
  log "  WARN kernel-trace wrapper still alive after ${TIMEOUT_S}s; killing"
  kill -KILL "$WPID" 2>/dev/null || true
fi
sleep 20
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 5

KT_RAW=$(ls -S "$OUT_DIR"/kt_*_kernel_trace.csv 2>/dev/null | head -1)
if [[ -z "$KT_RAW" || ! -s "$KT_RAW" ]]; then
  log "FATAL: no kernel-trace CSV produced under $OUT_DIR"
  ls -la "$OUT_DIR" || true
  tail -n 40 "$KT_LOG" || true
  exit 2
fi
KT_RECORDS=$(($(wc -l < "$KT_RAW") - 1))
log "  kernel-trace records=$KT_RECORDS  file=$KT_RAW"
cp "$KT_RAW" "$OUT_DIR/kernel_trace.csv"

# --- Pass 2: PMC ----------------------------------------------------------
PMC_LOG="$OUT_DIR/pmc.log"
log "Pass 2/2 PMC: counters from $PMC_FILE"
pgrep -f 'vllm.entrypoints' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 3
setsid nohup /opt/rocm/core-7.12/bin/rocprofv3 \
  -i "$PMC_FILE" \
  --process-sync \
  -d "$OUT_DIR" \
  -o "pmc_${path}_${cell}_pid%pid%" \
  --output-format csv \
  -- \
  "${HARNESS[@]}" \
  >"$PMC_LOG" 2>&1 &
PPID_=$!
log "  rocprofv3-pmc PID=$PPID_; log=$PMC_LOG"
# PMC replay: each pmc group replays the workload; 2 groups → up to 2× kernel-trace wall time.
PMC_TIMEOUT_S=4200
for i in $(seq 1 "$PMC_TIMEOUT_S"); do
  if ! kill -0 "$PPID_" 2>/dev/null; then
    log "  pmc wrapper exited after ${i}s"; break
  fi
  sleep 1
done
if kill -0 "$PPID_" 2>/dev/null; then
  log "  WARN pmc wrapper still alive after ${PMC_TIMEOUT_S}s; killing"
  kill -KILL "$PPID_" 2>/dev/null || true
fi
sleep 20
pgrep -f 'VLLM::EngineCore' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 5

# Concatenate per-group PMC counter CSVs into a single pmc.csv
PMC_PARTS=( "$OUT_DIR"/pmc_*/pmc_*_counter_collection.csv )
if [[ ! -s "${PMC_PARTS[0]:-}" ]]; then
  log "WARN: no PMC counter_collection.csv produced; pmc.csv will be a stub"
  echo "# rocprofv3 PMC capture failed — see pmc.log" > "$OUT_DIR/pmc.csv"
  PMC_RECORDS=0
else
  {
    head -n1 "${PMC_PARTS[0]}"
    for f in "${PMC_PARTS[@]}"; do
      tail -n +2 "$f"
    done
  } > "$OUT_DIR/pmc.csv"
  PMC_RECORDS=$(($(wc -l < "$OUT_DIR/pmc.csv") - 1))
  log "  pmc records=$PMC_RECORDS  parts=${#PMC_PARTS[@]} -> $OUT_DIR/pmc.csv"
fi

cat > "$OUT_DIR/capture_summary.json" <<EOF
{
  "path": "${path}",
  "cell": "${cell}",
  "model": "${MODEL}",
  "tp": ${TP},
  "tracer": "rocprofv3 1.2.0",
  "harness": "vllm bench throughput (offline) --num-prompts=20 --input-len=${INPUT_LEN} --output-len=${OUTPUT_LEN}",
  "kernel_trace_csv": "$OUT_DIR/kernel_trace.csv",
  "n_kernel_records": ${KT_RECORDS},
  "pmc_csv": "$OUT_DIR/pmc.csv",
  "n_pmc_records": ${PMC_RECORDS},
  "pmc_counters": "FETCH_SIZE WRITE_SIZE / VALUUtilization SQ_INSTS_MFMA",
  "env": {
    "KV_CACHE_DTYPE": "${KV_CACHE_DTYPE}",
    "ENABLE_CHUNKED_PREFILL": "${ENABLE_CHUNKED_PREFILL}",
    "MAX_NUM_BATCHED_TOKENS": "${MAX_NUM_BATCHED_TOKENS}",
    "NCCL_ALGO": "${NCCL_ALGO:-unset}",
    "VLLM_MI100_DISABLE_CUSTOM_AR": "${VLLM_MI100_DISABLE_CUSTOM_AR:-unset}",
    "CUDA_VISIBLE_DEVICES": "${CUDA_VISIBLE_DEVICES:-unset}"
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
log "  summary -> $OUT_DIR/capture_summary.json"
log "done."
exit 0
