#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Per-cell HBM-mission launch script — w4a16_tp4_c2
#
# VAL-FINAL-006: pinned env + M1+M2+M3 winners baked in for w4a16_tp4_c2.
# Reproduces the M4-measured synthetic-workload output_throughput within ±2 %.
#
# Sibling to scripts/launch_w4a16_tp4_c2.sh (production baseline). This HBM
# variant bakes in the cumulative MI100 HBM-Optimization mission winners
# as exported environment variables in the "Pinned environment" block:
#
#   KV_CACHE_DTYPE=int8_per_token_head           (M1 KV-INT8 winner)
#   ENABLE_CHUNKED_PREFILL=1                      (M2 chunked-prefill)
#   MAX_NUM_BATCHED_TOKENS=4096                       (M2 per-quant optimum)
#   NCCL_ALGO=(empty — rccl heuristic)                                (M3 per-cell winner, TP=4 only)
#
# Each of those env vars is read by the *same* CLI-flag-array logic that the
# production scripts use, so the resulting vLLM CLI is byte-identical
# whether the env var is exported at the top of this script or set by the
# caller. Disable path: `unset <FLAG>` then invoke this script — falls
# back to the corresponding non-HBM behavior on a flag-by-flag basis. The
# canonical "production baseline" disable is to invoke
# `scripts/launch_w4a16_tp4_c2.sh` instead (which carries the prior FINAL.md
# reference number); the disable-path smoke harness validates that
# unsetting any single M4 flag here reproduces the production
# output_throughput_toks_s within ±5 %.
#
# Usage:
#   launch_hbm_w4a16_tp4_c2.sh [--check]      # run the cell, ±2 % gate vs ref
#   launch_hbm_w4a16_tp4_c2.sh --serve-only   # start server, no bench
#
# Recorded reference (from /root/bench-int8-w4a16-hbm/m4-final/w4a16/w4a16_tp4_c2_synthetic.json):
#   output_throughput_toks_s (synthetic, num_prompts=200) = 106.113438
#
# Pinned versions:
#   vLLM commit:  85a6a0b751d5ce6a8386d5ed809bec80c6b6c162
#   ROCm:         7.12 at /opt/rocm/core-7.12
#   torch:        2.11.0+rocm7.2
#   Triton:       3.5.1
#   Tuning prov:  TensileLite library + per-shape JSONs (SHA256 pinned in
#                 /root/bench-int8-w4a16-hbm/m4-final/tuning_hashes_hbm.json)
set -euo pipefail

cell_id=w4a16_tp4_c2
model_path=/models/Qwen3.5-9B-w4a16
tp=4
conc=2
ref_tput=106.113438

# ---------------------------------------------------------------------------
# Pinned environment (HBM-mission stack: M1+M2+M3 winners)
# ---------------------------------------------------------------------------
export ROCM_PATH=/opt/rocm/core-7.12
export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
export PATH=/opt/rocm/core-7.12/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1

# M1 KV-INT8 winner ----------------------------------------------------------
export KV_CACHE_DTYPE=int8_per_token_head

# M2 chunked-prefill winner --------------------------------------------------
export ENABLE_CHUNKED_PREFILL=1
export MAX_NUM_BATCHED_TOKENS=4096

# M3 NCCL_ALGO winner --------------------------------------------------------
# (rccl heuristic default — leave NCCL_ALGO unset; sweep delta < 1 % for w4a16_tp4_c2)

# Tuning provenance — pinned to the merged TensileLite library + the
# per-shape JSONs that ship in-tree. SHA256 hashes pinned in
# /root/bench-int8-w4a16-hbm/m4-final/tuning_hashes_hbm.json; verify with
# `/opt/vllm-env/bin/python3 scripts/verify_tuning_hashes.py --hbm`.
export HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library
export TUNING_JSON_DIR=vllm/model_executor/kernels/configs/gfx908

# Pinned vLLM SHA / ROCm / torch / Triton  ----------------------------------
export PINNED_VLLM_COMMIT=85a6a0b751d5ce6a8386d5ed809bec80c6b6c162
export PINNED_ROCM=7.12
export PINNED_TORCH=2.11.0+rocm7.2
export PINNED_TRITON=3.5.1

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/cold-points-sit-rancb
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Optional additive env-var → CLI-flag overrides (matches production scripts)
# ---------------------------------------------------------------------------
KV_CACHE_DTYPE_FLAG=()
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  KV_CACHE_DTYPE_FLAG=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

MAX_NUM_BATCHED_TOKENS_FLAG=()
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  MAX_NUM_BATCHED_TOKENS_FLAG=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

ENABLE_CHUNKED_PREFILL_FLAG=()
if [[ -n "${ENABLE_CHUNKED_PREFILL:-}" ]]; then
  ENABLE_CHUNKED_PREFILL_FLAG=(--enable-chunked-prefill)
fi

CUDAGRAPH_MODE_FLAG=()
if [[ -n "${CUDAGRAPH_MODE:-}" ]]; then
  CUDAGRAPH_MODE_FLAG=(--compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}")
fi

OUT_ROOT=/root/bench-int8-w4a16-hbm/m4-final/launch_smoke
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$OUT_ROOT/${cell_id}}"
mkdir -p "$SERVER_LOG_DIR"
LOG="$SERVER_LOG_DIR/server.log"

# ---------------------------------------------------------------------------
# CLI arg parsing (backwards-compatible: no args → mode=--check, port=8000)
# ---------------------------------------------------------------------------
PORT=8000
mode=--check
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --serve-only|--check)
      mode="$1"
      shift
      ;;
    *)
      echo "[launch_hbm_$cell_id] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
stop_server() {
  # Default mode (no HIP_VISIBLE_DEVICES override, port 8000) keeps the
  # original global pkill teardown — required by AGENTS.md anti-pattern #8.
  # Parallel mode (caller pins HIP_VISIBLE_DEVICES and/or a non-8000 port)
  # kills only the server we spawned, so sibling cells survive.
  if [[ -z "${HIP_VISIBLE_DEVICES:-}" && "${PORT:-8000}" == "8000" ]]; then
    pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
    pkill -9 -f 'VLLM::' 2>/dev/null || true
    sleep 2
  elif [[ -n "${SERVER_PID:-}" ]]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
    sleep 1
  fi
}

trap stop_server EXIT
stop_server

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
export VLLM_MI100_DISABLE_CUSTOM_AR=1

/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --block-size 32 \
    --enable-prefix-caching \
    --language-model-only \
    --gpu-memory-utilization ${LAUNCH_GPU_MEM_UTIL:-0.93} \
    --port "$PORT"  \
    --disable-custom-all-reduce "${KV_CACHE_DTYPE_FLAG[@]}" "${MAX_NUM_BATCHED_TOKENS_FLAG[@]}" "${ENABLE_CHUNKED_PREFILL_FLAG[@]}" "${CUDAGRAPH_MODE_FLAG[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
echo "[launch_hbm_$cell_id] server PID=$SERVER_PID; log=$LOG"

# Health wait (default 300 s; override via LAUNCH_HEALTH_WAIT_SECS).
HEALTH_WAIT_SECS=${LAUNCH_HEALTH_WAIT_SECS:-300}
poll_count=$(( HEALTH_WAIT_SECS / 5 ))
for i in $(seq 1 "$poll_count"); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[launch_hbm_$cell_id] healthcheck OK after ${i} x5 s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[launch_hbm_$cell_id] FATAL: server exited; tail:"
    tail -n 60 "$LOG"
    exit 2
  fi
  sleep 5
done
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[launch_hbm_$cell_id] FATAL: server did not become healthy in ${HEALTH_WAIT_SECS} s"
  tail -n 60 "$LOG"
  exit 2
fi

if [[ "$mode" == "--serve-only" ]]; then
  trap - EXIT
  echo "[launch_hbm_$cell_id] server ready; pid=$SERVER_PID (you must kill it manually)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Canonical bench (synthetic random, NUM_PROMPTS=200, seed=42)
# ---------------------------------------------------------------------------
RAW_DIR="$OUT_ROOT/${cell_id}/bench_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RAW_DIR"

/opt/vllm-env/bin/python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$model_path" \
    --base-url "http://127.0.0.1:${PORT}" \
    --num-prompts 200 \
    --request-rate inf \
    --max-concurrency 2 \
    --seed 42 \
    --save-result \
    --result-dir "$RAW_DIR" \
    --result-filename raw.json \
    --trust-remote-code \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --metadata cell_id=${cell_id} workload=synthetic tp=4 concurrency=2 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --ignore-eos

result_json="$RAW_DIR/raw.json"
if [[ ! -f "$result_json" ]]; then
  echo "[launch_hbm_$cell_id] FATAL: bench produced no result.json"; exit 2
fi

actual_tput=$(/opt/vllm-env/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_throughput"])' "$result_json")

echo "[launch_hbm_$cell_id] M4 recorded reference output_throughput = $ref_tput tok/s"
echo "[launch_hbm_$cell_id] this run     output_throughput        = $actual_tput tok/s"

if [[ -z "$ref_tput" || "$ref_tput" == "None" ]]; then
  echo "[launch_hbm_$cell_id] no recorded reference — skipping ±2 % gate"
  exit 0
fi

verdict=$(/opt/vllm-env/bin/python3 -c "
import sys
ref=float(sys.argv[1]); act=float(sys.argv[2])
delta=(act-ref)/ref*100
print(f'delta={delta:+.2f}%')
sys.exit(0 if abs(delta)<=2.0 else 1)
" "$ref_tput" "$actual_tput")
rc=$?
echo "[launch_hbm_$cell_id] $verdict"
exit $rc
