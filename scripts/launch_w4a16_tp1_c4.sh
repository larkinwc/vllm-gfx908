#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Per-cell launch script — w4a16_tp1_c4
#
# VAL-FINAL-005: pinned env, vLLM commit, ROCm version, tuning-JSON
# provenance for w4a16 TP=1 concurrency=4.
#
# Reproduces the cell's recorded output throughput within ±2 %.
#
# Usage:
#   launch_w4a16_tp1_c4.sh [--check]   # run the cell, compare to recorded ref
#   launch_w4a16_tp1_c4.sh --serve-only # just start the server, no bench
#
# Recorded reference (from /root/bench-int8-w4a16/final/final_grid.csv):
#   winner path = +Triton
#   output_throughput_toks_s (synthetic, num_prompts=200) = 104.291868
#
# Optional env-var overrides (additive — empty/unset preserves production
# behavior; resulting vLLM CLI is byte-identical to the unmodified script):
#
#   KV_CACHE_DTYPE  (m1-kvint8): when non-empty, injects
#                                `--kv-cache-dtype $KV_CACHE_DTYPE` into the
#                                vllm.entrypoints.openai.api_server invocation.
#                                Recommended value on this vLLM build:
#                                `int8_per_token_head` (the only INT8-named
#                                CacheDType in vllm 0.20.2; see
#                                vllm/config/cache.py CacheDType Literal).
#                                Other accepted values include fp8, fp8_e4m3,
#                                fp8_e5m2, fp8_inc, fp8_per_token_head,
#                                fp8_ds_mla, nvfp4. Disable path:
#                                `unset KV_CACHE_DTYPE` or `KV_CACHE_DTYPE=`
#                                returns to the production baseline (FP16 KV);
#                                resulting CLI is byte-identical to the
#                                pre-extension script.
set -euo pipefail

cell_id=w4a16_tp1_c4
model_path=/models/Qwen3.5-9B-w4a16
tp=1
conc=4
ref_tput=104.291868

# ---------------------------------------------------------------------------
# Pinned environment
# ---------------------------------------------------------------------------
export ROCM_PATH=/opt/rocm/core-7.12
export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
export PATH=/opt/rocm/core-7.12/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1

# Tuning provenance — pinned to the merged TensileLite library + the
# per-shape JSONs that ship in-tree. Their SHA256 hashes are pinned in
# /root/bench-int8-w4a16/final/tuning_hashes.json; verify with
# `scripts/verify_tuning_hashes.py`.
export HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library
export TUNING_JSON_DIR=vllm/model_executor/kernels/configs/gfx908

# vLLM commit (M4 final): 003f7d6ec (post M5 negative-result commit).
# ROCm: 7.12. PyTorch: 2.11.0+rocm7.2. Triton: 3.5.1.
export PINNED_VLLM_COMMIT=003f7d6ec
export PINNED_ROCM=7.12
export PINNED_TORCH=2.11.0+rocm7.2
export PINNED_TRITON=3.5.1

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Optional additive env-var → CLI-flag overrides (m1-kvint8 et seq.)
# Each populates a bash array that expands to the corresponding CLI flag(s)
# when its env var is non-empty, and to *nothing* when unset/empty. This
# preserves byte-identical CLI vs the pre-extension script in the default
# (unset) case while letting workers opt in to KV-INT8 etc. additively.
# ---------------------------------------------------------------------------
KV_CACHE_DTYPE_FLAG=()
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  KV_CACHE_DTYPE_FLAG=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

OUT_ROOT=/root/bench-int8-w4a16/final/launch_smoke
mkdir -p "$OUT_ROOT/${cell_id}"
LOG="$OUT_ROOT/${cell_id}/server.log"

mode=${1:---check}

# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
stop_server() {
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  sleep 2
}

trap stop_server EXIT
stop_server

# ---------------------------------------------------------------------------
# Start server (matches services.yaml: vllm-w4a16-tp1)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0

/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 32 \
    --enable-prefix-caching \
    --language-model-only \
    --gpu-memory-utilization 0.93 \
    --port 8000  "${KV_CACHE_DTYPE_FLAG[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
echo "[launch_$cell_id] server PID=$SERVER_PID; log=$LOG"

# Health wait (up to 300 s).
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "[launch_$cell_id] healthcheck OK after ${i} x5 s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[launch_$cell_id] FATAL: server exited; tail:"
    tail -n 60 "$LOG"
    exit 2
  fi
  sleep 5
done
if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  echo "[launch_$cell_id] FATAL: server did not become healthy in 300 s"
  tail -n 60 "$LOG"
  exit 2
fi

if [[ "$mode" == "--serve-only" ]]; then
  trap - EXIT
  echo "[launch_$cell_id] server ready; pid=$SERVER_PID (you must kill it manually)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Canonical bench (synthetic random, NUM_PROMPTS=200, seed=42)
# ---------------------------------------------------------------------------
RAW_DIR="$OUT_ROOT/${cell_id}/bench_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RAW_DIR"

/opt/vllm-env/bin/python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$model_path" \
    --base-url http://127.0.0.1:8000 \
    --num-prompts 200 \
    --request-rate inf \
    --max-concurrency 4 \
    --seed 42 \
    --save-result \
    --result-dir "$RAW_DIR" \
    --result-filename raw.json \
    --trust-remote-code \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --metadata cell_id=${cell_id} workload=synthetic tp=1 concurrency=4 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --ignore-eos

result_json="$RAW_DIR/raw.json"
if [[ ! -f "$result_json" ]]; then
  echo "[launch_$cell_id] FATAL: bench produced no result.json"; exit 2
fi

actual_tput=$(/opt/vllm-env/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_throughput"])' "$result_json")

echo "[launch_$cell_id] recorded reference output_throughput = $ref_tput tok/s"
echo "[launch_$cell_id] this run     output_throughput        = $actual_tput tok/s"

if [[ -z "$ref_tput" || "$ref_tput" == "None" ]]; then
  echo "[launch_$cell_id] no recorded reference — skipping ±2 % gate"
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
echo "[launch_$cell_id] $verdict"
exit $rc
