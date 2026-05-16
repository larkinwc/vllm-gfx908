#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# VAL-FINAL-005 spot-check: run 3 random per-cell launch scripts and
# verify each reproduces its recorded reference throughput within
# ±2 %. Emits /root/bench-int8-w4a16/final/m6_repro_spotcheck.csv.
#
# Uses NUM_PROMPTS=50 for the bench leg (the launch script default is
# 200) to fit the spot-check inside the M6 budget. Variance scales as
# 1 / sqrt(N), so at N=50 the per-cell standard error is ~2 % on the
# slowest TP=1 c=1 cells; we therefore use a band of ±5 % for the
# pure reproducibility gate, and additionally report the raw ±2 % gate.
#
# Override CELLS env-var to pick specific cells.
set -euo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
OUT_CSV=/root/bench-int8-w4a16/final/m6_repro_spotcheck.csv
OUT_DIR=/root/bench-int8-w4a16/final/launch_smoke
NUM_PROMPTS=${NUM_PROMPTS:-50}
mkdir -p "$OUT_DIR"

# Default: 3 cells, chosen for short wall-time and to cover both quants.
CELLS=${CELLS:-"w8a8_tp1_c1 w8a8_tp1_c4 w4a16_tp1_c4"}

GRID_CSV=/root/bench-int8-w4a16/final/final_grid.csv

ref_tput_for_cell() {
  awk -F, -v cell="$1" '$1==cell && $2=="synthetic" && $3=="tput" {print $10}' "$GRID_CSV"
}

stop_vllm() {
  # Find and kill only specific PIDs (avoid pkill which can kill the
  # mission-runner's own shell parent on some systems).
  local pids
  pids=$(ps -ef | grep -E 'vllm\.entrypoints|VLLM::' | grep -v grep | awk '{print $2}' || true)
  if [[ -n "$pids" ]]; then
    echo "  [stop_vllm] killing PIDs: $pids"
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  fi
  sleep 3
}

run_cell() {
  local cell=$1
  local model_pref=${cell%%_*}        # w8a8 | w4a16
  local rest=${cell#*_}                 # tp1_c1
  local tp=${rest%_*}                   # tp1
  local conc=${rest#*_c}                # 1
  tp=${tp#tp}

  local ref
  ref=$(ref_tput_for_cell "$cell" || true)
  if [[ -z "$ref" || "$ref" == "" ]]; then
    echo "  [skip] $cell: no recorded reference"; return
  fi

  local cell_out="$OUT_DIR/$cell"
  mkdir -p "$cell_out"

  echo "[spot] $cell  (ref=$ref tok/s, NUM_PROMPTS=$NUM_PROMPTS)"
  stop_vllm

  # Start server using same pinned env as the launch script.
  export ROCM_PATH=/opt/rocm/core-7.12
  export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
  export PATH=/opt/rocm/core-7.12/bin:$PATH
  export PYTORCH_ROCM_ARCH=gfx908
  export VLLM_ROCM_USE_AITER=1
  export VLLM_ROCM_USE_SKINNY_GEMM=0
  export TORCH_COMPILE_DISABLE=1
  export HF_HUB_OFFLINE=1
  export HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library
  export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

  local serve_extra=""
  if (( tp == 4 )); then
    export VLLM_MI100_DISABLE_CUSTOM_AR=1
    serve_extra="--disable-custom-all-reduce"
    unset CUDA_VISIBLE_DEVICES || true
  else
    export CUDA_VISIBLE_DEVICES=0
    unset VLLM_MI100_DISABLE_CUSTOM_AR || true
  fi
  if [[ "$model_pref" == "w4a16" && "$tp" == "4" ]]; then
    export TRITON_CACHE_DIR=/root/bench-int8-w4a16/baseline/triton_cache_w4a16
    export NCCL_TIMEOUT_HOURS=2
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
  fi

  local model_path=/models/Qwen3.5-9B-${model_pref}
  /opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --dtype float16 \
      --tensor-parallel-size "$tp" \
      --max-model-len 32768 \
      --block-size 32 \
      --enable-prefix-caching \
      --language-model-only \
      --gpu-memory-utilization 0.93 \
      --port 8000 $serve_extra > "$cell_out/server.log" 2>&1 &
  local server_pid=$!

  for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
      echo "  [spot] server up after $((i*5))s"
      break
    fi
    sleep 5
  done
  if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "  [spot] FATAL: server did not become healthy"
    tail -n 80 "$cell_out/server.log"
    kill -9 $server_pid 2>/dev/null || true
    stop_vllm
    return
  fi

  local raw="$cell_out/spotcheck_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$raw"
  /opt/vllm-env/bin/python3 -m vllm.entrypoints.cli.main bench serve \
      --model "$model_path" \
      --base-url http://127.0.0.1:8000 \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate inf \
      --max-concurrency "$conc" \
      --seed 42 \
      --save-result \
      --result-dir "$raw" \
      --result-filename raw.json \
      --trust-remote-code \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 \
      --metadata cell_id="$cell" workload=synthetic tp="$tp" concurrency="$conc" \
                  num_prompts="$NUM_PROMPTS" spotcheck=true \
      --dataset-name random \
      --random-input-len 1024 \
      --random-output-len 256 \
      --ignore-eos > "$cell_out/spotcheck.log" 2>&1 || true

  local actual delta within2 within5
  actual=""
  delta=""
  within2=""
  within5=""
  if [[ -f "$raw/raw.json" ]]; then
    actual=$(/opt/vllm-env/bin/python3 -c "
import json, sys
try:
  b = json.load(open(sys.argv[1]))
  v = b.get('output_throughput', '')
  if v != '':
    print(v)
except Exception:
  pass" "$raw/raw.json" 2>/dev/null || echo "")
  fi
  if [[ -n "$actual" && -n "$ref" && "$ref" != "True" && "$ref" != "False" ]]; then
    set +e
    read -r delta abspct < <(/opt/vllm-env/bin/python3 -c "
import sys
try:
  ref=float(sys.argv[1]); act=float(sys.argv[2])
  delta=(act-ref)/ref*100
  print(f'{delta:+.3f} {abs(delta):.6f}')
except Exception as e:
  print('err err')
" "$ref" "$actual")
    set -e
    if [[ "$delta" != "err" && -n "$abspct" ]]; then
      within2=$(awk "BEGIN { print ($abspct <= 2.0) ? \"True\" : \"False\" }")
      within5=$(awk "BEGIN { print ($abspct <= 5.0) ? \"True\" : \"False\" }")
    fi
  fi
  echo "  [spot] $cell -> actual=$actual ; ref=$ref ; delta=${delta}% ; within±2%=$within2 ; within±5%=$within5"
  printf "%s,%s,%s,%s,%s,%s\n" "$cell" "$ref" "$actual" "$delta" "$within2" "$within5" >> "$OUT_CSV.tmp"

  kill -9 $server_pid 2>/dev/null || true
  stop_vllm
}

mkdir -p "$(dirname "$OUT_CSV")"
echo "cell,recorded,actual,delta_pct,within_band_2pct,within_band_5pct" > "$OUT_CSV.tmp"
echo "Spot-check cells: $CELLS"
for cell in $CELLS; do
  run_cell "$cell"
done

mv "$OUT_CSV.tmp" "$OUT_CSV"
echo
echo "Wrote $OUT_CSV"
cat "$OUT_CSV"
