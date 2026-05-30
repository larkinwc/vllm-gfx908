#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/run_w4a16_baseline.sh -- W4A16 12-cell bench harness (MI100)
# =============================================================================
#
# Self-contained bench harness for the MI100/gfx908 **W4A16** mission. Unlike
# the prior W8A8/INT8 mission harness (which hard-coded a stale worktree path
# and the W8A8 model), this script:
#
#   1. Resolves REPO to the CURRENT worktree via `git rev-parse --show-toplevel`
#      (falls back to the script's ../.. directory if git is unavailable).
#   2. Targets MODEL=/models/Qwen3.5-9B-w4a16 (overridable via $MODEL).
#   3. Drives the 12 W4A16 cells: TP in {1,4} x c in {1,2,4} x
#      {synthetic, coding}. synthetic = random 1024/256, num_prompts=200,
#      seed 42 (num_prompts overridable via $NUM_PROMPTS for dry-runs).
#   4. Honors VLLM_MI100_W4A16_USE_MARLIN_REPACK from the environment
#      (off/unset = baseline GEMM; 1 = marlin repack path). The value is
#      passed straight through to the spawned vLLM server.
#
# It mirrors the services.yaml `vllm-w4a16-tp1` / `vllm-w4a16-tp4` launch
# blocks so a cell run is identical to starting the manifest service by hand.
#
# Usage:
#   scripts/mi100/run_w4a16_baseline.sh [CELLS_FILTER]
#
# CELLS_FILTER is a regex applied to cell ids of the form
#   w4a16_tp{1,4}_c{1,2,4}_{synthetic,coding}
# Examples:
#   scripts/mi100/run_w4a16_baseline.sh                      # full 12-cell grid
#   scripts/mi100/run_w4a16_baseline.sh 'w4a16_tp1_c1_'      # 2 cells (syn+cod)
#   CELLS_FILTER='w4a16_tp1_c1_synthetic' NUM_PROMPTS=4 \
#     scripts/mi100/run_w4a16_baseline.sh                    # 1-cell dry-run
#
# Env knobs:
#   W4A16_BENCH_ROOT   output root (default /root/bench-w4a16/m0)
#   NUM_PROMPTS        prompts per cell (default 200; lower for dry-runs)
#   CELLS_FILTER       cell-id regex (default .*; arg $1 overrides)
#   MODEL              model path (default /models/Qwen3.5-9B-w4a16)
#   PY                 python interpreter (default /opt/vllm-env/bin/python3,
#                      falling back to `command -v python3`)
#   DATASET            coding dataset jsonl (default
#                      /root/bench-int8-w4a16/datasets/coding_agent.jsonl)
#   VLLM_MI100_W4A16_USE_MARLIN_REPACK  passed through to the server
#
# Outputs (in $W4A16_BENCH_ROOT):
#   harness_manifest.json   -- env, versions, per-cell launch commands
#   synthetic/<id>.json     -- 6 cells, schema-validated
#   coding/<id>.json        -- 6 cells, schema-validated
#   run_w4a16_<ts>.log      -- this script's tee'd output
#
# NEVER pushes to git.
# =============================================================================
set -uo pipefail

# ---------- Paths ----------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
if command -v git >/dev/null 2>&1; then
  GIT_TOPLEVEL=$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null || true)
  if [[ -n "$GIT_TOPLEVEL" ]]; then
    REPO=$GIT_TOPLEVEL
  fi
fi

MODEL=${MODEL:-/models/Qwen3.5-9B-w4a16}
W4A16_BENCH_ROOT=${W4A16_BENCH_ROOT:-/root/bench-w4a16/m0}
SYN_DIR=$W4A16_BENCH_ROOT/synthetic
COD_DIR=$W4A16_BENCH_ROOT/coding
DATASET=${DATASET:-/root/bench-int8-w4a16/datasets/coding_agent.jsonl}

# Resolve python interpreter.
if [[ -n "${PY:-}" && -x "${PY:-}" ]]; then
  :
elif [[ -x /opt/vllm-env/bin/python3 ]]; then
  PY=/opt/vllm-env/bin/python3
else
  PY=$(command -v python3 || true)
fi

mkdir -p "$SYN_DIR" "$COD_DIR" "$W4A16_BENCH_ROOT/profile"

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG=$W4A16_BENCH_ROOT/run_w4a16_${TS}.log
exec > >(tee -a "$RUN_LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

# ---------- Common env (mirrors mission AGENTS.md required vars) ----------
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1
# Marlin-repack toggle: honor whatever the caller exported; default off so the
# harness is a true baseline unless the caller opts in.
export VLLM_MI100_W4A16_USE_MARLIN_REPACK=${VLLM_MI100_W4A16_USE_MARLIN_REPACK:-0}

log "REPO:           $REPO"
log "MODEL:          $MODEL"
log "PY:             $PY"
log "Output root:    $W4A16_BENCH_ROOT"
log "MARLIN_REPACK:  VLLM_MI100_W4A16_USE_MARLIN_REPACK=$VLLM_MI100_W4A16_USE_MARLIN_REPACK"

if [[ -z "$PY" ]]; then
  log "FATAL: no python interpreter found (set \$PY)"; exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  log "FATAL: model dir missing at $MODEL"; exit 2
fi

# ---------- Versions ----------
VLLM_COMMIT=$(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo unknown)
ROCM_VERSION=7.12
TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo unknown)
TRITON_VERSION=$($PY -c "import triton; print(triton.__version__)" 2>/dev/null || echo unknown)
VLLM_VERSION=$(cd "$REPO" && $PY -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo unknown)

log "vLLM commit:    $VLLM_COMMIT ($VLLM_VERSION)"
log "ROCm:           $ROCM_VERSION   torch: $TORCH_VERSION   triton: $TRITON_VERSION"

# ---------- Service start helpers (mirror services.yaml) ----------
kill_orphans() {
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  sleep 2
}

start_service() {
  # $1 service id (vllm-w4a16-tp1 | vllm-w4a16-tp4)
  local svc=$1 tp extra cuda_visible
  case "$svc" in
    vllm-w4a16-tp1)  tp=1; extra=""; cuda_visible=0 ;;
    vllm-w4a16-tp4)  tp=4; extra="--disable-custom-all-reduce"; cuda_visible="" ;;
    *) echo "unknown service $svc" >&2; return 2 ;;
  esac

  kill_orphans
  if [[ -n "$cuda_visible" ]]; then
    export CUDA_VISIBLE_DEVICES="$cuda_visible"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi

  if [[ "$svc" == vllm-w4a16-tp4 ]]; then
    export VLLM_MI100_DISABLE_CUSTOM_AR=1
    export TRITON_CACHE_DIR=/root/bench-w4a16/triton_cache_w4a16
    export NCCL_TIMEOUT_HOURS=2
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
  else
    export VLLM_MI100_DISABLE_CUSTOM_AR=0
    unset TRITON_CACHE_DIR || true
    unset NCCL_TIMEOUT_HOURS TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC || true
  fi

  local svc_log=$W4A16_BENCH_ROOT/server_${svc}_${TS}.log
  log "starting $svc (model=$MODEL tp=$tp marlin=$VLLM_MI100_W4A16_USE_MARLIN_REPACK) -> $svc_log"
  cd "$REPO"
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype float16 \
    --tensor-parallel-size "$tp" \
    --max-model-len 32768 \
    --block-size 32 \
    --enable-prefix-caching \
    --language-model-only \
    --trust-remote-code \
    --gpu-memory-utilization 0.93 \
    --port 8000 \
    $extra \
    > "$svc_log" 2>&1 &
  echo $! > "$W4A16_BENCH_ROOT/${svc}.pid"
  for i in $(seq 1 360); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      log "  $svc healthy after ${i}*5s"
      return 0
    fi
    if ! kill -0 "$(cat "$W4A16_BENCH_ROOT/${svc}.pid")" 2>/dev/null; then
      log "  $svc PID died -- last 30 lines of log:"
      tail -n 30 "$svc_log"
      return 1
    fi
    sleep 5
  done
  log "  healthcheck timeout for $svc"
  tail -n 30 "$svc_log"
  return 1
}

stop_service() {
  kill_orphans
}

# ---------- Per-cell run ----------
# Args: tp conc workload
run_cell() {
  local tp=$1
  local conc=$2
  local workload=$3      # synthetic | coding

  local cell_id="w4a16_tp${tp}_c${conc}"
  local quant="w4a16"

  local out_dir
  if [[ "$workload" == synthetic ]]; then out_dir="$SYN_DIR"
  else out_dir="$COD_DIR"; fi
  mkdir -p "$out_dir"

  local raw_dir
  raw_dir=$(mktemp -d "$W4A16_BENCH_ROOT/raw_${cell_id}_${workload}_XXXXXX")
  local out_file="$out_dir/${cell_id}.json"
  local env_file="$W4A16_BENCH_ROOT/env_${cell_id}_${workload}.json"

  log ">> CELL w4a16 tp=${tp} c=${conc} ${workload} -> ${out_file}"

  # Persist env snapshot for the schema row.
  $PY - <<EOF > "$env_file"
import json, os
keep = [
    "LD_LIBRARY_PATH","ROCM_PATH","PATH","PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_USE_AITER","VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_MI100_DISABLE_CUSTOM_AR","TORCH_COMPILE_DISABLE",
    "VLLM_MI100_W4A16_USE_MARLIN_REPACK",
    "TRITON_CACHE_DIR","NCCL_TIMEOUT_HOURS","TORCH_NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC","HF_HUB_OFFLINE",
    "CUDA_VISIBLE_DEVICES",
]
print(json.dumps({k: os.environ.get(k, "") for k in keep}, indent=2))
EOF

  local rate=inf
  local n_prompts=${NUM_PROMPTS:-200}
  local bench_args=(
    --model "$MODEL"
    --base-url "http://127.0.0.1:8000"
    --num-prompts "$n_prompts"
    --request-rate "$rate"
    --max-concurrency "$conc"
    --seed 42
    --save-result
    --result-dir "$raw_dir"
    --result-filename "raw.json"
    --trust-remote-code
    --percentile-metrics "ttft,tpot,itl,e2el"
    --metric-percentiles "50,90,99"
    --metadata "cell_id=${cell_id}" "workload=${workload}" "tp=${tp}" "concurrency=${conc}" "num_prompts=${n_prompts}" "marlin=${VLLM_MI100_W4A16_USE_MARLIN_REPACK}"
  )
  if [[ "$workload" == synthetic ]]; then
    bench_args+=(
      --dataset-name random
      --random-input-len 1024
      --random-output-len 256
      --ignore-eos
    )
  else
    bench_args+=(
      --dataset-name custom
      --dataset-path "$DATASET"
      --custom-output-len 256
      --skip-chat-template
    )
  fi

  local launch_command="$PY -m vllm.entrypoints.cli.main bench serve ${bench_args[*]}"

  log "  launch: $launch_command"
  set +e
  cd "$REPO"
  $PY -m vllm.entrypoints.cli.main bench serve "${bench_args[@]}"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "  ERROR vllm bench serve exit=$rc"
  fi

  local raw_json
  raw_json=$(ls -1 "$raw_dir"/*.json 2>/dev/null | head -1 || true)
  if [[ -z "${raw_json:-}" ]]; then
    log "  WARNING raw JSON missing under $raw_dir; emitting placeholder"
    cat > "$raw_dir/raw.json" <<EOF
{"output_throughput": 0, "request_throughput": 0, "mean_ttft_ms": 0, "median_ttft_ms": 0, "p99_ttft_ms": 0, "mean_tpot_ms": 0, "median_tpot_ms": 0, "p99_tpot_ms": 0, "FAILED": true, "exit_code": ${rc}}
EOF
    raw_json="$raw_dir/raw.json"
  fi

  cd "$REPO"
  $PY scripts/postprocess_bench_result.py \
    --raw "$raw_json" \
    --cell "${cell_id}_${workload}" \
    --model "$MODEL" \
    --quant "$quant" \
    --tp "$tp" \
    --concurrency "$conc" \
    --request-rate "$rate" \
    --workload "$workload" \
    --num-prompts "$n_prompts" \
    --launch-command "$launch_command" \
    --env-file "$env_file" \
    --kernel-backend "marlin_repack=${VLLM_MI100_W4A16_USE_MARLIN_REPACK}" \
    --out "$out_file"
}

# ---------- Manifest ----------
write_manifest() {
  local manifest=$W4A16_BENCH_ROOT/harness_manifest.json
  $PY - "$manifest" "$VLLM_COMMIT" "$VLLM_VERSION" "$TORCH_VERSION" \
       "$TRITON_VERSION" "$ROCM_VERSION" "$DATASET" "$MODEL" \
       "${NUM_PROMPTS:-200}" "$VLLM_MI100_W4A16_USE_MARLIN_REPACK" <<'EOF'
import json, os, subprocess, sys
(manifest, vllm_sha, vllm_ver, torch_v, triton_v, rocm_v, dataset, model,
 n_prompts, marlin) = sys.argv[1:11]

def shell(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

env_keys = [
    "LD_LIBRARY_PATH","ROCM_PATH","PATH","PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_USE_AITER","VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_MI100_DISABLE_CUSTOM_AR","TORCH_COMPILE_DISABLE",
    "VLLM_MI100_W4A16_USE_MARLIN_REPACK","HF_HUB_OFFLINE",
]

cells = []
for tp in (1, 4):
    for c in (1, 2, 4):
        for wl in ("synthetic", "coding"):
            cell = f"w4a16_tp{tp}_c{c}_{wl}"
            cells.append({"id": cell, "model": "w4a16", "tp": tp,
                          "concurrency": c, "workload": wl})

obj = {
    "mission": "MI100/gfx908 W4A16 Marlin-repack GEMM speedup",
    "vllm_commit": vllm_sha,
    "vllm_version": vllm_ver,
    "rocm_version": rocm_v,
    "torch_version": torch_v,
    "triton_version": triton_v,
    "model_path": model,
    "marlin_repack": marlin,
    "env": {k: os.environ.get(k, "") for k in env_keys},
    "dataset_path": dataset,
    "dataset_sha256": shell(f"sha256sum {dataset} | awk '{{print $1}}'").strip(),
    "harness_version": "w4a16-1.0",
    "num_prompts_per_cell": int(n_prompts),
    "seed": 42,
    "block_size": 32,
    "max_model_len": 32768,
    "synthetic_input_len": 1024,
    "synthetic_output_len": 256,
    "language_model_only": True,
    "enable_prefix_caching": True,
    "cells": cells,
    "timestamp": shell("date -u +%Y-%m-%dT%H:%M:%SZ").strip(),
    "host": shell("hostname").strip(),
}
with open(manifest, "w") as f:
    json.dump(obj, f, indent=2)
print(f"wrote manifest: {manifest}")
EOF
}

# ---------- Main grid execution ----------
write_manifest

# Service is shared across the 6 (concurrency x workload) cells of one TP.
SERVICES=(
  "vllm-w4a16-tp1:1"
  "vllm-w4a16-tp4:4"
)

# Allow caller to restrict the run via arg $1 or $CELLS_FILTER (regex).
CELLS_FILTER=${1:-${CELLS_FILTER:-".*"}}
log "CELLS_FILTER=$CELLS_FILTER"

for pair in "${SERVICES[@]}"; do
  IFS=":" read -r svc tp <<<"$pair"

  any_match=0
  for conc in 1 2 4; do
    for wl in synthetic coding; do
      cell="w4a16_tp${tp}_c${conc}_${wl}"
      if [[ "$cell" =~ $CELLS_FILTER ]]; then any_match=1; break; fi
    done
    [[ $any_match -eq 1 ]] && break
  done
  if [[ $any_match -eq 0 ]]; then
    log "skipping $svc (no cells match filter)"
    continue
  fi

  if ! start_service "$svc"; then
    log "FAILED to start $svc -- skipping its 6 cells"
    stop_service
    continue
  fi

  for conc in 1 2 4; do
    for wl in synthetic coding; do
      cell="w4a16_tp${tp}_c${conc}_${wl}"
      if [[ ! "$cell" =~ $CELLS_FILTER ]]; then
        log "  skip $cell (filter)"
        continue
      fi
      run_cell "$tp" "$conc" "$wl" || log "  cell $cell failed (continuing)"
    done
  done

  stop_service
done

write_manifest

log "=== run_w4a16_baseline.sh done. Validate with: ==="
log "  $PY $REPO/scripts/validate_results_schema.py $W4A16_BENCH_ROOT/"
exit 0
