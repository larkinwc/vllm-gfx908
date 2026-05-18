#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/run_grid_hbm.sh — MI100 HBM-mission per-milestone grid wrapper
# =============================================================================
#
# Sequel to scripts/mi100/run_grid.sh + /root/bench-int8-w4a16/baseline/
# run_baseline.sh, scoped to the MI100 (gfx908) HBM Optimization mission
# (KV-INT8 / chunked-prefill / TP topology bundle).
#
# Runs the 24-cell × 2-workload grid for one of the HBM milestones:
#   {model ∈ w8a8, w4a16} × {tp ∈ 1, 4} × {concurrency ∈ 1, 2, 4}
#                         × {workload ∈ synthetic, coding}
#
# Per cell:
#   1. Stop any running vLLM (pkill vllm.entrypoints + sleep 2).
#   2. Start the matching api_server with the milestone-specific env layered
#      on top of services.yaml's baseline env. KV_CACHE_DTYPE (m1-kvint8) is
#      injected as --kv-cache-dtype <value> at server startup.
#   3. Wait up to 1800 s for /health.
#   4. Run `vllm bench serve` per workload (synthetic random + coding-agent
#      custom dataset) with NUM_PROMPTS=200, --request-rate inf,
#      --max-concurrency <c>, --seed 42, ttft/tpot/itl/e2el percentiles.
#   5. Post-process the raw JSON to a schema-conformant per-cell file at:
#        /root/bench-int8-w4a16-hbm/<milestone>/<model>/<cell>_<workload>.json
#      and ALSO place a symlink (or duplicate) under:
#        /root/bench-int8-w4a16-hbm/<milestone>/<workload>/<model>_<cell>.json
#      so scripts/validate_results_schema.py finds 24 files under
#      <root>/{synthetic,coding}/.
#   6. Stop the server before launching the next service.
#
# If hipErrorLaunchFailure (or healthcheck timeout) appears on any cell,
# we kill all vllm + sleep 5 s + retry the cell once. If still wedged,
# the cell is marked FAILED (placeholder JSON) and the script continues.
# A non-zero count of FAILED cells is reflected in the final exit code.
#
# Outputs (in /root/bench-int8-w4a16-hbm/<milestone>):
#   harness_manifest.json     -- env, versions, per-cell launch commands
#   w8a8/<cell>_<wl>.json     -- 12 cells (canonical per-quant layout)
#   w4a16/<cell>_<wl>.json    -- 12 cells (canonical per-quant layout)
#   synthetic/<cell>.json     -- 12 cells (validator layout, symlinked)
#   coding/<cell>.json        -- 12 cells (validator layout, symlinked)
#   schema_check.log          -- written by Step 3 (validator)
#   cells_complete.txt        -- 24-row summary (path + tput) written at end
#   run_grid_hbm_<ts>.log     -- this script's tee'd output
#   server_<svc>_<ts>.log     -- per-service api_server stderr/stdout
#
# Reruns: idempotent. Existing per-cell JSON files are overwritten.
#
# NEVER pushes to git (per VAL-CROSS-001).
#
# Usage:
#   scripts/mi100/run_grid_hbm.sh m1-kvint8 [CELLS_FILTER]
#
# CELLS_FILTER is a regex matched against "<model>_tp<tp>_c<c>_<wl>".
# Examples:
#   scripts/mi100/run_grid_hbm.sh m1-kvint8                   # full 24-cell grid
#   scripts/mi100/run_grid_hbm.sh m1-kvint8 'w8a8_tp1_c4_'    # 2 cells
#   scripts/mi100/run_grid_hbm.sh m1-kvint8 'w8a8_tp[14]_c4_' # 4 cells
# =============================================================================
set -uo pipefail

# ---------- Args ----------
milestone=${1:-}
if [[ -z "$milestone" ]]; then
  echo "usage: $0 <milestone> [CELLS_FILTER]" >&2
  echo "  milestone in {m1-kvint8, m2-chunked, m3-tp, m4-final}" >&2
  exit 2
fi
case "$milestone" in
  m1-kvint8|m2-chunked|m3-tp|m4-final) ;;
  *) echo "unknown milestone $milestone" >&2; exit 2 ;;
esac
CELLS_FILTER=${2:-.*}

# ---------- Paths ----------
REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
ROOT=/root/bench-int8-w4a16-hbm/${milestone}
PY=/opt/vllm-env/bin/python3
DATASET=/root/bench-int8-w4a16/datasets/coding_agent.jsonl
SERVICES_YAML=/root/.factory/missions/f74e8645-0bfb-462a-963d-84d7196b0f6c/services.yaml

mkdir -p "$ROOT/w8a8" "$ROOT/w4a16" "$ROOT/synthetic" "$ROOT/coding"

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG=$ROOT/run_grid_hbm_${milestone}_${TS}.log
exec > >(tee -a "$RUN_LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

if [[ ! -f "$DATASET" ]]; then
  log "FATAL: coding-agent dataset missing at $DATASET"
  exit 2
fi

# ---------- Common env (mirrors AGENTS.md required vars) ----------
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_MI100_DISABLE_CUSTOM_AR=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# ---------- Milestone-specific env (server-side flags + env vars) ----------
# kv_cache_dtype value applied for this milestone (empty = production FP16 KV).
KV_CACHE_DTYPE_VAL=""
# m2-chunked per-quant chunk size (max-num-batched-tokens). Empty = no chunked
# prefill. Populated for m2-chunked / m4-final from chunk_sweep.json.
declare -A CHUNK_PER_QUANT=([w8a8]="" [w4a16]="")
ENABLE_CHUNKED_PREFILL_FLAG_VAL=""
case "$milestone" in
  m1-kvint8)
    KV_CACHE_DTYPE_VAL=int8_per_token_head
    ;;
  m2-chunked)
    # M2 stack sits on top of M1 KV-INT8 per orchestrator decision.
    KV_CACHE_DTYPE_VAL=int8_per_token_head
    ENABLE_CHUNKED_PREFILL_FLAG_VAL=1
    # Read per-quant optimum from the chunk-size sweep emitted by
    # m2-chunk-size-sweep. Fail fast if absent or malformed.
    CHUNK_SWEEP_JSON=/root/bench-int8-w4a16-hbm/m2-chunked/chunk_sweep.json
    if [[ ! -f "$CHUNK_SWEEP_JSON" ]]; then
      log "FATAL: chunk_sweep.json missing at $CHUNK_SWEEP_JSON"
      exit 2
    fi
    CHUNK_W8A8=$($PY -c "import json; d=json.load(open('$CHUNK_SWEEP_JSON')); print(d['optimum_per_quant']['w8a8'])" 2>/dev/null || echo "")
    CHUNK_W4A16=$($PY -c "import json; d=json.load(open('$CHUNK_SWEEP_JSON')); print(d['optimum_per_quant']['w4a16'])" 2>/dev/null || echo "")
    if [[ -z "$CHUNK_W8A8" || -z "$CHUNK_W4A16" ]]; then
      log "FATAL: could not read optimum_per_quant.{w8a8,w4a16} from $CHUNK_SWEEP_JSON"
      exit 2
    fi
    CHUNK_PER_QUANT[w8a8]="$CHUNK_W8A8"
    CHUNK_PER_QUANT[w4a16]="$CHUNK_W4A16"
    log "  m2-chunked: optimum chunk sizes w8a8=$CHUNK_W8A8 w4a16=$CHUNK_W4A16"
    ;;
  m3-tp)
    # M3 stack sits on top of M1 KV-INT8 + M2 chunked-prefill (per
    # orchestrator decision in feature m3-tp-bench-and-update). Per-cell
    # NCCL_ALGO is read from /root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json
    # in start_service (TP=4 only — TP=1 services skip the export).
    KV_CACHE_DTYPE_VAL=int8_per_token_head
    ENABLE_CHUNKED_PREFILL_FLAG_VAL=1
    CHUNK_SWEEP_JSON=/root/bench-int8-w4a16-hbm/m2-chunked/chunk_sweep.json
    if [[ ! -f "$CHUNK_SWEEP_JSON" ]]; then
      log "FATAL: chunk_sweep.json missing at $CHUNK_SWEEP_JSON (needed by m3-tp to stack M2 winners)"
      exit 2
    fi
    CHUNK_W8A8=$($PY -c "import json; d=json.load(open('$CHUNK_SWEEP_JSON')); print(d['optimum_per_quant']['w8a8'])" 2>/dev/null || echo "")
    CHUNK_W4A16=$($PY -c "import json; d=json.load(open('$CHUNK_SWEEP_JSON')); print(d['optimum_per_quant']['w4a16'])" 2>/dev/null || echo "")
    if [[ -z "$CHUNK_W8A8" || -z "$CHUNK_W4A16" ]]; then
      log "FATAL: could not read optimum_per_quant.{w8a8,w4a16} from $CHUNK_SWEEP_JSON"
      exit 2
    fi
    CHUNK_PER_QUANT[w8a8]="$CHUNK_W8A8"
    CHUNK_PER_QUANT[w4a16]="$CHUNK_W4A16"
    NCCL_SWEEP_JSON=/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json
    if [[ ! -f "$NCCL_SWEEP_JSON" ]]; then
      log "FATAL: nccl_sweep.json missing at $NCCL_SWEEP_JSON (needed by m3-tp for per-cell NCCL_ALGO)"
      exit 2
    fi
    export M3_NCCL_SWEEP_JSON="$NCCL_SWEEP_JSON"
    log "  m3-tp: stacking M1 KV-INT8 (int8_per_token_head) + M2 chunked-prefill"
    log "  m3-tp: M2 optimum chunk sizes w8a8=$CHUNK_W8A8 w4a16=$CHUNK_W4A16"
    log "  m3-tp: per-cell NCCL_ALGO from $NCCL_SWEEP_JSON"
    ;;
  m4-final)
    # M4 cumulative stack: M1 KV-INT8 + M2 chunked-prefill per-quant optimum
    # + M3 per-cell NCCL_ALGO winner (TP=4 cells only; TP=1 cells leave
    # NCCL_ALGO unset since it's a no-op at single-rank).
    KV_CACHE_DTYPE_VAL=int8_per_token_head
    ENABLE_CHUNKED_PREFILL_FLAG_VAL=1
    CHUNK_SWEEP_JSON=/root/bench-int8-w4a16-hbm/m2-chunked/chunk_sweep.json
    if [[ ! -f "$CHUNK_SWEEP_JSON" ]]; then
      log "FATAL: chunk_sweep.json missing at $CHUNK_SWEEP_JSON (needed by m4-final to stack M2 winners)"
      exit 2
    fi
    CHUNK_W8A8=$($PY -c "import json; d=json.load(open('$CHUNK_SWEEP_JSON')); print(d['optimum_per_quant']['w8a8'])" 2>/dev/null || echo "")
    CHUNK_W4A16=$($PY -c "import json; d=json.load(open('$CHUNK_SWEEP_JSON')); print(d['optimum_per_quant']['w4a16'])" 2>/dev/null || echo "")
    if [[ -z "$CHUNK_W8A8" || -z "$CHUNK_W4A16" ]]; then
      log "FATAL: could not read optimum_per_quant.{w8a8,w4a16} from $CHUNK_SWEEP_JSON"
      exit 2
    fi
    CHUNK_PER_QUANT[w8a8]="$CHUNK_W8A8"
    CHUNK_PER_QUANT[w4a16]="$CHUNK_W4A16"
    NCCL_SWEEP_JSON=/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json
    if [[ ! -f "$NCCL_SWEEP_JSON" ]]; then
      log "FATAL: nccl_sweep.json missing at $NCCL_SWEEP_JSON (needed by m4-final for per-cell NCCL_ALGO)"
      exit 2
    fi
    export M3_NCCL_SWEEP_JSON="$NCCL_SWEEP_JSON"
    log "  m4-final: stacking M1 KV-INT8 (int8_per_token_head) + M2 chunked-prefill + M3 NCCL_ALGO"
    log "  m4-final: M2 optimum chunk sizes w8a8=$CHUNK_W8A8 w4a16=$CHUNK_W4A16"
    log "  m4-final: per-cell NCCL_ALGO from $NCCL_SWEEP_JSON (TP=4 cells only)"
    ;;
esac
log "milestone=$milestone  KV_CACHE_DTYPE=${KV_CACHE_DTYPE_VAL:-(unset)}"
log "  ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL_FLAG_VAL:-(unset)}  CHUNK_PER_QUANT=w8a8:${CHUNK_PER_QUANT[w8a8]:-(unset)} w4a16:${CHUNK_PER_QUANT[w4a16]:-(unset)}"

# ---------- Versions ----------
VLLM_COMMIT=$(cd "$REPO" && git rev-parse HEAD)
ROCM_VERSION=7.12
TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo unknown)
TRITON_VERSION=$($PY -c "import triton; print(triton.__version__)" 2>/dev/null || echo unknown)
VLLM_VERSION=$(cd "$REPO" && $PY -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo unknown)

log "vLLM commit:    $VLLM_COMMIT ($VLLM_VERSION)"
log "ROCm:           $ROCM_VERSION   torch: $TORCH_VERSION   triton: $TRITON_VERSION"
log "Output root:    $ROOT"
log "CELLS_FILTER:   $CELLS_FILTER"

# ---------- Lifecycle helpers ----------
kill_orphans() {
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  sleep 2
}

start_service() {
  # $1 service id (vllm-w8a8-tp1 | vllm-w8a8-tp4 | vllm-w4a16-tp1 | vllm-w4a16-tp4)
  # $2 (optional) per-cell NCCL_ALGO value. Empty/unset -> NCCL_ALGO is
  #               unexported (rccl heuristic default). Used by m3-tp where the
  #               per-cell winner from nccl_sweep.json drives the env var.
  local svc=$1 model tp extra cuda_visible quant_short
  local nccl_algo_per_cell=${2:-}
  case "$svc" in
    vllm-w8a8-tp1)   model=/models/Qwen3.5-9B-w8a8;   tp=1; extra=""; cuda_visible=0;  quant_short=w8a8 ;;
    vllm-w8a8-tp4)   model=/models/Qwen3.5-9B-w8a8;   tp=4; extra="--disable-custom-all-reduce"; cuda_visible=""; quant_short=w8a8 ;;
    vllm-w4a16-tp1)  model=/models/Qwen3.5-9B-w4a16;  tp=1; extra=""; cuda_visible=0;  quant_short=w4a16 ;;
    vllm-w4a16-tp4)  model=/models/Qwen3.5-9B-w4a16;  tp=4; extra="--disable-custom-all-reduce"; cuda_visible=""; quant_short=w4a16 ;;
    *) log "unknown service $svc"; return 2 ;;
  esac

  # NCCL_ALGO: per-cell override for m3-tp; otherwise unset.
  if [[ -n "$nccl_algo_per_cell" ]]; then
    export NCCL_ALGO="$nccl_algo_per_cell"
    log "  per-cell NCCL_ALGO=$nccl_algo_per_cell"
  else
    unset NCCL_ALGO || true
  fi
  export ACTIVE_NCCL_ALGO="$nccl_algo_per_cell"

  kill_orphans
  if [ -n "$cuda_visible" ]; then
    export CUDA_VISIBLE_DEVICES="$cuda_visible"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi

  # Service-specific NCCL / Triton-cache safety net (TP=4 only).
  if [[ "$svc" == vllm-w4a16-tp4 ]]; then
    export VLLM_MI100_DISABLE_CUSTOM_AR=1
    # Pre-populated Triton cache from prior mission's M1 pretune step.
    export TRITON_CACHE_DIR=/root/bench-int8-w4a16/baseline/triton_cache_w4a16
    export NCCL_TIMEOUT_HOURS=2
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
  elif [[ "$svc" == vllm-w8a8-tp4 ]]; then
    export VLLM_MI100_DISABLE_CUSTOM_AR=1
    unset TRITON_CACHE_DIR || true
    unset NCCL_TIMEOUT_HOURS TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC || true
  else
    export VLLM_MI100_DISABLE_CUSTOM_AR=0
    unset TRITON_CACHE_DIR || true
    unset NCCL_TIMEOUT_HOURS TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC || true
  fi

  # Optional --kv-cache-dtype injection.
  local kv_flag=()
  if [[ -n "$KV_CACHE_DTYPE_VAL" ]]; then
    kv_flag=(--kv-cache-dtype "$KV_CACHE_DTYPE_VAL")
  fi

  # Optional chunked-prefill injection (m2-chunked / m4-final).
  local chunked_flag=()
  if [[ -n "$ENABLE_CHUNKED_PREFILL_FLAG_VAL" ]]; then
    chunked_flag+=(--enable-chunked-prefill)
  fi
  local chunk_val=${CHUNK_PER_QUANT[$quant_short]:-}
  if [[ -n "$chunk_val" ]]; then
    chunked_flag+=(--max-num-batched-tokens "$chunk_val")
  fi
  # Export for run_cell's env snapshot.
  export ACTIVE_MAX_NUM_BATCHED_TOKENS="$chunk_val"
  export ACTIVE_ENABLE_CHUNKED_PREFILL="$ENABLE_CHUNKED_PREFILL_FLAG_VAL"

  local svc_log=$ROOT/server_${svc}_${TS}.log
  log "starting $svc (model=$model tp=$tp kv=${KV_CACHE_DTYPE_VAL:-fp16} chunked=${ENABLE_CHUNKED_PREFILL_FLAG_VAL:-0} max-num-batched-tokens=${chunk_val:-default}) -> $svc_log"
  cd "$REPO"
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model "$model" \
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
    "${kv_flag[@]}" \
    "${chunked_flag[@]}" \
    > "$svc_log" 2>&1 &
  echo $! > "$ROOT/${svc}.pid"

  for i in $(seq 1 360); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      log "  $svc healthy after ${i}*5s"
      return 0
    fi
    if ! kill -0 "$(cat "$ROOT/${svc}.pid")" 2>/dev/null; then
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
# Args: model_short svc tp conc workload
run_cell() {
  local model_short=$1   # w8a8 | w4a16
  local svc=$2           # service id (already started)
  local tp=$3
  local conc=$4
  local workload=$5      # synthetic | coding

  local cell_id="${model_short}_tp${tp}_c${conc}"
  local quant model
  if [[ "$model_short" == w8a8 ]]; then
    quant="w8a8-int8"; model=/models/Qwen3.5-9B-w8a8
  else
    quant="w4a16"; model=/models/Qwen3.5-9B-w4a16
  fi

  local raw_dir
  raw_dir=$(mktemp -d "$ROOT/raw_${cell_id}_${workload}_XXXXXX")

  # Canonical per-quant output file (task spec).
  local out_file="$ROOT/${model_short}/${cell_id}_${workload}.json"
  # Validator-compatible symlink (scripts/validate_results_schema.py expects
  # files under <root>/{synthetic,coding}/).
  local link_file="$ROOT/${workload}/${cell_id}.json"
  local env_file="$ROOT/env_${cell_id}_${workload}.json"

  log ">> CELL ${model_short} tp=${tp} c=${conc} ${workload} -> ${out_file}"

  # Persist env snapshot for the schema row.
  $PY - <<EOF > "$env_file"
import json, os
keep = [
    "LD_LIBRARY_PATH","ROCM_PATH","PATH","PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_USE_AITER","VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_MI100_DISABLE_CUSTOM_AR","TORCH_COMPILE_DISABLE",
    "TRITON_CACHE_DIR","NCCL_TIMEOUT_HOURS","TORCH_NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC","HF_HUB_OFFLINE",
    "CUDA_VISIBLE_DEVICES","NCCL_ALGO",
]
extra = {
    "KV_CACHE_DTYPE": "${KV_CACHE_DTYPE_VAL}",
    "milestone": "${milestone}",
    "ENABLE_CHUNKED_PREFILL": "${ACTIVE_ENABLE_CHUNKED_PREFILL:-}",
    "MAX_NUM_BATCHED_TOKENS": "${ACTIVE_MAX_NUM_BATCHED_TOKENS:-}",
    "NCCL_ALGO": "${ACTIVE_NCCL_ALGO:-}",
}
d = {k: os.environ.get(k, "") for k in keep}
d.update(extra)
print(json.dumps(d, indent=2))
EOF

  local rate=inf
  local n_prompts=${NUM_PROMPTS:-200}
  local bench_args=(
    --model "$model"
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
    --metadata "cell_id=${cell_id}" "workload=${workload}" "tp=${tp}" \
               "concurrency=${conc}" "num_prompts=${n_prompts}" \
               "kv_cache_dtype=${KV_CACHE_DTYPE_VAL:-auto}" \
               "milestone=${milestone}" \
               "enable_chunked_prefill=${ACTIVE_ENABLE_CHUNKED_PREFILL:-0}" \
               "max_num_batched_tokens=${ACTIVE_MAX_NUM_BATCHED_TOKENS:-default}" \
               "nccl_algo=${ACTIVE_NCCL_ALGO:-default}"
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

  # Find the raw JSON. vllm bench serve creates a timestamped file.
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
    --model "$model" \
    --quant "$quant" \
    --tp "$tp" \
    --concurrency "$conc" \
    --request-rate "$rate" \
    --workload "$workload" \
    --num-prompts "$n_prompts" \
    --launch-command "$launch_command" \
    --env-file "$env_file" \
    --kernel-backend "stock" \
    --out "$out_file"

  # Mirror into the validator-friendly <workload>/<cell>.json path as a
  # relative symlink so `scripts/validate_results_schema.py <root>/` finds
  # exactly 24 files under {synthetic,coding}/.
  rm -f "$link_file"
  ln -s "../${model_short}/${cell_id}_${workload}.json" "$link_file"
}

# ---------- Manifest ----------
write_manifest() {
  local manifest=$ROOT/harness_manifest.json
  $PY - "$manifest" "$VLLM_COMMIT" "$VLLM_VERSION" "$TORCH_VERSION" \
       "$TRITON_VERSION" "$ROCM_VERSION" "$DATASET" "$SERVICES_YAML" \
       "$milestone" "$KV_CACHE_DTYPE_VAL" <<'EOF'
import json, os, subprocess, sys
(manifest, vllm_sha, vllm_ver, torch_v, triton_v, rocm_v, dataset,
 services_yaml, milestone, kv_cache_dtype) = sys.argv[1:11]

def shell(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL)
    except Exception:
        return ""

env_keys = [
    "LD_LIBRARY_PATH","ROCM_PATH","PATH","PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_USE_AITER","VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_MI100_DISABLE_CUSTOM_AR","TORCH_COMPILE_DISABLE",
    "HF_HUB_OFFLINE",
]

cells = []
for ms in ("w8a8","w4a16"):
    for tp in (1,4):
        for c in (1,2,4):
            for wl in ("synthetic","coding"):
                cell = f"{ms}_tp{tp}_c{c}_{wl}"
                cells.append({"id": cell, "model": ms, "tp": tp,
                              "concurrency": c, "workload": wl})

obj = {
    "milestone": milestone,
    "kv_cache_dtype": kv_cache_dtype or "auto",
    "vllm_commit": vllm_sha,
    "vllm_version": vllm_ver,
    "rocm_version": rocm_v,
    "torch_version": torch_v,
    "triton_version": triton_v,
    "gpu_topo": shell("rocm-smi --showtopo"),
    "rocm_smi_product": shell("rocm-smi --showproductname"),
    "rocm_smi_temp": shell("rocm-smi --showtemp"),
    "env": {k: os.environ.get(k, "") for k in env_keys},
    "dataset_path": dataset,
    "dataset_sha256": shell(f"sha256sum {dataset} | awk '{{print $1}}'").strip(),
    "services_yaml": services_yaml,
    "harness_version": "hbm-1.0",
    "num_prompts_per_cell": int(os.environ.get("NUM_PROMPTS", "200")),
    "seed": 42,
    "block_size": 32,
    "max_model_len": 32768,
    "cudagraph_mode": "FULL_DECODE_ONLY",
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

SERVICES=(
  "w8a8:vllm-w8a8-tp1:1"
  "w8a8:vllm-w8a8-tp4:4"
  "w4a16:vllm-w4a16-tp1:1"
  "w4a16:vllm-w4a16-tp4:4"
)

failed_cells=()

# Helper: read per-cell NCCL_ALGO winner from nccl_sweep.json (m3-tp only).
# Echoes the winner_algo string (possibly empty for "default") or empty if
# the cell isn't a TP=4 cell or the file is absent.
m3_nccl_winner_for() {
  # $1 = cell id like w8a8_tp4_c1
  local cell=$1
  if [[ -z "${M3_NCCL_SWEEP_JSON:-}" || ! -f "$M3_NCCL_SWEEP_JSON" ]]; then
    echo ""
    return 0
  fi
  $PY - "$M3_NCCL_SWEEP_JSON" "$cell" <<'PYEOF'
import json, sys
sweep = json.load(open(sys.argv[1]))
cell_id = sys.argv[2]
for c in sweep.get("cells", []):
    if c.get("cell_id") == cell_id:
        w = c.get("winner_algo", "")
        # The convention in nccl_sweep.json: 'default' means the rccl
        # heuristic, which we represent on the wire by leaving NCCL_ALGO
        # unset / empty. 'Ring' / 'Tree' are passed through verbatim.
        if w == "default":
            print("")
        else:
            print(w)
        break
PYEOF
}

if [[ "$milestone" == "m3-tp" || "$milestone" == "m4-final" ]]; then
  # m3-tp / m4-final diverge from the standard loop because the TP=4 cells
  # need the server restarted PER concurrency cell so the baked per-cell
  # NCCL_ALGO winner is honored by rccl (which reads NCCL_ALGO once at
  # engine init).
  #
  # m3-tp scope: TP=4 cells only (NCCL_ALGO is the only knob being swept).
  # m4-final scope: all 12 cells (TP=1 + TP=4); TP=1 cells use the standard
  # one-service-per-(quant,tp) pass since NCCL_ALGO is a no-op at single-rank.
  if [[ "$milestone" == "m4-final" ]]; then
    # m4-final: first run the TP=1 cells via the standard single-service path
    # for each quant, then fall through to the TP=4 per-cell-restart loop.
    log "m4-final main loop (phase 1/2): TP=1 cells via single-service pass per quant"
    TP1_SERVICES=(
      "w8a8:vllm-w8a8-tp1:1"
      "w4a16:vllm-w4a16-tp1:1"
    )
    for triple in "${TP1_SERVICES[@]}"; do
      IFS=":" read -r model_short svc tp <<<"$triple"

      any_match=0
      for conc in 1 2 4; do
        for wl in synthetic coding; do
          cell="${model_short}_tp${tp}_c${conc}_${wl}"
          if [[ "$cell" =~ $CELLS_FILTER ]]; then any_match=1; break; fi
        done
        [[ $any_match -eq 1 ]] && break
      done
      if [[ $any_match -eq 0 ]]; then
        log "skipping $svc (no cells match filter)"
        continue
      fi

      if ! start_service "$svc"; then
        log "  start_service $svc failed; killing all vLLM, sleeping 5s, retrying once"
        kill_orphans
        sleep 5
        if ! start_service "$svc"; then
          log "FAILED to start $svc after retry — marking its TP=1 cells as FAILED"
          stop_service
          for conc in 1 2 4; do
            for wl in synthetic coding; do
              cell="${model_short}_tp${tp}_c${conc}_${wl}"
              if [[ "$cell" =~ $CELLS_FILTER ]]; then
                failed_cells+=("$cell")
              fi
            done
          done
          continue
        fi
      fi

      for conc in 1 2 4; do
        for wl in synthetic coding; do
          cell="${model_short}_tp${tp}_c${conc}_${wl}"
          if [[ ! "$cell" =~ $CELLS_FILTER ]]; then
            log "  skip $cell (filter)"
            continue
          fi
          if ! run_cell "$model_short" "$svc" "$tp" "$conc" "$wl"; then
            log "  run_cell $cell failed; continuing"
            failed_cells+=("$cell")
          fi
        done
      done

      stop_service
    done
    log "m4-final main loop (phase 2/2): TP=4 cells with per-cell NCCL_ALGO winner"
  else
    log "m3-tp main loop: TP=4 cells only, per-cell server restart for NCCL_ALGO winner"
  fi
  TP4_SERVICES=(
    "w8a8:vllm-w8a8-tp4:4"
    "w4a16:vllm-w4a16-tp4:4"
  )
  for triple in "${TP4_SERVICES[@]}"; do
    IFS=":" read -r model_short svc tp <<<"$triple"
    for conc in 1 2 4; do
      any_match=0
      for wl in synthetic coding; do
        cell="${model_short}_tp${tp}_c${conc}_${wl}"
        if [[ "$cell" =~ $CELLS_FILTER ]]; then any_match=1; break; fi
      done
      if [[ $any_match -eq 0 ]]; then
        log "  skipping ${model_short}_tp${tp}_c${conc} (no cells match filter)"
        continue
      fi
      cell_id_no_wl="${model_short}_tp${tp}_c${conc}"
      winner=$(m3_nccl_winner_for "$cell_id_no_wl")
      if [[ -z "$winner" ]]; then
        log "  cell $cell_id_no_wl: NCCL_ALGO=default (rccl heuristic)"
      else
        log "  cell $cell_id_no_wl: NCCL_ALGO=$winner"
      fi
      if ! start_service "$svc" "$winner"; then
        log "  start_service $svc (NCCL_ALGO=${winner:-default}) failed; killing all vLLM, sleeping 5s, retrying once"
        kill_orphans
        sleep 5
        if ! start_service "$svc" "$winner"; then
          log "FAILED to start $svc for $cell_id_no_wl after retry — marking its 2 workload cells as FAILED"
          stop_service
          for wl in synthetic coding; do
            cell="${model_short}_tp${tp}_c${conc}_${wl}"
            if [[ "$cell" =~ $CELLS_FILTER ]]; then
              failed_cells+=("$cell")
            fi
          done
          continue
        fi
      fi
      for wl in synthetic coding; do
        cell="${model_short}_tp${tp}_c${conc}_${wl}"
        if [[ ! "$cell" =~ $CELLS_FILTER ]]; then
          log "  skip $cell (filter)"
          continue
        fi
        if ! run_cell "$model_short" "$svc" "$tp" "$conc" "$wl"; then
          log "  run_cell $cell failed; continuing"
          failed_cells+=("$cell")
        fi
      done
      stop_service
    done
  done
else
  for triple in "${SERVICES[@]}"; do
    IFS=":" read -r model_short svc tp <<<"$triple"

    any_match=0
    for conc in 1 2 4; do
      for wl in synthetic coding; do
        cell="${model_short}_tp${tp}_c${conc}_${wl}"
        if [[ "$cell" =~ $CELLS_FILTER ]]; then any_match=1; break; fi
      done
      [[ $any_match -eq 1 ]] && break
    done
    if [[ $any_match -eq 0 ]]; then
      log "skipping $svc (no cells match filter)"
      continue
    fi

    # Try to start service; on hipErrorLaunchFailure / healthcheck timeout
    # kill all + sleep 5 + retry once.
    if ! start_service "$svc"; then
      log "  start_service $svc failed; killing all vLLM, sleeping 5s, retrying once"
      kill_orphans
      sleep 5
      if ! start_service "$svc"; then
        log "FAILED to start $svc after retry — marking its cells as FAILED"
        stop_service
        for conc in 1 2 4; do
          for wl in synthetic coding; do
            cell="${model_short}_tp${tp}_c${conc}_${wl}"
            if [[ "$cell" =~ $CELLS_FILTER ]]; then
              failed_cells+=("$cell")
            fi
          done
        done
        continue
      fi
    fi

    for conc in 1 2 4; do
      for wl in synthetic coding; do
        cell="${model_short}_tp${tp}_c${conc}_${wl}"
        if [[ ! "$cell" =~ $CELLS_FILTER ]]; then
          log "  skip $cell (filter)"
          continue
        fi
        if ! run_cell "$model_short" "$svc" "$tp" "$conc" "$wl"; then
          log "  run_cell $cell failed; continuing"
          failed_cells+=("$cell")
        fi
      done
    done

    stop_service
  done
fi

# Re-write manifest at end so timestamp is post-run.
write_manifest

# ---------- cells_complete.txt ----------
COMPLETE=$ROOT/cells_complete.txt
$PY - "$ROOT" "$COMPLETE" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
out = Path(sys.argv[2])
lines = []
for ms in ("w8a8", "w4a16"):
    for tp in (1, 4):
        for c in (1, 2, 4):
            for wl in ("synthetic", "coding"):
                cell = f"{ms}_tp{tp}_c{c}_{wl}"
                p = root / ms / f"{cell}.json"
                if p.is_file():
                    try:
                        d = json.loads(p.read_text())
                        tput = d.get("output_throughput_toks_s", 0.0)
                    except Exception as e:
                        tput = f"ERR({e})"
                else:
                    tput = "MISSING"
                lines.append(f"{p}\t{tput}")
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out} with {len(lines)} rows")
EOF

# Summary.
n_failed=${#failed_cells[@]}
log "=== run_grid_hbm.sh ($milestone) done. failed_cells=$n_failed ==="
if (( n_failed > 0 )); then
  log "failed cells:"
  for c in "${failed_cells[@]}"; do log "  - $c"; done
fi
log "Validate with:"
log "  $PY $REPO/scripts/validate_results_schema.py $ROOT/"

exit $n_failed
