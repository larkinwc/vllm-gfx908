#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# scripts/mi100/run_grid_fused_parallel.sh
#
# Thin orchestrator that drives the M4 fused-act-quant bench grid with
# 4-way parallel TP=1 cells (one per MI100) followed by serial TP=4 cells.
#
# Concurrency model
# -----------------
#   TP=1 cells:  up to 4 launched concurrently
#     - GPU 0 → port 8000
#     - GPU 1 → port 8001
#     - GPU 2 → port 8002
#     - GPU 3 → port 8003
#   TP=4 cells:  run serially after the TP=1 batch drains (each TP=4 cell
#     uses all four MI100s, so they cannot overlap with anything else).
#
# Each per-cell launch shell is invoked via:
#   HIP_VISIBLE_DEVICES=<gpu> bash scripts/launch_hbm_<quant>_tp1_c<c>.sh \
#       --port <port> [--check|--serve-only]
#
# That contract was introduced by the m4-parallel-bench-harness work item;
# see VAL-M4-001 in validation-contract.md. The launch scripts remain
# backwards-compatible: calling them with no args still pins GPU 0 and
# port 8000.
#
# Modes
# -----
#   --smoke   Launch the 4 TP=1 W8A8 cells in --check mode but with their
#             bench step replaced by a /health probe (no `vllm bench serve`).
#             Verifies parallel launchability + lifecycle.  Persists a log
#             at /root/bench-int8-w4a16-fused/m4-bench/parallel_harness_test.log.
#   --milestone <m4-fused-act-quant>
#             Full 24-cell grid execution. For each TP=1 cell config a server
#             is launched per (quant, conc) pair and both synthetic + coding
#             workloads run back-to-back via vllm bench serve. With
#             PARALLEL_CAP=N, up to N (quant, conc) pairs run concurrently
#             (each consuming 1 GPU). TP=4 cells require 4 GPUs and run
#             serially after the TP=1 batch; on a host with fewer than 4
#             visible GPUs the orchestrator emits schema-conformant
#             placeholder JSONs marking those cells as FAILED.
#             Per-cell server.log is routed via SERVER_LOG_DIR into
#             /root/bench-int8-w4a16-fused/m4-bench/<quant>/<cell>/ so
#             parallel sibling servers do not clobber each other's log.
#
# Lifecycle guarantees
# --------------------
#   - All per-cell server processes are tracked by PID and torn down on EXIT.
#   - After successful completion, `pgrep -f vllm.entrypoints` reports
#     no surviving processes.
#   - The smoke pass writes per-cell {pid, port, gpu, exit_code, wall_s}
#     to the harness log.
#
# Usage:
#   bash scripts/mi100/run_grid_fused_parallel.sh --smoke
#
# NEVER pushes to git.
# =============================================================================
set -uo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/cold-points-sit-rancb
OUT_ROOT=/root/bench-int8-w4a16-fused/m4-bench
mkdir -p "$OUT_ROOT"
LOG_FILE="$OUT_ROOT/parallel_harness_test.log"

# Detect visible GPU count via rocm-smi (per library/fused-act-quant-mission-context.md
# §16: cap parallelism at min(detected, 4)). Honors any caller-side HIP_VISIBLE_DEVICES
# clamp by re-counting under that clamp.
detect_visible_gpus() {
  if command -v rocm-smi >/dev/null 2>&1; then
    # Count distinct GPU[<n>] indices reported by --showproductname.
    rocm-smi --showproductname 2>/dev/null \
      | awk '/^GPU\[/ { gsub(/[^0-9]/, "", $1); print $1 }' \
      | sort -u \
      | wc -l
  else
    echo 0
  fi
}
VISIBLE_GPUS=$(detect_visible_gpus)
if [[ -z "$VISIBLE_GPUS" || "$VISIBLE_GPUS" -lt 1 ]]; then
  VISIBLE_GPUS=1
fi
PARALLEL_CAP=$(( VISIBLE_GPUS < 4 ? VISIBLE_GPUS : 4 ))

MODE=""
MILESTONE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE=--smoke
      shift
      ;;
    --milestone)
      MODE=--milestone
      MILESTONE="$2"
      shift 2
      ;;
    "")
      shift
      ;;
    *)
      echo "usage: $0 [--smoke | --milestone m4-fused-act-quant]" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" == "--milestone" && "$MILESTONE" != "m4-fused-act-quant" ]]; then
  echo "unsupported milestone: $MILESTONE (only m4-fused-act-quant)" >&2
  exit 2
fi

# Cell → GPU/port mapping for the 4-way TP=1 batch. The smoke pass only
# exercises 4 cells (3 W8A8 + 1 W4A16, one per MI100); the full grid
# extends this to W4A16 in additional 4-way waves and the TP=4 cells
# serial after the TP=1 batch drains.
# shellcheck disable=SC2034  # exported as informational lists for the
# full-grid stub path below; future workers consume them when the
# non-smoke orchestrator path is wired up.
TP1_CELLS=(
  "w8a8_tp1_c1" "w8a8_tp1_c2" "w8a8_tp1_c4"
  "w4a16_tp1_c1" "w4a16_tp1_c2" "w4a16_tp1_c4"
)
# shellcheck disable=SC2034
TP4_CELLS=(
  "w8a8_tp4_c1" "w8a8_tp4_c2" "w8a8_tp4_c4"
  "w4a16_tp4_c1" "w4a16_tp4_c2" "w4a16_tp4_c4"
)

# Smoke-pass GPU/port assignment for the 4 W8A8 TP=1 cells.
declare -A SMOKE_GPU=(
  ["w8a8_tp1_c1"]=0
  ["w8a8_tp1_c2"]=1
  ["w8a8_tp1_c4"]=2
  ["w4a16_tp1_c1"]=3
)
declare -A SMOKE_PORT=(
  ["w8a8_tp1_c1"]=8000
  ["w8a8_tp1_c2"]=8001
  ["w8a8_tp1_c4"]=8002
  ["w4a16_tp1_c1"]=8003
)

# ---------------------------------------------------------------------------
# Lifecycle: track per-cell PIDs and tear them down on EXIT.
# ---------------------------------------------------------------------------
declare -a CELL_PIDS=()

# shellcheck disable=SC2329  # invoked indirectly via `trap ... EXIT`
cleanup() {
  for pid in "${CELL_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  # Belt-and-braces: any lingering server processes get a hard kill.
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
  sleep 1
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Smoke pass
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--smoke" ]]; then
  : > "$LOG_FILE"
  {
    echo "===== run_grid_fused_parallel.sh --smoke ====="
    echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo:          $REPO"
    echo "out_root:      $OUT_ROOT"
    echo "log_file:      $LOG_FILE"
    echo "visible_gpus:  $VISIBLE_GPUS"
    echo "parallel_cap:  $PARALLEL_CAP  (min(visible, 4))"
    echo
    echo "rocm-smi snapshot (pre):"
    rocm-smi --showuse 2>/dev/null | sed 's/^/  /' || echo "  rocm-smi unavailable"
    echo
  } | tee -a "$LOG_FILE"

  # First, make sure no stale vllm processes are around — the launch
  # scripts in parallel mode do NOT global-pkill, so we must do it once
  # before the parallel fan-out.
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
  sleep 2

  declare -A CELL_PID=()
  declare -A CELL_START=()

  # Fan-out: launch up to $PARALLEL_CAP TP=1 cells in --serve-only mode
  # in parallel. The canonical smoke set is 4 cells (3 W8A8 + 1 W4A16),
  # but environmental GPU-count disparities (see
  # library/fused-act-quant-mission-context.md §16) may drop the cap to
  # 2 or 3. Each child writes its own log + (per-cell) bench server.log.
  ALL_SMOKE_CELLS=("w8a8_tp1_c1" "w8a8_tp1_c2" "w8a8_tp1_c4" "w4a16_tp1_c1")
  SMOKE_CELLS=("${ALL_SMOKE_CELLS[@]:0:PARALLEL_CAP}")
  echo "[parent] smoke cells (cap=$PARALLEL_CAP): ${SMOKE_CELLS[*]}" \
    | tee -a "$LOG_FILE"
  for cell in "${SMOKE_CELLS[@]}"; do
    gpu=${SMOKE_GPU[$cell]}
    port=${SMOKE_PORT[$cell]}
    case "$cell" in
      w8a8_*) script="$REPO/scripts/launch_hbm_w8a8_tp1_c${cell##*_c}.sh" ;;
      w4a16_*) script="$REPO/scripts/launch_hbm_w4a16_tp1_c${cell##*_c}.sh" ;;
    esac
    cell_log="$OUT_ROOT/${cell}/parallel_smoke.log"
    mkdir -p "$(dirname "$cell_log")"
    echo "[parent] launching $cell on GPU $gpu port $port  (script=$script)" | tee -a "$LOG_FILE"
    HIP_VISIBLE_DEVICES="$gpu" \
      bash "$script" --port "$port" --serve-only \
      > "$cell_log" 2>&1 &
    pid=$!
    CELL_PID["$cell"]=$pid
    CELL_START["$cell"]=$(date +%s)
    CELL_PIDS+=("$pid")
    echo "[parent]   $cell pid=$pid log=$cell_log" | tee -a "$LOG_FILE"
  done

  # Wait for each cell's /health to come up OR the launch to fail.
  # Launch scripts run --serve-only as a foreground wait — but they
  # background the server and then exit 0 once health is up. So we wait
  # for the launch process to exit (success), then poll /health from
  # the orchestrator side as the user-facing PASS gate.
  declare -A CELL_VERDICT=()
  declare -A CELL_EXIT=()
  declare -A CELL_WALL=()
  for cell in "${SMOKE_CELLS[@]}"; do
    pid=${CELL_PID[$cell]}
    start_s=${CELL_START[$cell]}
    port=${SMOKE_PORT[$cell]}
    if wait "$pid"; then
      rc=0
    else
      rc=$?
    fi
    end_s=$(date +%s)
    wall=$((end_s - start_s))
    CELL_EXIT["$cell"]=$rc
    CELL_WALL["$cell"]=$wall
    if [[ "$rc" -eq 0 ]] && curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      CELL_VERDICT["$cell"]=PASS
    else
      CELL_VERDICT["$cell"]=FAIL
    fi
  done

  # Emit per-cell summary.
  echo | tee -a "$LOG_FILE"
  echo "===== per-cell smoke results =====" | tee -a "$LOG_FILE"
  printf "%-18s %-6s %-6s %-6s %-9s %-8s %s\n" \
    "cell" "gpu" "port" "pid" "exit" "wall_s" "verdict" | tee -a "$LOG_FILE"
  pass_count=0
  fail_count=0
  for cell in "${SMOKE_CELLS[@]}"; do
    gpu=${SMOKE_GPU[$cell]}
    port=${SMOKE_PORT[$cell]}
    pid=${CELL_PID[$cell]}
    rc=${CELL_EXIT[$cell]}
    wall=${CELL_WALL[$cell]}
    verdict=${CELL_VERDICT[$cell]}
    if [[ "$verdict" == "PASS" ]]; then
      pass_count=$((pass_count + 1))
    else
      fail_count=$((fail_count + 1))
    fi
    printf "%-18s %-6d %-6d %-6d %-9d %-8d %s\n" \
      "$cell" "$gpu" "$port" "$pid" "$rc" "$wall" "$verdict" | tee -a "$LOG_FILE"
  done
  echo | tee -a "$LOG_FILE"
  echo "pass=${pass_count} fail=${fail_count}" | tee -a "$LOG_FILE"

  # Tear everything down NOW (smoke only — we do not run any bench step).
  echo | tee -a "$LOG_FILE"
  echo "===== teardown =====" | tee -a "$LOG_FILE"
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
  sleep 3
  if pgrep -af 'vllm.entrypoints' 2>/dev/null; then
    echo "WARNING: lingering vllm.entrypoints processes after teardown" \
      | tee -a "$LOG_FILE"
  else
    echo "no surviving vllm.entrypoints processes (good)" | tee -a "$LOG_FILE"
  fi
  echo "rocm-smi snapshot (post):" | tee -a "$LOG_FILE"
  rocm-smi --showuse 2>/dev/null | sed 's/^/  /' | tee -a "$LOG_FILE" \
    || echo "  rocm-smi unavailable" | tee -a "$LOG_FILE"

  # Disable EXIT trap (we already cleaned up explicitly).
  trap - EXIT
  total=$((pass_count + fail_count))
  if [[ "$fail_count" -eq 0 ]]; then
    echo "smoke PASS (${pass_count}/${total}, cap=${PARALLEL_CAP})" \
      | tee -a "$LOG_FILE"
    exit 0
  else
    echo "smoke FAIL (${pass_count}/${total}, cap=${PARALLEL_CAP})" \
      | tee -a "$LOG_FILE"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Full grid — milestone m4-fused-act-quant
# ---------------------------------------------------------------------------
# Caveats:
#   * `VLLM_MI100_DISABLE_FUSED_ACT_QUANT` MUST be unset in the calling env;
#     this script does NOT touch it. The default-on state is the production
#     fused path being measured by this grid.
#   * NUM_PROMPTS=200 and --seed 42 are fixed by the mission spec; do not
#     parameterize them.
#   * TP=4 cells require 4 visible GPUs. If fewer are exposed the cell is
#     marked FAILED via a schema-conformant placeholder JSON.

GRID_LOG="$OUT_ROOT/run_grid_fused_parallel_$(date -u +%Y%m%dT%H%M%SZ).log"
: > "$GRID_LOG"
PY=/opt/vllm-env/bin/python3
DATASET=/root/bench-int8-w4a16/datasets/coding_agent.jsonl

{
  echo "===== run_grid_fused_parallel.sh --milestone $MILESTONE ====="
  echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo:          $REPO"
  echo "out_root:      $OUT_ROOT"
  echo "grid_log:      $GRID_LOG"
  echo "visible_gpus:  $VISIBLE_GPUS"
  echo "parallel_cap:  $PARALLEL_CAP  (min(visible, 4))"
  echo "fused_off_env: ${VLLM_MI100_DISABLE_FUSED_ACT_QUANT:-(unset, default-on)}"
  echo
  echo "rocm-smi snapshot (pre):"
  rocm-smi --showuse 2>/dev/null | sed 's/^/  /' || echo "  rocm-smi unavailable"
  echo
} | tee -a "$GRID_LOG"

# Sanity: refuse to run if the disable env-var is set — m4-bench-grid is the
# default-on (fused) grid. The disable-path smoke is a sibling feature
# (m4-disable-path-smoke).
if [[ "${VLLM_MI100_DISABLE_FUSED_ACT_QUANT:-0}" != "0" || \
      "${VLLM_DISABLE_FUSED_ACT_QUANT:-0}" != "0" ]]; then
  {
    echo "[grid] FATAL: VLLM_MI100_DISABLE_FUSED_ACT_QUANT or"
    echo "       VLLM_DISABLE_FUSED_ACT_QUANT is set; m4-bench-grid must"
    echo "       run with the default-on fused state. Unset and retry."
  } | tee -a "$GRID_LOG"
  exit 2
fi

# Disable EXIT cleanup trap during the grid so per-cell teardowns own the
# cleanup; we restore + invoke once at the end.
trap - EXIT
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
sleep 2

GIT_SHA=$(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo unknown)

# Cell metadata for output JSONs.
quant_name_for() {
  case "$1" in
    w8a8) echo "w8a8-int8" ;;
    w4a16) echo "w4a16" ;;
  esac
}
model_path_for() {
  case "$1" in
    w8a8) echo "/models/Qwen3.5-9B-w8a8" ;;
    w4a16) echo "/models/Qwen3.5-9B-w4a16" ;;
  esac
}

# Run one (quant, tp, conc, workload) bench against an already-healthy server
# on $1 (PORT). Writes the schema-conformant per-cell JSON.
# Args: quant tp conc workload port raw_dir out_dir
run_one_bench() {
  local quant=$1 tp=$2 conc=$3 wl=$4 port=$5 raw_dir=$6 out_dir=$7
  local cell_id="${quant}_tp${tp}_c${conc}"
  local model
  model=$(model_path_for "$quant")
  local quant_full
  quant_full=$(quant_name_for "$quant")
  mkdir -p "$raw_dir" "$out_dir"
  local env_file="$raw_dir/env.json"
  $PY - >"$env_file" <<EOF
import json, os
keep = [
    "LD_LIBRARY_PATH","ROCM_PATH","PATH","PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_USE_AITER","VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_MI100_DISABLE_CUSTOM_AR","TORCH_COMPILE_DISABLE",
    "HF_HUB_OFFLINE","CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES",
    "NCCL_ALGO","KV_CACHE_DTYPE","ENABLE_CHUNKED_PREFILL",
    "MAX_NUM_BATCHED_TOKENS",
    "VLLM_MI100_DISABLE_FUSED_ACT_QUANT","VLLM_DISABLE_FUSED_ACT_QUANT",
]
d = {k: os.environ.get(k, "") for k in keep}
d["milestone"] = "m4-fused-act-quant"
d["fused_act_quant_state"] = (
    "off" if (
        os.environ.get("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", "0") == "1"
        or os.environ.get("VLLM_DISABLE_FUSED_ACT_QUANT", "0") == "1"
    ) else "on"
)
print(json.dumps(d, indent=2))
EOF

  local bench_args=(
    --model "$model"
    --base-url "http://127.0.0.1:${port}"
    --num-prompts 200
    --request-rate inf
    --max-concurrency "$conc"
    --seed 42
    --save-result
    --result-dir "$raw_dir"
    --result-filename "raw.json"
    --trust-remote-code
    --percentile-metrics "ttft,tpot,itl,e2el"
    --metric-percentiles "50,90,99"
    --metadata "cell_id=${cell_id}" "workload=${wl}" "tp=${tp}" \
               "concurrency=${conc}" "num_prompts=200" \
               "milestone=m4-fused-act-quant" \
               "fused_act_quant_state=${d_fused_state:-on}"
  )
  if [[ "$wl" == synthetic ]]; then
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
  echo "  bench: $cell_id/$wl port=$port" | tee -a "$GRID_LOG"
  cd "$REPO"
  local bench_log="$raw_dir/bench.log"
  set +e
  $PY -m vllm.entrypoints.cli.main bench serve "${bench_args[@]}" >"$bench_log" 2>&1
  local rc=$?
  set -e
  local raw_json="$raw_dir/raw.json"
  if [[ $rc -ne 0 || ! -f "$raw_json" ]]; then
    echo "  bench FAILED rc=$rc cell=${cell_id}_${wl}" | tee -a "$GRID_LOG"
    cat >"$raw_json" <<EOF
{"output_throughput": 0, "request_throughput": 0, "mean_ttft_ms": 0,
 "median_ttft_ms": 0, "p99_ttft_ms": 0, "mean_tpot_ms": 0,
 "median_tpot_ms": 0, "p99_tpot_ms": 0, "FAILED": true,
 "exit_code": ${rc}}
EOF
  fi

  local out_file="$out_dir/${cell_id}_${wl}.json"
  $PY "$REPO/scripts/postprocess_bench_result.py" \
    --raw "$raw_json" \
    --cell "${cell_id}_${wl}" \
    --model "$model" \
    --quant "$quant_full" \
    --tp "$tp" \
    --concurrency "$conc" \
    --request-rate inf \
    --workload "$wl" \
    --num-prompts 200 \
    --launch-command "$launch_command" \
    --env-file "$env_file" \
    --kernel-backend "fused-act-quant" \
    --out "$out_file"
}

# Emit a schema-conformant placeholder JSON for a cell that could not be run
# (e.g., TP=4 on a host with <4 visible GPUs).
# Args: quant tp conc workload reason
emit_placeholder() {
  local quant=$1 tp=$2 conc=$3 wl=$4 reason=$5
  local cell_id="${quant}_tp${tp}_c${conc}"
  local model
  model=$(model_path_for "$quant")
  local quant_full
  quant_full=$(quant_name_for "$quant")
  local out_dir="$OUT_ROOT/$quant"
  mkdir -p "$out_dir"
  local raw_dir="$OUT_ROOT/$quant/${cell_id}/placeholder_${wl}"
  mkdir -p "$raw_dir"
  local env_file="$raw_dir/env.json"
  $PY - >"$env_file" <<EOF
import json, os
d = {
    "milestone": "m4-fused-act-quant",
    "fused_act_quant_state": (
        "off" if (
            os.environ.get("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", "0") == "1"
            or os.environ.get("VLLM_DISABLE_FUSED_ACT_QUANT", "0") == "1"
        ) else "on"
    ),
    "VLLM_MI100_DISABLE_FUSED_ACT_QUANT": os.environ.get("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", ""),
    "VLLM_DISABLE_FUSED_ACT_QUANT": os.environ.get("VLLM_DISABLE_FUSED_ACT_QUANT", ""),
    "placeholder_reason": "${reason}",
}
print(json.dumps(d, indent=2))
EOF
  local raw_json="$raw_dir/raw.json"
  cat >"$raw_json" <<EOF
{"output_throughput": 0, "request_throughput": 0, "mean_ttft_ms": 0,
 "median_ttft_ms": 0, "p99_ttft_ms": 0, "mean_tpot_ms": 0,
 "median_tpot_ms": 0, "p99_tpot_ms": 0, "FAILED": true,
 "placeholder_reason": "${reason}"}
EOF
  local out_file="$out_dir/${cell_id}_${wl}.json"
  $PY "$REPO/scripts/postprocess_bench_result.py" \
    --raw "$raw_json" \
    --cell "${cell_id}_${wl}" \
    --model "$model" \
    --quant "$quant_full" \
    --tp "$tp" \
    --concurrency "$conc" \
    --request-rate inf \
    --workload "$wl" \
    --num-prompts 200 \
    --launch-command "PLACEHOLDER (${reason})" \
    --env-file "$env_file" \
    --kernel-backend "fused-act-quant" \
    --out "$out_file"
  echo "  placeholder: $cell_id/$wl  reason='${reason}'" | tee -a "$GRID_LOG"
}

# Wait for /health on $1 (PORT) with timeout in seconds ($2). Echoes 0/1.
wait_health() {
  local port=$1 timeout=${2:-600}
  local deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo 1; return
    fi
    sleep 5
  done
  echo 0
}

# Build the per-cell server-log dir under $OUT_ROOT/<quant>/<cell>/ so each
# parallel server writes its log into a distinct path.
server_log_dir_for() {
  local quant=$1 tp=$2 conc=$3
  echo "$OUT_ROOT/${quant}/${quant}_tp${tp}_c${conc}"
}

# ---------------------------------------------------------------------------
# TP=1 fan-out: 4 (quant, conc) pairs per quant × 2 quants = 8 distinct
# (quant, conc) pairs. Each pair runs both workloads back-to-back on a
# single GPU; up to $PARALLEL_CAP pairs run concurrently. 4 concurrencies
# {1, 2, 4, 8} per quant per TP.
# ---------------------------------------------------------------------------
TP1_PAIRS=(
  "w8a8:1" "w8a8:2" "w8a8:4"
  "w4a16:1" "w4a16:2" "w4a16:4"
)

# Run one TP=1 (quant, conc) pair on the given GPU + port: launch server,
# bench synthetic + coding, tear down.
# Args: quant conc gpu port
run_tp1_pair() {
  local quant=$1 conc=$2 gpu=$3 port=$4
  local cell_id="${quant}_tp${conc/#/c}"
  cell_id="${quant}_tp1_c${conc}"
  local svc_log_dir
  svc_log_dir=$(server_log_dir_for "$quant" 1 "$conc")
  mkdir -p "$svc_log_dir"
  local svc_log="$svc_log_dir/server.log"
  local launch_script="$REPO/scripts/launch_hbm_${quant}_tp1_c${conc}.sh"
  if [[ ! -x "$launch_script" ]]; then
    echo "  FATAL: missing launch script $launch_script" | tee -a "$GRID_LOG"
    return 2
  fi
  echo "[tp1-pair] start $cell_id gpu=$gpu port=$port log=$svc_log" | tee -a "$GRID_LOG"
  # Run the launch script in --serve-only mode. It backgrounds the server and
  # exits once /health is up. SERVER_LOG_DIR points server.log into the
  # per-cell dir under $OUT_ROOT.
  SERVER_LOG_DIR="$svc_log_dir" \
    HIP_VISIBLE_DEVICES="$gpu" \
    LAUNCH_HEALTH_WAIT_SECS=900 \
    bash "$launch_script" --port "$port" --serve-only \
    >"$svc_log_dir/launcher.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  launcher exit=$rc for $cell_id; emitting placeholders" | tee -a "$GRID_LOG"
    emit_placeholder "$quant" 1 "$conc" synthetic "launcher_exit_${rc}"
    emit_placeholder "$quant" 1 "$conc" coding "launcher_exit_${rc}"
    pkill -9 -P $$ 2>/dev/null || true
    return $rc
  fi
  # Run both workloads back-to-back on this server.
  local raw_base="$OUT_ROOT/${quant}/${cell_id}"
  local out_dir="$OUT_ROOT/${quant}"
  for wl in synthetic coding; do
    local raw_dir="$raw_base/raw_${wl}"
    mkdir -p "$raw_dir"
    if ! run_one_bench "$quant" 1 "$conc" "$wl" "$port" "$raw_dir" "$out_dir"; then
      echo "  bench $cell_id/$wl returned non-zero" | tee -a "$GRID_LOG"
    fi
  done
  # Tear down THIS server (port-scoped). Use `fuser` to find the listener
  # PID and kill the entire process group; this confines the kill to our
  # server and leaves sibling cells on other ports running.
  fuser -k -TERM "${port}/tcp" 2>/dev/null || true
  sleep 3
  fuser -k -KILL "${port}/tcp" 2>/dev/null || true
  # Belt-and-braces: pkill any vllm process whose cmdline references this
  # port (api_server has --port <N> in its argv).
  pkill -9 -f -- "--port ${port}" 2>/dev/null || true
  sleep 2
  echo "[tp1-pair] done  $cell_id" | tee -a "$GRID_LOG"
}

# Slot-tracking GPU/port assignment across PARALLEL_CAP workers. Each slot
# pairs (GPU n -> port 8000+n). A slot is "free" when its previous worker
# PID has exited. New pairs are dispatched to the next free slot.
echo "===== TP=1 fan-out ($PARALLEL_CAP workers) =====" | tee -a "$GRID_LOG"
declare -a SLOT_PID
declare -a SLOT_PAIR
for ((s = 0; s < PARALLEL_CAP; s++)); do
  SLOT_PID[$s]=0
  SLOT_PAIR[$s]=""
done

acquire_slot() {
  # Echo the index of the first free slot; block until one is free.
  while :; do
    for ((s = 0; s < PARALLEL_CAP; s++)); do
      local pid="${SLOT_PID[$s]}"
      if [[ "$pid" == "0" ]] || ! kill -0 "$pid" 2>/dev/null; then
        # Reap if a worker just exited.
        if [[ "$pid" != "0" ]]; then
          wait "$pid" 2>/dev/null || true
          SLOT_PID[$s]=0
          SLOT_PAIR[$s]=""
        fi
        echo "$s"
        return
      fi
    done
    sleep 5
  done
}

for pair in "${TP1_PAIRS[@]}"; do
  IFS=":" read -r q conc <<<"$pair"
  slot=$(acquire_slot)
  gpu="$slot"
  port=$(( 8000 + slot ))
  run_tp1_pair "$q" "$conc" "$gpu" "$port" &
  SLOT_PID[$slot]=$!
  SLOT_PAIR[$slot]="$pair"
  echo "[scheduler] dispatched $pair to slot=$slot gpu=$gpu port=$port pid=${SLOT_PID[$slot]}" \
    | tee -a "$GRID_LOG"
  # Small stagger to reduce concurrent model-load HBM spike on a single
  # GPU bus when launching sibling cells back-to-back.
  sleep 2
done

# Drain all live slots.
for ((s = 0; s < PARALLEL_CAP; s++)); do
  pid="${SLOT_PID[$s]}"
  if [[ "$pid" != "0" ]]; then
    wait "$pid" 2>/dev/null || true
    SLOT_PID[$s]=0
  fi
done

# Belt-and-braces: kill anything lingering between phases.
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
sleep 5
echo "===== TP=1 fan-out drained =====" | tee -a "$GRID_LOG"

# ---------------------------------------------------------------------------
# TP=4 serial cells (4 concurrencies × 2 quants = 8 cell configs × 2
# workloads = 16 cell-workloads). Each TP=4 cell requires 4 visible GPUs;
# on a host with fewer, emit placeholders.
# ---------------------------------------------------------------------------
echo "===== TP=4 serial cells =====" | tee -a "$GRID_LOG"
if (( VISIBLE_GPUS < 4 )); then
  echo "[tp4] visible_gpus=$VISIBLE_GPUS < 4 -- emitting placeholders for all TP=4 cells" \
    | tee -a "$GRID_LOG"
  for quant in w8a8 w4a16; do
    for conc in 1 2 4; do
      for wl in synthetic coding; do
        emit_placeholder "$quant" 4 "$conc" "$wl" \
          "tp4_requires_4_gpus_visible_${VISIBLE_GPUS}"
      done
    done
  done
else
  for quant in w8a8 w4a16; do
    for conc in 1 2 4; do
      cell_id="${quant}_tp4_c${conc}"
      svc_log_dir=$(server_log_dir_for "$quant" 4 "$conc")
      mkdir -p "$svc_log_dir"
      launch_script="$REPO/scripts/launch_hbm_${quant}_tp4_c${conc}.sh"
      if [[ ! -x "$launch_script" ]]; then
        echo "  FATAL: missing launch script $launch_script" | tee -a "$GRID_LOG"
        emit_placeholder "$quant" 4 "$conc" synthetic "launch_script_missing"
        emit_placeholder "$quant" 4 "$conc" coding "launch_script_missing"
        continue
      fi
      echo "[tp4] start $cell_id" | tee -a "$GRID_LOG"
      SERVER_LOG_DIR="$svc_log_dir" \
        LAUNCH_HEALTH_WAIT_SECS=900 \
        bash "$launch_script" --port 8000 --serve-only \
        >"$svc_log_dir/launcher.log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "  launcher exit=$rc for $cell_id; placeholders" | tee -a "$GRID_LOG"
        emit_placeholder "$quant" 4 "$conc" synthetic "launcher_exit_${rc}"
        emit_placeholder "$quant" 4 "$conc" coding "launcher_exit_${rc}"
        pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
        sleep 3
        continue
      fi
      raw_base="$OUT_ROOT/${quant}/${cell_id}"
      out_dir="$OUT_ROOT/${quant}"
      for wl in synthetic coding; do
        raw_dir="$raw_base/raw_${wl}"
        mkdir -p "$raw_dir"
        run_one_bench "$quant" 4 "$conc" "$wl" 8000 "$raw_dir" "$out_dir" || true
      done
      pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
      pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
      sleep 5
      echo "[tp4] done  $cell_id" | tee -a "$GRID_LOG"
    done
  done
fi

# ---------------------------------------------------------------------------
# Schema-validator-friendly layout: <root>/{synthetic,coding}/<cell>.json
# (relative symlinks). aggregate_hbm.py and validate_results_schema.py both
# benefit from this layout — the latter scans <root>/{synthetic,coding}/.
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT/synthetic" "$OUT_ROOT/coding"
for quant in w8a8 w4a16; do
  for tp in 1 4; do
    for conc in 1 2 4; do
      for wl in synthetic coding; do
        cell_id="${quant}_tp${tp}_c${conc}"
        src_rel="../${quant}/${cell_id}_${wl}.json"
        link="$OUT_ROOT/${wl}/${cell_id}.json"
        if [[ -f "$OUT_ROOT/${quant}/${cell_id}_${wl}.json" ]]; then
          ln -sfn "$src_rel" "$link"
        fi
      done
    done
  done
done

# ---------------------------------------------------------------------------
# Final lifecycle sweep.
# ---------------------------------------------------------------------------
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
sleep 5
{
  echo
  echo "===== run_grid_fused_parallel.sh DONE ====="
  echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "rocm-smi snapshot (post):"
  rocm-smi --showuse 2>/dev/null | sed 's/^/  /' || echo "  rocm-smi unavailable"
  echo "git_sha: $GIT_SHA"
} | tee -a "$GRID_LOG"
exit 0
