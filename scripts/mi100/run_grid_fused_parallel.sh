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
#   (default) Full grid — currently STUB: only emits an explanatory message.
#             The m4-bench-grid worker is expected to flesh out the
#             non-smoke path in a follow-up commit; this orchestrator only
#             needs to support the smoke gate for VAL-M4-001.
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

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
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

MODE=${1:-}
case "$MODE" in
  --smoke|"") ;;
  *)
    echo "usage: $0 [--smoke]" >&2
    exit 2
    ;;
esac

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
# Full grid (stub)
# ---------------------------------------------------------------------------
cat <<'EOF' >&2
[run_grid_fused_parallel.sh] full-grid mode is not yet wired up in this
orchestrator. Use the m4-bench-grid worker's harness, or invoke this script
with --smoke for the parallel-harness gate (VAL-M4-001). TP=1 cells:
${TP1_CELLS[*]}; TP=4 cells (serial): ${TP4_CELLS[*]}.
EOF
trap - EXIT
exit 3
