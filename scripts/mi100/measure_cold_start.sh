#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/mi100/measure_cold_start.sh — MI100 M2 Producer Wire-In, M1-F3.
#
# Verifies VAL-M1-003: cold-start engine load < 5 min on a wiped
# TRITON_CACHE_DIR with the M1-F2 pinned autotune JSONs in place.
#
# This is the ONE AND ONLY feature in the mission allowed to wipe
# TRITON_CACHE_DIR (mission AGENTS.md anti-pattern #13). Do NOT invoke
# the production fused-on path (m2-producer-wire-in) before this gate
# confirms cold-start is bounded; this feature deliberately precedes
# m2-producer-wire-in by design.
#
# Steps:
#   1. Teardown any existing vLLM server (triad pkill — anti-pattern #8).
#   2. Wipe TRITON_CACHE_DIR (default $HOME/.triton/cache).
#   3. Launch scripts/launch_hbm_w8a8_tp1_c1.sh --serve-only in background.
#   4. Poll http://localhost:8000/health until OK or 360 s timeout.
#   5. Record delta_s = health_ts - start_ts.
#   6. Teardown again (triad pkill).
#   7. Write cold_start_timing.json into the output dir.
#   8. ASSERT delta_s < 300 (5 min); fail the script otherwise.
#
# Usage:
#   bash scripts/mi100/measure_cold_start.sh [out_dir]
#
# Default out_dir: /root/bench-int8-w4a16-m2-producer/m1-cold-start/

set -euo pipefail

OUT_DIR="${1:-/root/bench-int8-w4a16-m2-producer/m1-cold-start}"
mkdir -p "$OUT_DIR"

REPO_ROOT="/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/thin-hands-smell-2bxf5"
LAUNCH_SCRIPT="${REPO_ROOT}/scripts/launch_hbm_w8a8_tp1_c1.sh"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-360}"
PORT=8000

TIMING_JSON="$OUT_DIR/cold_start_timing.json"
SERVER_LOG="$OUT_DIR/server.log"

log() { echo "[measure_cold_start] $*"; }

# --- Triad teardown (anti-pattern #8) -------------------------------------
teardown() {
  pkill -f 'vllm.entrypoints.cli.main'        2>/dev/null || true
  pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  pkill -f 'VLLM::EngineCore'                 2>/dev/null || true
  pkill -f 'multiprocessing.resource_tracker' 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! ss -ltnp 2>/dev/null | grep -qE ":${PORT}\b"; then
      break
    fi
    sleep 2
  done
}

# Make sure we never leave a server behind, even on early exit/error.
trap teardown EXIT INT TERM

log "out_dir=$OUT_DIR"
log "triton_cache_dir=$TRITON_CACHE_DIR"
log "health_timeout_secs=$HEALTH_TIMEOUT_SECS"

# --- Step 1: pre-run teardown --------------------------------------------
log "pre-run teardown"
teardown

# --- Step 2: wipe TRITON_CACHE_DIR ---------------------------------------
mkdir -p "$TRITON_CACHE_DIR"
log "wiping $TRITON_CACHE_DIR/* (ONLY feature allowed to do so — anti-pattern #13)"
# Use find to avoid blowing up on huge cache dirs and to skip dotfiles cleanly.
find "$TRITON_CACHE_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# --- Step 3: record start_ts + launch server in background ---------------
START_TS=$(date +%s)
log "start_ts=$START_TS"

# SERVER_LOG_DIR is honored by the launch script for its own server.log path.
export SERVER_LOG_DIR="$OUT_DIR/launch_logs"
mkdir -p "$SERVER_LOG_DIR"

bash "$LAUNCH_SCRIPT" --serve-only >"$SERVER_LOG" 2>&1 &
LAUNCH_PID=$!
log "launch PID=$LAUNCH_PID (launch wrapper); log=$SERVER_LOG"

# --- Step 4: poll /health ------------------------------------------------
HEALTH_TS=0
DEADLINE=$((START_TS + HEALTH_TIMEOUT_SECS))
while :; do
  NOW=$(date +%s)
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    HEALTH_TS=$NOW
    log "healthcheck OK at health_ts=$HEALTH_TS (elapsed=$((HEALTH_TS - START_TS)) s)"
    break
  fi
  if (( NOW >= DEADLINE )); then
    log "FATAL: server did not become healthy within ${HEALTH_TIMEOUT_SECS} s"
    log "--- tail of launch wrapper log ---"
    tail -n 80 "$SERVER_LOG" || true
    HEALTH_TS=$NOW
    break
  fi
  sleep 2
done

# --- Step 5: teardown ----------------------------------------------------
log "post-health teardown"
teardown

# --- Step 6: compute delta and write JSON --------------------------------
DELTA_S=$((HEALTH_TS - START_TS))
HEAD_SHA=$(cd "$REPO_ROOT" && git rev-parse HEAD)

/opt/vllm-env/bin/python3 - "$START_TS" "$HEALTH_TS" "$DELTA_S" "$HEAD_SHA" "$TRITON_CACHE_DIR" "$TIMING_JSON" <<'PY'
import json
import sys

start_ts, health_ts, delta_s, head_sha, triton_cache_dir, out_path = sys.argv[1:7]
payload = {
    "start_ts": int(start_ts),
    "health_ts": int(health_ts),
    "delta_s": int(delta_s),
    "head_sha": head_sha,
    "triton_cache_dir": triton_cache_dir,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"[measure_cold_start] wrote {out_path}: {payload}")
PY

# --- Step 7: assert < 300 s ----------------------------------------------
log "delta_s=$DELTA_S"
if (( DELTA_S >= 300 )); then
  log "FAIL: delta_s=$DELTA_S >= 300 — cold-start gate FAILED."
  log "Surface as blocker: return to m1-autotune-jsons to expand pinned-shape coverage."
  exit 1
fi
log "PASS: delta_s=$DELTA_S < 300 — cold-start gate satisfied (VAL-M1-003)."
exit 0
