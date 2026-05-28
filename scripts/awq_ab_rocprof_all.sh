#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/awq_ab_rocprof_all.sh — M3-F1 driver: runs all 6 rocprofv3
# captures sequentially (3 paths × 2 cells).
set -uo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/puny-animals-go-wp0to
RUNNER_LOG=/root/bench-w4a16-ab/rocprof/m3f1_runner.log
mkdir -p "$(dirname "$RUNNER_LOG")"

CELLS=(
  "a w4a16_tp1_c1_synthetic"
  "a w4a16_tp4_c4_coding"
  "b w4a16_tp1_c1_synthetic"
  "b w4a16_tp4_c4_coding"
  "c w4a16_tp1_c1_synthetic"
  "c w4a16_tp4_c4_coding"
)

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ) M3F1] $*" | tee -a "$RUNNER_LOG"; }

log "==== M3-F1 rocprofv3 capture grid begin ===="
log "host=$(hostname) pwd=$(pwd) rocprofv3=$(/opt/rocm/core-7.12/bin/rocprofv3 --version | tr -d '\n')"

i=0
total=${#CELLS[@]}
for entry in "${CELLS[@]}"; do
  i=$((i+1))
  read -r path cell <<<"$entry"
  log "==== ($i/$total) path=$path cell=$cell ===="
  t0=$(date +%s)
  bash "$REPO/scripts/awq_ab_rocprof.sh" "$path" "$cell" 2>&1 | tee -a "$RUNNER_LOG"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  dt=$((t1 - t0))
  log "==== path=$path cell=$cell rc=$rc elapsed=${dt}s ===="
  # Thermal/cooldown between captures
  log "  cooldown 60s"
  sleep 60
done

log "==== M3-F1 rocprofv3 capture grid done ===="
exit 0
