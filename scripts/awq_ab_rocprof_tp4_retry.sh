#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Re-runs the 3 TP=4 captures using the LLM-direct harness.
set -uo pipefail
REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/puny-animals-go-wp0to
RUNNER_LOG=/root/bench-w4a16-ab/rocprof/m3f1_tp4_retry.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ) M3F1-TP4-RETRY] $*" | tee -a "$RUNNER_LOG"; }

for path in a b c; do
  log "==== retry path=$path cell=w4a16_tp4_c4_coding ===="
  t0=$(date +%s)
  bash "$REPO/scripts/awq_ab_rocprof.sh" "$path" "w4a16_tp4_c4_coding" 2>&1 | tee -a "$RUNNER_LOG"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  log "==== path=$path TP4 rc=$rc elapsed=$((t1-t0))s ===="
  sleep 60
done
log "==== TP4 retry done ===="
