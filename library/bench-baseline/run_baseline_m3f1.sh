#!/usr/bin/env bash
# M3-F1 baseline harness wrapper: re-routes REPO + BASELINE_ROOT in the
# locked M1 harness /root/bench-int8-w4a16/baseline/run_baseline.sh so it
# targets THIS worktree (loose-rats-clean-phm2k) at pre-sync-baseline,
# and writes outputs into /root/bench-int8-w4a16/m3-f1-baseline/.
#
# After this script finishes, post-hoc copy:
#   cp -a /root/bench-int8-w4a16/m3-f1-baseline/* \
#         <worktree>/library/bench-baseline/grid/
set -uo pipefail

REPO_NEW=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k
ROOT_NEW=/root/bench-int8-w4a16/m3-f1-baseline
M1_HARNESS=/root/bench-int8-w4a16/baseline/run_baseline.sh
PATCHED=$ROOT_NEW/run_m3f1.sh

mkdir -p "$ROOT_NEW"/{synthetic,coding,profile}

# Patch the locked harness:
#   REPO=...fuzzy-hornets...     -> this worktree
#   BASELINE_ROOT=...baseline    -> $ROOT_NEW
sed \
  -e "s|^REPO=.*|REPO=$REPO_NEW|" \
  -e "s|^BASELINE_ROOT=.*|BASELINE_ROOT=$ROOT_NEW|" \
  "$M1_HARNESS" > "$PATCHED"
chmod +x "$PATCHED"

echo "[run_baseline_m3f1] patched harness: $PATCHED"
echo "[run_baseline_m3f1] REPO=$REPO_NEW"
echo "[run_baseline_m3f1] BASELINE_ROOT=$ROOT_NEW"
echo "[run_baseline_m3f1] CELLS_FILTER=${CELLS_FILTER:-.*}"
echo "[run_baseline_m3f1] kicking off"

bash "$PATCHED"
exit $?
