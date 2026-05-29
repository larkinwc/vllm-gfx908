#!/usr/bin/env bash
# M3-F2 post-sync harness wrapper — identical structure to
# library/bench-baseline/run_baseline_m3f1.sh, just retargets the
# BASELINE_ROOT into /root/bench-int8-w4a16/m3-f2-postsync. Run from
# upstream-sync-2026-05-28 HEAD with the same env, seeds, and prompts.
set -uo pipefail

REPO_NEW=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k
ROOT_NEW=/root/bench-int8-w4a16/m3-f2-postsync
M1_HARNESS=/root/bench-int8-w4a16/baseline/run_baseline.sh
PATCHED=$ROOT_NEW/run_m3f2.sh

mkdir -p "$ROOT_NEW"/{synthetic,coding,profile}

sed \
  -e "s|^REPO=.*|REPO=$REPO_NEW|" \
  -e "s|^BASELINE_ROOT=.*|BASELINE_ROOT=$ROOT_NEW|" \
  "$M1_HARNESS" > "$PATCHED"
chmod +x "$PATCHED"

echo "[run_postsync_m3f2] patched harness: $PATCHED"
echo "[run_postsync_m3f2] REPO=$REPO_NEW"
echo "[run_postsync_m3f2] BASELINE_ROOT=$ROOT_NEW"
echo "[run_postsync_m3f2] CELLS_FILTER=${CELLS_FILTER:-.*}"
echo "[run_postsync_m3f2] kicking off"

bash "$PATCHED"
exit $?
