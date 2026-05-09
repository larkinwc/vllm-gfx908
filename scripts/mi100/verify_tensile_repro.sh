#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/verify_tensile_repro.sh -- VAL-TENSILE-009 evidence collector
# =============================================================================
#
# Re-run TensileLite tuning for ONE canary (M, N, K) shape and assert the
# new logic YAML is byte-identical to the previously committed version.
#
# Pinned versions (recorded in hipblaslt_tuned_shapes.json):
#   - ROCm 7.12
#   - hipBLASLt rocm-7.2.0 (sha 90de4d11... in /root/hipblaslt-src)
#   - TensileLite (in-tree)
#   - gfx908
# =============================================================================
set -uo pipefail

CANARY=${CANARY:-tune_M512_N4096_K4096}
CFG_DIR=${CFG_DIR:-/root/bench-int8-w4a16/tensilelite/configs}
LOGIC_DIR=${LOGIC_DIR:-/root/bench-int8-w4a16/tensilelite/logic/gfx908}
REPRO_DIR=${REPRO_DIR:-/root/bench-int8-w4a16/tensilelite/repro}
DIFF_OUT=${DIFF_OUT:-/root/bench-int8-w4a16/tensilelite/repro_canary.diff}

mkdir -p "$REPRO_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

original="$LOGIC_DIR/${CANARY}.yaml"
if [[ ! -f "$original" ]]; then
  echo "FATAL: canary baseline missing: $original" >&2
  exit 2
fi

cfg="$CFG_DIR/${CANARY}.yaml"
if [[ ! -f "$cfg" ]]; then
  echo "FATAL: canary config missing: $cfg" >&2
  exit 2
fi

log "re-running TensileLite for canary $CANARY"
log "  cfg       = $cfg"
log "  baseline  = $original"

# Drive the tuner into a sibling run dir (avoid clobbering main run output).
RUNS_DIR="$REPRO_DIR/runs" \
LOGIC_DIR="$REPRO_DIR/logic" \
LOGS_DIR="$REPRO_DIR/logs" \
SUMMARY_JSON="$REPRO_DIR/repro_summary.json" \
bash "$(dirname "$0")/run_tensilelite.sh" "$cfg"
rc=$?
if [[ $rc -ne 0 ]]; then
  log "FAIL: repro tune exit=$rc"
  exit $rc
fi

repro="$REPRO_DIR/logic/${CANARY}.yaml"
if [[ ! -f "$repro" ]]; then
  log "FAIL: repro logic YAML missing at $repro"
  exit 2
fi

if diff -u "$original" "$repro" > "$DIFF_OUT" 2>&1; then
  log "PASS: byte-identical repro for $CANARY"
  log "  diff -> $DIFF_OUT (empty)"
  exit 0
else
  log "FAIL: repro logic YAML differs from baseline"
  log "  diff -> $DIFF_OUT"
  head -40 "$DIFF_OUT" | sed 's/^/    | /'
  exit 1
fi
