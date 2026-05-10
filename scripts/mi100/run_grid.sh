#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/run_grid.sh -- M2 (and beyond) grid bench wrapper
# =============================================================================
#
# Wraps the M1 baseline harness (/root/bench-int8-w4a16/baseline/run_baseline.sh)
# with the M2 hipBLASLt env-vars so the spawned vLLM picks up the freshly-
# built libhipblaslt.so + our merged TensileLite library.
#
# Usage:
#   scripts/mi100/run_grid.sh m2 [CELLS_FILTER]
#
# CELLS_FILTER is a regex passed to the M1 harness's CELLS_FILTER var.
# Examples:
#   scripts/mi100/run_grid.sh m2                   # full 24-cell grid
#   scripts/mi100/run_grid.sh m2 'w8a8_tp1_c4_'    # 2 cells (synth+coding)
#   scripts/mi100/run_grid.sh m2 'w8a8_tp[14]_c4_' # 4 cells
#
# Outputs: same as M1 baseline harness, but rooted at
#   /root/bench-int8-w4a16/m2/{synthetic,coding}/<cell>.json
#
# NOTE: The M1 harness writes results into $BASELINE_ROOT (hard-coded).
#       This wrapper invokes a copy with BASELINE_ROOT redirected to
#       /root/bench-int8-w4a16/m2/, by patching the inherited env.
# =============================================================================
set -uo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
M1_HARNESS=/root/bench-int8-w4a16/baseline/run_baseline.sh
M2_ROOT=/root/bench-int8-w4a16/m2
M1_REBASELINE_ROOT=/root/bench-int8-w4a16/m1-rebaseline
MERGED_LIB_DIR=/root/bench-int8-w4a16/tensilelite/merged_library/library
BUILD_LIB_DIR=/root/hipblaslt-src/build/release/library

milestone=${1:-m2}
filter=${2:-.*}

case "$milestone" in
  m2) ROOT=$M2_ROOT ;;
  m1) ROOT=/root/bench-int8-w4a16/baseline ;;
  m1-rebaseline) ROOT=$M1_REBASELINE_ROOT ;;
  *) echo "unknown milestone $milestone" >&2; exit 2 ;;
esac

mkdir -p "$ROOT"/{synthetic,coding}

if [[ ! -f "$M1_HARNESS" ]]; then
  echo "FATAL: M1 harness missing at $M1_HARNESS" >&2
  exit 2
fi
if [[ "$milestone" == "m2" ]]; then
  if [[ ! -d "$MERGED_LIB_DIR" ]]; then
    echo "FATAL: merged library missing at $MERGED_LIB_DIR" >&2
    echo "       Run scripts/mi100/merge_tensile_logic.sh --foreground first." >&2
    exit 2
  fi
fi
if [[ "$milestone" == "m1-rebaseline" ]]; then
  if [[ ! -f "$BUILD_LIB_DIR/libhipblaslt.so" ]]; then
    echo "FATAL: M2 build libhipblaslt.so missing at $BUILD_LIB_DIR" >&2
    exit 2
  fi
fi

# Copy harness to milestone root and patch BASELINE_ROOT to point here.
M2_HARNESS=$ROOT/run_${milestone}.sh
sed "s|^BASELINE_ROOT=.*$|BASELINE_ROOT=$ROOT|" "$M1_HARNESS" > "$M2_HARNESS"
chmod +x "$M2_HARNESS"

# M2 env-vars layered on top of M1.
if [[ "$milestone" == "m2" ]]; then
  export LD_LIBRARY_PATH="$BUILD_LIB_DIR:/opt/rocm/core-7.12/lib"
  export HIPBLASLT_TENSILE_LIBPATH="$MERGED_LIB_DIR"
  echo "[run_grid] M2 env exported:"
  echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "  HIPBLASLT_TENSILE_LIBPATH=$HIPBLASLT_TENSILE_LIBPATH"
elif [[ "$milestone" == "m1-rebaseline" ]]; then
  # Same M2-build libhipblaslt.so + LD_LIBRARY_PATH, but DO NOT export
  # HIPBLASLT_TENSILE_LIBPATH so hipBLASLt loads its own prebuilt
  # I8I8 default kernel (the one that ships with the M2 build).
  # This isolates the libhipblaslt build/system-swap delta from the
  # actual TensileLite tuning gain delivered by the merged library.
  export LD_LIBRARY_PATH="$BUILD_LIB_DIR:/opt/rocm/core-7.12/lib"
  unset HIPBLASLT_TENSILE_LIBPATH
  echo "[run_grid] m1-rebaseline env exported:"
  echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "  HIPBLASLT_TENSILE_LIBPATH=(unset)"
fi

# Filter the cells.
export CELLS_FILTER="$filter"
echo "[run_grid] CELLS_FILTER=$CELLS_FILTER"
echo "[run_grid] kicking off $M2_HARNESS"

bash "$M2_HARNESS"
exit $?
