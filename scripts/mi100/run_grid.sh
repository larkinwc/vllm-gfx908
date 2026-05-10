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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
if command -v git >/dev/null 2>&1; then
  GIT_TOPLEVEL=$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null || true)
  if [[ -n "$GIT_TOPLEVEL" ]]; then
    REPO=$GIT_TOPLEVEL
  fi
fi
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
  m3-w8a8-autotune) ROOT=/root/bench-int8-w4a16/m3/w8a8/autotune ;;
  m3-w8a8-heuristic) ROOT=/root/bench-int8-w4a16/m3/w8a8/heuristic ;;
  m3-w4a16-mi100) ROOT=/root/bench-int8-w4a16/m3/w4a16/mi100 ;;
  m3-w4a16-generic) ROOT=/root/bench-int8-w4a16/m3/w4a16/generic ;;
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
elif [[ "$milestone" == "m3-w8a8-autotune" || "$milestone" == "m3-w8a8-heuristic" ]]; then
  # M3 W8A8 cells: keep the M2-build libhipblaslt + merged library
  # in flight so the dispatcher's "hipBLASLt > Triton" priority is
  # exercised exactly as it ships. The M3 mi100_int8 Triton kernel
  # picks up its persisted autotune configs from configs/gfx908/.
  if [[ ! -d "$MERGED_LIB_DIR" ]]; then
    echo "FATAL: merged library missing at $MERGED_LIB_DIR" >&2
    exit 2
  fi
  export LD_LIBRARY_PATH="$BUILD_LIB_DIR:/opt/rocm/core-7.12/lib"
  export HIPBLASLT_TENSILE_LIBPATH="$MERGED_LIB_DIR"
  if [[ "$milestone" == "m3-w8a8-heuristic" ]]; then
    export VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1
    echo "[run_grid] M3 heuristic env: VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1"
  else
    unset VLLM_MI100_DISABLE_AUTOTUNE_CONFIG || true
  fi
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  echo "[run_grid] M3 W8A8 env:"
  echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "  HIPBLASLT_TENSILE_LIBPATH=$HIPBLASLT_TENSILE_LIBPATH"
  echo "  VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=${VLLM_MI100_DISABLE_AUTOTUNE_CONFIG:-(unset)}"
  echo "  PYTHONPATH=$PYTHONPATH"
elif [[ "$milestone" == "m3-w4a16-mi100" || "$milestone" == "m3-w4a16-generic" ]]; then
  # M3 W4A16 cells: same libhipblaslt as M2 (irrelevant for w4a16
  # path but keeps env consistent). Toggle the mi100_w4a16 kernel
  # via VLLM_DISABLE_MI100_W4A16.
  export LD_LIBRARY_PATH="$BUILD_LIB_DIR:/opt/rocm/core-7.12/lib"
  unset HIPBLASLT_TENSILE_LIBPATH
  if [[ "$milestone" == "m3-w4a16-generic" ]]; then
    export VLLM_DISABLE_MI100_W4A16=1
    echo "[run_grid] M3 W4A16 generic env: VLLM_DISABLE_MI100_W4A16=1"
  else
    unset VLLM_DISABLE_MI100_W4A16 || true
  fi
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  echo "[run_grid] M3 W4A16 env:"
  echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "  VLLM_DISABLE_MI100_W4A16=${VLLM_DISABLE_MI100_W4A16:-(unset)}"
  echo "  PYTHONPATH=$PYTHONPATH"
fi

# Filter the cells.
export CELLS_FILTER="$filter"
echo "[run_grid] CELLS_FILTER=$CELLS_FILTER"
echo "[run_grid] kicking off $M2_HARNESS"

bash "$M2_HARNESS"
exit $?
