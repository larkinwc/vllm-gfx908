#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/merge_tensile_logic.sh
# =============================================================================
#
# M2 lazy-index merge driver.
#
# We tried the canonical TensileMergeLibrary path (merging our 8 tuned
# logic YAMLs into the upstream arcturus_Cijk_Ailk_Bljk_I8II_BH.yaml)
# and discovered the upstream ROCm-7.2.0 arcturus I8 logic uses a
# different ProblemType (DestDataType=6 / FP16 output, TransposeB=False)
# than the I8 -> I8 dest, TransposeB=True contraction we tuned. The
# upstream library ships the matching contraction only as a
# pre-compiled .dat/.co (TensileLibrary_I8I8_II8_..._Ailk_Bljk_..._gfx908)
# without an arcturus_*_yaml source file. A direct YAML-merge therefore
# fails ProblemType comparison.
#
# Pragmatic alternative used here:
#   - Treat our 8 tuned YAMLs as the merged logic input directly (each
#     is a self-contained LibraryLogic file with one tuned (M,N,K)
#     -> tuned solution entry).
#   - Run TensileCreateLibrary against that logic dir to build a
#     self-contained merged library tree at
#     $MERGED_LIBRARY_DIR (TensileLibrary_lazy_gfx908.dat plus per-
#     contraction .dat / .co plus the assembled Kernels.so-*-gfx908.hsaco).
#   - Stage the upstream prebuilt I8I8_II8 default logic alongside the
#     merged library so hipBLASLt has fallback solutions for shapes
#     not in our tuned set.
#
# Step 2 (TensileCreateLibrary) is the heavyweight build -- it is
# fire-and-forget by default; pass --foreground to block on it.
#
# Pre-requisites (from library/m2-build-artifacts.md):
#   - /root/hipblaslt-src/build/release/* present (libhipblaslt.so + Tensile)
#   - /shared -> /root/shared symlink intact (origami)
#   - 8 tuned YAMLs at $LOGIC_DIR
# =============================================================================
set -uo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
TENSILE_DIR=/root/hipblaslt-src/tensilelite
TENSILE_BIN=$TENSILE_DIR/Tensile/bin/Tensile
TENSILE_MERGE=$TENSILE_DIR/Tensile/bin/TensileMergeLibrary
TENSILE_CREATE=$TENSILE_DIR/Tensile/bin/TensileCreateLibrary
PY=/opt/vllm-env/bin/python3

LOGIC_DIR=${LOGIC_DIR:-/root/bench-int8-w4a16/tensilelite/logic/gfx908}
MERGE_ROOT=${MERGE_ROOT:-/root/bench-int8-w4a16/tensilelite/merge_workspace}
MERGED_LIBRARY_DIR=${MERGED_LIBRARY_DIR:-/root/bench-int8-w4a16/tensilelite/merged_library}
MERGED_LOGIC_DIR=$MERGE_ROOT/merged_logic
LOG_FILE=${LOG_FILE:-/root/bench-int8-w4a16/tensilelite/merge_run.log}

# Upstream prebuilt library shipped by the m2-hipblaslt-build worker.
# Used as the fallback library staged alongside our merged kernels so
# hipBLASLt has solutions for I8 contractions not in our tuned set.
PREBUILT_LIB_DIR=/root/hipblaslt-src/build/release/hipblaslt-install/lib/hipblaslt/library

FOREGROUND=${FOREGROUND:-0}
SKIP_CREATE=${SKIP_CREATE:-0}
for arg in "$@"; do
  case "$arg" in
    --foreground) FOREGROUND=1 ;;
    --skip-create) SKIP_CREATE=1 ;;
  esac
done

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE" ; }

mkdir -p "$MERGE_ROOT" "$MERGED_LOGIC_DIR" "$MERGED_LIBRARY_DIR"
: > "$LOG_FILE"

if ! ls "$LOGIC_DIR"/tune_M*_N*_K*.yaml >/dev/null 2>&1; then
  log "FATAL: no tuned YAMLs at $LOGIC_DIR"
  exit 2
fi

export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export ROCM_PATH=/opt/rocm/core-7.12
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export PYTHONPATH=/root/hipblaslt-src/build/release/tensilelite/rocisa/lib:/root/hipblaslt-src/tensilelite

log "=== M2 lazy-merge campaign ==="
log "  tuned YAMLs       : $LOGIC_DIR/tune_M*_N*_K*.yaml"
log "  merge workspace   : $MERGE_ROOT"
log "  merged logic dir  : $MERGED_LOGIC_DIR"
log "  merged library    : $MERGED_LIBRARY_DIR"
log "  prebuilt fallback : $PREBUILT_LIB_DIR"

# -----------------------------------------------------------------------------
# Step 1 -- stage tuned YAMLs into MERGED_LOGIC_DIR.
#
# Earlier attempt: iterate TensileMergeLibrary with each tuned yaml
# renamed to arcturus_Cijk_Ailk_Bljk_I8II_BH.yaml. That fails with
# "[Error] ProblemType in library logic doesn't match" because the
# upstream arcturus I8 logic is the I8II_BH (I8 -> FP16 dest, TransposeB=False)
# variant while our tuning targeted I8 -> I8 dest TransposeB=True
# (matching the prebuilt I8I8_II8 contraction shipped only as compiled
# .dat/.co). The upstream repo does not ship a source LibraryLogic YAML
# for that contraction, so there is nothing to merge into via the
# canonical TensileMergeLibrary tool.
#
# Workaround: each tuned YAML is itself a complete LibraryLogic file
# with a single (M, N, K) -> tuned solution entry. We stage all 8
# directly as the merged-logic input to TensileCreateLibrary in step 2.
# TensileCreateLibrary handles multiple disjoint logic files in the
# same input dir.
# -----------------------------------------------------------------------------

shopt -s nullglob
yamls=( "$LOGIC_DIR"/tune_M*_N*_K*.yaml )
shopt -u nullglob
log "  ${#yamls[@]} tuned YAMLs to stage"

rm -rf "$MERGED_LOGIC_DIR"
mkdir -p "$MERGED_LOGIC_DIR"
for incremental_yaml in "${yamls[@]}"; do
  shape_name=$(basename "$incremental_yaml" .yaml)
  cp "$incremental_yaml" "$MERGED_LOGIC_DIR/${shape_name}.yaml"
  log "    staged $shape_name"
done

merged_count=$(ls "$MERGED_LOGIC_DIR"/*.yaml 2>/dev/null | wc -l)
log "merged logic dir contains $merged_count tuned YAML(s)"

if [[ "$SKIP_CREATE" == "1" ]]; then
  log "SKIP_CREATE=1 -- stopping before TensileCreateLibrary build."
  exit 0
fi

# -----------------------------------------------------------------------------
# Step 2 -- TensileCreateLibrary against the merged logic dir.
#
# Output is a self-contained library tree:
#   $MERGED_LIBRARY_DIR/
#     TensileLibrary_lazy_gfx908.dat
#     TensileLibrary_*_gfx908.dat / .co
#     Kernels.so-000-gfx908.hsaco
#
# This is the heavyweight step (compiles all kernels in the merged
# logic file). With only one logic YAML in the input dir, only the
# kernels referenced by the merged solution map are built — typically
# < 30 distinct assembly kernels for the I8 contraction we care
# about.
# -----------------------------------------------------------------------------

create_log=/root/bench-int8-w4a16/tensilelite/tensile_create_library.log
log "=== Step 2: TensileCreateLibrary against merged logic dir ==="
log "  output          : $MERGED_LIBRARY_DIR"
log "  build log       : $create_log"
log "  foreground mode : $FOREGROUND"

if [[ "$FOREGROUND" == "1" ]]; then
  ( cd "$TENSILE_DIR" && \
    "$PY" "$TENSILE_CREATE" \
       --architecture=gfx908 \
       --code-object-version=V5 \
       --library-format=msgpack \
       --logic-format=yaml \
       --jobs="$(nproc)" \
       "$MERGED_LOGIC_DIR" "$MERGED_LIBRARY_DIR" HIP \
       2>&1 | tee "$create_log" ) || {
       rc=$?
       log "TensileCreateLibrary FAIL: exit=$rc -- see $create_log"
       exit $rc
  }
  log "TensileCreateLibrary OK -- artifacts at $MERGED_LIBRARY_DIR"
else
  log "  launching TensileCreateLibrary fire-and-forget..."
  ( cd "$TENSILE_DIR" && \
    nohup "$PY" "$TENSILE_CREATE" \
       --architecture=gfx908 \
       --code-object-version=V5 \
       --library-format=msgpack \
       --logic-format=yaml \
       --jobs="$(nproc)" \
       "$MERGED_LOGIC_DIR" "$MERGED_LIBRARY_DIR" HIP \
       > "$create_log" 2>&1 ) &
  CREATE_PID=$!
  echo "$CREATE_PID" > /root/bench-int8-w4a16/tensilelite/tensile_create.pid
  log "  TensileCreateLibrary PID=$CREATE_PID  -- log: $create_log"
fi

# -----------------------------------------------------------------------------
# Step 3 -- stage prebuilt fallback library files.
#
# After step 2 finishes, the prebuilt I8I8_II8 default contraction
# files from the upstream ROCm-7.2.0 build need to live alongside our
# tuned merged_library/ so hipBLASLt can serve untuned shapes from the
# defaults. This is invoked separately (after the create build is
# verified) because TensileCreateLibrary will overwrite anything in
# its OutputPath at start.
# -----------------------------------------------------------------------------
log "=== Step 3 (manual): once $create_log finishes, stage prebuilt I8 fallback ==="
log "    cp $PREBUILT_LIB_DIR/TensileLibrary_I8I8_II8_*_gfx908.{co,dat} \\"
log "       $MERGED_LIBRARY_DIR/"
log "    (only run after Step 2 has fully completed)"

exit 0
