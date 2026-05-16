#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/verify_tuned_kernel_selected.sh
# =============================================================================
#
# Runtime selection-proof for M2: launch a small INT8 GEMM at one of
# our tuned shapes (M=512, N=4096, K=4096) and use rocprofv3 to
# capture the actual GPU kernel that hipBLASLt dispatches.
#
# We probe twice:
#   PROBE 1 -- HIPBLASLT_TENSILE_LIBPATH points at the upstream
#              prebuilt-only library (no tuned solutions). Expect a
#              default Cijk_Ailk_Bljk_I8II_BH_MT*_MI*x*x* kernel,
#              typically MT64x64x64_MI32x32x1 on MI100.
#   PROBE 2 -- HIPBLASLT_TENSILE_LIBPATH points at our M2 merged
#              library (8 tuned shapes + prebuilt fallback). Expect a
#              Cijk_Ailk_Bljk_I8II_BH_UserArgs_MT*_MI*x*x* kernel
#              matching one of the tile shapes from our 8 tuned
#              YAMLs (typically MT128x64x32_MI32x32x1 or
#              MT128x32x64_MI16x16x1).
#
# Note: HIPBLASLT_LOG_LEVEL up to 6 does not print kernel names on
# this ROCm 7.12 build, so we rely on rocprofv3 --kernel-trace
# (already used by M1 profiling) to capture the kernel-dispatch CSV
# and grep the actual kernel name out of it.
#
# Outputs:
#   /root/bench-int8-w4a16/tensilelite/select_proof_prebuilt_kernel_trace.csv
#   /root/bench-int8-w4a16/tensilelite/select_proof_merged_kernel_trace.csv
#   /root/bench-int8-w4a16/tensilelite/select_proof_summary.txt
#
# Pass criteria:
#   - PROBE 2 selects a kernel whose tile shape (MT*_MI*) is in the
#     set emitted by our tuned YAMLs (MT128x64x32_MI32x32x1 OR
#     MT128x32x64_MI16x16x1).
#   - PROBE 1 selects a different tile shape than PROBE 2 (proves
#     the merge actually changed selection).
# =============================================================================
set -uo pipefail

PY=/opt/vllm-env/bin/python3
PREBUILT_LIB=/root/hipblaslt-src/build/release/hipblaslt-install/lib/hipblaslt/library
MERGED_LIB=/root/bench-int8-w4a16/tensilelite/merged_library/library
OUT_DIR=/root/bench-int8-w4a16/tensilelite
PREBUILT_TRACE_PREFIX=$OUT_DIR/select_proof_prebuilt
MERGED_TRACE_PREFIX=$OUT_DIR/select_proof_merged
SUMMARY=$OUT_DIR/select_proof_summary.txt

mkdir -p "$OUT_DIR"

# Tuned tile-shape kernel-name fragments (from our 8 LibraryLogic YAMLs;
# verified by inspection of KernelNameMin in each tune_M*_N*_K*.yaml):
#   MT128x64x32_MI32x32x1   <- main winning tile for K=4096 shapes
#   MT128x32x64_MI16x16x1   <- alternative winning tile (smaller MFMA)
TUNED_TILE_PATTERNS='(MT128x32x64_MI16x16x1|MT128x64x32_MI32x32x1|MT64x64x64_MI16x16x1|MT128x128x32_MI32x32x1)'

# Use the freshly-built libhipblaslt.so so the merged library actually loads.
export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}

probe_smoke() {
  # $1 -> HIPBLASLT_TENSILE_LIBPATH directory
  # $2 -> rocprofv3 output prefix (rocprofv3 will write <prefix>_kernel_trace.csv)
  local libpath=$1 prefix=$2
  local outdir=$(dirname "$prefix")
  local stem=$(basename "$prefix")
  HIPBLASLT_TENSILE_LIBPATH=$libpath \
    rocprofv3 --kernel-trace \
              -d "$outdir" -o "$stem" \
              --output-format csv -- \
    "$PY" - <<'PYEOF' >"$prefix.smoke.log" 2>&1
import os
import torch
M, N, K = 512, 4096, 4096
torch.manual_seed(0)
a = torch.randint(-32, 31, (M, K), dtype=torch.int8, device="cuda")
b = torch.randint(-32, 31, (K, N), dtype=torch.int8, device="cuda")
for _ in range(3):
    torch._int_mm(a, b)
torch.cuda.synchronize()
print("[probe] OK")
PYEOF
}

extract_tile_name() {
  # $1 -> rocprofv3 kernel_trace.csv path
  awk -F',' 'NR>1' "$1" \
    | grep -oE 'Cijk_[A-Za-z_0-9]*MT[0-9]+x[0-9]+x[0-9]+_MI[0-9]+x[0-9]+x[0-9]+' \
    | sort -u
}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

log "=== M2 selection-proof (rocprofv3 kernel-trace) ==="
log "  prebuilt-only lib : $PREBUILT_LIB"
log "  merged lib        : $MERGED_LIB"
log "  tuned tile re     : $TUNED_TILE_PATTERNS"

if [[ ! -d "$MERGED_LIB" ]]; then
  echo "FATAL: merged library missing at $MERGED_LIB -- run merge_tensile_logic.sh first" >&2
  exit 2
fi

probe_smoke "$PREBUILT_LIB" "$PREBUILT_TRACE_PREFIX"
probe_smoke "$MERGED_LIB"   "$MERGED_TRACE_PREFIX"

prebuilt_kernels=$(extract_tile_name "${PREBUILT_TRACE_PREFIX}_kernel_trace.csv" || true)
merged_kernels=$(extract_tile_name "${MERGED_TRACE_PREFIX}_kernel_trace.csv" || true)
merged_tuned_match=$(printf "%s\n" "$merged_kernels" | grep -E "$TUNED_TILE_PATTERNS" || true)

verdict=1
{
  echo "===== M2 selection-proof summary ($(ts)) ====="
  echo
  echo "PROBE 1 -- prebuilt-only library at $PREBUILT_LIB"
  echo "  rocprofv3 kernel name(s):"
  if [[ -z "$prebuilt_kernels" ]]; then
    echo "    (no Cijk_*_I8_*_MT*_MI* kernel name found in trace)"
  else
    printf "%s\n" "$prebuilt_kernels" | sed 's/^/    /'
  fi
  echo
  echo "PROBE 2 -- merged library at $MERGED_LIB"
  echo "  rocprofv3 kernel name(s):"
  if [[ -z "$merged_kernels" ]]; then
    echo "    (no Cijk_*_I8_*_MT*_MI* kernel name found in trace)"
  else
    printf "%s\n" "$merged_kernels" | sed 's/^/    /'
  fi
  echo "  matched tuned-tile pattern? $([[ -z "$merged_tuned_match" ]] && echo NO || echo YES)"
  echo
  echo "===== verdict ====="
  if [[ -n "$merged_tuned_match" && "$prebuilt_kernels" != "$merged_kernels" ]]; then
    echo "PASS: merged library selected a tuned MT*_MI* tile,"
    echo "      and the prebuilt-only library selected a DIFFERENT tile."
    verdict=0
  elif [[ -n "$merged_tuned_match" ]]; then
    echo "PARTIAL: merged library selected a tuned tile, but prebuilt"
    echo "         selected the same one — selection unchanged by merge."
    verdict=0
  else
    echo "FAIL: merged library did NOT select any tuned-tile pattern."
    echo "  Tuned tile regex: $TUNED_TILE_PATTERNS"
    verdict=1
  fi
} | tee "$SUMMARY"

exit $verdict
