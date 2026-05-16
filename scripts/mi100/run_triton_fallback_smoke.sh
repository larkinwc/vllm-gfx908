#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/run_triton_fallback_smoke.sh -- VAL-TENSILE-008 evidence
# =============================================================================
#
# Run one cell of the W8A8 grid (TP=1, c=1, synthetic) with
# VLLM_DISABLE_HIPBLASLT=1 set. The dispatcher must force every W8A8
# call onto the Triton mi100_int8 path; output throughput is expected
# to be within ±0.5% of the M1 baseline at the same cell.
#
# Outputs the result JSON at:
#   /root/bench-int8-w4a16/tensilelite/triton_fallback_smoke.json
#
# Compared in BENCH_INT8_W4A16_M2.md "Triton fallback" section.
# =============================================================================
set -uo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
M1_HARNESS=/root/bench-int8-w4a16/baseline/run_baseline.sh
OUT_ROOT=/root/bench-int8-w4a16/tensilelite/triton_fallback
mkdir -p "$OUT_ROOT/synthetic" "$OUT_ROOT/coding"

# Patched harness redirected to OUT_ROOT.
PATCHED=$OUT_ROOT/run_triton_fallback.sh
sed "s|^BASELINE_ROOT=.*$|BASELINE_ROOT=$OUT_ROOT|" "$M1_HARNESS" > "$PATCHED"
chmod +x "$PATCHED"

export VLLM_DISABLE_HIPBLASLT=1
# Use the prebuilt-only library so we know hipBLASLt has no tuned solutions
# loaded — this isolates the dispatcher fallback decision.
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
unset HIPBLASLT_TENSILE_LIBPATH || true
export CELLS_FILTER='w8a8_tp1_c1_synthetic'

echo "[fallback] VLLM_DISABLE_HIPBLASLT=$VLLM_DISABLE_HIPBLASLT"
echo "[fallback] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[fallback] CELLS_FILTER=$CELLS_FILTER"
echo "[fallback] kicking off $PATCHED"

bash "$PATCHED"
rc=$?

result_json=$OUT_ROOT/synthetic/w8a8_tp1_c1.json
copy_to=/root/bench-int8-w4a16/tensilelite/triton_fallback_smoke.json
if [[ -f "$result_json" ]]; then
  cp "$result_json" "$copy_to"
  echo "[fallback] wrote $copy_to"
else
  echo "[fallback] WARNING: result missing at $result_json"
fi
exit $rc
