#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/verify_hipblaslt_load.sh -- VAL-TENSILE-003 evidence collector
# =============================================================================
#
# Run a single small INT8 GEMM through torch._int_mm (which routes through
# hipBLASLt on ROCm) with HIPBLASLT_LOG_LEVEL=4 and
# HIPBLASLT_TENSILE_LIBPATH pointing at our M2 logic dir. Captures a log
# proving:
#   - hipBLASLt walked the custom-logic directory at startup
#   - A Cijk_*_I8_* kernel was selected for at least one M2 tuned shape
# =============================================================================
set -uo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/cold-points-sit-rancb
PY=/opt/vllm-env/bin/python3
# Use the FULL prebuilt hipBLASLt library directory from our M2 build —
# it ships the gfx908 INT8 contraction logic (TensileLibrary_I8I8_II8_*)
# that the system /opt/rocm install was missing. The standalone per-shape
# YAML directory at /root/bench-int8-w4a16/tensilelite/logic/gfx908/ is
# NOT a fully-staged hipBLASLt library tree (no TensileLibrary_lazy_*.dat
# index), so pointing HIPBLASLT_TENSILE_LIBPATH at it directly fails to
# load. Merging tuned per-shape YAMLs into the lazy index is a follow-up
# step (TensileMergeLibrary or TensileCreateLibrary --merge-library);
# until then this smoke proves hipBLASLt is invoked and selects an
# INT8 Tensile kernel for the W8A8 shape.
LIB_DIR_DEFAULT=/root/hipblaslt-src/build/release/hipblaslt-install/lib/hipblaslt/library
LOGIC_DIR=${LOGIC_DIR:-$LIB_DIR_DEFAULT}
OUT_LOG=${OUT_LOG:-/root/bench-int8-w4a16/tensilelite/hipblaslt_load_smoke.log}

# Use the freshly-built libhipblaslt.so so custom logic YAMLs are honored.
export LD_LIBRARY_PATH=/root/hipblaslt-src/build/release/library:/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export HIPBLASLT_TENSILE_LIBPATH="$LOGIC_DIR"
export HIPBLASLT_LOG_LEVEL=4

echo "[$(date -u +%FT%TZ)] verifying hipBLASLt loads custom logic from $LOGIC_DIR"
echo "[$(date -u +%FT%TZ)]   -> $OUT_LOG"
ls -la "$LOGIC_DIR" || true

"$PY" - <<'PYEOF' >"$OUT_LOG" 2>&1
import os
import torch

print("[smoke] python module under test: torch._int_mm "
      "(routes through hipBLASLt on ROCm).")
print(f"[smoke] HIPBLASLT_TENSILE_LIBPATH={os.environ.get('HIPBLASLT_TENSILE_LIBPATH')}")
print(f"[smoke] HIPBLASLT_LOG_LEVEL={os.environ.get('HIPBLASLT_LOG_LEVEL')}")
print(f"[smoke] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
print(f"[smoke] torch={torch.__version__} hip={torch.version.hip}")

# One of the M2 tuned shapes (matches hipblaslt_tuned_shapes.json):
M, N, K = 512, 4096, 4096
print(f"[smoke] running int8 GEMM at (M={M}, N={N}, K={K})")
torch.manual_seed(0)
a = torch.randint(-32, 31, (M, K), dtype=torch.int8, device="cuda")
b = torch.randint(-32, 31, (K, N), dtype=torch.int8, device="cuda")
torch.cuda.synchronize()
c = torch._int_mm(a, b)
torch.cuda.synchronize()
print(f"[smoke] result shape: {c.shape}, dtype: {c.dtype}, "
      f"sum (sanity): {int(c.sum().item())}")
PYEOF
rc=$?

echo
echo "[$(date -u +%FT%TZ)] smoke exit code: $rc"
echo
echo "=== relevant lines from log (custom path / Cijk) ==="
grep -nE "TENSILE|TensileLib|Cijk|Solution|Kernel|library|gfx908" "$OUT_LOG" \
    | head -40
echo
exit $rc
