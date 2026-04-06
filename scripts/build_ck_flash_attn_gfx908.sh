#!/usr/bin/env bash
# Build CK flash attention for gfx908 (MI100)
#
# The upstream ROCm/flash-attention setup.py does not include gfx908
# in its allowed_archs list. This script clones the repo, applies
# a one-line patch, and builds the CK backend for MI100.
#
# Usage:
#   ./scripts/build_ck_flash_attn_gfx908.sh [--opt-dim 128] [--clone-dir /tmp/flash-attention]
#
# Requirements:
#   - ROCm 6.2+ (tested with 7.2)
#   - PyTorch with ROCm support
#   - hipcc in PATH (or ROCM_PATH set)

set -euo pipefail

OPT_DIM="${OPT_DIM:-128}"
CLONE_DIR="${CLONE_DIR:-/tmp/rocm-flash-attention}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
FLASH_ATTN_BRANCH="${FLASH_ATTN_BRANCH:-tridao}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --opt-dim) OPT_DIM="$2"; shift 2 ;;
        --clone-dir) CLONE_DIR="$2"; shift 2 ;;
        --max-jobs) MAX_JOBS="$2"; shift 2 ;;
        --branch) FLASH_ATTN_BRANCH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Building CK flash attention for gfx908 (MI100) ==="
echo "  Clone dir:  $CLONE_DIR"
echo "  OPT_DIM:    $OPT_DIM"
echo "  MAX_JOBS:   $MAX_JOBS"
echo "  Branch:     $FLASH_ATTN_BRANCH"
echo ""

# Detect ROCm
if [ -n "${ROCM_PATH:-}" ]; then
    ROCM="$ROCM_PATH"
elif [ -d "/opt/rocm" ]; then
    ROCM="/opt/rocm"
else
    echo "ERROR: Cannot find ROCm. Set ROCM_PATH." >&2
    exit 1
fi
echo "  ROCm path:  $ROCM"

# Verify gfx908 GPU is present
if command -v rocm-smi &>/dev/null; then
    if ! rocm-smi --showproductname 2>/dev/null | grep -qi "MI100\|gfx908"; then
        echo "WARNING: No MI100 (gfx908) GPU detected. Building anyway with GPU_ARCHS=gfx908."
    fi
fi

# Clone if needed
if [ ! -d "$CLONE_DIR" ]; then
    echo "Cloning ROCm/flash-attention..."
    git clone --depth 1 --branch "$FLASH_ATTN_BRANCH" \
        https://github.com/ROCm/flash-attention.git "$CLONE_DIR"
fi

cd "$CLONE_DIR"

# Initialize composable_kernel submodule (required for CK backend)
echo "Initializing composable_kernel submodule..."
git submodule update --init --depth 1 csrc/composable_kernel

# Apply the gfx908 patch
echo "Patching setup.py to allow gfx908..."
if grep -q '"gfx908"' setup.py; then
    echo "  Already patched."
else
    sed -i 's/allowed_archs = \["native", "gfx90a"/allowed_archs = ["native", "gfx908", "gfx90a"/' setup.py
    if grep -q '"gfx908"' setup.py; then
        echo "  Patch applied successfully."
    else
        echo "ERROR: Failed to patch setup.py. The format may have changed." >&2
        echo "  Manually add \"gfx908\" to the allowed_archs list in setup.py" >&2
        exit 1
    fi
fi

# Build
echo ""
echo "Building flash_attn with CK backend for gfx908..."
echo "  This compiles ~500 GPU kernels and may take 30-60 minutes."
echo ""

export GPU_ARCHS=gfx908
export OPT_DIM="$OPT_DIM"
export MAX_JOBS="$MAX_JOBS"
export ROCM_PATH="$ROCM"
export HIP_PATH="$ROCM"
export PATH="$ROCM/bin:$PATH"
export LD_LIBRARY_PATH="${ROCM}/lib:${LD_LIBRARY_PATH:-}"

pip install --no-build-isolation -e .

echo ""
echo "=== Build complete ==="
echo ""

# Verify
python3 -c "
import flash_attn
import flash_attn.flash_attn_interface as fai
backend = 'CK' if hasattr(fai, 'flash_attn_2_cuda') else 'Triton (fallback)'
print(f'flash_attn {flash_attn.__version__} installed with {backend} backend')
"

echo ""
echo "CK flash attention is now available for MI100."
echo "Models using standard attention (Llama, Mistral, etc.) will"
echo "benefit from ~5x faster attention compute during prefill."
