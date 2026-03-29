#!/bin/bash
# Environment setup for vLLM MI100 optimization mission
# Idempotent - safe to run multiple times

set -e

export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export VLLM_ROCM_USE_AITER=1
export PATH=/opt/rocm/core-7.12/bin:$PATH
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_SKINNY_GEMM=0

# Verify GPUs are accessible
if command -v rocm-smi &> /dev/null; then
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo "0")
    if [ "$GPU_COUNT" -lt 4 ]; then
        echo "WARNING: Expected 4 GPUs, found $GPU_COUNT"
    else
        echo "OK: $GPU_COUNT MI100 GPUs detected"
    fi
fi

# Verify vLLM env
if [ -f /opt/vllm-env/bin/python3 ]; then
    echo "OK: vLLM virtualenv exists"
    /opt/vllm-env/bin/python3 -c "import vllm; print(f'OK: vLLM {vllm.__version__}')" 2>/dev/null || echo "WARNING: vLLM import failed"
else
    echo "ERROR: /opt/vllm-env/bin/python3 not found"
    exit 1
fi

# Verify models directory
if [ -d /models ]; then
    echo "OK: /models directory exists"
    ls /models/
else
    echo "WARNING: /models directory not found"
fi

# Create benchmark results directory
mkdir -p /root/benchmark-results

echo "Init complete."
