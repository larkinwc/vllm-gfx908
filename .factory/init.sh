#!/bin/bash
set -e

# Verify ROCm PyTorch is intact (TurboQuant pip install can replace it)
HIP_VERSION=$(/opt/vllm-env/bin/python3 -c "import torch; print(torch.version.hip or 'NONE')" 2>/dev/null)
if [ "$HIP_VERSION" = "NONE" ]; then
    echo "ERROR: ROCm PyTorch not found! Reinstalling..."
    /opt/vllm-env/bin/pip install --pre torch==2.11.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2
fi

# Ensure TurboQuant is installed (editable)
/opt/vllm-env/bin/pip install -e /opt/turboquant 2>/dev/null || true

# Re-verify ROCm PyTorch after TQ install
HIP_VERSION=$(/opt/vllm-env/bin/python3 -c "import torch; print(torch.version.hip or 'NONE')" 2>/dev/null)
if [ "$HIP_VERSION" = "NONE" ]; then
    echo "ROCm PyTorch replaced by TQ install, fixing..."
    /opt/vllm-env/bin/pip uninstall torch triton -y
    /opt/vllm-env/bin/pip install --pre torch==2.11.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2
fi

# Verify key imports
/opt/vllm-env/bin/python3 -c "
import torch
import vllm
import turboquant
print(f'PyTorch: {torch.__version__} (HIP: {torch.version.hip})')
print(f'vLLM: {vllm.__version__}')
print(f'TurboQuant: {turboquant.__version__}')
print(f'GPUs: {torch.cuda.device_count()}')
"

# Ensure benchmark directories exist
mkdir -p /root/benchmark-results
mkdir -p /root/benchmark-scripts

# Stop any running vLLM instance
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
sleep 1

echo "Environment ready."
