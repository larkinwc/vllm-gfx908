# vLLM on AMD Instinct MI100 (gfx908) -- Setup Guide

Native build of vLLM for MI100 GPUs using ROCm 7.12 and PyTorch 2.11.

## Reference Hardware

| Component | Details |
|-----------|---------|
| Motherboard | HUANANZHI H12D-8D (EPYC) |
| CPU | AMD EPYC 7742 (64C/64T) |
| RAM | 64 GB DDR4 |
| GPUs | 4x AMD Instinct MI100 (32 GB HBM2, gfx908) |
| Interconnect | XGMI Infinity Bridge (full mesh, 1-hop) |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.17.0-19-generic |

## 1. BIOS Configuration

These settings are required for multi-GPU PCIe passthrough and large BAR support.

| Setting | Value |
|---------|-------|
| Above 4G Decoding | **Enabled** |
| Re-Size BAR Support | **Enabled** |
| CSM Support | **Disabled** |
| IOMMU | **Enabled** |

## 2. Kernel Command Line

Edit `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet iommu=pt"
```

Then apply:

```bash
sudo update-grub
sudo reboot
```

**Do not** use `intel_iommu=on` (AMD platform), `amdgpu.noretry=0`, or `ppfeaturemask` flags.
The `iommu=pt` passthrough mode is sufficient.

## 3. Install amdgpu-dkms Driver

The kernel driver must be version 6.19.0+ for MI100 stability on newer kernels.

```bash
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
  sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
  https://repo.radeon.com/amdgpu/31.20/ubuntu noble main" | \
  sudo tee /etc/apt/sources.list.d/amdgpu.list

sudo apt update
sudo apt install amdgpu-dkms amdgpu-dkms-firmware
sudo reboot
```

Verify GPUs are visible:

```bash
rocm-smi --showproductname
# Should list all MI100 GPUs with GFX Version: gfx908
```

## 4. Install ROCm 7.12

```bash
wget -qO - https://repo.amd.com/rocm/rocm.gpg.key | \
  sudo gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] \
  https://repo.amd.com/rocm/packages/ubuntu2404 stable main" | \
  sudo tee /etc/apt/sources.list.d/amdrocm.list

sudo apt update
sudo apt install amdrocm7.12-gfx908
```

This installs the full ROCm 7.12 stack for gfx908 to `/opt/rocm/core-7.12/`.

### 4a. Install ROCm dev packages (for building from source)

```bash
sudo apt install \
  amdrocm-core-dev7.12-gfx908 \
  amdrocm-blas-dev7.12-gfx908 \
  amdrocm-fft-dev7.12-gfx908 \
  amdrocm-rand-dev7.12-gfx908 \
  amdrocm-sparse-dev7.12-gfx908 \
  amdrocm-solver-dev7.12-gfx908 \
  amdrocm-dnn-dev7.12-gfx908 \
  amdrocm-rccl-dev7.12-gfx908 \
  amdrocm-llvm-dev7.12
```

### 4b. Create ROCm symlinks

The non-standard ROCm 7.12 install layout needs symlinks for cmake to find things:

```bash
sudo ln -sf /opt/rocm/core-7.12/bin /opt/rocm/bin
sudo ln -sf /opt/rocm/core-7.12/include /opt/rocm/include
sudo ln -sf /opt/rocm/core-7.12/lib /opt/rocm/lib
```

### 4c. Fix AMD cmake packaging bug

ROCm 7.12 dev packages ship cmake configs that reference ROCm 6.3 library versions
(e.g. `librocblas.so.4.3.60304` instead of the actual installed version). This must be
fixed before building anything against these libraries.

```bash
cd /opt/rocm/core-7.12/lib/cmake
for cmake_file in $(grep -rl '\.60304\|\.60300' . 2>/dev/null); do
  sudo sed -i \
    -e 's/\.so\.4\.3\.60304/.so.4.5.71200/g' \
    -e 's/\.so\.4\.1\.60304/.so.4.5.71200/g' \
    -e 's/\.so\.1\.0\.60304/.so.1.0.71200/g' \
    -e 's/\.so\.0\.4\.60304/.so.0.4.71200/g' \
    -e 's/\.so\.1\.1\.60304/.so.1.1.71200/g' \
    -e 's/\.so\.0\.1\.60304/.so.0.1.71200/g' \
    -e 's/\.so\.2\.1\.60304/.so.2.1.71200/g' \
    -e 's/\.so\.60304/.so.71200/g' \
    -e 's/\.so\.60300/.so.71200/g' \
    "$cmake_file"
done
```

> **Note**: The exact version numbers may differ in future ROCm releases.
> Run `ls /opt/rocm/core-7.12/lib/librocblas.so.*` to find the actual installed
> version, then adjust the sed replacements accordingly.

Also remove the hipsolver Fortran library reference (not shipped):

```bash
sudo sed -i '/hipsolver_fortran/d' \
  /opt/rocm/core-7.12/lib/cmake/hipsolver/hipsolver-targets-release.cmake
```

## 5. Set Up GPU Power and Performance

MI100 defaults to 178W power cap. For full performance, set 250W and high perf mode.

### One-time (live)

```bash
for card in /sys/class/drm/card*/device; do
  [ -f "$card/power_dpm_force_performance_level" ] && \
    echo high > "$card/power_dpm_force_performance_level"
done

for hwmon in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
  [ -f "$hwmon" ] && echo 250000000 > "$hwmon"
done
```

### Persistent (systemd service)

```bash
cat > /etc/systemd/system/gpu-perf-high.service << 'EOF'
[Unit]
Description=Set AMD GPU performance to high and power cap to 250W
After=systemd-modules-load.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
  for card in /sys/class/drm/card*/device; do \
    [ -f "$card/power_dpm_force_performance_level" ] && \
      echo high > "$card/power_dpm_force_performance_level"; \
  done; \
  for hwmon in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do \
    [ -f "$hwmon" ] && echo 250000000 > "$hwmon"; \
  done'

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now gpu-perf-high.service
```

## 6. Create Python Virtual Environment

```bash
python3 -m venv /opt/vllm-env
```

## 7. Install PyTorch for ROCm

PyTorch must be the ROCm 7.2 build (closest available to host ROCm 7.12):

```bash
/opt/vllm-env/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

Verify:

```bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
/opt/vllm-env/bin/python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('HIP:', torch.version.hip)
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### Fix Triton

PyTorch may pull in upstream `triton` (CUDA-only) as a dependency. Remove it and
ensure only the ROCm version is installed:

```bash
/opt/vllm-env/bin/pip uninstall -y triton
/opt/vllm-env/bin/pip install pytorch-triton-rocm \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  --force-reinstall --no-deps
```

Also remove any CUDA packages that may have been pulled in:

```bash
/opt/vllm-env/bin/pip uninstall -y \
  flashinfer flashinfer-python \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
```

## 8. Install amdsmi

```bash
/opt/vllm-env/bin/pip install \
  /opt/rocm/core-7.12/lib/amdsmi/amdsmi_cli/amdsmi*.whl
```

## 9. Build vLLM from Source

```bash
git clone https://github.com/larkinwc/vllm-gfx908.git
cd vllm-gfx908
git checkout mi100-fixes

export PYTORCH_ROCM_ARCH=gfx908
export VLLM_TARGET_DEVICE=rocm
export ROCM_PATH=/opt/rocm/core-7.12
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export PATH=/opt/rocm/core-7.12/bin:$PATH
export MAX_JOBS=$(nproc)

/opt/vllm-env/bin/pip install --no-build-isolation --no-deps -e .
```

Build takes 20-40 minutes depending on CPU core count. It compiles HIP kernels
(paged_attention, `_C`, `_rocm_C`, `_moe_C`) targeting gfx908.

Verify the build:

```bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
/opt/vllm-env/bin/python3 -c "
import vllm; print('vLLM:', vllm.__version__)
import vllm._rocm_C; print('_rocm_C: OK')
from vllm.platforms import current_platform
print('Platform:', type(current_platform).__name__)
"
```

Expected output:

```
vLLM: 0.18.1.dev4+...
_rocm_C: OK
Platform: RocmPlatform
```

## 10. Download a Model

Example with Qwen3.5-27B quantized (INT4, fits in 4x 32 GB):

```bash
mkdir -p /models
huggingface-cli download Qwen/Qwen3.5-27B-AWQ-BF16-INT4 \
  --local-dir /models/Qwen3.5-27B-AWQ-BF16-INT4
```

## 11. Launch vLLM

Create a launch script (e.g. `/root/launch-vllm.sh`):

```bash
#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export VLLM_ROCM_USE_AITER=1
export PATH=/opt/rocm/core-7.12/bin:$PATH
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export TORCH_COMPILE_DISABLE=1
export VLLM_ROCM_USE_SKINNY_GEMM=1

exec /opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.5-27B-AWQ-BF16-INT4 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --dtype float16 \
  --port 8000 \
  --trust-remote-code \
  --enforce-eager \
  --language-model-only
```

```bash
chmod +x /root/launch-vllm.sh
/root/launch-vllm.sh
```

### Environment Variables

| Variable | Value | Why |
|----------|-------|-----|
| `TORCH_COMPILE_DISABLE=1` | Disable torch.compile/inductor | Avoids `KernelMetadata.cluster_dims` error on gfx908 |
| `VLLM_ROCM_USE_SKINNY_GEMM=1` | Enable skinny GEMM kernels | `wvSplitK`/`LLMM1` now compiled for gfx908 (compile guard added) |
| `VLLM_ROCM_USE_AITER=1` | Enable AITER Triton kernels | Triton-based kernels that work on gfx908 |
| `PYTORCH_ROCM_ARCH=gfx908` | Target GPU architecture | Ensures correct code generation |

### Server Flags

| Flag | Why |
|------|-----|
| `--dtype float16` | ExllamaLinearKernel (INT4 dequant) requires float16 activations |
| `--enforce-eager` | Disable torch.compile graph capture (not stable on MI100) |
| ~~`--disable-custom-all-reduce`~~ | No longer needed: quickreduce custom all-reduce now supports gfx908 |
| `--language-model-only` | Skip loading vision encoder (saves memory for text-only use) |

## 12. Test Inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3.5-27B-AWQ-BF16-INT4",
    "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

## 13. Monitor

A status monitor script is included at `tools/vllm-monitor.py`:

```bash
python3 tools/vllm-monitor.py        # default 2s refresh
python3 tools/vllm-monitor.py 0.5    # 500ms refresh
```

Shows GPU temps, power, utilization, VRAM usage, and live vLLM tok/s, TTFT, KV cache.

## Troubleshooting

### GPUs not showing in rocm-smi

- Check `lspci | grep -i amd` -- GPUs should appear as `Arcturus GL-XL [Instinct MI100]`
- Check `dmesg | grep -i atombios` -- "atombios stuck" means BIOS settings are wrong
  (enable Above 4G Decoding, Re-Size BAR, disable CSM)
- Ensure `amdgpu-dkms` version >= 6.19.0

### vLLM segfaults in libamdhip64.so

PyTorch ROCm version must match host ROCm major version. PyTorch rocm6.3 + host
ROCm 7.12 = ABI mismatch. Use `torch+rocm7.2` with ROCm 7.x host.

### skinny_gemms.hip:530 Device-side assertion

The skinny GEMM kernels (`wvSplitK`) originally excluded gfx908 from the compile
guard. The fork adds `__gfx908__` support; set `VLLM_ROCM_USE_SKINNY_GEMM=1`.

### Exllama only supports float16 activations

AWQ/GPTQ INT4 models use ExllamaLinearKernel which requires `--dtype float16`,
not `auto` (which defaults to bfloat16).

### InductorError: KernelMetadata has no attribute cluster_dims

Set `TORCH_COMPILE_DISABLE=1`. The inductor backend does not support gfx908.

### libtorch_cuda.so: cannot open shared object file

flashinfer (CUDA-only) is installed. Remove it:

```bash
/opt/vllm-env/bin/pip uninstall -y flashinfer flashinfer-python
```

### Workers crash with NCCL/RCCL timeout

Verify XGMI topology with `rocm-smi --showtopo`. All GPUs should show 1-hop
distance to each other. If not, check physical Infinity Bridge cables.

## Known Working Package Versions

| Package | Version |
|---------|---------|
| Ubuntu | 24.04.4 LTS |
| Kernel | 6.17.0-19-generic |
| amdgpu-dkms | 6.19.0 |
| ROCm | 7.12.0 |
| PyTorch | 2.11.0+rocm7.2 |
| pytorch-triton-rocm | 3.5.1 |
| vLLM | 0.18.1.dev4 (mi100-fixes branch) |
| amdsmi | 26.3.0 |
