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

```text
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

```text
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

MI100 auto-detection (`rocm.py`) handles most settings automatically:

- torch.compile disabled (Inductor fusions unavailable on ROCm)
- FULL_DECODE_ONLY CUDA graphs (PIECEWISE hangs at TP>1)
- Custom all-reduce via XGMI enabled (validated on PyTorch 2.11+rocm7.2)
- KV cache block_size=32

The launch scripts below set only what isn't auto-detected.

### TP=4: Max Performance (recommended for 9B+ models)

```bash
#!/bin/bash
# launch-tp4.sh -- Max performance on 4x MI100
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_SKINNY_GEMM=1

# TunableOp: replay pre-tuned rocBLAS algorithm selections.
# +14% throughput at batch>=8. Tune first with PYTORCH_TUNABLEOP_TUNING=1.
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_FILENAME=/root/tunableop-results/tunableop_results%d.csv

exec /opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.5-9B \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --block-size 32 \
  --enable-prefix-caching \
  --trust-remote-code \
  --language-model-only \
  --port 8000
```

**Expected decode throughput (Qwen3.5-9B FP16, TP=4):**

| Scenario | Latency | Output tok/s | TPOT |
|----------|---------|-------------|------|
| batch=1, in=128, out=128 | 0.90s | 142 | 7.1 ms |
| batch=8, in=128, out=128 | 1.28s | 800 | 10.0 ms |
| Coding: in=8k, out=2k | 15.0s | 136 | 7.5 ms |
| Coding: in=32k, out=4k | 34.1s | 120 | 7.5 ms |
| 2 users: in=8k, out=2k | 17.0s | 240 combined | 8.3 ms |

### TP=1: Single GPU (for models <= 20B params in FP16)

```bash
#!/bin/bash
# launch-tp1.sh -- Single MI100, max decode speed (no NCCL overhead)
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_SKINNY_GEMM=1
export CUDA_VISIBLE_DEVICES=0

exec /opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.5-0.8B \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --block-size 32 \
  --enable-prefix-caching \
  --trust-remote-code \
  --language-model-only \
  --port 8000
```

**Expected decode throughput (Qwen3.5-0.8B FP16, TP=1):**

| Scenario | TPOT | Output tok/s |
|----------|------|-------------|
| batch=1, in=128, out=128 | 3.1 ms | 325 |

TP=1 eliminates all NCCL overhead. Use for models that fit in a single 32 GB GPU.

### Environment Variables

| Variable | Default | Why |
|----------|---------|-----|
| `VLLM_ROCM_USE_SKINNY_GEMM=1` | Required | Enables `wvSplitK`/`LLMM1` decode GEMM kernels (gfx908 compile guard added) |
| `PYTORCH_ROCM_ARCH=gfx908` | Required | Correct HIP code generation target |
| `PYTORCH_TUNABLEOP_ENABLED=1` | Optional | Replay tuned rocBLAS algorithm selections (+14% at batch>=8) |
| `PYTORCH_TUNABLEOP_FILENAME=...%d.csv` | Optional | Per-GPU tuning result files (one per TP rank) |
| `VLLM_MI100_TORCH_COMPILE=1` | Optional | Re-enable torch.compile (disabled by default, adds overhead on ROCm) |
| `VLLM_MI100_DISABLE_CUSTOM_AR=1` | Optional | Fall back to pynccl (custom AR is enabled by default, -17% if disabled) |

### Server Flags

| Flag | Why |
|------|-----|
| `--dtype float16` | Required for INT4 models (ExllamaLinearKernel). Also fine for FP16 models. |
| `--enable-prefix-caching` | 85-99% TTFT reduction on repeated prompts |
| `--language-model-only` | Skip vision encoder loading (saves memory for text-only use) |
| ~~`--enforce-eager`~~ | Not needed: FULL_DECODE_ONLY graphs auto-detected |
| ~~`--disable-custom-all-reduce`~~ | Not needed: custom all-reduce works correctly on PyTorch 2.11+rocm7.2 |

### TunableOp First-Time Tuning

To generate tuning results for your specific model (run once, takes ~10 minutes):

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_FILENAME=/root/tunableop-results/tunableop_results%d.csv
# Launch server normally, send ~50 requests, then stop.
# Results are saved per-GPU. Set TUNING=0 for production.
```

## 12. Test Inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3.5-9B",
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

This was fixed in PyTorch 2.11+rocm7.2. If you see this on an older PyTorch
version, set `TORCH_COMPILE_DISABLE=1` or upgrade PyTorch.

### libtorch_cuda.so: cannot open shared object file

flashinfer (CUDA-only) is installed. Remove it:

```bash
/opt/vllm-env/bin/pip uninstall -y flashinfer flashinfer-python
```

### Workers crash with NCCL/RCCL timeout

Verify XGMI topology with `rocm-smi --showtopo`. All GPUs should show 1-hop
distance to each other. If not, check physical Infinity Bridge cables.

### GPU memory not freed after kill -9

`kill -9` leaves orphaned worker processes holding GPU memory. Always use
graceful shutdown: `kill -15 $(lsof -ti :8000)`. If GPUs are stuck, reboot.

## CK Flash Attention (Optional)

The upstream ROCm flash-attention package excludes gfx908 from its allowed
build targets, but the CK kernels compile and run correctly on MI100. This
provides ~5x faster attention compute (34% vs 7% MFMA efficiency) for models
that use standard quadratic attention (Llama, Mistral, etc.).

**Note:** Qwen3.5 uses linear attention (GDN) and does NOT benefit from this.

To build CK flash attention for MI100:

```bash
./scripts/build_ck_flash_attn_gfx908.sh
```

This clones ROCm/flash-attention, patches `setup.py` to allow gfx908, and
builds the CK backend. Takes 30-60 minutes. Options:

```bash
# Custom clone directory and parallelism
CLONE_DIR=/path/to/flash-attention MAX_JOBS=8 ./scripts/build_ck_flash_attn_gfx908.sh

# Build only hdim=128 kernels (faster build, covers most models)
./scripts/build_ck_flash_attn_gfx908.sh --opt-dim 128
```

Verify installation:

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
# Should print version without "falling back to Triton" warning
```

## Optimizations Tested and NOT Recommended

| Optimization | Result | Reason |
|---|---|---|
| AWQ INT4 quantization | -8% to -16% slower | MI100 lacks native INT4 MFMA; Exllama dequant overhead exceeds bandwidth savings |
| TP=2 (instead of TP=4) | -29% at batch=1 | Larger GEMMs per GPU outweigh NCCL savings |
| torch.compile | -1% to -15% | Inductor fusions disabled on ROCm; compile overhead without benefit |
| NCCL_ALGO=Ring | -5% | Default algorithm already optimal for XGMI topology |
| NCCL_PROTO=Simple | -13% | LL (low latency) protocol already best for small messages |
| MTP speculative decoding | -25% to -45% | Incompatible with HIP graph mode |
| TurboQuant KV compression | -6% to -49% | Not worth it for Qwen3.5 (only 8/32 full attention layers) |

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
