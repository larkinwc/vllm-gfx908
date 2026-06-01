# vLLM on AMD gfx900 (Vega10: V340 / MI25) -- Work In Progress

This branch (`gfx900-support`) extends the MI100/gfx908 fork to AMD **gfx900**
(Vega10) GPUs such as the Radeon Pro V340 and Instinct MI25.

## Hardware context

Validated on: 8x AMD Radeon Pro V340 (= 16x gfx900 GPUs, 8GB VRAM each),
dual Xeon E5-2640 v3, Ubuntu 24.04, ROCm 7.2.4.

## Why gfx900 is a distinct tier

gfx900 is **gfx908 minus key hardware**:

| Feature        | gfx908 (MI100) | gfx900 (Vega10) |
|----------------|----------------|-----------------|
| MFMA matrix cores | yes         | **no**          |
| Native FP8     | no             | no              |
| XGMI / Infinity Fabric | yes (some) | **no** (PCIe only) |

The fork's custom paged-attention (`csrc/rocm/attention.cu`, 85 MFMA ops),
skinny GEMMs, CK flash-attn, and quickreduce all assume MFMA and/or XGMI that
gfx900 lacks. So gfx900 is modeled as a **separate tier** that routes around
all MFMA/XGMI kernels to portable Triton + rocBLAS + PCIe paths.

## Changes in this branch

- `vllm/platforms/rocm.py`: add `_ON_GFX900` / `on_gfx900()` (NOT in `_ON_GFX9`,
  so all `if _ON_GFX9:` MFMA paths are skipped automatically).
- `vllm/platforms/rocm.py`: gfx900 forces the `TRITON_ATTN` attention backend.
- `CMakeLists.txt`: add `gfx900` to `HIP_SUPPORTED_ARCHS`. The MFMA kernels are
  guarded by `__HIP__GFX9__` (gfx908/90a/942/950 only), so they compile-guard
  out for gfx900 automatically.

## Dependency stack status (ROCm 7.2.4)

| Layer | gfx900 status | Fix |
|-------|---------------|-----|
| ROCm runtime | works | stock 7.2.4 |
| rocBLAS | works | copy `*gfx900*` TensileLibrary from system rocBLAS (incl. `TensileLibrary_lazy_gfx900.dat`) into torch bundle |
| RCCL | needs rebuild | build RCCL (rocm-7.2.x) with `--amdgpu_targets gfx900`; swap into torch/lib |
| PyTorch | works | `torch==2.12.0+rocm7.2` (ABI-matched to system ROCm 7.2.x) |
| amdsmi | needed | copy `/opt/rocm/share/amd_smi/amdsmi` into site-packages (vLLM platform detection) |
| vLLM csrc | builds | this branch (`PYTORCH_ROCM_ARCH=gfx900`) |
| **Triton** | **BLOCKED** | prebuilt Triton MLIR backend rejects gfx900 (`unsupported target: gfx900` in `ConvertWarpPipeline`). Needs Triton built from source with gfx900 re-enabled. See refs. |

## Triton gfx900 -- the remaining blocker

vLLM requires Triton for attention and sampling. The prebuilt Triton 3.7.0
(bundled with torch rocm wheels) fails on gfx900 even for trivial kernels:

```
error: unsupported target: gfx900
note: Pipeline failed while executing [ConvertWarpPipeline]
RuntimeError: PassManager::run failed
```

The underlying LLVM (22.0.0) DOES support gfx900 codegen -- the rejection is in
Triton's own AMD MLIR passes. Prior art for first-gen GCN/Vega Triton:

- https://github.com/Said-Akbar/triton-gcn5  (explicitly MI25 = gfx900, Triton 3.1.0)
- https://github.com/nlzy/triton-gfx906  (newer; "mark gfx906 as CDNA, fix permlanex16 intrinsic")

Plan: fork one of these as `triton-gfx900`, version-aligned to vLLM's Triton API,
add gfx900 to the AMD backend passes + LLVM intrinsic selection.

## Bandwidth / topology notes (this box)

- All GPUs PCIe 3.0 x16. Large-BAR already enabled (full 8G VRAM BAR).
- 2 NUMA domains (GPU 0-7 socket0, 8-15 socket1).
- Intra-socket P2P D2D ~8 GB/s; **cross-socket P2P ~0.26 GB/s** (Xeon E5 v3 QPI
  hardware limit -- not fixable via IOMMU/BIOS). Keep TP groups within a socket.
- NUMA-aware host pinned memory restores +66% H2D on socket1 GPUs.
- Recommended layout for 16 GPUs: **TP=8 within a socket, PP=2 across sockets**.
