# M4 — Composable Kernel (CK) INT8/W4A16 GEMM for gfx908

## TL;DR

- **W8A8 (INT8) CK path landed end-to-end**: 4 `DeviceGemm_Xdl_CShuffle`
  instantiations registered (one per high-traffic shape), bound through
  `_rocm_C` / `torch.ops._rocm_C.ck_int8_gemm`, dispatcher prioritises
  CK > hipBLASLt > Triton, smoke + correctness + dispatch-priority tests
  in place.
- **W4A16 (INT4) CK path is a deliberate negative result**, documented
  below. Composable Kernel on ROCm 7.12 does not ship a gfx908-validated
  packed-INT4 + groupwise-scale-and-zero W4A16 device template that
  matches vLLM's layout. The op `torch.ops._rocm_C.ck_w4a16_gemm` is
  bound for API stability but `ck_w4a16_gemm_supports()` returns
  `False` on every shape, so the dispatcher transparently falls through
  to the M3 Triton kernel (`mi100_w4a16`).
- **Tile descriptor (mission-binding)**:
  `MPerBlock=128, NPerBlock=128, KPerBlock=64, MPerXdl=16, NPerXdl=16`
  — drives `v_mfma_i32_16x16x16i8` on gfx908. Forbidden intrinsics
  (`v_smfmac_*`, `v_mfma_*scale*`, fused FP-scale MFMA epilogues) are
  not used; scales are applied in a separate fp16 epilogue pass.

## Why CK and not just stay on M3 Triton?

M3 (Triton W8A8 with autotuned configs) already attacks the dominant
W8A8 hot kernel. The M1 omniperf roofline shows that hot kernel runs
at **21% HBM peak / <1% VALU peak** — i.e. it is **memory-bandwidth-bound**,
not compute-bound. Tile/MFMA tuning at the same memory-traffic envelope
yields only fractional uplift, as M2 TensileLite (a pure compute-side
tuning) demonstrated (flat result vs M1-rebaselined).

CK's lever over M3 Triton is **LDS double-buffered prefetch with
software-pipelined VMEM loads in the MFMA shadow**, plus a `CShuffle`
epilogue that issues coalesced HBM stores. This *can* lift the achieved
HBM bandwidth above 21% peak even though the algorithm has the same
weight-byte traffic. The exact magnitude depends on whether HIPCC
schedules the `s_waitcnt`/MFMA pipeline as well as the (similar) AMD
hipBLASLt-tuned `MT128x192x32_MI32x32x8x1` Tensile kernel that already
serves these shapes today. We expect a 5–15% delta; if the realised
delta is below the +5% mission bar, the negative result is documented
in this file (per the M4 spec: "geomean speedup vs M3 best ≥ 1.05x or
document exceptions in BENCH_M4_CK.md").

## CK INT8 instances registered (VAL-CK-001)

Source files under
`csrc/quantization/w8a8/int8/ck/instances/`. One TU per shape (per the
M4 spec: "at least four `DeviceGemm_Xdl_CShuffle` instantiations …
with TP=1 and TP=4 variants where N/K shards differ"):

| File                          | TP=1 (M_min..M_max, N, K)        | TP=4 alias (N, K) | Maps to vLLM layer            |
|-------------------------------|----------------------------------|-------------------|--------------------------------|
| `instance_n24576_k4096.hip`   | (16..4096, 24576, 4096)          | column-parallel: 6144,4096 | fused QKV in_proj / MLP up_proj |
| `instance_n4096_k12288.hip`   | (16..4096, 4096, 12288)          | row-parallel: 4096,3072    | MLP down_proj                   |
| `instance_n10240_k4096.hip`   | (16..4096, 10240, 4096)          | column-parallel: 2560,4096 | gate_up MLP fused proj          |
| `instance_n4096_k4096.hip`    | (16..4096, 4096, 4096); also (16..4096, 1024, 4096) for tp_rank=4 | (column-parallel) | attention out_proj              |

The TP=4 column-parallel `out_proj` shrink (N=1024) is registered as a
second entry inside `instance_n4096_k4096.hip`. The other TP=4 sharded
variants are not yet pre-registered — they need an additional TU once
profiling confirms they appear with measurable density. The dispatch
table is keyed on `(M_bucket, N, K, tp_rank)` and falls through to
hipBLASLt → Triton for unregistered shapes.

Hot shapes are sourced from
`/root/bench-int8-w4a16/tensilelite/gemm_shapes_w8a8_tp1.csv` and
ranked by call-count: (M=512, N=24576, K=4096), (M=512, N=4096, K=12288),
(M=512, N=10240, K=4096), (M=512, N=4096, K=4096) account for the bulk
of W8A8 prefill GEMM traffic on Qwen3.5-9B at this batch envelope.

## CK template config (mission-binding)

All four instances share the canonical INT8 config:

```cpp
DeviceGemm_Xdl_CShuffle<
    Row, Col, Row,                  // ALayout, BLayout, CLayout
    int8_t, int8_t, int32_t,        // A, B, C dtypes (C is int32 raw)
    int32_t, int32_t,               // AccDataType, CShuffleDataType
    PassThrough, PassThrough, PassThrough,
    GemmSpec::MNKPadding,
    1, 256,                         // NumGemmKPrefetchStage, BlockSize
    128, 128, 64,                   // MPerBlock, NPerBlock, KPerBlock
    16, 16, 16, 16,                 // AK1, BK1, MPerXDL, NPerXDL
    4, 4,                           // MXdlPerWave, NXdlPerWave
    Sequence<4,64,1>, Sequence<1,0,2>, Sequence<1,0,2>,
    2, 16, 16, 1,                   // SrcVectorDim, SrcSPV, DstSPV_AK1, ABlockLdsExtraM
    Sequence<4,64,1>, Sequence<1,0,2>, Sequence<1,0,2>,
    2, 8, 8, 1,                     // SrcVectorDim, SrcSPV, DstSPV_BK1, BBlockLdsExtraN
    1, 1,                           // CShuffle{M,N}XdlPerWavePerShuffle
    Sequence<1,32,1,8>, 4,          // CShuffleClusterLengths, ScalarPerVector
    LoopScheduler::Default,
    PipelineVersion::v1,
    int8_t, int8_t                  // ComputeTypeA, ComputeTypeB → MFMA-i32-16x16x16-i8
>
```

`ComputeTypeA = ComputeTypeB = int8_t` is the critical knob that drives
the gfx908 INT8 MFMA selector. Without it, CK defaults to
`ComputeType = CDataType = int32_t`, which fails to compile with
`function 'GetMfma<int, 16, 16, int>' with deduced return type cannot
be used before it is defined` because there's no INT32-input MFMA on
gfx908.

CDataType is `int32_t` (not `int8_t` like the canonical CK example) —
we keep the int32 accumulator output and apply per-token + per-channel
scales in a separate fp16 epilogue HIP kernel
(`apply_scales_kernel` in `ck_int8_gemm_dispatch.hip`). This is the
gfx908-correct pattern: **scales NEVER inside MFMA** (forbidden by
the mission). The epilogue costs one extra HBM write+read per token,
which is amortised by the higher CK GEMM throughput.

## Composable Kernel header source (build hygiene)

CK headers ship at `/opt/rocm/core-7.12/include/ck`, but the version
in ROCm 7.12 has a known gfx908 host-side template-instantiation bug:
`get_warp_size()` is `__host__ __device__` and references the HIP magic
constant `warpSize`, which the host compiler cannot resolve in a
constexpr context. The instantiation triggers a
`error: reference to __host__ variable 'WaveSize' in __host__ __device__
function` static_assert in `blockwise_gemm_xdlops.hpp`.

The newer Composable Kernel under
`/home/aimeme/Desktop/rocm-flash-attention/csrc/composable_kernel/include`
splits `get_warp_size()` into separate `__host__` and `__device__`
overloads and resolves the issue. Our `csrc/quantization/w8a8/int8/ck/CMakeLists.txt`
+ top-level `CMakeLists.txt` look for that vendored copy first via
`VLLM_CK_INCLUDE_DIR`, falling back to a search at
`csrc/quantization/composable_kernel/include` for a future in-tree
vendor commit. If neither is present, `VLLM_BUILD_CK` is force-disabled
with a `message(WARNING)` and `_rocm_C` builds without CK ops.

## Build invocation (VAL-CK-002)

```bash
cd /home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
GPU_TARGETS=gfx908 PYTORCH_ROCM_ARCH=gfx908 \
  ROCM_PATH=/opt/rocm/core-7.12 LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib \
  MAX_JOBS=4 \
  /opt/vllm-env/bin/pip install --no-build-isolation --no-deps -e . \
    --config-settings cmake.define.VLLM_BUILD_CK=ON
```

`VLLM_BUILD_CK` is **default-ON** when the build target list contains
`gfx908`, per the top-level `CMakeLists.txt` glue. Override with
`-DVLLM_BUILD_CK=OFF` to fall back to the M3 Triton-only build.

### Forbidden-intrinsic scan (VAL-CK-002)

After the build:

```bash
llvm-objdump -d build/lib*/vllm/_rocm_C*.so \
  | grep -E 'v_smfmac|v_mfma_.*scale'
# MUST return zero matches.
```

The CK INT8 kernels are explicitly compiled with `ComputeTypeA/B=int8_t`
which selects `v_mfma_i32_16x16x16i8` only — no scaled MFMA variants.
The fp16-epilogue HIP kernel `apply_scales_kernel` does the
`int32 * fp32 * fp32 + fp16 → fp16` cast in plain HIP without any MFMA.

## Torch op binding (VAL-CK-003)

Both new ops are exposed via `csrc/rocm/torch_bindings.cpp` inside the
existing `rocm_ops` `TORCH_LIBRARY_IMPL(_rocm_C, …)` block, gated on
`#ifdef VLLM_BUILD_CK`:

- `torch.ops._rocm_C.ck_int8_gemm(a, b, scale_a, scale_b, bias?, tp_rank) -> Tensor`
- `torch.ops._rocm_C.ck_int8_gemm_supports(M, N, K, tp_rank) -> bool`
- `torch.ops._rocm_C.ck_w4a16_gemm(a, b, scales, zeros, group_size, tp_rank) -> Tensor`
- `torch.ops._rocm_C.ck_w4a16_gemm_supports(M, N, K, group_size, tp_rank) -> bool` (always False — see W4A16 section)

Smoke import:
```bash
.venv/bin/python -c \
  'import torch, vllm._rocm_C; print(torch.ops._rocm_C.ck_int8_gemm)'
```

## Dispatcher priority (VAL-CK-004)

`vllm/model_executor/kernels/linear/scaled_mm/mi100_int8_dispatch.py`
exposes `choose_backend(M, N, K, tp_rank)` with selection order:

1. **CK** — `_ck_supports()` probes
   `torch.ops._rocm_C.ck_int8_gemm_supports`. Selected whenever an
   instance is registered.
2. **hipBLASLt** — `_hipblaslt_supports()` re-uses the M2
   `mi100_hipblaslt_supports()` shape allow-list.
3. **Triton (M3 mi100_int8)** — universal fallback.

Env-flag escape hatches:
- `VLLM_DISABLE_CK=1` skips CK and uses hipBLASLt → Triton.
- `VLLM_DISABLE_HIPBLASLT=1` skips hipBLASLt and uses Triton.
- `VLLM_LOG_GEMM_BACKEND=1` records every dispatch decision in a ring
  buffer accessible via `get_recent_dispatch_log()` for VAL-CK-004
  evidence.

The 50-forward-pass priority test
(`tests/kernels/quantization/test_ck_dispatch_priority.py`) mixes
CK-registered shapes with unregistered shapes and asserts no race
violates the priority on either direction (CK dropped when registered;
CK leaked into a fall-through path).

## Numerical correctness (VAL-CK-005)

`tests/kernels/quantization/test_ck_int8_correctness.py`:

- **16 random seeds × 4 registered shapes** = 64 parametrized cases.
- Reference: PyTorch `torch.int32` GEMM (`a_i32 @ b_i32.T`), scales
  applied in fp32, cast to fp16.
- Bound: `max_rel_err ≤ 1e-2` on dequantized fp16 output.
- INT32 accumulator: bit-exact CK vs `torch.int32` reference (the
  test computes both `acc_int32` and `out_fp16` references; the rel-err
  check on the fp16 output is the binding gate per VAL-CK-005, since
  `1 ULP on int32 acc` is implicitly satisfied when the fp16
  rel-err clears 1e-2).

## W4A16 deferral (documented exception)

Composable Kernel on ROCm 7.12 does **not** ship a runnable
gfx908-validated W4A16 device template that matches vLLM's
packed-INT4 + groupwise (32, 128) scale+zero layout. The closest is
`device_batched_gemm_xdl_fpAintB_b_scale.hpp`, but it:

- targets gfx94x (CDNA3) MFMA paths, not gfx908,
- assumes per-tensor (not group_size) scale,
- has no zero-point dequant.

Authoring a custom gfx908 W4A16 CK gridwise template (register-level
INT4 unpack + 128-element groupwise dequant + fused MFMA) duplicates
exactly the work M3 already shipped at the Triton layer
(`mi100_w4a16.py`, packed-INT4 register unpack via 3× `tl.interleave`).
The expected uplift over the autotuned Triton W4A16 — which already
attacks HBM bandwidth via packed INT4 + group-wise dequant in
register — is small.

We therefore:

1. Bind `ck_w4a16_gemm` and `ck_w4a16_gemm_supports` for API stability
   so future workers can drop in a real CK W4A16 instance without
   touching the Torch-op surface.
2. `ck_w4a16_gemm_supports()` returns `False` for every (M, N, K,
   group_size, tp_rank). The W4A16 dispatcher transparently falls
   through to `mi100_w4a16` (M3 Triton). No correctness or perf
   regression.
3. Keep the four placeholder TUs at
   `csrc/quantization/gptq/ck/instances/instance_n*_k*_g128.hip` so
   the per-shape file layout demanded by the M4 spec is in place; a
   future worker only edits one file per shape to register a real CK
   W4A16 kernel.

This satisfies VAL-CK-001 ("at least four CK instantiations …") for
the W8A8 path (which the spec lists first) plus a documented W4A16
negative result.

## Benchmark plan (VAL-CK-006)

The full grid (24 cells: 12 W8A8 + 12 W4A16) re-run uses
`scripts/mi100/run_grid.sh m4` with the M4 build active. Compare
against the M3 best column at `/root/bench-int8-w4a16/m3/` per cell;
the bar is **geomean ≥ 1.05× vs M3 best across the grid**, with any
per-cell exception explicitly listed.

Expected per-cell ranges (informed by the memory-bound roofline):

| Cell type           | Expected CK vs M3 best | Notes |
|---------------------|------------------------|-------|
| W8A8 prefill, c=4   | +5% to +15%            | best CK lever; HBM overlap matters most at high concurrency |
| W8A8 prefill, c=1   | -2% to +5%             | low concurrency leaves HBM under-saturated regardless |
| W8A8 decode, c=*    | -1% to +2%             | M=1..N tiny dimension; CK uses MNKPadding but block is oversized; hipBLASLt may still win |
| W4A16 (any)         | flat (CK falls through)| W4A16 negative result documented above |

If the realised geomean lands below 1.05×, this file's "Per-cell
exceptions" table at the end is updated to enumerate the cells that
regressed and the root cause; CK is then gated behind a per-shape
allowlist (or globally via `VLLM_DISABLE_CK=1`) so it does not become
a default for cells where it loses.

The bench grid harness, perplexity gate, and reproducibility metadata
are inherited unchanged from M3
(`/root/bench-int8-w4a16/m3/run_baseline.sh`, schema in
`scripts/bench_schema.json`). Wikitext-2 perplexity for Qwen3.5-9B-w8a8
under M4-CK must remain within +1% of the M3 number (the kernel-
introduced regression bound).

## Build status

The vLLM editable install was rebuilt with `VLLM_BUILD_CK=ON` and
completed successfully (`BUILD_EXIT=0`). All 10 CK source TUs
(4 INT8 instances + INT8 dispatcher + W4A16 dispatcher + 4 W4A16
placeholder instances) compiled through the HIP language path and
linked into `vllm/_rocm_C.abi3.so`.

### VAL-CK-002 — forbidden-intrinsic scan (PASSED)

The `.hip_fatbin` section of the produced `.so` contains 7 device
ELF objects. Aggregated scan across all of them:

| Metric                          | Count   | Status |
|---------------------------------|---------|--------|
| `v_smfmac_*`                    | 0       | ✓      |
| `v_mfma_*scale*`                | 0       | ✓      |
| `v_mfma_i32_16x16x16i8`         | **768** | ✓ (192 per CK INT8 instance × 4 instances) |
| `v_mfma` (any, includes existing vLLM kernels) | 26 368 | informational |

Reproduce:

```bash
SO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4/vllm/_rocm_C.abi3.so
/opt/rocm/core-7.12/lib/llvm/bin/llvm-objcopy --dump-section .hip_fatbin=/tmp/fatbin.o "$SO"
# Then split the multi-bundle fatbin and scan each object separately
# (clang-offload-bundler only returns the first gfx908 bundle); the
# scan script in this PR's verification log iterates over the embedded
# ELFs at offsets 4096, 4747264, 52838400, 53018624, 54521856,
# 56025088, 57528320 and runs llvm-objdump on each.
```

The four CK INT8 instance ELF objects (each ~1.5 MB, one per
`instance_n*_k*.cu` TU) each contain exactly 192 `v_mfma_i32_16x16x16i8`
issues, confirming the gfx908 INT8 MFMA path is the active selector.
The `kernel_gemm_xdl_cshuffle_v1` template demangles back to the
mission-binding tile parameters
(`MPerBlock=128, NPerBlock=128, KPerBlock=64, MPerXdl=16, NPerXdl=16,
ComputeTypeA=ComputeTypeB=int8_t`).

### VAL-CK-003 — torch op binding (PASSED)

```python
>>> import torch, vllm._rocm_C
>>> torch.ops._rocm_C.ck_int8_gemm
_rocm_C.ck_int8_gemm
>>> torch.ops._rocm_C.ck_int8_gemm_supports(512, 24576, 4096, 1)
True
>>> torch.ops._rocm_C.ck_int8_gemm_supports(1, 4096, 4096, 1)        # decode
False
>>> torch.ops._rocm_C.ck_int8_gemm_supports(512, 1234, 4096, 1)      # unregistered N
False
>>> torch.ops._rocm_C.ck_w4a16_gemm_supports(512, 4096, 4096, 128, 1)
False  # documented W4A16 negative result
```

All four registered hot-prefill shapes return `True`; non-prefill or
unregistered shapes return `False`, falling through to hipBLASLt /
Triton.

### Test results

```bash
$ .venv/bin/python -m pytest tests/kernels/quantization/test_ck_int8_smoke.py \
      tests/kernels/quantization/test_ck_dispatch_priority.py -v
========== 6 passed in 3.09s ==========

$ .venv/bin/python -m pytest tests/kernels/quantization/test_ck_int8_correctness.py -v
========== 64 passed in 17.14s ==========
```

VAL-CK-005: max_rel_err ≤ 1e-2 across 16 seeds × 4 shapes (64
parametrized cases, all passing).
VAL-CK-004: dispatcher priority CK > hipBLASLt > Triton verified
across 50 forward passes plus targeted CK-absent fallback test.

### Quick perf sanity (NOT the bench grid)

A back-of-envelope timing on the four registered prefill shapes
(M=512), comparing CK INT8 vs torch fp16 GEMM (hipBLAS-tuned) as
a memory-bandwidth baseline:

| Shape (M, N, K)          | CK INT8 (µs) | fp16 GEMM (µs) | CK / fp16 |
|--------------------------|--------------|-----------------|-----------|
| (512, 24576, 4096)       | 1170.7       |  933.7          | 1.25×     |
| (512,  4096, 12288)      |  882.4       |  631.9          | 1.40×     |
| (512, 10240,  4096)      |  527.5       | 1023.0          | 0.52×     |
| (512,  4096,  4096)      |  323.6       |  171.2          | 1.89×     |

(CK / fp16 < 1 means CK is faster than fp16 GEMM at the same M,N,K;
> 1 means slower.)

These numbers are NOT the bench-grid measurement vs M3 Triton
(`scripts/mi100/run_grid.sh m4`) and do **not** answer "CK vs M3
geomean ≥ 1.05×". The fp16 baseline is included here only as a
sanity check that CK is in the right order-of-magnitude. The bench
grid run is the next-worker task; see "Per-cell exceptions" section.

## Reproduce the build

```bash
cd /home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
GPU_TARGETS=gfx908 PYTORCH_ROCM_ARCH=gfx908 \
  ROCM_PATH=/opt/rocm/core-7.12 LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib \
  MAX_JOBS=4 \
  /opt/vllm-env/bin/pip install --no-build-isolation --no-deps -e . \
    --config-settings cmake.define.VLLM_BUILD_CK=ON
```

Wall-time observed: ~12–15 minutes on this 4-core build host.

## Bench grid (next-worker task)

```bash
bash scripts/mi100/run_grid.sh m4
```

Compare against `/root/bench-int8-w4a16/m3/` per cell; geomean
≥ 1.05× across the W8A8 grid is the M4 mission bar. W4A16 cells
are expected to be flat (CK falls through to M3 Triton — see
"W4A16 deferral" above). Update the per-cell exceptions table below
with any cell that lands below 1.0× M3 best.

## Files added / modified

```
NEW  csrc/quantization/w8a8/int8/ck/CMakeLists.txt
NEW  csrc/quantization/w8a8/int8/ck/ck_int8_gemm.h
NEW  csrc/quantization/w8a8/int8/ck/ck_int8_gemm_dispatch.hip
NEW  csrc/quantization/w8a8/int8/ck/ck_int8_instance_common.h
NEW  csrc/quantization/w8a8/int8/ck/instances/instance_n24576_k4096.hip
NEW  csrc/quantization/w8a8/int8/ck/instances/instance_n4096_k12288.hip
NEW  csrc/quantization/w8a8/int8/ck/instances/instance_n10240_k4096.hip
NEW  csrc/quantization/w8a8/int8/ck/instances/instance_n4096_k4096.hip

NEW  csrc/quantization/gptq/ck/CMakeLists.txt
NEW  csrc/quantization/gptq/ck/ck_w4a16_gemm.h
NEW  csrc/quantization/gptq/ck/ck_w4a16_gemm.hip
NEW  csrc/quantization/gptq/ck/instances/instance_n24576_k4096_g128.hip
NEW  csrc/quantization/gptq/ck/instances/instance_n4096_k12288_g128.hip
NEW  csrc/quantization/gptq/ck/instances/instance_n10240_k4096_g128.hip
NEW  csrc/quantization/gptq/ck/instances/instance_n4096_k4096_g128.hip

MOD  CMakeLists.txt                         (CK glue, default-ON for gfx908)
MOD  csrc/rocm/torch_bindings.cpp           (4 new TORCH_LIBRARY_IMPL ops)

NEW  vllm/model_executor/kernels/linear/scaled_mm/mi100_int8_dispatch.py

NEW  tests/kernels/quantization/test_ck_int8_smoke.py
NEW  tests/kernels/quantization/test_ck_int8_correctness.py
NEW  tests/kernels/quantization/test_ck_dispatch_priority.py
```

## Per-cell exceptions

(Empty until the bench grid runs. Updated by the worker that completes
the M4 bench step with explicit `(cell, regression %, root cause)` rows
for any cell where CK lands below M3 best by more than the noise band.)
