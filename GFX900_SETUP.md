# vLLM on AMD gfx900 (Vega10: V340 / MI25) -- Work In Progress

This branch (`gfx900-support`) extends the MI100/gfx908 fork to AMD **gfx900**
(Vega10) GPUs such as the Radeon Pro V340 and Instinct MI25.

> **Reproducibility status (2026-07):** historical hardware statements below are
> evidence, not platform defaults. Capture a live manifest and validate its
> `platform_sha256` before using any recipe:
>
> ```bash
> .venv/bin/python -m scripts.gfx900 manifest \
>   --profile scripts/gfx900/profiles/c4130-2.json \
>   --output <run>/manifest.json --reference
> .venv/bin/python -m scripts.gfx900 validate <run>/manifest.json
> ```
>
> The profile pins the Qwen3.5 reference revisions and records actual
> device/VRAM/topology, ROCm/PyTorch/Triton/RCCL hashes, power state, NUMA
> placement, and gfx900 reset method. A mismatch is `INCOMPARABLE`, never a
> normalized performance comparison. The capture intentionally fails closed
> unless `amdgpu.reset_method=2`; it does not change host settings.

## Historical c4130-2 reference record (2026-07-13)

Historical system note: earlier testing described 8 Radeon Pro V340 cards
(16 gfx900 dies) on dual Xeon E5-2640 v3 with ROCm 7.2.4. The selected
2026-07-13 capture used the eight-die c4130-2 group; do not infer unselected
inventory, VRAM, or topology from the historical description. It was captured
from clean source commit `3973e0ec9cd10b95f4663096237c025806efdfdb` with vLLM
`0.1.dev17289+g3973e0ec9`, ROCm 7.2.4, PyTorch 2.12.1+rocm7.2, and Triton
3.7.0. The selected group is eight 56-CU gfx900 dies (8,573,157,376 bytes
each), all PCIe-attached on NUMA node 0. Its compatibility digest is
`61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`.

The 2026-07-19 clean source commit
`689cbbba3ae5400bd583e1435177464e48f1f94a` reproduced that platform digest,
but its capacity and dense-graph screens are awaiting independent-launch
confirmation. Treat the historical figures as comparable context, not an
accepted recommendation. Recapture rather than reusing any figure after a
hardware, ROCm, library, rank-order, or topology change.

### Reproducing the completed c4130-2 campaign

Use `scripts/gfx900/profiles/c4130-2.json` and its declared device group rather
than copying an old `HIP_VISIBLE_DEVICES` recipe. The TP8 topology control uses
`Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, FP16,
`--language-model-only`, `--max-model-len 4352`, and
`--gpu-memory-utilization 0.85` in eager mode. The separate 8,192-input
capacity and prefix cells use `--max-model-len 8448`.

The measured workload-specific serving features are prefix caching and
decode-only CUDAGraphs. Prefix-caching promotion evidence is the deterministic
`prefix_repetition` workload: 4,096 shared-prefix tokens, 256 unique suffix
tokens, 128 requested output tokens, four prefixes, 32 prompts, and c=8. The
reproducible option is `--enable-prefix-caching`; do not infer a benefit for
unrelated prompt distributions.

For latency-sensitive decode-dominant serving, the TP8
`FULL_DECODE_ONLY` graph result is historical pending independent-launch
confirmation. The clean c=1 screen measured 57.651 output tok/s in graph mode
versus 12.707 eager, with p99 TPOT 15.939 versus 79.490 ms; it is not yet a
recommended profile. Do not apply the historical AWQ graph result: on the
clean substrate AWQ only reached eager readiness with text-only multimodal
limits and `--max-num-batched-tokens 512`. Every tested VLLM_COMPILE mode
hung during warmup, with and without CUDAGraph capture.

This is not a burst/open-loop default. In the TP8 DecodeBenchConnector
4,096-input / 256-output burst control at 0.0827 RPS and burstiness 0.25, graph
and eager had nearly equal output throughput (18.119 versus 18.064 tok/s);
graph p99 TPOT was slightly lower (126.93 versus 129.48 ms), but its p99 TTFT
was 2,431 versus 1,621 ms. Use the graph profile only for the demonstrated
low-concurrency, decode-dominant workload, and validate application arrival
traces before enabling it.

The 32,768-token numerical quality check is intentionally a separate,
single-sequence launch because a 0.85 GPU-memory-utilization TurboQuant server
OOMed in the prefill activation path. For that check only, use
`--max-model-len 34816 --max-num-seqs 1 --max-num-batched-tokens 4096
--gpu-memory-utilization 0.75`, then run:

```bash
.venv/bin/python scripts/eval_needle.py \
  --base-url http://127.0.0.1:<port> --ctx 32768 --probes 5 \
  --model Qwen/Qwen3.5-9B --out <run>/needle-32768.json
```

Both auto KV and `turboquant_k8v4` passed this 5/5 probe under the identical
single-sequence settings. The verified WikiText-2 Parquet shard bypasses the
incompatible legacy `datasets` loader, but the `echo=true` prompt-logprob API
still intentionally skips prefix-cache reads; retain it as a corpus-loader and
prefill-numerics smoke test only.

The valid cache-read perplexity gate is
`.venv/bin/python -m scripts.gfx900.cache_read_ppl`: it prefills 256 tokens
from each fixed 512-token chunk, teacher-forces the remaining 256 corpus tokens
through actual decode steps, and records raw target logits before masking the
sampled output. Across 50 chunks (12,800 scored tokens), auto PPL was
7.7323505 and `turboquant_k8v4` PPL was 7.7242287 (-0.105%). That difference
is quality-neutral within measurement noise, not a quality gain, and is below
the +1% gate. The clean c=32 screen measured 20.521 output tok/s for
`turboquant_k8v4` versus 18.583 for auto, with zero request failures; it still
requires independent-launch confirmation before becoming a profile option.
See `GFX900_RECOMMENDED.md` for measured launch recommendations and
`PERF_GFX900.md` for the chronological campaign evidence.

## Historical hardware context

Historical system observed: 8x AMD Radeon Pro V340 (= 16x gfx900 GPUs, 8GB
VRAM each), dual Xeon E5-2640 v3, Ubuntu 24.04, ROCm 7.2.4. Use the accepted
manifest above rather than this prose for a new run.

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
| **Triton** | **WORKS** | custom `triton-gfx900` fork (Triton v3.7.0 + gfx900 ISA family). See below. |

## STATUS: Qwen3.5-9B RUNS ON gfx900 (TP=8)

The original goal is achieved. **Qwen3.5-9B** -- a multimodal hybrid model
(Gated DeltaNet linear-attention + full attention, 32 layers) -- runs across
**8x gfx900 GPUs** with tensor parallelism, text-only mode:

```
GPU KV cache size: 368,208 tokens
'The capital of France is' => ' Paris.\nThe capital of France is Paris...'
```

Notably the FLA Gated-DeltaNet Triton kernels (`chunk_gated_delta_rule`,
`causal_conv1d`, `l2norm_fwd`) -- the most advanced kernels in the stack --
compiled and ran on Vega10 with **zero additional gfx900 fixes** beyond the
GFX900 ISA family. First run is slow due to Triton autotuning (~130 kernels
JIT-compiled + disk-cached); subsequent runs reuse the cache.

Run config: `tensor_parallel_size=8, dtype=float16` (gfx900 has no native BF16,
so cast BF16->FP16), `enforce_eager=True`, `language_model_only=True`,
`max_model_len=4096`, GPUs 0-7 (single socket -- keeps TP within one NUMA node).

## STATUS: vLLM RUNS ON gfx900 (smaller models)

Validated end-to-end on a Radeon Pro V340 (gfx900:xnack-):
opt-125m at TP=1/2/4. Multi-GPU all-reduce uses PYNCCL over our custom gfx900
RCCL (XGMI custom all-reduce is correctly disabled -- Vega10 has no XGMI).

```
[rocm.py] Using TRITON_ATTN backend out of potential backends: ['TRITON_ATTN'].
'The capital of France is' => ' the capital of the French Republic...'
```

## Triton gfx900 -- SOLVED (custom fork)

vLLM requires Triton for attention and sampling. Stock Triton 3.7.0 rejects
gfx900 even for trivial kernels (`unsupported target: 'gfx900'` because
`deduceISAFamily()` returns `Unknown`). The underlying LLVM DOES support gfx900
codegen -- the rejection was purely in Triton's AMD MLIR backend.

Fix: a `triton-gfx900` fork based on **triton v3.7.0** (matches our stack),
adding a dedicated **GFX900 ISA family** (8 + 6 lines):

- `TargetUtils.h`: add `GFX900` to `ISAFamily` enum.
- `TargetUtils.cpp` `deduceISAFamily`: map gfx900/gfx902 -> `ISAFamily::GFX900`.
  A *dedicated* family (NOT gfx906's VEGA20) is essential: gfx906 enables V_DOT
  and DirectToLds that Vega10 lacks -- reusing it crashes with ILLEGAL_INSTRUCTION
  (the same reason `HSA_OVERRIDE_GFX_VERSION=9.0.6` spoofing failed).
- `TargetInfo.cpp` `getWarpSize()`: GFX900 is wave64.
- `TargetInfo.cpp` `warpReduce()`: bail out for GFX900 so it uses the generic
  `shuffleXor` (ds_swizzle/ds_bpermute) reduction instead of the DPP-broadcast +
  `permlanex16` path. `permlanex16` is RDNA-only and is illegal on Vega10 -- this
  was the second blocker after the "unsupported target" gate.

gfx900 thus routes entirely through portable wave64 `ds_swizzle`/`ds_bpermute`
paths: no MFMA, no V_DOT, no DPP-broadcast, no permlane. Validated kernels:
trivial add (PASS), softmax cross-lane reduction (PASS), matmul `tl.dot` FMA (PASS).

Build: `TRITON_BUILD_WITH_CCACHE=true pip wheel --no-build-isolation --no-deps .`
(LLVM is downloaded prebuilt and already supports gfx900 -- only Triton's C++ is
recompiled). Prior art: Said-Akbar/triton-gcn5 (MI25=gfx900), nlzy/triton-gfx906.

## Bandwidth / topology notes (this box)

- All GPUs PCIe 3.0 x16. Large-BAR already enabled (full 8G VRAM BAR).
- 2 NUMA domains (GPU 0-7 socket0, 8-15 socket1).
- Intra-socket P2P D2D ~8 GB/s; **cross-socket P2P ~0.26 GB/s** (Xeon E5 v3 QPI
  hardware limit -- not fixable via IOMMU/BIOS). Keep TP groups within a socket.
- NUMA-aware host pinned memory restores +66% H2D on socket1 GPUs.
- Recommended layout for 16 GPUs: **TP=8 within a socket, PP=2 across sockets**.
