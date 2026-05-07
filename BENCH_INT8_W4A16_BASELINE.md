# BENCH — INT8 / W4A16 Baseline (gfx908 / MI100)

This document aggregates Milestone 0 + Milestone 1 baseline numbers for the
`vllm-project/vllm` MI100 quant-kernel optimization mission. Milestone 1
content will be added by the M1 worker once Milestone 0's GO/NO-GO decision
is resolved by the user/orchestrator.

## Hardware / software manifest

| | |
|---|---|
| Hosts | 1× node, 4× AMD Instinct MI100 (gfx908), 32 GB VRAM each |
| Driver | amdgpu-dkms 6.19.0+, perf=high, 250 W cap |
| ROCm | 7.12 (`/opt/rocm/core-7.12`) |
| PyTorch | 2.11.0+rocm7.2 |
| pytorch-triton-rocm | 3.5.1 |
| vLLM | 0.20.2rc1.dev93+g85994b2e3 (mission worktree, editable install) |
| Models | `/models/Qwen3.5-9B-{w8a8,w4a16}`, FP16 ref `/models/Qwen3.5-9B` |

## Milestone 0 — Quality Gate (hard gate)

All evidence in `/root/bench-int8-w4a16/baseline/`. See `M0_DECISION.md` for
the full per-assertion table.

| Gate | w8a8 | w4a16 |
|------|------|-------|
| Artifact present + valid `quantization_config` | PASS | PASS |
| vLLM smoke-load TP=1 (`/v1/models` 200) | PASS | PASS |
| vLLM smoke-load TP=4 (4 GPUs at 96+% VRAM) | PASS | PASS (eager) |
| Coherence ≥8/10 syntax, no >50% n-gram repetition (TP=1, TP=4) | 8/10, 8/10 | 8/10, 8/10 |
| NIAH 5/5 verbatim @ 8k | 5/5 | 5/5 |
| Determinism + no NaN/Inf | PASS | PASS |
| Wikitext-2 perplexity Δ ≤ +1% vs FP16 | **FAIL +1.37 %** | **FAIL +2.91 %** |

### Wikitext-2 perplexity (50 × 512 tokens, seed 0)

| Model | ppl |
|-------|------|
| FP16 (`Qwen/Qwen3.5-9B`) | 9.5254 |
| W8A8 (`RedHatAI/Qwen3.5-9B-quantized.w8a8`) | 9.6561 (Δ +1.37 %) |
| W4A16 (`apolo13x/Qwen3.5-9B-quantized.w4a16`) | 9.8030 (Δ +2.91 %) |

### M0 decision

**`DECISION: NO-GO`** — Wikitext-2 perplexity gate failed for both artifacts.

Generation quality (coherence, needle retrieval, determinism) PASSES on both
artifacts. The prior-round W8A8 GatedDeltaNet failure mode (gibberish output)
**is NOT reproduced** by these RedHatAI / apolo13x artifacts.

The Δ values (+1.37 %, +2.91 %) are within published norms for INT8 channel
+ INT4 group-128 PTQ. They merely exceed the contract's strict +1 % gate.

Mission paused at M0; orchestrator returned the decision to the user for a
pivot-or-relax-gate choice. See `/root/bench-int8-w4a16/baseline/M0_DECISION.md`.

## Milestone 1 — Baseline numbers

*(Pending M0 GO; will be filled in by the M1 worker.)*

### M1 pre-step — AOT-pre-tune of stock Triton W4A16 kernel (feature `m1-aot-pretune-w4a16`)

**Status: SUCCESS — graph mode recovered for W4A16 TP=4.**

M0 documented that W4A16 TP=4 could not capture FULL_DECODE_ONLY cudagraphs
because Triton's W4A16 JIT compile fan-out across 4 workers exceeded NCCL's
default 10-min `all_gather` timeout during `profile_run`, forcing the M0
worker to keep `--enforce-eager` for that cell. This pre-step recovers
graph mode by populating the Triton compile cache once at TP=1 and sharing
the cache directory with all four TP=4 workers via `TRITON_CACHE_DIR`.

Mechanism: the in-tree W4A16 kernel
(`vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`) is
a single `@triton.jit` function whose constexpr meta-parameters are
`(BLOCK_M, BLOCK_N, BLOCK_K, HAS_ZP, ZP_BIAS)` plus launch-side
`num_warps=4, num_stages=2` on MI100. None of the runtime M/N/K dimensions
participate in Triton's compile-cache key, so a TP=1 run that exercises
all three M-bucket branches in the kernel's MI100 dispatch table:

```
M ≤ 16        -> (BLOCK_M=16,  BLOCK_N=64,  BLOCK_K=32)   # decode hot path
16 < M ≤ 64   -> (BLOCK_M=32,  BLOCK_N=64,  BLOCK_K=32)
M > 64        -> (BLOCK_M=64,  BLOCK_N=128, BLOCK_K=32)   # prefill
```

produces cache entries that each TP=4 rank can hit byte-for-byte at
profile_run time without any compile work.

#### Artifacts

| Artifact | Path |
|----------|------|
| Pretune script (loads model TP=1, drives prefill+decode) | `scripts/mi100/pretune_w4a16.py` |
| Persisted Triton compile cache | `/root/bench-int8-w4a16/baseline/triton_cache_w4a16/` |
| Cache manifest (file counts, sizes) | `/root/bench-int8-w4a16/baseline/triton_cache_manifest.json` |
| TP=4 smoke server log | `/root/bench-int8-w4a16/baseline/tp4_pretune_smoke.log` |
| Structured evidence JSON | `/root/bench-int8-w4a16/baseline/m1_pretune_evidence.json` |

#### Cache contents

```
n_subdirs: 163
n_files:   1087
total:     ~51 MB
.hsaco:    132   (HIP code-object binaries — Triton kernels)
.amdgcn:   132   (AMDGCN assembly)
.ttir:     132   (Triton IR)
.ttgir:    132   (Triton GPU IR)
.llir:     132   (LLVM IR)
.json:     270   (kernel metadata)
.so:       25    (Triton runtime libs)
```

#### TP=4 startup verification

After updating `services.yaml::vllm-w4a16-tp4` to set
`TRITON_CACHE_DIR=/root/bench-int8-w4a16/baseline/triton_cache_w4a16`
and dropping `--enforce-eager`:

| Check | Value | Gate |
|-------|------:|------|
| Time from process start to `/health` 200 | **152 s** | < 300 s ✓ |
| `enforce_eager` in engine config | `False` | required ✓ |
| `cudagraph_mode` in engine config | `FULL_DECODE_ONLY` | required ✓ |
| Decode CUDA graphs captured | **35 / 35** | "captured N graphs" ✓ |
| `init engine (profile, create kv cache, warmup model)` | 100.84 s | well below 600 s NCCL bar |
| 100-step decode probe via graph replay | 25/25 OK in 3.8 s | required ✓ |

Engine config excerpt from
`/root/bench-int8-w4a16/baseline/tp4_pretune_smoke.log`:

```
enforce_eager=False, ...
cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>, ...
cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, ..., 496, 512],
max_cudagraph_capture_size=512
```

```
Capturing CUDA graphs (decode, FULL): 100%|██████████| 35/35
init engine (profile, create kv cache, warmup model) took 100.84 s
INFO:     Application startup complete.
```

This unblocks Milestone 1 baseline measurements: W4A16 TP=4 cells now run
with the **same** graph mode as W8A8 TP=4, and decode latency is no longer
inflated by per-iteration kernel-launch overhead. The
`NCCL_TIMEOUT_HOURS=2 / TORCH_NCCL_BLOCKING_WAIT=1 /
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` env triple is retained in
`services.yaml::vllm-w4a16-tp4` as defence-in-depth for cold-start cache
misses.

#### Reproducing the pretune

```bash
cd /home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
/opt/vllm-env/bin/python3 scripts/mi100/pretune_w4a16.py \
  --cache-dir /root/bench-int8-w4a16/baseline/triton_cache_w4a16 \
  --max-wait-sec 1500
```

Idempotent: re-running with `--keep-existing-cache` is a no-op when the
kernel source hasn't changed; without that flag the cache dir is wiped
and rebuilt from scratch (~3 minutes wall-clock at TP=1).
