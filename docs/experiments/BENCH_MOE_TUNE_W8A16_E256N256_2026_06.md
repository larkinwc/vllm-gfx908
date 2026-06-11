<!-- markdownlint-disable MD013 -->
# Tuned fused-MoE config for int8_w8a16 E=256/N=256: a concurrency-gated WIN (2026-06)

Follow-up to the W8A16 35B capacity work (`BENCH_W8A16_35B_MOE_2026_06.md`),
which logged `Using default MoE config ... E=256,N=256,...,int8_w8a16.json not
found`. Two prior MoE-tuning notes are the relevant prior art:

- `BENCH_MOE_TUNED_CONFIG_NEGATIVE_2026_06.md`: the **inherited** `Arcturus_GL-XL`
  configs regress **−38%** vs Triton defaults on our stack (stale large-M tiles
  register-spill). Lesson: **ship only a config that A/B-beats the default.**
- `BENCH_MOE_RETUNE_FROM_SCRATCH_2026_06.md`: a from-scratch re-tune of
  **int4_w4a16 E=512/N=128** found **no headroom** (±2%): that shape is
  fixed-overhead-bound at decode M.

**Why this is not duplicate work:** the W8A16 model is a *different* shape/dtype
— `int8_w8a16`, **E=256, N=256** (TP=2) — vs the flat int4/N=128 case. N=256 is
2× the per-expert GEMM width and weight-only int8 dequant differs from int4
wna16, so tile sensitivity is genuinely untested. It turns out **this shape
*does* have headroom** — the opposite of the int4 result.

## Model / shape

`/models/Qwen3.5-35B-A3B-GPTQ-8bit` → `Qwen3_5MoeForConditionalGeneration`,
E=256, top-8, hidden=2048, moe_intermediate=512. At **TP=2** the per-shard
fused-MoE shape is **N=256** (`2*512/2/2`), group_size 32, dtype `int8_w8a16`
(weight-only: int8 weights, fp16 activations, no activation quant).

The runtime uses the **Triton** wna16 kernel at all decode M
(`should_moe_wna16_use_cuda=False` for M·topk at our E/bit), so tile params are
genuinely tunable (if it used the fixed CUDA wna16 path, they would not be).

## Tuning method

Reused the subprocess-isolated tuner from the int4 retune
(`/root/w8a16-35b/moe_tune_w8a16.py`, derived from
`BENCH_MOE_RETUNE_FROM_SCRATCH`), repointed at the rebuilt v0.22.1 worktree and
restricted to **GPUs 1,2** (thermal). Three stale-API bugs in this version's
`benchmarks/kernels/benchmark_moe.py` had to be fixed first (they break the
int8_w8a16 path entirely — it had never been exercised on this build):

1. `disable_inplace()` referenced but **never defined** (fork commit `7c12dd8a7`)
   → added a definition (default in-place, opt-out via `VLLM_MOE_TUNE_DISABLE_INPLACE`).
2. `fused_experts(..., inplace=...)` — the modern API dropped the `inplace`
   kwarg → removed it from the non-deep-gemm call.
3. The int8_w8a16 weights/scales were built as **w8a8 activation-quant** (int8
   weights, fp32 per-tensor scales, routed through `scaled_int8_quant`) instead
   of **weight-only grouped** (uint8 weights, grouped fp16 scales via
   `int8_w8a16_moe_quant_config`, `block_quant_shape=[0, group_size]`,
   `SPLIT_K=1`). Rebuilt to match the `moe_wna16` runtime contract.

Decode batch sizes M=1,2,4,8,16,24,32 tuned exhaustively (304–3280 configs each,
`num_iters=20`, eager timing via `VLLM_MOE_TUNE_NO_CUDAGRAPH=1`).

## Kernel A/B — tuned-best vs Triton default (eager, best-of-3, 40 iters)

| M | default us | tuned us | tuned/def | verdict |
|---|---:|---:|---:|---|
| 1 | 439.7 | 450.1 | 1.024 | tie |
| 2 | 438.8 | 448.0 | 1.021 | tie |
| 4 | 440.7 | 445.2 | 1.010 | tie |
| 8 | 441.6 | 449.0 | 1.017 | tie |
| 16 | 454.1 | 456.6 | 1.005 | tie |
| 24 | 608.9 | 464.3 | **0.763** | **win** |
| 32 | 718.8 | 466.3 | **0.649** | **win** |

**geomean tuned/default = 0.914, zero losses.** Root cause of the win: the
default heuristic switches to `BLOCK_SIZE_M=32` at M≥24, which is poor for
N=256; the tuned config keeps `BLOCK_SIZE_M=16` with an explicit N=32/K=64 tile
(~35% faster at M=32). At M≤16 decode is fixed-overhead-bound (matches int4),
so the tuned tile only ties.

### Prefill-safety probe (guards the −38% int4 failure mode)

The runtime's `try_get_optimal_moe_config` picks the **closest M key**, so M>32
(prefill / high concurrency) uses the tuned M=32 tile. Timed that tile vs the
default at larger M:

| M | default us | tuned32 us | tuned/def | verdict |
|---|---:|---:|---:|---|
| 48 | 734.1 | 520.9 | 0.710 | win |
| 64 | 808.8 | 568.7 | 0.703 | win |
| 96 | 884.3 | 625.2 | 0.707 | win |
| 128 | 930.0 | 657.8 | 0.707 | win |
| 256 | 971.0 | 703.9 | 0.725 | win |
| 512 | 1052.1 | 957.7 | 0.910 | win |

**Zero losses at M>32** — the tuned tile (`BLOCK_M=16`) does not register-spill
like the inherited int4 large-M tiles did. Safe to ship without exhaustive
prefill tuning (the int4 note already showed exhaustive prefill tuning is not
worth the GPU-hours).

## Server A/B — the decisive gate (default vs tuned, same v0.22.1 binary, TP=2 GPUs 1,2)

FULL_DECODE_ONLY (no compile) both arms, synthetic random 1024/256, 200 prompts,
seed 42. The default arm is the W8A16 doc's baseline; the tuned arm has
`VLLM_TUNED_CONFIG_FOLDER` pointed at the tuned JSON (`Using default MoE config`
warning count 0 confirms it loaded).

| cell | default tok/s | tuned tok/s | **Δ** | tuned TPOT p50 |
|---|---:|---:|---:|---:|
| c1 | 55.12 | 48.71 | **−11.6 %** | 19.95 ms |
| c2 | 74.47 | 85.90 | **+15.3 %** | 22.26 ms |
| c4 | 115.27 | 142.64 | **+23.7 %** | 26.66 ms |

**Concurrency-dependent tradeoff:** the config **regresses single-stream
(−11.6 %) but wins at c≥2 (+15 % / +24 %)**, the win growing with concurrency.
This is consistent with the kernel A/B (tie at small decode M, win at M≥24): c=1
is decode-bound (M=1, the tile loses slightly + worse TPOT), while c≥2 hits the
larger batch/prefill M where the tuned tile is far better.

## Decision: ship concurrency-gated (opt-in), do NOT install globally

The config-file lookup (`get_moe_configs`) is **unconditional** — installing the
JSON in the global `configs/` dir would load it for every deployment and regress
c=1. Instead, following the fork's existing concurrency-gated torch.compile
policy:

- Tuned JSON staged at **`scripts/moe_configs_w8a16/`** (not in the global
  configs dir).
- Opt in per-launch via **`LAUNCH_TUNED_MOE=1`** on the W8A16 launch scripts,
  which sets `VLLM_TUNED_CONFIG_FOLDER` (checked *before* the built-in configs).
- Recommended for **c≥2 / batched-serving** deployments; leave off for
  latency-critical single-stream.

This preserves the prior-art rule (ship only a measured win) while not paying the
c=1 regression: each deployment opts in based on its concurrency regime.

## Caveats / follow-ups

- **TP=4 (N=128) is a different shape** and is *not* covered here — that is the
  int4-equivalent N=128 width, where the int4 retune found no headroom; whether
  int8/N=128 has headroom is untested. Tie it to the deferred TP=4 W8A16 re-run.
- The win is partly a *default-heuristic* artifact (`BLOCK_M=32` at M≥24 is bad
  for N=256). A better upstream default for wna16/N≥256 would shrink the gap;
  worth a follow-up issue rather than per-shape JSONs.
- One stock-file change shipped: `benchmark_moe.py` int8_w8a16 path fixes
  (3 bugs above) — these make the tuner usable for *any* int8_w8a16 shape, not
  just ours.

## Artifacts / reproduction

```bash
# Tune (GPUs 1,2; ~2-3 h for the full decode sweep with crash-recovery):
PYTHONPATH=<worktree> /opt/vllm-env/bin/python3 /root/w8a16-35b/moe_tune_w8a16.py \
  --model /models/Qwen3.5-35B-A3B-GPTQ-8bit --tp-size 2 --use-int8 \
  --gpu-ids 1 2 --num-gpus 2 --batch-sizes 1 2 4 8 16 24 32 \
  --save-dir /root/w8a16-35b/moe_tuned_w8a16

# Kernel A/B + prefill-safety probe:
PYTHONPATH=<worktree> /opt/vllm-env/bin/python3 /root/w8a16-35b/moe_ab_probe.py
PYTHONPATH=<worktree> /opt/vllm-env/bin/python3 /root/w8a16-35b/moe_prefill_safety_probe.py

# Server A/B (gated config):
LAUNCH_TUNED_MOE=1 HIP_VISIBLE_DEVICES=1,2 LAUNCH_TP=2 LAUNCH_NO_COMPILE=1 \
  scripts/launch_w8a16_moe_tp4_c2.sh --serve-only
```

Build: vLLM `0.22.1rc1.dev466+g7133b783a` (happy-carrots, rebuilt gfx908), torch
`2.11.0+rocm7.2`, Triton `3.5.1`, ROCm 7.12, 4× MI100 gfx908 (runs pinned to
GPUs 1,2 for thermal).

---

*Generated 2026-06-10. AI assistance was used.*
