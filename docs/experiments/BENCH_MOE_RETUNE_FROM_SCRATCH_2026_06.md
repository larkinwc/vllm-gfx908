<!-- markdownlint-disable MD013 -->
# Re-tuning the W4A16 MoE kernel from scratch: no headroom on gfx908 (2026-06)

Follow-up to `BENCH_MOE_TUNED_CONFIG_NEGATIVE_2026_06.md`, which found the
inherited `Arcturus_GL-XL` MoE configs *regress* −38% vs Triton defaults on our
stack. Open question left there: **would a from-scratch tune on Triton 3.5.1 /
ROCm 7.12 actually beat the default?** This note answers it: **no — there is no
meaningful tuning headroom for the int4_w4a16 MoE kernel at decode sizes.**

## Two blockers had to be solved first

### 1. The stock tuner crashes ~50 configs in (uncatchable C++ abort)

`benchmarks/kernels/benchmark_moe.py` dies mid-search on this stack:
```
at::cuda::CUDAGraph::~CUDAGraph() -> c10_cuda_check_implementation() -> abort()
ray.exceptions.ActorDiedError: ... Worker exit type: SYSTEM_ERROR
```
Ruled out, by experiment:
- **Not a single bad config** — every individual tile (incl. `BLOCK_K=256`,
  `num_warps=1`) runs fine in isolation (~450 us).
- **Not OOM, not the device-guard, not the cache-clear interval** — the crash
  index drifts (42–99) with those held constant, and reproduces single-GPU
  *without* Ray.
- **Root cause: cumulative GPU-context corruption.** Sustained Triton MoE
  JIT+exec in one long-lived process eventually triggers an unrecoverable HIP
  fault, surfaced in a C++ destructor that calls `abort()`. Python `try/except`
  cannot catch it; CUDA-graph capture defers the fault to `~CUDAGraph()`.

### 2. Subprocess-isolated harness (the fix)

Built `/root/fp16-bench/moe_tune_isolated.py`: a parent orchestrator that tunes
each batch size in short-lived **worker subprocesses**, each of which
- times configs eagerly (new `VLLM_MOE_TUNE_NO_CUDAGRAPH=1` path added to
  `benchmark_moe.py` so faults are catchable `RuntimeError`s, not graph-dtor
  aborts),
- checkpoints every `(idx, time, config)` to a JSONL (flushed per config),
- `os._exit(0)`s to dodge the gfx908 CUDA-teardown segfault.

On a worker crash the parent reads the checkpoint and respawns a fresh worker
from `last_done+1` (a fresh process gets ~50–99 more configs); an index that
kills two fresh workers with zero progress is recorded as a hard-crash skip.
Validated end-to-end: M=1 (304 configs) tunes to a correct, runtime-named JSON.

## Result: tuned ≈ default (within ±2% = noise)

Completed batch sizes M=1,2,4 (decode-relevant; M = concurrency × topk is small
for decode). Direct kernel A/B, tuned-best vs Triton default heuristic, 20 iters:

| M | default | tuned-best | tuned/default | verdict |
|---|---:|---:|---:|---|
| 1 | 452.1 us | 461.5 us | 1.021 | tuned 2.1% **slower** |
| 2 | 463.6 us | 455.8 us | 0.983 | tuned 1.7% faster |
| 4 | 446.0 us | 453.4 us | 1.017 | tuned 1.7% **slower** |

The best achievable time is **flat at ~440–460 us across M=1,2,4 regardless of
tile config** — and the per-M sweep itself showed the same flatness (best times
within a few us across hundreds of tile shapes).

## Why there's no headroom

The `int4_w4a16` decode kernel at small M is **fixed-overhead-bound, not
tile-efficiency-bound**:
- 512 experts with top-10 routing → the gather/scatter + per-expert dispatch and
  the wna16 dequant path dominate the ~450 us, and those costs are independent
  of `BLOCK_SIZE_*` / `num_warps`.
- The actual per-expert GEMM at M≤4 is tiny (a handful of rows), so tile tuning —
  which only changes GEMM efficiency — has almost nothing to optimize.
- vLLM's default heuristic already does the one thing that matters here: set
  `BLOCK_SIZE_M = min(16, M)` for the decode regime and let the wna16 kernel
  pick N/K. (`get_default_config`, `fused_moe.py:1223`.)

This is consistent with the earlier full-server finding: the inherited hand-tuned
configs *lost* to the default by −38% (their larger-M tiles register-spill),
and now we see even an exhaustive correct re-tune only **matches** the default
at the sizes that matter.

## Decision

- **Do not ship any tuned int4_w4a16 MoE config.** The default is already optimal
  (±2%) for decode; a tuned file can only risk regressions (as the inherited one
  did) for zero upside.
- **Stopped the full sweep** of the large prefill batch sizes (M≥512, ~5k–7k
  configs each, multi-hour): they are not decode-relevant, and the kernel
  flatness + the −38% prior result make a win there implausible and not worth
  the GPU-hours.
- The real MoE-throughput lever on gfx908 is **not** Triton tile tuning — it is
  the fixed per-expert routing/dequant overhead (a kernel-algorithm problem,
  e.g. a fused gather+dequant+GEMM or an AITER path), which is out of scope for
  config tuning.

## Artifacts / reproduction

- Harness: `/root/fp16-bench/moe_tune_isolated.py` (kept for future use; e.g.
  re-tuning a *different* dtype/shape where headroom may exist).
- One stock-file change: `benchmarks/kernels/benchmark_moe.py` gains an opt-in
  `VLLM_MOE_TUNE_NO_CUDAGRAPH=1` eager-timing path (no behavior change when
  unset) so the tuner is usable at all on gfx908.
- Checkpoints: `/root/fp16-bench/moe_tuned/ckpt_M{1,2,4}.jsonl` (complete),
  others partial.
- Tuned-vs-default kernel A/B is reproducible with the inline probe in the
  session log (eager timing, `VLLM_MOE_TUNE_NO_CUDAGRAPH=1`, GPU 2).

Build: vLLM `0.20.2rc1.dev107+gd960f21e4`, torch `2.11.0+rocm7.2`, Triton
`3.5.1`, ROCm 7.12, 4× MI100 gfx908. Model `/models/Qwen3-Coder-Next-AWQ-4bit`
(`qwen3_next`, E=512/top-10, W4A16 gs=32; per-shard E=512, N=128 at TP=4).

---

*Generated 2026-06-03. AI assistance was used.*
