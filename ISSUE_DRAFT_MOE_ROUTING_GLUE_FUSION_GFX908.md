<!-- markdownlint-disable MD013 -->
# Issue draft: [gfx908] Fuse MoE routing/glue kernels to cut decode overhead

> Ready-to-file issue for the `larkinwc/vllm-gfx908` fork. Candidate #2 from the
> `PERF_GFX908.md` decode-triage. File after the split-K GEMM work concludes.

## Summary

On gfx908 (MI100) W4A16 MoE decode (M=1), the int4 expert **GEMM is only ~63%**
of the per-layer MoE GPU time; the remaining **~37% is a chain of small
"routing/glue" kernels** that are launch/overhead-bound, not compute-bound. These
are the next structural target after the GEMM. This issue proposes fusing them
to cut the fixed per-layer overhead.

## Evidence (from this fork's decode profiling)

rocprof `--kernel-trace` of the real `fused_experts` at M=1 (Qwen3-Coder-Next,
E=512/top-10, H=2048, N=128/shard at TP=4, gs=32), excluding `torch.randn`
harness noise — per layer, per token:

| kernel | µs/iter | share | role |
|---|---:|---:|---|
| `fused_moe_kernel_gptq_awq` | ~57 | ~63% | int4 expert GEMM (separate effort) |
| `reduce_kernel` (at::native) | ~12 | ~13% | routing softmax / topk-weight reduce |
| `moe_align_block_size_kernel` | ~10 | ~11% | sort tokens by expert + pad to block |
| `act_and_mul_kernel` | ~5 | ~6% | SiLU(gate) * up |
| `count_and_sort_expert_tokens_kernel` | ~4.6 | ~5% | routing histogram/sort |
| `topkGating` | ~1.4 | ~2% | router top-k |

The glue total (~33 µs) is **~37% of the ~90 µs MoE GPU time per layer**. With 48
MoE layers and an ~18.5 ms TPOT, the glue is **48 × 33 µs ≈ 1.6 ms ≈ ~8.6% of
decode** — a larger end-to-end pool than the GEMM's reclaimable slice.

Why they're expensive at M=1: each is a tiny kernel (1 token × top-10) whose
runtime is dominated by **launch latency**, not work. The fixed ~5–12 µs each
pays the gfx908 kernel-launch overhead repeatedly.

## Proposed work

Apply the `PERF_GFX908.md` evidence-driven loop, but the lever here is **fusion**
(fewer launches), not split-K (occupancy). Candidate fusions, in rough order of
expected value / risk:

1. **Fold `act_and_mul` (SiLU·up) into the gate_up GEMM epilogue.** The gate_up
   kernel already produces the [.., 2N] output; doing SiLU·mul in the epilogue
   before writing `intermediate_cache2` removes one full kernel launch (~5 µs)
   and one round-trip of the intermediate to HBM. Lowest risk; self-contained in
   the Triton kernel.
2. **Fuse `moe_align_block_size` + `count_and_sort_expert_tokens`.** These two
   (~15 µs combined) both walk the top-k assignment; they can likely share a
   single pass that produces sorted_token_ids, expert_ids and the padded counts
   together. Medium risk (indexing correctness).
3. **Fold the routed-weight `reduce_kernel` into the down-proj epilogue.** The
   final `* topk_weight` + expert-sum reduction (~12 µs) can be done in the
   down-GEMM epilogue / store, avoiding a separate reduce launch.

Each fusion must:
- Keep a non-gfx908 fallback (gate on `on_mi100()` or make the fused kernel arch-
  neutral and just default-on if it's a pure win everywhere).
- Pass correctness vs the current multi-kernel path (rel-err ≤ 1e-3) across M =
  1, 2, 4, 8 and across expert counts.
- Show a net **CUDA-graph-timed** win on full `fused_experts` (isolated-kernel
  wins must survive integration — see the GEMM split-K cautionary tale below).

## Critical lesson from the GEMM attempt (read before starting)

The sibling split-K GEMM effort got a **1.5× isolated-GEMM microbench win that
did NOT survive integration** (net ~1.0× at M=1, regressions at M≥2). Root
causes that will also threaten this work:
- **Launch overhead is the enemy at M=1.** Any *added* kernel (e.g. a separate
  zeroing or reduction pass) costs a fixed ~7 µs launch that can erase the win.
  Fusion is attractive precisely because it *removes* launches — but do not
  replace one launch with two.
- **Validate across M, not just M=1.** The decode batch is M = concurrency; a
  win at M=1 that regresses at M=4/8 is a net loss under real serving.
- **Measure full `fused_experts`, CUDA-graph-timed**, never eager (eager dispatch
  ≈ 800 µs/op swamps everything) and never the isolated kernel alone.

## Done when

- One or more glue kernels are fused, correctness-validated, and show a net
  CUDA-graph-timed win on `fused_experts` across M = 1–8, **and** an end-to-end
  decode tok/s A/B on the real model confirms it (see the harness in
  `PERF_GFX908.md` §4).
- Or a documented negative result explaining why fusion didn't net out (as we did
  for tuning and the GEMM split-K).

## References in-repo
- `PERF_GFX908.md` — hardware model, the loop, the decode triage (§6).
- `BENCH_MOE_HANDKERNEL_GEMV_2026_06.md` — the GEMM split-K story (microbench win
  + integration reality).
- `bench_scripts/moe_int4_gemv_bench.py`, `bench_scripts/decode_profile.py` —
  microbench + whole-model profiler templates.

---

*Draft prepared 2026-06-03. AI assistance was used.*
