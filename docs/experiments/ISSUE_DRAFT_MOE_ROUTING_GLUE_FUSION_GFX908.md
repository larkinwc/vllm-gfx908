<!-- markdownlint-disable MD013 -->
# Issue draft: [gfx908] Fuse MoE routing/glue kernels to cut decode overhead

> Ready-to-file issue for the `larkinwc/vllm-gfx908` fork. Candidate #2 from the
> `PERF_GFX908.md` decode-triage.
>
> **STATUS: DE-PRIORITIZED (2026-06).** A graph-timed feasibility pass (below)
> shows the realistic end-to-end ceiling is **~1% TPOT per fusable kernel**, with
> real risk of slowing the 66% GEMM. Filed for the record, not recommended as
> active work unless the cost model changes. Read the "Reality check" section
> before starting.

## Reality check — graph-timed ceilings (READ FIRST)

The original motivation (below) used an **eager-mode** rocprof, which is inflated
by per-kernel launch gaps (~7 µs each on gfx908). **Real serving runs under CUDA
graphs, which eliminate those gaps.** Re-measured graph-timed (the numbers that
actually matter), per layer at M=1:

| kernel | µs/layer (graph) | % of MoE | 48-layer E2E ceiling | fusability |
|---|---:|---:|---:|---|
| `fused_moe_kernel_gptq_awq` (GEMM) | 43.7 | 66% | — | already optimal (split-K failed) |
| `moe_align_block_size` | 7.2 | 11% | ~1.9% TPOT | C++; scans all 512 experts; hard |
| `reduce_kernel` | 4.9 | 7% | ~1.3% TPOT | maybe → down-GEMM epilogue |
| `act_and_mul` | 4.4 | 7% | ~1.1% TPOT | **structurally hard** (gate/up tiles N apart) |
| `count_and_sort_expert_tokens` | 4.3 | 7% | ~1.1% TPOT | C++ |
| **total glue** | **~21** | **34%** | **~5.4% TPOT (unreachable)** | — |

Key facts that sink the value case:

- **act_and_mul can't sit in the gate_up GEMM epilogue cheaply.** `silu_and_mul`
  computes `out[i] = silu(c1[i]) * c1[i+N]`; the gate half (cols 0..N) and up half
  (cols N..2N) live in *different* N-tiles, so a program computing one tile lacks
  the matching tile. Fusing requires each program to compute *both* tiles (two
  `tl.dot` regions) — a real GEMM restructure that risks slowing the 66% kernel
  for a ~1.1% glue saving.
- **Launch overhead is already hidden by graphs.** The isolated `silu_and_mul`
  is ~2 µs of true GPU time; the ~7 µs seen eagerly was launch latency that
  cudagraph_mode removes for free in production. There is no "launch tax" left to
  reclaim by fusion in the graphed regime.
- **Each remaining target nets ~1% TPOT** and several are deep C++ (`moe_align`,
  `count_and_sort`) that scan all 512 experts. The integration-tax lesson from
  the split-K GEMM (a real micro-saving erased on the full path) applies here at
  even smaller absolute scale.

**Recommendation:** leave MoE decode as-is. The GEMM is optimal and the glue is
small under graphs. Pursue this only if (a) a higher-concurrency regime shifts
the glue share materially, or (b) a single fused align+count+act kernel can be
shown to net > a few % without touching the GEMM tiles.

---

## Original motivation (eager-profile; superseded by the reality check above)

On gfx908 (MI100) W4A16 MoE decode (M=1), an eager rocprof suggested the int4
expert **GEMM is ~63%** of per-layer MoE GPU time with **~37% routing/glue**.
That framing over-counts the glue because eager timing includes launch gaps that
graphs remove (see above). Retained for context:

| kernel | µs/iter (eager) | share | role |
|---|---:|---:|---|
| `fused_moe_kernel_gptq_awq` | ~57 | ~63% | int4 expert GEMM (separate effort) |
| `reduce_kernel` (at::native) | ~12 | ~13% | routing softmax / topk-weight reduce |
| `moe_align_block_size_kernel` | ~10 | ~11% | sort tokens by expert + pad to block |
| `act_and_mul_kernel` | ~5 | ~6% | SiLU(gate) * up |
| `count_and_sort_expert_tokens_kernel` | ~4.6 | ~5% | routing histogram/sort |
| `topkGating` | ~1.4 | ~2% | router top-k |

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

## Done when (if ever revived)

- One or more glue kernels are fused, correctness-validated, and show a net
  CUDA-graph-timed win on `fused_experts` across M = 1–8, **and** an end-to-end
  decode tok/s A/B on the real model confirms it (see the harness in
  `PERF_GFX908.md` §4).
- Or a documented negative result explaining why fusion didn't net out (as we did
  for tuning and the GEMM split-K).

Given the ~1% ceilings, the only version worth attempting is a **single fused
align+count(+act) kernel** that removes multiple launches at once *without*
touching the GEMM tiles — and only if a quick prototype clears a few % net.

## References in-repo

- `PERF_GFX908.md` — hardware model, the loop, the decode triage (§6).
- `BENCH_MOE_HANDKERNEL_GEMV_2026_06.md` — the GEMM split-K story (microbench win + integration reality).
- `bench_scripts/moe_int4_gemv_bench.py`, `bench_scripts/decode_profile.py` —
  microbench + whole-model profiler templates.

---

*Draft prepared 2026-06-03. AI assistance was used.*
