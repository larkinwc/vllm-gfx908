<!-- markdownlint-disable MD013 -->
# Hand-built int4 MoE decode kernel for gfx908: split-K MFMA — microbench WIN, integration NEGATIVE (2026-06)

Follow-up to `RESEARCH_MOE_HANDKERNEL_FEASIBILITY_2026_06.md`. We ran the #58
loop to completion **including integration**, which is where the story turns.

**Two-line verdict:**

1. **Microbench: WIN.** A `tl.dot`/MFMA kernel with split-K beats the in-tree
   `fused_moe_kernel_gptq_awq` at M=1 decode — 1.63× gate_up, 1.22× down, **1.51×
   combined GEMM** (44.7 → 29.6 µs/layer), head-to-head, same CUDA-graph harness,
   rel-err < 4e-4.
2. **Integration: NEGATIVE — reverted.** Wired into the real `fused_experts`, the
   win collapses to ~1.0× at M=1 and **regresses to 0.67–0.74× at M=4–8**. The
   atomic-accumulation's mandatory workspace `zero_()` (~7 µs fixed launch) ≈
   cancels the GEMM saving at M=1, and the baseline is no longer under-occupied
   at M≥2. **Not shipped; vLLM source reverted to pristine.** (Details in
   *Disposition*.)

The methodology was sound end-to-end; the honest outcome is that this particular
optimization does not net out on the real path. Both halves are recorded because
the *reason* it failed integration is the reusable lesson.

## Two designs — the first was wrong for the right-sounding reason

The diagnosis (`RESEARCH_MOE_HANDKERNEL_FEASIBILITY`) was correct: the in-tree
GEMM is **under-occupied — 80 workgroups on 120 CUs, VGPR=108 (~1 wave/CU)**.

**Iteration 1 (LOSS).** We copied #57's *choice* — a `tl.sum` GEMV with no
`tl.dot` — because that was the gfx900 win. On gfx908 it lost 0.63× and was
ALU-bound (29× above roofline at 42 GB/s) even at 1280 workgroups / VGPR=44.
**The mistake:** porting gfx900's *answer* (drop MFMA) instead of applying the
*method* (fix the diagnosed problem). gfx900 had no MFMA so a GEMV was right
there; gfx908 has MFMA, so dropping it throws away the matrix units.

**Iteration 2 (WIN).** Apply the method to the actual diagnosis: the problem is
*occupancy*, not the primitive. Keep `tl.dot` (MFMA), and **split the K
contraction into SPLIT_K partial GEMMs** so the grid becomes
`E_act × n_n_tiles × SPLIT_K` — turning 80 workgroups into 320 and filling the
idle CUs, while every tile still runs on the matrix units. Partials atomic-add
into the fp32 output.

## Results (CUDA-graph timed = true GPU µs)

Head-to-head vs the **real** in-tree kernel (`invoke_fused_moe_wna16_triton_kernel`),
same shapes, same harness, full 10-active-expert M=1 decode:

| proj | N | K | in-tree | hand MFMA+split-K | config | rel-err | speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| gate_up | 256 | 2048 | 34.1 µs | **20.9 µs** | BLOCK_N=64, BLOCK_K=32, SPLIT_K=8 | 3.8e-4 | **1.63×** |
| down | 2048 | 128 | 10.6 µs | **8.7 µs** | BLOCK_N=64, BLOCK_K=32, SPLIT_K=1 | 2.8e-4 | **1.22×** |
| **GEMM/layer** | | | **44.7 µs** | **29.6 µs** | | | **1.51×** |

(For reference, the discarded GEMV: 70.6 µs gate_up, 0.63× — a loss.)

## Why it wins — the occupancy fix, confirmed by profile

rocprof, hand v3 (gate_up) vs in-tree:

```text
in-tree : grid=20480  -> 80  workgroups, VGPR=108, dur=54 µs (rocprof) / 34 µs (graph)
hand v3 : grid=81920  -> 320 workgroups, VGPR=76,  dur=39 µs (rocprof) / 21 µs (graph)
```

Split-K does exactly what the diagnosis predicted: **80 → 320 workgroups** (fills
the previously-idle ⅓ of CUs) and **VGPR 108 → 76** (room for more waves/CU),
*while keeping MFMA*. The kernel is still ~8× above the HBM roofline (M=1 can't
be bandwidth-bound — the single token under-feeds the matrix units), but it now
extracts ~1.5× more of the available MFMA throughput than the un-split kernel.

## Correctness (#58 step 4) — PASS

Validated two ways: vs torch fp16 dequant+matvec reference (rel-err 2.4e-7), and
vs the **real in-tree kernel's output** across all 10 active experts (rel-err
3.8e-4 gate_up / 2.8e-4 down, < 1e-3 bar). Packing reproduced exactly: B uint8
`[E,N,K/2]`, low-nibble even-k / high-nibble odd-k, no zero-point, dequant
`(nibble-8)*scale`, scale fp16 `[E,N,K/gs]`.

## End-to-end ceiling (#58 step 9, honest)

- Isolated GEMM: 44.7 → 29.6 µs/layer (saves 15.1 µs).
- 48 MoE layers → saves ~0.72 ms of the 18.46 ms TPOT.
- **~3.9% lower TPOT ≈ ~3.9% more decode tok/s.**

Real and positive, but modest — the GEMM is only ~12% of decode, so even a 1.5×
GEMM can't move the whole model much. The routing/glue kernels (~37% of MoE GPU
time) remain the larger untouched pool, addressable only by fusion.

## Disposition

- **Integrated, measured, then REVERTED — the microbench win did not survive.**
  The split-K was wired into the in-tree `fused_moe_kernel_gptq_awq` itself (the
  kernel already had an unused `SPLIT_K` constexpr + `tl.dot`; the change made it
  honor split-K via a `program_id(1)` axis + `tl.atomic_add`, gated on
  `on_mi100()` + M≤8 + int4 + a tall-K/small-N shape test, SPLIT_K=1 byte-
  identical everywhere else). Correctness passed (rel-err 9.5e-4 vs baseline).
  **But on the full `fused_experts`:**

  | M | baseline | split-K | result |
  |---|---:|---:|---|
  | 1 | 66.4 µs | 65.2 µs | 1.02× (noise) |
  | 2 | 107.9 µs | 109.4 µs | 0.99× |
  | 4 | 141.1 µs | 190.3 µs | **0.74× regress** |
  | 8 | 241.0 µs | 360.0 µs | **0.67× regress** |

  Root causes (rocprof confirmed): (a) the atomic accumulation needs the reused
  workspace pre-zeroed, and a separate `C.zero_()` costs a **fixed ~7 µs launch**
  that ≈ cancels the ~5 µs the gate_up GEMM actually saved in-context (the
  isolated 1.5× shrank to ~7% once the real bs=1 block config + the unaffected
  down-proj are included); (b) at M≥2 the baseline already fills the CUs (more
  tokens ⇒ more M-tiles), so split-K only adds redundant grid + zeroing + atomic
  traffic ⇒ growing regression. Net: **not worth wiring in.** The vLLM source is
  reverted to pristine.
- Kernels + benches kept: `bench_scripts/moe_int4_gemv_bench.py` (all designs +
  sweep), `bench_scripts/moe_int4_headtohead.py` (vs real in-tree kernel),
  `bench_scripts/moe_splitk_integration_test.py` (the integration A/B that
  produced the table above).

### The deeper lesson: isolated wins must clear the integration tax

The 1.5× isolated-GEMM result was real, but an isolated microbench omits three
real costs that erased it: the **workspace-zeroing launch** the atomic needs,
the **unaffected sibling GEMM** (down-proj), and the **M≥2 regime** where the
baseline is no longer under-occupied. On gfx908 at M=1 the absolute GEMM slice is
small (~5 µs/layer) and **launch overhead dominates** — so any optimization that
*adds* a launch (here, `zero_()`) tends to net out flat. This is the same
"launch-overhead-bound at M=1" wall seen in the tuner (eager 800 µs/op) and the
routing-glue kernels.

## The portable lesson

On gfx908, when the #58 diagnosis says **under-occupancy** for an MFMA kernel,
the fix is **split-K (and/or more N-tiles) to fill the CUs while keeping
`tl.dot`** — not dropping MFMA. Only drop `tl.dot` when the chip lacks MFMA
(gfx900). Same methodology, opposite kernel choice. The "verified negative" from
iteration 1 was a negative for *one design*, not for the approach.

## Reproduction

```bash
cd <compiled-worktree>
PYTHONPATH=. HIP_VISIBLE_DEVICES=2 \
  /opt/vllm-env/bin/python3 bench_scripts/moe_int4_gemv_bench.py     # all designs
PYTHONPATH=. HIP_VISIBLE_DEVICES=2 \
  /opt/vllm-env/bin/python3 bench_scripts/moe_int4_headtohead.py     # vs in-tree
```

Build: vLLM `0.20.2rc1.dev107+gd960f21e4`, torch `2.11.0+rocm7.2`, Triton
`3.5.1`, ROCm 7.12, 4× MI100 gfx908. Model `/models/Qwen3-Coder-Next-AWQ-4bit`.

---

*Generated 2026-06-03. AI assistance was used.*
