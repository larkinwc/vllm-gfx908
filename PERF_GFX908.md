<!-- markdownlint-disable MD013 -->
# PERF_GFX908 — living guidance for gfx908 (MI100) perf research

> **Living doc.** This is the gfx908 analog of the gfx900 `PERF_GFX900.md` /
> issue #58 methodology. It captures (a) the hardware cost model, (b) the
> evidence-driven loop we use, (c) environment gotchas that will otherwise eat
> hours, and (d) a running ledger of what has been tried and the verdict.
> **Update it whenever a kernel/optimization is investigated.**

---

## 0. TL;DR for the next investigation

1. **Profile before coding.** rocprofv3 `--kernel-trace` the *real* model shape;
   get per-kernel µs + launch geometry (grid/block/VGPR). Never optimize a
   kernel you haven't seen dominate a real trace.
2. **Roofline-gate it.** bytes-moved / 1.2 TB/s = floor. If the kernel is within
   ~5× of the floor it's bandwidth-bound — stop, there's nothing to win. If it's
   10–100× above, find out *why* (occupancy? ALU? overhead?) before assuming a
   rewrite helps.
3. **gfx908 HAS MFMA — keep it, fix occupancy instead.** Do not port gfx900's
   "drop `tl.dot`, use a GEMV at M=1" *kernel choice*; a pure GEMV is ALU-bound
   and loses here (0.63×, see §5). But the #58 *method* still works: the in-tree
   int4 MoE GEMM at M=1 is **under-occupied** (80 WG / 120 CUs), and a hand
   **`tl.dot`/MFMA + split-K** kernel that fills the CUs **wins 1.5×** on the
   isolated GEMM. Same diagnosis, opposite primitive vs gfx900.
4. **CUDA-graph timing only.** Eager Python dispatch is ~800 µs/op on this stack
   and will swamp real kernel times of 10–90 µs. Use a captured graph (see
   `time_us_graph` in `bench_scripts/moe_int4_gemv_bench.py`).
5. **Tile-tuning the int4 MoE kernel is a dead end** at decode sizes (±2% =
   noise; see §5) — but a **structural rewrite is not**: split-K to fix
   occupancy beats it 1.5× (§5). Tuning reshuffles tiles within a kernel that
   can't fill the machine; only changing the launch structure does.

---

## 1. Hardware cost model (MI100 / gfx908 / CDNA1)

| property | value | implication |
|---|---|---|
| Compute Units | **120 CUs** | a kernel needs ≥120 workgroups (ideally several×) to fill the machine; <120 leaves CUs idle |
| Matrix units | **MFMA present** (CDNA1) | `tl.dot` lowers to real matrix instructions — fast. This is the #1 difference from gfx900 (Vega10, no MFMA) |
| Wavefront | **64 lanes** (wave64) | `num_warps` max 16 (32 is invalid); reductions are over 64-lane waves |
| HBM bandwidth | **~1.2 TB/s** | the roofline denominator for every memory-bound kernel |
| VRAM | 32 GB ×4 | TP=4 is the standard layout for big models |
| dtype | **fp16 only** (no bf16 kernels) | always launch servers/benches with `--dtype float16` |
| VGPR/occupancy | 256 VGPR/lane budget | VGPR=108 → ~1 wave/CU; VGPR≤64 → 4+ waves/CU. Watch this in rocprof |

**Roofline reflex:** for a weight-bound op, bytes ≈ weight bytes (+ scales +
activations). int4 = K·N/2 bytes; fp16 scales = K·N/gs·2 bytes. Divide by 1.2e12,
multiply by 1e6 → µs floor. Compare to measured. Record the ratio.

---

## 2. The evidence-driven loop (gfx908 form of #58)

1. **Find the hot kernel.** `rocprofv3 --kernel-trace --output-format csv` on a
   short script that runs the *real* shapes; aggregate duration by kernel name.
   Confirm it's on the decode path (M small), not prefill.
2. **Roofline check.** Bytes / 1.2 TB/s. >5–10× above floor ⇒ potential; near
   floor ⇒ stop.
3. **Diagnose WHY it's above floor** (this is the gfx908-specific step that
   decides whether a rewrite can help):
   - **Launch geometry** from the same trace: workgroups = grid/block. <120 ⇒
     under-occupancy. VGPR high ⇒ occupancy-capped.
   - **Scaling probe:** vary the parameter you suspect (e.g. #experts, batch) and
     see if time tracks it. Flat ⇒ fixed-overhead-bound; linear ⇒ work-bound.
   - **Primitive check:** is it ALU-bound (reduction/dequant) or memory-bound?
     If GB/s ≪ 1.2 TB/s *and* occupancy is already high, it's ALU/primitive-bound
     and a structural rewrite won't reach roofline.
4. **Reproduce in isolation** with real shapes + correct packing.
5. **Correctness FIRST** — vs torch ref AND vs the in-tree kernel you'd replace;
   rel-err ≤ 1e-3.
6. **Iterate structure, then tune** — occupancy (fill 120 CUs), split-K for
   tall-K/small-N, VGPR reduction, dtype. Keep the correctness check in the loop.
7. **Compare against the RIGHT baseline** — the in-tree kernel's true GPU time
   (CUDA-graph timed / rocprof), not eager wall-clock and not just FP16.
8. **Wire in behind a narrow gate** — `on_gfx908()` + the M/shape regime where it
   wins; fall back everywhere else (reversible, other arches untouched).
9. **Validate end-to-end** — real model, output correctness, decode tok/s A/B.
   Microbench wins must survive integration.

**End-to-end ceiling math (do it at step 3, not step 9):** kernel µs × layers /
TPOT µs = the fraction of decode you can touch. If the hot kernel is 15% of
decode, the *best possible* whole-model win is 15%. Decide if that's worth it
before building.

---

## 3. Environment gotchas (these will eat hours)

- **`import vllm` fails by default.** The editable `.pth` points at a deleted
  worktree. Fix: `cd` into a *compiled* worktree and set `PYTHONPATH=.` for
  **every** python invocation (benches, servers, profilers). Primary compiled
  build: `fuzzy-hornets-see-szfl4` = `0.20.2rc1.dev107+gd960f21e4`.
- **CUDA teardown segfaults** (ROCm 7.12 / torch 2.11: `c10::cuda::SetDevice` in
  `__del__`). End scripts with `os._exit(0)` — **except** when a tool must flush
  on atexit (rocprof, file writers): then exit naturally (`sys.stdout.flush()`),
  or rocprof's CSVs will be empty/truncated.
- **Eager dispatch ≈ 800 µs/op.** Always CUDA-graph-time kernels in the 10–100 µs
  range or the measurement is pure Python overhead. (This is why the MoE tuner
  saw everything "flat at ~450 µs" — graphs gave the real ~57 µs.)
- **Triton tuner HIP-abort.** Sustained Triton JIT+exec in one long process
  eventually hits an uncatchable C++ `abort()` (surfaces in `~CUDAGraph()`);
  cumulative GPU-context corruption, ~50–99 configs in. Work around with the
  subprocess-isolated harness `/root/fp16-bench/moe_tune_isolated.py` (short-lived
  workers, checkpoint per config, respawn from last index). Add
  `VLLM_MOE_TUNE_NO_CUDAGRAPH=1` (eager-timing path in `benchmark_moe.py`) so
  faults are catchable `RuntimeError`s.
- **Servers/zombies:** launch via `nohup` + `fireAndForget`; kill by **exact
  PID** (pkill patterns catch the launching shell). The Execute tool rejects
  background commands containing `rm -rf` or pipe-to-sh — wrap such logic in a
  `.sh` file first.
- **MoE config device-name key:** runtime builds the lookup key as
  `AMD_Instinct_MI100`; inherited config files named `Arcturus_GL-XL_...` never
  load (silent default fallback). If you *do* install configs, name them to match
  (`fused_moe.py:1022`). Note the mismatch was historically *protecting* us — see
  §5.
- **rocprof parsing:** columns are `Kernel_Name`, `Start_Timestamp`,
  `End_Timestamp` (ns), `Grid_Size_X`, `Workgroup_Size_X`, `VGPR_Count`,
  `LDS_Block_Size`, `Scratch_Size`. workgroups = grid/block.

---

## 4. Standard benchmark harness

```bash
# serve A/B (decode-heavy)
bench serve --num-prompts 200 --request-rate inf --seed 42 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --ignore-eos
# server: cd into compiled worktree, --dtype float16, nohup + fireAndForget
```

Reusable artifacts in `/root/fp16-bench/`: `moe_tune_isolated.py`,
`run_qmoe_config_ab.sh`, compile A/B runners, result dirs.
Microbench template: `bench_scripts/moe_int4_gemv_bench.py` (correctness +
CUDA-graph timing + roofline + occupancy).

---

## 5. Ledger — what's been tried on gfx908 (verdicts)

| date | investigation | result | doc |
|---|---|---|---|
| 2026-06 | **torch.compile, dense 9B fp16** | +1.81% geomean; TP=4 +4.5/+7.2%, TP=1 −3% ⇒ concurrency-gate | `BENCH_FP16_COMPILE_PORT_0202_*` |
| 2026-06 | **torch.compile, fp16 35B-A3B MoE (TP=4)** | **+10.14%** geomean, no c=1 regression ⇒ enable | `BENCH_FP16_COMPILE_MOE_*` |
| 2026-06 | **torch.compile, W4A16 512e MoE** | **+9.85%** geomean, TTFT −28/−49/−32% ⇒ enable | `BENCH_FP16_COMPILE_QUANT_MOE_*` |
| 2026-06 | **Dynamo guard fix** (skip fused-silu-quant stash under compile) | unblocks compile on 0.20.2; zero effect unless `VLLM_MI100_TORCH_COMPILE=1` | committed `24ac8f64e` |
| 2026-06 | **Inherited MoE tuned configs** (`Arcturus_GL-XL`) | **−38% geomean** if correctly named; never loaded due to device-name mismatch (accidental protection) ⇒ do not install | `BENCH_MOE_TUNED_CONFIG_NEGATIVE_*` |
| 2026-06 | **Re-tune int4 MoE from scratch** (M=1,2,4) | tuned ≈ default ±2% = noise; kernel is fixed-overhead-bound at decode ⇒ no headroom | `BENCH_MOE_RETUNE_FROM_SCRATCH_*` |
| 2026-06 | **Hand int4 MoE-GEMV** (drop `tl.dot`, `tl.sum` reduction) | **0.63× LOSS**; ALU-bound even at 1280 WG / VGPR=44 ⇒ wrong design (copied #57's choice, not its method) | `BENCH_MOE_HANDKERNEL_GEMV_*` |
| 2026-06 | **Hand int4 MoE GEMM, MFMA + split-K** (keep `tl.dot`, fix occupancy) | **microbench 1.51× WIN** isolated GEMM (1.63× gate_up, 1.22× down), rel-err <4e-4; 80→320 WG, VGPR 108→76 — **BUT integration NEGATIVE: ~1.0× at M=1, 0.67–0.74× regress at M=4–8** (atomic needs `zero_()` ~7 µs launch ≈ cancels saving; baseline already CU-filled at M≥2). **Reverted, not shipped.** | `BENCH_MOE_HANDKERNEL_GEMV_*` + `RESEARCH_MOE_HANDKERNEL_FEASIBILITY_*` |

### Distilled rules from the ledger
- **torch.compile is the biggest realized win on gfx908** (~10% on MoE serving).
  Enable for MoE; concurrency-gate for dense.
- **Don't tile-tune the int4 MoE decode GEMM** (±2% = noise). Split-K *does* win
  1.5× on the **isolated** GEMM, but it **does not survive integration** (the
  atomic's `zero_()` launch cancels the saving at M=1, and it regresses at M≥2
  where the baseline already fills the CUs). Net: leave the in-tree kernel as-is.
- **Isolated microbench wins must clear the "integration tax."** Before
  celebrating a kernel microbench, account for: (a) any *added* launch the
  integration needs (zeroing/reduction) — a fixed ~7 µs on gfx908, often fatal at
  M=1; (b) sibling ops the change doesn't help; (c) the M≥2 regime. Always finish
  with a **CUDA-graph-timed full-`fused_experts` A/B across M=1–8**, not the
  isolated kernel.
- **MFMA is the dividing line vs gfx900 — port the *method*, not the *choice*.**
  On gfx900 (no MFMA) the M=1 win was a GEMV (drop `tl.dot`). On gfx908 the M=1
  win is the *opposite kernel* (keep `tl.dot`, add split-K) reached by the *same
  diagnosis* (under-occupancy). A pure GEMV here is ALU-bound and loses 0.63×.
- **Always re-diagnose, never transplant.** The trap that cost an iteration:
  assuming #57's kernel *choice* was the answer instead of re-deriving the fix
  from the gfx908 profile. The diagnosis (occupancy) was portable; the kernel
  was not.

### Open / unexplored (candidate next work)
- **MoE routing-glue fusion — DE-PRIORITIZED (2026-06).** The "~37% glue" was an
  *eager* rocprof inflated by launch gaps. **Graph-timed** (real serving), per
  layer at M=1: GEMM 43.7 µs (66%), `moe_align` 7.2 µs (11%), `reduce` 4.9 µs
  (7%), `act_and_mul` 4.4 µs (7%), `count_and_sort` 4.3 µs (7%) — glue ~21 µs
  (34%) of true GPU time. End-to-end ceilings (48 layers / 18.5 ms TPOT): each
  fusable kernel ≈ **~1% TPOT**, all glue ≈ 5.4% (unreachable). And `act_and_mul`
  won't sit in the gate_up epilogue cheaply — gate (cols 0..N) and up (cols
  N..2N) are in different N-tiles, so fusion means a GEMM restructure that risks
  the 66% kernel. Verdict: not worth it. Details in
  `ISSUE_DRAFT_MOE_ROUTING_GLUE_FUSION_GFX908.md` (Reality-check section).
- **AITER / CK paths** for the int4 MoE GEMM (if available for this shape) —
  would beat Triton-MFMA only if they target the M=1 padding waste in hardware.

---

## 6. Whole-model decode triage — where the hand-kernel candidates are

Decode-step rocprof of a **dense** model (Llama-2-7B fp16, TP=1, 128 decode
steps; corrected 3D-grid geometry) ranks every per-token kernel and answers
"which other paths are worth hand-building?":

| kernel | % decode | total WG | VGPR | verdict |
|---|---:|---:|---:|---|
| `wvSplitK_hf_sml` (ROCm skinny GEMM, the dense decode matmul) | **40.8%** | 1920 | 64 | **already optimal** — ROCm's split-K-to-fill-CUs C++ kernel; full occupancy |
| `Cijk_*` (rocBLAS/Tensile GEMMs) | ~26% (sum) | 16–192 | 128–256 | **prefill** tiles, not decode hot path |
| `paged_attention_ll4mi_QKV` | 2.3% | 8192 | 84 | well-occupied; memory-bound |
| `reshape_and_cache` | 2.1% | 256 | 32 | tiny (3.4 µs), well-occupied |
| `paged_attention_ll4mi_reduce` | 1.1% | 8192 | 68 | well-occupied |
| `act_and_mul`, `rotary_embedding` | ~2% | — | — | tiny elementwise, fused already |

**The decisive insight:** the dense decode GEMM uses **`wvSplitK`** — a ROCm
C++ kernel that *takes `cu_count` as an argument* and split-K's specifically to
fill all 120 CUs (there is also `wvSplitKQ` for quantized). So the dense fp16 /
W4A16 / W8A8 matmul paths **already have the exact optimization** we hand-built
for MoE — they are not under-occupied. The MoE win existed only because the
Triton `fused_moe_kernel_gptq_awq` does **not** route through `wvSplitK` (it's a
generic `tl.dot` tile kernel) and so launched just 80 workgroups.

### Candidate ranking for hand-built kernels (by remaining structural headroom)

1. **DONE (negative) — int4 MoE decode GEMM** → split-K MFMA won 1.51×
   isolated but did NOT survive integration (§5). Closed.
2. **DE-PRIORITIZED — MoE routing-glue fusion.** Graph-timed, the glue is ~34%
   of MoE GPU time but each fusable kernel caps at **~1% TPOT** end-to-end, and
   `act_and_mul` fusion would require restructuring the 66% GEMM (gate/up tiles
   are N apart). The launch overhead the eager profile attributed to glue is
   already hidden by cudagraphs in production. Closed unless the cost model
   shifts. Full reality-check: `ISSUE_DRAFT_MOE_ROUTING_GLUE_FUSION_GFX908.md`.
3. **NOT WORTH IT — dense decode GEMM** (fp16/W4A16/W8A8): already `wvSplitK`,
   full occupancy. A hand kernel would have to beat ROCm's tuned C++; expected
   loss. (Consistent with prior `BENCH_W4A16_*` notes finding dense quant
   near-optimal and Marlin-repack a regression.)
4. **NOT WORTH IT — paged attention / reshape_cache / rope / act_and_mul**:
   well-occupied and either memory-bound or trivially small; no roofline gap.

### Triage rule (add to the loop)
> Before hand-building any decode kernel, check whether it already routes through
> **`wvSplitK`/`wvSplitKQ`** (dense GEMM) or a well-occupied attention kernel.
> If it does, it's already CU-filled — stop. The hand-kernel opportunity on
> gfx908 is specifically **Triton tile kernels that bypass `wvSplitK`** (MoE,
> and any future custom-quant path) where the launch geometry starves the CUs.

Reproduction: `bench_scripts/decode_profile.py <model> <tp>` under
`rocprofv3 --kernel-trace`. (Note: Qwen tokenizers trip a transformers-v4
validation bug in the in-process `LLM()` API on this stack — use a
Llama-family model for the dense triage, or profile Qwen via the server path.)

### Standing conclusion (2026-06)
The MoE decode well is **mostly dry** at the kernel level: the int4 expert GEMM
is already occupancy-optimal (split-K wins isolated but not integrated), the
dense paths already use `wvSplitK`, and the routing/glue is ~34% of MoE GPU time
but only ~1% TPOT per fusable kernel under cudagraphs. **The one realized MoE
serving win remains `torch.compile` (~10% geomean on MoE TP=4, ledger above).**
Further kernel-level MoE decode work is not recommended unless the cost model
changes (new ROCm/Triton, higher concurrency regime, or a fused multi-launch
glue kernel that demonstrably clears a few % net).

---

*Living doc. Last updated 2026-06-03. AI assistance was used for the entries
above; each links to a standalone writeup with full reproduction.*
