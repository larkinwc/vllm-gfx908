<!-- markdownlint-disable MD013 -->
# Porting torch.compile + PIECEWISE to the 0.20.2 build (gfx908/MI100, 2026-06)

Follow-up to `BENCH_FP16_TORCH_COMPILE_AB_2026_06.md`, which found the
community compile config gives +3.64% FP16 decode on 0.19.2 but **crashed on
our 0.20.2 production build**. This note ports the fix and re-benches.

## Root cause of the 0.20.2 crash

torch.compile (`mode=3`) aborted engine init with:
> `torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor` on
> `torch.cuda.is_current_stream_capturing()`

Traced to a **gfx908-specific** producer added in the 0.20.2 tree (absent in
0.19.2), called inside the now-compiled MoE forward:

```
qwen2_moe.py:129  Qwen2MoeMLP.forward
  → try_stash_fused_silu_quant_int8(gate_up, self.down_proj)
    → fused_silu_quant_int8.py:252  if torch.cuda.is_current_stream_capturing():  ← Dynamo can't trace
```

`is_current_stream_capturing()` returns a Python `bool`; under
`@support_torch_compile` Dynamo tries to fold it into the FX graph output and
fails. This is the M1 silu→quant fusion (issue #33), a W8A8-only optimization —
but it sits in the shared MLP forward, so it trips compile even on FP16 runs.

## The fix (1 line + comment)

`vllm/model_executor/kernels/quantization/fused_silu_quant_int8.py`, as the
first branch of `try_stash_fused_silu_quant_int8`:

```python
if torch.compiler.is_compiling():
    return False
```

Why this is correct and minimal:
- `torch.compiler.is_compiling()` **is** Dynamo-traceable and constant-folds to
  `True` during capture, so it short-circuits **before** the untraceable
  `is_current_stream_capturing()` call.
- Returning `False` under compile is the right behavior: the helper is a
  Python-side producer that stashes a cache via `setattr` — it cannot run inside
  a traced graph anyway. The subsequent `act_fn(gate_up)` + `down_proj` sequence
  is preserved, so the compiled path is **byte-identical** to the legacy
  `scaled_int8_quant` path (same fall-through the existing CUDA-graph-capture
  skip already relies on).
- At eager runtime `is_compiling()` is `False` (verified), so the W8A8 fused
  act-quant feature is **completely unaffected** when not compiling.
- Same idiom already used in-tree at `fused_batched_moe.py:664`
  (`if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()`).

## Result: compile now works on 0.20.2

- Smoke: `mode=3` + `FULL_AND_PIECEWISE` compiles in **38.7 s** and serves
  healthy (was: instant crash).
- A/B on the **fixed 0.20.2 build** (fuzzy-hornets, matches current BENCH docs),
  decode subset, same HBM-identical harness:

| cell | baseline (FULL_DECODE_ONLY) | compile (mode3+FULL_AND_PIECEWISE) | Δ tput | Δ TTFT |
|---|---:|---:|---:|---:|
| tp1_c1 | 44.55 | 43.12 | **−3.21%** | −4.6% |
| tp1_c2 | 81.27 | 80.58 | **−0.85%** | −6.0% |
| tp4_c1 | 72.56 | 75.81 | **+4.47%** | −6.2% |
| tp4_c2 | 127.70 | 136.84 | **+7.16%** | −4.2% |
| **decode geomean** | **76.11** | **77.48** | **+1.81%** | — |

## Interpretation

**The port succeeds and the high-concurrency win transfers cleanly** (TP=4:
+4.47% / +7.16%, essentially identical to 0.19.2's +4.88% / +6.92%). TTFT
improves on every cell.

**But on 0.20.2 the low-concurrency cells now REGRESS** (tp1_c1 −3.21%,
tp1_c2 −0.85%), which they did not on 0.19.2 (+0.06% / +2.82%). Net geomean is
**+1.81%** vs 0.19.2's +3.64% — compile's TP=1 behavior is worse on the newer
build. Likely the newer Inductor/Dynamo path adds per-launch overhead that the
small c=1 batch can't amortize, while the FULL_DECODE_ONLY graph the baseline
uses is already near-optimal there.

This makes the deploy recommendation **concurrency-gated**, not global.

## Recommendation

1. **Ship compile for TP=4 / multi-stream serving on 0.20.2** — clean +4.5–7.2%
   with better TTFT. This is the real production-serving regime.
2. **Keep FULL_DECODE_ONLY (no compile) for TP=1 / single-stream** — compile
   regresses there on 0.20.2.
3. The fix is safe to merge regardless (no effect unless
   `VLLM_MI100_TORCH_COMPILE=1`), since it only unblocks the option; it does not
   change default behavior. It also makes `VLLM_MI100_TORCH_COMPILE` actually
   usable on the current build, which it wasn't before.
4. **Bigger upside still expected on MoE / larger models** (the community's
   headline compile wins were on 35B-A3B MoE). Re-test once a larger GPTQ-8bit
   MoE model is loadable — ties into the W8A16 capacity work.

## Files changed
- `vllm/model_executor/kernels/quantization/fused_silu_quant_int8.py` — added
  `torch.compiler.is_compiling()` guard (1 line + explanatory comment).

## Reproduction
```bash
# fix is in the fuzzy-hornets 0.20.2 worktree
/root/fp16-bench/run_compile_ab_0202.sh
# results: /root/fp16-bench/compile_ab_0202/{baseline,compile}_tp{1,4}_c{1,2}/raw.json
```

Build: vLLM `0.20.2rc1.dev107+gd960f21e4`, torch `2.11.0+rocm7.2`, Triton
`3.5.1`, ROCm 7.12, 4× MI100 gfx908.

---

*Generated 2026-06-02. AI assistance was used.*
