<!-- markdownlint-disable MD013 -->
# The gfx908 fused-MoE tuned configs are a NET REGRESSION on our stack (2026-06)

Investigation triggered by a fair question on `BENCH_FP16_COMPILE_QUANT_MOE_2026_06.md`:
**a 3B-active MoE getting only ~62 tok/s single-stream is too slow** — a dense
9B (3× the active params) got 44 tok/s, so the MoE is barely ahead. That points
at the MoE kernel, not compute.

## What the logs showed

Both MoE servers (fp16 35B and W4A16 512e) logged, every run:

```text
WARNING fused_moe.py:1091 Using default MoE config. Performance might be
sub-optimal! Config file not found at .../configs/
E=512,N=128,device_name=AMD_Instinct_MI100,dtype=int4_w4a16.json
```

So the Triton fused-MoE kernel ran on **default tile params** — which for a
3B-active model (where fused-MoE *is* the decode hot loop) plausibly explains
the weak throughput.

## Two real bugs found in the merged config support

The fork *does* ship gfx908 MoE configs (commit `477ae544d`, from
btbtyler09's `mi100-optimized`). But:

1. **Dead lookup (device-name mismatch).** Configs are named
   `device_name=Arcturus_GL-XL_[Instinct_MI100]`, but the runtime builds the
   lookup key from `current_platform.get_device_name().replace(" ","_")` =
   **`AMD_Instinct_MI100`** (ROCm 7.12). The two never match, so the tuned
   configs were **never loaded** — every MoE run silently used Triton defaults.
   (`fused_moe.py:1022`, `get_config_file_name`.)
2. **Worktree drift.** The configs exist only in the `five-poems` doc worktree,
   not in `fuzzy-hornets` (the compiled build where all benchmarks ran), so even
   the dead-named files weren't present during benching.

The W4A16 model needs exactly `E=512, N=128, int4_w4a16`
(`num_experts=512`, `moe_intermediate_size=512 / TP4 = 128`) — and the merged
set **contains that exact shape**. So this looked like a clean win waiting to be
unlocked: just install the file under the runtime-expected name.

## The fix made it WORSE — the configs are actively harmful

Installed the four configs under their `AMD_Instinct_MI100` names in
`fuzzy-hornets` and A/B'd **default vs tuned** with everything else identical
(no compile, FULL_DECODE_ONLY, TP=4, same harness). The tuned arm confirmed the
warning disappeared (`Using default MoE config` count: 1 → 0), so the config was
genuinely picked up.

| cell | default (Triton) | tuned config | Δ tput | tuned tpot |
|---|---:|---:|---:|---:|
| tp4_c1 | 51.47 | 49.93 | −2.99% | 19.1 ms |
| tp4_c2 | 83.62 | 78.23 | −6.45% | 24.8 ms |
| tp4_c4 | 147.74 | 38.38 | **−74.0%** | **104.4 ms** |
| **geomean** | **85.99** | **53.12** | **−38.2%** | — |

The c4 (M≈512) regression is catastrophic and **not noise**: 200/200 requests
completed, tpot pinned at 104 ms vs the default's ~26 ms — a steady **2.7×
slowdown**. The selected larger-M entries use
`BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2`, a tile that
register-spills / occupancy-starves on our **Triton 3.5.1 / ROCm 7.12** stack.

## Conclusion

- The autotuned configs were tuned on **btbtyler09's older ROCm/Triton**
  (hence the `Arcturus_GL-XL` device string). On our newer Triton 3.5.1 / ROCm
  7.12 they are a **net −38% regression**, with a pathological larger-batch tile.
- **The device-name mismatch was accidentally protecting us.** "Fixing" the
  lookup to load these configs would *degrade* every gfx908 MoE deployment on
  the current stack.
- **Triton's built-in default heuristic beats these stale configs here.** The
  weak 3B-active throughput is therefore NOT an un-tuned-config problem — it's
  the realistic ceiling for the current Triton MoE kernel on gfx908 at this
  shape. (Separately, attention also falls back: `Cannot use ROCm custom paged
  attention kernel, falling back to Triton` — another gfx908 slow path, not
  addressed here.)

## Actions taken

- **Did NOT install / commit the configs** into the build. Removed the copies;
  `fuzzy-hornets` is back to only the committed Dynamo-guard change.
- The compile A/B numbers in the prior docs **stand and are fair** — both arms
  ran Triton defaults, so the +9–10% compile delta is real and independent of
  this config issue. (The absolute tok/s is simply the Triton-default ceiling,
  not inflated or deflated by a tuned config.)

## Recommendations

1. **Leave the merged `Arcturus_GL-XL` configs as dead files** (or delete them),
   but **do not** rename them to `AMD_Instinct_MI100` / normalize the device
   lookup — that would silently regress MoE on this stack. If the lookup is ever
   normalized, these specific JSONs must be **re-tuned or dropped** first.
2. **Re-tune from scratch on Triton 3.5.1 / ROCm 7.12** with
   `benchmarks/kernels/benchmark_moe.py` for the shapes we actually serve
   (`E=512,N=128,int4_w4a16` and `E=256,N=128` fp16) if we want to beat the
   Triton default. Only ship configs that **A/B-beat the default** on this stack
   — gate every new JSON behind a measured win, since the default is a real
   competitor here.
3. **Add a guard to the config docstring/commit** noting that gfx908 MoE configs
   are Triton/ROCm-version-sensitive and must be re-validated per stack.

## Reproduction

```bash
/root/fp16-bench/run_qmoe_config_ab.sh   # toggles the config file, default vs tuned
# results: /root/fp16-bench/qmoe_config_ab/{default,tuned}_tp4_c{1,2,4}/raw.json
```

Model `/models/Qwen3-Coder-Next-AWQ-4bit` (W4A16, 512e/10a). Build: vLLM
`0.20.2rc1.dev107+gd960f21e4`, torch `2.11.0+rocm7.2`, Triton `3.5.1`, ROCm
7.12, 4× MI100 gfx908. Harness: 200 prompts, seed 42, random 1024/256,
`--dtype float16`, max-model-len 16384, gpu-mem-util 0.92, FULL_DECODE_ONLY.

---

*Generated 2026-06-03. AI assistance was used.*
