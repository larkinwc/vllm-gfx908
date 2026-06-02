# FP16 vs Quant — Current-Build Decode Comparison (2026-06)

**Motivation:** The §Latest quant grids in `BENCH.md` and the §Historical FP16
numbers were never directly comparable (different binary `0.18.1` vs `0.20.2`,
different harness). This run closes that gap: a fresh FP16 measurement on the
**same build and the byte-identical bench harness** as the M4 HBM quant grids,
so the FP16-vs-quant delta can finally be stated honestly.

## Setup

| Item | Value |
|---|---|
| Hardware | 4× MI100 (gfx908) 32GB, XGMI mesh |
| vLLM build | `0.20.2rc1.dev107+gd960f21e4` (worktree `fuzzy-hornets-see-szfl4`) — **same version BENCH.md records for the quant grids** |
| FP16 model | `/models/Qwen3.5-9B`, `--dtype float16` |
| FP16 config | **production as-shipped** (`/root/launch-vllm-optimized.sh`): skinny GEMM + TunableOp replay + custom all-reduce + FULL_DECODE_ONLY graphs + block-size 32 + prefix caching, `--gpu-memory-utilization 0.93 --max-model-len 32768` |
| Bench harness | byte-identical to the HBM launch scripts: `bench serve --num-prompts 200 --request-rate inf --seed 42 --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos` |
| Scope | **decode-dominated geomean subset**: `tp{1,4} × c{1,2}`, synthetic workload (the 4 cells that define the failed 1.5× DoD bar) |
| Raw JSON | `/root/fp16-bench/results/fp16_tp{1,4}_c{1,2}_synthetic/raw.json` |
| Runner | `/root/fp16-bench/run_fp16_decode_subset.sh` |

> **Caveat (accepted):** the FP16 build differs by patch-level from the literal
> SHA the quant cells were captured at (`85a6a0b…`), and the quant numbers below
> are the BENCH.md headline figures, not re-run here. This is a documented
> **near-apples-to-apples** comparison, not an exact-binary A/B. The harness,
> model family, prompts, seed, and bench flags are identical.

## FP16 results (this run)

| Cell | out tok/s | TTFT p50 (ms) | TPOT p50 (ms) | req/s |
|---|---:|---:|---:|---:|
| `fp16_tp1_c1` | **45.02** | 265.58 | 21.20 | 0.176 |
| `fp16_tp1_c2` | **82.75** | 452.72 | 22.57 | 0.323 |
| `fp16_tp4_c1` | **72.15** | 156.77 | 13.70 | 0.282 |
| `fp16_tp4_c2` | **126.02** | 148.33 | 15.41 | 0.492 |

**FP16 decode-dominated geomean = 76.29 tok/s**

## Head-to-head (vs BENCH.md headline quant grids)

| Cell | FP16 | W8A8 | W8A8/FP16 | W4A16 | W4A16/FP16 |
|---|---:|---:|:-:|---:|:-:|
| tp1_c1 | 45.02 | 40.11 | **0.89×** | 30.88 | **0.69×** |
| tp1_c2 | 82.75 | 74.61 | **0.90×** | 56.76 | **0.69×** |
| tp4_c1 | 72.15 | 63.57 | **0.88×** | 55.25 | **0.77×** |
| tp4_c2 | 126.02 | 122.75 | **0.97×** | 106.11 | **0.84×** |

| Quant | Decode geomean | Ratio vs FP16 |
|---|---:|:-:|
| **FP16** | 76.29 | 1.00× (reference) |
| **W8A8 INT8** | 69.52 | **0.91×** (−9 %) |
| **W4A16 INT4** | 56.62 | **0.74×** (−26 %) |

## Verdict

**Your instinct was correct.** On current-build decode throughput, **FP16 is
faster than both quant paths** — W8A8 runs at 0.91× FP16 and W4A16 at 0.74×.
The quant stack is *not* a throughput win on these decode cells; it is a
throughput **regression** that we have been optimizing *within* (the +13–20 %
"wins" in BENCH.md are quant-vs-earlier-quant, not vs FP16).

This is fully consistent with the roofline finding in `BENCH_INT8_W4A16_HBM.md`:
the hot GEMMs are **HBM-bandwidth bound** (~21–32 % of HBM peak, VALU <1 % of
compute peak). At low concurrency, dequant + per-token INT8/INT32 conversion
overhead is not hidden behind enough arithmetic, so the smaller weight
footprint does not translate into faster decode. The gap narrows as concurrency
rises (W8A8 reaches 0.97× at tp4_c2), which is the only regime where weight-byte
savings start paying off.

### What quant is actually for (and it is not decode tok/s)

- **VRAM / capacity:** W4A16 saves ~75 % of weight bytes, W8A8 ~50 %. That buys
  longer context, more concurrent sequences, or fitting larger models — the real
  reason to quantize on a 32GB MI100, and a dimension this throughput bench does
  not capture.
- **Quality is preserved** (PPL Δ ≤ +3 %, 9/10 coding, 5/5 needle@32k per
  BENCH.md), so the capacity gain is essentially free on quality.

### Recommendation

1. **Stop framing the quant work as a throughput win vs FP16** — by this
   measurement it is a 9–26 % decode regression. Recent issue-driven kernel work
   (Marlin repack, MoE configs, etc.) was fighting an HBM wall, which explains
   the marginal/negative results.
2. **If raw decode tok/s is the goal, ship FP16** for cells that fit in VRAM.
3. **If capacity is the goal, keep quant** but re-benchmark on the axis where it
   wins: *max usable context / max concurrent sequences at fixed 32GB*, not
   decode tok/s.
4. **Next data to collect** (not in this run): the coding workload (4 more
   cells) and a VRAM-headroom / max-context comparison to quantify the capacity
   win that justifies quant.

## How to reproduce

```bash
# Runs the 4 decode-subset FP16 cells against the compiled 0.20.2 worktree
/root/fp16-bench/run_fp16_decode_subset.sh
# Raw results land in /root/fp16-bench/results/fp16_tp{1,4}_c{1,2}_synthetic/raw.json
```

---

*Generated 2026-06-02. FP16 cells measured this run; quant figures cited from
`BENCH.md` §Latest production figures. AI assistance was used.*
