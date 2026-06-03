# Recommended inference setup — gfx900 (V340 / Vega10), Qwen3.5-9B

This is the **prescriptive "how to run"** companion to the deep-dive in
`PERF_GFX900.md`. It distills everything we measured on this box (8× Radeon Pro
V340 = 16× gfx900 dies, ~8 GiB each, wave64, no MFMA, ~365 GB/s HBM/die) into the
single most performant, validated path and the knobs that actually move the needle.

---

## TL;DR — the best path

**For maximum capability per GPU (the recommended default):**

```bash
# int4 weights (custom gfx900 GEMV) + TurboQuant KV cache, on one socket.
HIP_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V1=1 \
vllm serve QuantTrio/Qwen3.5-9B-AWQ \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --kv-cache-dtype turboquant_k8v4 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --compilation-config '{"mode":0,"cudagraph_mode":[2,0],"cudagraph_capture_sizes":[1]}'
# NOTE: use CUDA graphs (do NOT pass --enforce-eager) — 3.6x decode on gfx900 (#63).
# --max-num-seqs must be <= the model's Mamba/GDN cache-block budget or capture fails.
```

This stacks the gfx900 wins we built and validated (LLMM1 M=1 GEMV is on by default via `VLLM_ROCM_USE_SKINNY_GEMM=1`):

| Layer | Mechanism | Win | Issue |
|---|---|---|---|
| **Weights** | int4 AWQ via hand-written gfx900 GEMV (sidesteps MFMA-less `tl.dot`) | **~1.5× decode vs FP16 on half the GPUs** | #57 |
| **All M=1 FP16 GEMMs** | LLMM1 skinny GEMV (lm_head + qkv/o/gate_up projections) | **2.2× FP16 decode (150→67 ms/step)** | #59 |
| **Per-step dispatch** | HIP CUDA graphs (FULL_DECODE_ONLY) once GPU-busy is small | **3.6× decode on top of LLMM1** | #63 |
| **KV cache** | TurboQuant `turboquant_k8v4` (FP8 keys + 4-bit values) | **2.34× context/concurrency per VRAM, ~0 decode cost** | #62 |

Both verified active together (coherent output; decode 9.97 tok/s ≈ standalone AWQ
10.46). They are orthogonal and compose cleanly.

---

## Why these specific choices (each is measured, not assumed)

### 1. `--dtype float16` (never bf16)
gfx900 has no usable bf16 path for this model. FP16 is mandatory.

### 2. CUDA graphs ON (do NOT use `--enforce-eager`) — updated, see #63
HIP CUDA graphs give a **3.6× decode speedup** on gfx900 (eager 67 → graph 18.4
ms/tok) once the LLMM1 GEMV (#59) shrinks GPU-busy time so the host dispatch gap
(59% of the step) dominates. Use `mode=NONE` (Inductor/compile stays off — it
crashes on ROCm) + `cudagraph_mode=FULL_DECODE_ONLY` + `capture_sizes=[1]`.
**Constraint:** Qwen3.5 is hybrid (Mamba/GDN); `--max-num-seqs` must be ≤ the
available Mamba cache blocks or capture fails. (Earlier docs said eager was best —
that was true before LLMM1; the bottleneck moved. See `PERF_GFX900.md` § #63.)

### 3. `--kv-cache-dtype turboquant_k8v4` (FP8 keys — NOT a K-quant preset)
Qwen3 family is **"quirky-K"**: key-side quantization is catastrophic
(`tq2k` = +150% PPL on qwen3:8b — output destroyed). `k8v4` keeps keys at FP8
(effectively unquantized) and only compresses values. **Do not** use
`turboquant_3bit_nc` / other aggressive-K presets on Qwen3.5.

**Quality is measured, not assumed.** WikiText-2 reference perplexity (12,264 tokens, 24x512 windows, prefill logprobs), Qwen3.5-9B TP4: FP16 = 9.8885, turboquant_k8v4 = 9.8879 (**-0.006%, lossless within noise**). Safe production default: 2.34x KV capacity at zero measurable quality cost. (PPL measures quantizer reconstruction quality, not long-decode drift.)

### 4. int4 weights via AWQ (`QuantTrio/Qwen3.5-9B-AWQ`)
Stock INT4 on gfx900 is *slower* than FP16 because the Triton path dequants then
runs `tl.dot` through a padded FP32 GEMM (no MFMA). Our hand-written M≤8 GEMV
(#57) sidesteps `tl.dot` entirely → 2.8–4.0× the MLP matmuls vs FP16, 10–15× vs
the stock int4 path. It auto-dispatches when `on_gfx900()` and `group_size ∈
{32,64,128}` and `M≤8`; high-batch falls back to FP16 automatically.

### 5. Layout: TP within one socket
GPUs 0–7 = socket 0, 8–15 = socket 1. **Cross-socket P2P is 0.26 GB/s** (Xeon
E5 v3 QPI HW limit) vs ~8 GB/s intra-socket. **Never span sockets in one TP/PP
group.** Use `HIP_VISIBLE_DEVICES=0,1,2,3` (or 0–7) for socket 0.

---

## Choosing a layout for your workload

Decode does **not** scale with bandwidth here (it sits at ~5–9% of the HBM
roofline, gated by per-token all-reduce + GDN kernel latency). So:

| Goal | Recommended layout | Why |
|---|---|---|
| **Single-stream latency** | TP4 (or TP2) one socket | TP8 only ~1.7× TP4 — diminishing |
| **Aggregate throughput** | **batch/concurrency** is the only real lever — c1→c4 gives 2.5–3.4× | decode per-step time is ~flat with batch up to ~64–96 |
| **Decode-optimized serving** | **TP2×PP4** within one socket | PP removes the per-layer all-reduce that dominates decode latency; best aggregate we measured (~26 tok/s, c4 coding) |
| **Max context per VRAM** | add `turboquant_k8v4` | 2.34× KV capacity |

**Realistic planning numbers** (not the parallel spec): ~24 TFLOPS/socket usable
prefill; decode in the **tens of tok/s aggregate**. Prefill scales well (~70%
efficiency, TP-friendly); decode is the bottleneck.

---

## VRAM ceiling — important

Each die is **~8 GiB**. AWQ(TP2) + 16k-token KV **OOMs at batch>1**. The
compression lets you push context/concurrency further per GiB, but does not remove
the per-die cap. For long context **and** batching with the AWQ model, use **TP4**
(more aggregate VRAM) or a shorter `--max-model-len`.

---

## Model facts that drive all of the above (Qwen3.5-9B)
- 9.65B params (19.3 GB FP16), 32 layers, hidden 4096, vocab 248320, 16 attn / 4 KV heads.
- **Hybrid attention**: `full_attention_interval=4` → only **8/32 layers** are full
  attention with a KV cache; the rest are **GDN linear-attn** (no KV cache). So the
  KV footprint is already small — TurboQuant's 2.34× applies to that 1/4 of layers.
- GDN **decode** uses `fused_recurrent` (0 `tl.dot`) — already gfx900-appropriate.
  The `tl.dot`-heavy FLA ops are **prefill-only**.
- Dense hybrid, **not** MoE. Use `--language-model-only`.

---

## Operational notes
- **Cold start is slow** (~647s) due to GDN/FLA Triton autotune + TurboQuant
  centroid init; warm is ~60s. Autotune results cache.
- `amdgpu.reset_method=2` (mode1) is set so a GPU hang auto-recovers instead of
  needing a reboot. (`/etc/modprobe.d/amdgpu-reset.conf`; reversible.)
- Pure-Triton TurboQuant kernels run on wave64 — the wave32 `__shfl_sync` blocker
  that stops the llama.cpp HIP port does **not** apply to us.

## Repro / benchmarks
- `bench_scripts/tq_measure.py {auto|turboquant_k8v4}` — KV size + decode tok/s.
- `bench_scripts/combo_dec.py turboquant_k8v4` — int4 weights + TQ KV together.
- `bench_scripts/batch_sweep.py` — concurrency scaling.
- Deep dives: `PERF_GFX900.md`, `BENCH_GFX900.md`, `GFX900_SETUP.md`.
