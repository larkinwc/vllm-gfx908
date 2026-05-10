# Milestone 3 — Triton W8A8 / W4A16 Custom Kernels

> Path 2 of 4 (TensileLite → **Triton** → CK → Hand-ISA).
> M3 lands the in-tree custom Triton kernels; benchmark grid + perplexity
> are tracked as deferred follow-ups (see "Status" below).

## What landed

### W8A8: extended `mi100_int8.py`

* Kernel signature already exposed `scale_a [M, 1]` (per-token) and
  `scale_b [N, 1]` (per-channel). M3 verifies and locks this contract
  via `tests/kernels/quantization/test_mi100_int8_features.py`.
* Accumulator path is `tl.dot(..., out_dtype=tl.int32)` mapping to
  `v_mfma_i32_*_i8` on gfx908.
* Epilogue is now **single-pass fused**:
  `acc.int32 -> fp32 -> * scale_a -> * scale_b -> + bias (fp32) -> cast -> store`.
  The pre-M3 code added bias **after** the FP16 cast which loses
  per-channel bias precision; the M3 version keeps bias in FP32.
* Tile selection now consults
  `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M{M}_N{N}_K{K}.json`
  before falling back to the M2 heuristic. Loaded JSON keys forwarded
  to Triton: `BLOCK_M/N/K`, `GROUP_SIZE_M`, `matrix_instr_nonkdim`,
  `kpack`, `waves_per_eu`, `num_warps`, `num_stages`.

### W4A16: new `mi100_w4a16.py`

* Public entry point: `mi100_w4a16_gemm(a, b_q, scales, qzeros,
  group_size, zp_bias)`. Matches the in-tree generic Triton W4A16
  contract (b_q layout `[K, N//8] int32`, scales `[K//G, N]`,
  qzeros `[K//G, N//8]`).
* **Register-level int4 unpack:** `tl.interleave` replicates each
  packed int32 8× along N, then `(packed >> shifts) & 0xF` extracts
  the right nibble per output column. No global-memory hop.
* Group sizes 32 and 128 supported and exercised by tests.
* `tl.dot(a_fp16, w_fp16, out_dtype=tl.float32)` -> `v_mfma_f32_*x*x*16f16`.
* Same per-shape config-loader hook as W8A8.
* Wired into the dispatcher: `triton_w4a16_gemm` checks `on_mi100() &&
  group_size in (32, 128) && not VLLM_DISABLE_MI100_W4A16` and forwards
  to the new kernel; otherwise falls through to the generic
  `triton_w4a16_gemm_kernel`.
* `VLLM_DISABLE_MI100_W4A16=1` forces the generic path for A/B testing
  / regression triage.

### Autotune sweep

* `scripts/mi100/autotune_sweep.py --kernel {w8a8,w4a16} --shape ...`
* Cartesian product: 768 configs across `BLOCK_M × BLOCK_N × BLOCK_K ×
  matrix_instr_nonkdim × kpack × waves_per_eu × num_stages` per the M3
  brief.
* Coverage per shape (rest pruned with logged reason: LDS overflow,
  illegal MFMA tile, `BLOCK_K > group_size`):
  * W8A8: **87.5%** (672/768) per shape.
  * W4A16, group_size=128: **80.2%** (616/768) per shape.
  * W4A16, group_size=32: 29.2% (224/768) — tail of `BLOCK_K ∈ {64,
    128}` is a hard kernel invariant violation
    (`block_k_gt_group_*`); each excluded config is logged with reason
    per the spec ("rest pruned with logged reason").
* Persisted JSONs at `vllm/model_executor/kernels/configs/gfx908/`:
  * 6 W8A8 configs (M ∈ {1, 32, 512}; N ∈ {4096, 10240, 24576};
    K ∈ {4096, 12288}).
  * 6 W4A16 configs (group sizes 32 and 128 mixed).
  * Each JSON carries: `config`, `measured_ms_per_iter`,
    `measured_tok_s`, `autotune_runs_evaluated`,
    `autotune_runs_pruned`, `vllm_commit`, `kernel_source_sha256`,
    `timestamp`. (See `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M1_N4096_K4096.json`.)

## Tests added

| File | Purpose |
| --- | --- |
| `tests/kernels/quantization/test_mi100_int8_features.py` | VAL-TRITON-001: signature + epilogue static check; FP32 reference smoke. |
| `tests/kernels/quantization/test_mi100_w4a16.py` | VAL-TRITON-002: register-unpack pattern + groupwise smoke (g=32, g=128). |
| `tests/kernels/quantization/test_mi100_int8_correctness.py` | VAL-TRITON-005: 14 hot shapes × 3 seeds = 42 cases, atol=1e-2 / rtol=5e-2. |
| `tests/kernels/quantization/test_mi100_w4a16_correctness.py` | VAL-TRITON-006: 10 hot shapes × {g=32, g=128} × 3 seeds = 60 cases. |
| `tests/kernels/quantization/test_config_loader.py` | VAL-TRITON-004: per-shape JSON pickup, group-size keying, env override. |

All 112 M3 tests + the 184 pre-existing `test_mi100_int8.py` tests pass.

## CUDA-graph compatibility (VAL-TRITON-008)

`scripts/mi100/cudagraph_smoke.sh {w8a8,w4a16} {1,4}` launches vLLM with
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, waits
for `/health`, asserts the log contains both
`cudagraph_mode=FULL_DECODE_ONLY` and a graph-capture marker
("Graph capturing finished" / "captured N graphs"), runs a 100-token
decode probe via `/v1/completions`, and checks the log for
`HIP error` / `RuntimeError`.

| Quant | TP | Result | Graphs captured |
| --- | --- | --- | --- |
| w8a8  | 1 | **PASS** | 35 |
| w8a8  | 4 | **PASS** | 35 |
| w4a16 | 1 | **PASS** | 35 |
| w4a16 | 4 | **PASS** | 35 |

## Cache provenance (VAL-TRITON-002 follow-up from m1-aot-pretune-w4a16)

Because the new W4A16 kernel is a different `@triton.jit` source,
Triton's compile-cache key changes. M3 re-runs
`scripts/mi100/pretune_w4a16.py` to repopulate
`/root/bench-int8-w4a16/baseline/triton_cache_w4a16/` (134 hsaco
binaries, 51 MiB).

The cache manifest at
`/root/bench-int8-w4a16/baseline/triton_cache_manifest.json` is now
extended with `kernel_source_sha256` covering:

* `vllm/model_executor/kernels/linear/scaled_mm/mi100_w4a16.py`
* `vllm/model_executor/kernels/linear/scaled_mm/mi100_int8.py`
* `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`
* `scripts/mi100/pretune_w4a16.py`
* `vllm/model_executor/kernels/configs/gfx908/config_loader.py`

If any of these files change in M4/M5, regenerate the cache via
`/opt/vllm-env/bin/python3 scripts/mi100/pretune_w4a16.py` and re-pin
the manifest.

The W4A16 TP=4 cudagraph smoke succeeded under 360 s with the
freshly-regenerated cache, confirming the M1-pretune feature still
delivers FULL_DECODE_ONLY on TP=4 with the new kernel.

## Status

| Deliverable | Status |
| --- | --- |
| Extended W8A8 kernel (signature + fused epilogue) | **DONE** |
| New W4A16 kernel (register unpack, g=32 / g=128) | **DONE** |
| Autotune sweep (≥80% coverage W8A8 + W4A16 g=128) | **DONE** |
| Per-shape JSON configs persisted | **DONE** (12 shapes) |
| Config-loader unit test | **DONE** |
| Correctness vs FP32 PyTorch reference (≥3 seeds) | **DONE** (42 W8A8 + 60 W4A16) |
| FULL_DECODE_ONLY cudagraph capture (TP=1, TP=4 × both) | **DONE** (4/4 PASS) |
| Triton AOT-pretune cache regenerated, manifest re-pinned | **DONE** |
| Full 24-cell bench grid (M3 vs M2 / M1-rebaseline) | **DEFERRED** — runs ~2 h; spawn a follow-up `m3-bench-grid` worker. |
| Wikitext-2 perplexity Δ check (≤+1% vs M2 W8A8 / vs PyTorch ref W4A16) | **DEFERRED** — depends on grid completing; spawn `m3-perplexity` worker after grid run. |

The kernels themselves are landed, correct, autotune-guided, and
graph-compat. The remaining benchmark deltas are pure measurement
follow-ups that do not invalidate the M3 implementation if a regression
shows up — they would only motivate a kernel-tuning re-run, gated by
`VLLM_DISABLE_MI100_W4A16=1` to revert to the generic path.

## Reproduction

```bash
REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
cd $REPO

# Baseline kernel correctness (CPU-only static checks + GPU correctness)
PYTHONPATH=. /opt/vllm-env/bin/python3 -m pytest \
  tests/kernels/quantization/test_mi100_int8.py \
  tests/kernels/quantization/test_mi100_int8_features.py \
  tests/kernels/quantization/test_mi100_int8_correctness.py \
  tests/kernels/quantization/test_mi100_w4a16.py \
  tests/kernels/quantization/test_mi100_w4a16_correctness.py \
  tests/kernels/quantization/test_config_loader.py \
  -v --timeout=180

# Autotune sweep (subset of hot shapes; takes ~10 min per kernel)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/vllm-env/bin/python3 \
  scripts/mi100/autotune_sweep.py --kernel w8a8 \
  --shape 1,4096,4096 --shape 1,10240,4096 \
  --shape 32,10240,4096 --shape 512,4096,4096 \
  --shape 512,4096,12288 --shape 512,24576,4096

# Cudagraph smoke (manifest-managed start/stop)
bash scripts/mi100/cudagraph_smoke.sh w8a8 1
bash scripts/mi100/cudagraph_smoke.sh w8a8 4
bash scripts/mi100/cudagraph_smoke.sh w4a16 1
bash scripts/mi100/cudagraph_smoke.sh w4a16 4

# Triton AOT-pretune (refreshes the W4A16 TP=4 cache)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/vllm-env/bin/python3 \
  scripts/mi100/pretune_w4a16.py
```

To force a clean comparison against the generic path:

```bash
VLLM_DISABLE_MI100_W4A16=1 ...    # routes W4A16 to the existing path
VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1 ...   # routes W8A8/W4A16 to heuristic tiles
```
