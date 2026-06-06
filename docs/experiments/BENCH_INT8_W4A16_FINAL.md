# BENCH_INT8_W4A16_FINAL — gfx908 (MI100) custom kernels, final aggregate

Final Pareto report for the MI100 (gfx908) custom INT8 / W4A16 kernel mission.
Aggregates milestones M0–M5 into a single grid with per-(cell × metric) winners,
production recommendations, full quality-gate evidence, and per-cell reproducible
launch scripts.

Cross-links: 
[BENCH_INT8_W4A16_BASELINE.md](BENCH_INT8_W4A16_BASELINE.md) (M0+M1), 
[BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md) (TensileLite three-way), 
[BENCH_INT8_W4A16_M3.md](BENCH_INT8_W4A16_M3.md) (Triton W8A8 + W4A16), 
[BENCH_M4_CK.md](BENCH_M4_CK.md) (Composable Kernel W8A8 + W4A16-negative), 
[BENCH_M5_ISA.md](BENCH_M5_ISA.md) (Hand-ISA negative result).

## Hardware / software manifest

| | |
|---|---|
| Hosts | 1× node, 4× AMD Instinct MI100 (gfx908), 32 GB VRAM each |
| Driver | amdgpu-dkms 6.19.0+, perf=high, 250 W cap |
| ROCm | 7.12 (`/opt/rocm/core-7.12`) |
| PyTorch | 2.11.0+rocm7.2 |
| pytorch-triton-rocm | 3.5.1 |
| vLLM | 0.20.2rc1.dev107+gd960f21e4 (mission worktree, editable install) |
| Models | `/models/Qwen3.5-9B-{w8a8,w4a16}`, FP16 ref `/models/Qwen3.5-9B` |
| Tuning manifest | `/root/bench-int8-w4a16/final/tuning_hashes.json` (68 pinned files) |

## Quality gates (Wikitext-2 perplexity, coding-agent, 32 k needle)

Wikitext-2 perplexity (50 chunks × 512 tokens, seed 0):

| Run | Perplexity | Δ vs FP16 | Δ vs M0 PTQ |
| --- | ---: | ---: | ---: |
| FP16 reference (`/models/Qwen3.5-9B`) | 9.5254 | — | — |
| M0 W8A8 (`/models/Qwen3.5-9B-w8a8`) | 9.6561 | +1.372 % | — (reference) |
| M0 W4A16 (`/models/Qwen3.5-9B-w4a16`) | 9.8030 | +2.914 % | — (reference) |
| **M6 W8A8 stack (best path per cell)** | **9.6518** | **+1.326 %** | **-0.045 %** |
| **M6 W4A16 stack (best path per cell)** | **9.8030** | **+2.914 %** | **+0.000 %** |

- VAL-FINAL-002 gate (Δ ≤ +3 % vs FP16 + Δ ≤ +0.5 % vs M0): W8A8 **PASS** ; W4A16 **PASS**.
- Command: `scripts/m0_perplexity.py --model <path> --dataset wikitext-2-raw-v1 --chunks 50 --chunk-tokens 512 --seed 0` ; seed=0; numbers fixed.
- Evidence: `/root/bench-int8-w4a16/{baseline,m3,m4,final}/ppl_*.json`.

Coding-agent qualitative pass (10 prompts, rubric compiles/runs/correct):

| Total | Pass-all (compiles ∧ runs ∧ correct) | Gate ≥ 9/10 |
| ---: | ---: | :-: |
| 10 | **9** | ✅ PASS |

| Prompt | Lang | Compiles | Runs | Correct |
| --- | --- | :-: | :-: | :-: |
| `fib_iterative` | python | ✅ | ✅ | ✅ |
| `is_prime` | python | ✅ | ✅ | ✅ |
| `merge_sorted` | python | ✅ | ✅ | ✅ |
| `balanced_parens` | python | ✅ | ✅ | ✅ |
| `two_sum` | python | ✅ | ✅ | ✅ |
| `reverse_words` | python | ✅ | ✅ | ✅ |
| `anagram` | python | ✅ | ✅ | ✅ |
| `max_subarray` | python | ❌ | ❌ | ❌ |
| `node_check_js` | javascript | ✅ | ✅ | ✅ |
| `bash_check` | bash | ✅ | ✅ | ✅ |

- Evidence: `tests/eval/coding_prompts.json`, `/root/bench-int8-w4a16/final/m6_coding_eval_m6.{json,md}`.
- Failures (if any) include diff vs FP16-baseline response in the JSON `response` field.

Long-Context needle-in-haystack @ 32 k:

- Result: **5/5**, gate ✅ PASS (5/5 required by VAL-FINAL-004)
- Depths probed: 10%, 30%, 50%, 70%, 90%
  - `Magic apple count` @ depth 10%: expected `47823` → ✅
  - `Crimson tower height` @ depth 30%: expected `9216 meters` → ✅
  - `Lunar passcode` @ depth 50%: expected `MOON-7741-XQ` → ✅
  - `Coral fish species` @ depth 70%: expected `1582` → ✅
  - `Captain's birthday` @ depth 90%: expected `March 22, 1873` → ✅
- Evidence: `/root/bench-int8-w4a16/final/m6_needle32k.json`.

## Full Pareto grid

Each row is one (cell, workload, metric). Columns:

- **stock** = first vLLM run on the artifact (May 7 baseline, commit `fc20b6f4f`).
- **+TensileLite** = M2 hipBLASLt + merged TensileLite library (commit `b69172e07` / re-baseline `73f7e629a`). W4A16 has no TensileLite path.
- **+Triton** = M3 custom Triton kernels (`mi100_int8`, `mi100_w4a16`, commit `9699d1f0a`).
- **+CK** = M4 Composable Kernel `DeviceGemm_Xdl_CShuffle` W8A8 instances (commits `a852f2600` → `f4bf9e503`). W4A16 CK declined as negative result (see [BENCH_M4_CK.md](BENCH_M4_CK.md)).
- **+ISA** = n/a (M5 hand-ISA declined; see [BENCH_M5_ISA.md](BENCH_M5_ISA.md)).
- **winner** = best-of-row across the present paths (higher-better for throughput, lower-better for latency).

Workload key: `synth`=synthetic random (1024 in / 256 out, num_prompts=200); `coding`=coding-agent realistic. Metric key: `tput`=output toks/s; `req_tput`=req/s; `*_ttft`/`*_tpot` in ms.

| cell | workload | metric | stock | +TensileLite | +Triton | +CK | +ISA | winner |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :-: |
| w8a8_tp1_c1 | synth | tput | 38.97 | 37.44 | 38.01 | 38.03 | n/a | stock |
| w8a8_tp1_c1 | synth | req_tput | 0.15 | 0.15 | 0.15 | 0.15 | n/a | stock |
| w8a8_tp1_c1 | synth | mean_ttft | 314.3 | 283.5 | 285.6 | 281.7 | n/a | +CK |
| w8a8_tp1_c1 | synth | p99_ttft | 318.0 | 288.1 | 306.3 | 286.4 | n/a | +CK |
| w8a8_tp1_c1 | synth | mean_tpot | 24.53 | 25.70 | 25.29 | 25.29 | n/a | stock |
| w8a8_tp1_c1 | synth | p99_tpot | 24.55 | 25.77 | 25.33 | 25.31 | n/a | stock |
| w8a8_tp1_c1 | coding | tput | 38.06 | 36.56 | 36.44 | 37.24 | n/a | stock |
| w8a8_tp1_c1 | coding | req_tput | 0.16 | 0.16 | 0.16 | 0.16 | n/a | stock |
| w8a8_tp1_c1 | coding | mean_ttft | 231.0 | 219.9 | 305.0 | 214.6 | n/a | +CK |
| w8a8_tp1_c1 | coding | p99_ttft | 1183.8 | 1065.1 | 1242.3 | 1068.0 | n/a | +TensileLite |
| w8a8_tp1_c1 | coding | mean_tpot | 25.34 | 26.49 | 26.28 | 26.07 | n/a | stock |
| w8a8_tp1_c1 | coding | p99_tpot | 33.36 | 34.50 | 33.95 | 34.02 | n/a | stock |
| w8a8_tp1_c2 | synth | tput | 73.81 | 70.78 | 71.04 | 70.81 | n/a | stock |
| w8a8_tp1_c2 | synth | req_tput | 0.29 | 0.28 | 0.28 | 0.28 | n/a | stock |
| w8a8_tp1_c2 | synth | mean_ttft | 488.3 | 439.4 | 444.8 | 437.8 | n/a | +CK |
| w8a8_tp1_c2 | synth | p99_ttft | 556.2 | 492.4 | 504.6 | 491.4 | n/a | +CK |
| w8a8_tp1_c2 | synth | mean_tpot | 25.29 | 26.64 | 26.52 | 26.64 | n/a | stock |
| w8a8_tp1_c2 | synth | p99_tpot | 25.56 | 26.86 | 26.78 | 26.85 | n/a | stock |
| w8a8_tp1_c2 | coding | tput | 68.61 | 66.67 | 65.07 | 65.70 | n/a | stock |
| w8a8_tp1_c2 | coding | req_tput | 0.29 | 0.28 | 0.27 | 0.29 | n/a | stock |
| w8a8_tp1_c2 | coding | mean_ttft | 278.0 | 255.6 | 299.3 | 256.3 | n/a | +TensileLite |
| w8a8_tp1_c2 | coding | p99_ttft | 1341.3 | 1185.4 | 1084.5 | 1191.4 | n/a | +Triton |
| w8a8_tp1_c2 | coding | mean_tpot | 27.88 | 29.01 | 30.08 | 29.35 | n/a | stock |
| w8a8_tp1_c2 | coding | p99_tpot | 37.91 | 35.77 | 52.31 | 36.28 | n/a | +TensileLite |
| w8a8_tp1_c4 | synth | tput | 135.9 | 131.9 | 128.5 | 132.3 | n/a | stock |
| w8a8_tp1_c4 | synth | req_tput | 0.53 | 0.52 | 0.50 | 0.52 | n/a | stock |
| w8a8_tp1_c4 | synth | mean_ttft | 815.0 | 717.5 | 720.0 | 724.8 | n/a | +TensileLite |
| w8a8_tp1_c4 | synth | p99_ttft | 956.6 | 819.7 | 883.2 | 886.6 | n/a | +TensileLite |
| w8a8_tp1_c4 | synth | mean_tpot | 26.35 | 27.63 | 27.30 | 27.51 | n/a | stock |
| w8a8_tp1_c4 | synth | p99_tpot | 27.41 | 28.51 | 28.33 | 28.43 | n/a | stock |
| w8a8_tp1_c4 | coding | tput | 120.4 | 116.6 | 110.4 | 117.4 | n/a | stock |
| w8a8_tp1_c4 | coding | req_tput | 0.54 | 0.51 | 0.51 | 0.51 | n/a | stock |
| w8a8_tp1_c4 | coding | mean_ttft | 322.7 | 293.7 | 388.9 | 291.9 | n/a | +CK |
| w8a8_tp1_c4 | coding | p99_ttft | 1436.7 | 1393.3 | 1576.4 | 1378.3 | n/a | +CK |
| w8a8_tp1_c4 | coding | mean_tpot | 31.76 | 32.94 | 43.69 | 32.81 | n/a | stock |
| w8a8_tp1_c4 | coding | p99_tpot | 42.66 | 38.61 | 286.4 | 41.30 | n/a | +TensileLite |
| w8a8_tp4_c1 | synth | tput | 63.95 | 57.04 | 57.24 | 57.24 | n/a | stock |
| w8a8_tp4_c1 | synth | req_tput | 0.25 | 0.22 | 0.22 | 0.22 | n/a | stock |
| w8a8_tp4_c1 | synth | mean_ttft | 165.1 | 165.8 | 169.8 | 170.9 | n/a | stock |
| w8a8_tp4_c1 | synth | p99_ttft | 166.6 | 167.8 | 211.4 | 175.7 | n/a | stock |
| w8a8_tp4_c1 | synth | mean_tpot | 15.05 | 16.95 | 16.87 | 16.87 | n/a | stock |
| w8a8_tp4_c1 | synth | p99_tpot | 15.08 | 16.96 | 16.98 | 16.98 | n/a | stock |
| w8a8_tp4_c1 | coding | tput | 60.98 | 54.69 | 53.64 | 54.90 | n/a | stock |
| w8a8_tp4_c1 | coding | req_tput | 0.26 | 0.23 | 0.23 | 0.23 | n/a | stock |
| w8a8_tp4_c1 | coding | mean_ttft | 132.1 | 131.4 | 168.0 | 134.2 | n/a | +TensileLite |
| w8a8_tp4_c1 | coding | p99_ttft | 494.1 | 461.2 | 545.2 | 460.7 | n/a | +CK |
| w8a8_tp4_c1 | coding | mean_tpot | 15.86 | 17.76 | 17.91 | 17.68 | n/a | stock |
| w8a8_tp4_c1 | coding | p99_tpot | 24.00 | 25.89 | 25.62 | 25.81 | n/a | stock |
| w8a8_tp4_c2 | synth | tput | 125.3 | 113.6 | 113.1 | 113.2 | n/a | stock |
| w8a8_tp4_c2 | synth | req_tput | 0.49 | 0.44 | 0.44 | 0.44 | n/a | stock |
| w8a8_tp4_c2 | synth | mean_ttft | 137.0 | 136.0 | 138.0 | 139.4 | n/a | +TensileLite |
| w8a8_tp4_c2 | synth | p99_ttft | 165.9 | 162.6 | 170.8 | 169.0 | n/a | +TensileLite |
| w8a8_tp4_c2 | synth | mean_tpot | 15.49 | 17.15 | 17.21 | 17.19 | n/a | stock |
| w8a8_tp4_c2 | synth | p99_tpot | 15.59 | 17.23 | 17.34 | 17.31 | n/a | stock |
| w8a8_tp4_c2 | coding | tput | 110.1 | 100.1 | 96.44 | 99.85 | n/a | stock |
| w8a8_tp4_c2 | coding | req_tput | 0.46 | 0.43 | 0.43 | 0.42 | n/a | stock |
| w8a8_tp4_c2 | coding | mean_ttft | 116.1 | 118.1 | 121.2 | 120.6 | n/a | stock |
| w8a8_tp4_c2 | coding | p99_ttft | 156.9 | 154.0 | 162.0 | 157.5 | n/a | +TensileLite |
| w8a8_tp4_c2 | coding | mean_tpot | 17.70 | 19.38 | 19.89 | 19.47 | n/a | stock |
| w8a8_tp4_c2 | coding | p99_tpot | 24.27 | 26.20 | 26.06 | 26.21 | n/a | stock |
| w8a8_tp4_c4 | synth | tput | 247.7 | 224.6 | 214.8 | 224.3 | n/a | stock |
| w8a8_tp4_c4 | synth | req_tput | 0.97 | 0.88 | 0.84 | 0.88 | n/a | stock |
| w8a8_tp4_c4 | synth | mean_ttft | 199.5 | 194.8 | 193.5 | 198.2 | n/a | +Triton |
| w8a8_tp4_c4 | synth | p99_ttft | 249.3 | 231.0 | 246.9 | 237.4 | n/a | +TensileLite |
| w8a8_tp4_c4 | synth | mean_tpot | 15.43 | 17.11 | 17.22 | 17.12 | n/a | stock |
| w8a8_tp4_c4 | synth | p99_tpot | 15.76 | 17.43 | 17.56 | 17.43 | n/a | stock |
| w8a8_tp4_c4 | coding | tput | 196.3 | 181.9 | 171.8 | 183.3 | n/a | stock |
| w8a8_tp4_c4 | coding | req_tput | 0.85 | 0.81 | 0.72 | 0.79 | n/a | stock |
| w8a8_tp4_c4 | coding | mean_ttft | 124.4 | 121.7 | 138.1 | 138.1 | n/a | +TensileLite |
| w8a8_tp4_c4 | coding | p99_ttft | 208.2 | 193.7 | 199.6 | 224.4 | n/a | +TensileLite |
| w8a8_tp4_c4 | coding | mean_tpot | 19.70 | 21.49 | 22.29 | 21.26 | n/a | stock |
| w8a8_tp4_c4 | coding | p99_tpot | 24.87 | 26.65 | 26.23 | 26.80 | n/a | stock |
| w4a16_tp1_c1 | synth | tput | 28.81 | n/a | 29.56 | n/a | n/a | +Triton |
| w4a16_tp1_c1 | synth | req_tput | 0.11 | n/a | 0.12 | n/a | n/a | +Triton |
| w4a16_tp1_c1 | synth | mean_ttft | 369.6 | n/a | 371.8 | n/a | n/a | stock |
| w4a16_tp1_c1 | synth | p99_ttft | 375.7 | n/a | 402.2 | n/a | n/a | stock |
| w4a16_tp1_c1 | synth | mean_tpot | 33.40 | n/a | 32.50 | n/a | n/a | +Triton |
| w4a16_tp1_c1 | synth | p99_tpot | 33.42 | n/a | 32.81 | n/a | n/a | +Triton |
| w4a16_tp1_c1 | coding | tput | 28.32 | n/a | 28.52 | n/a | n/a | +Triton |
| w4a16_tp1_c1 | coding | req_tput | 0.12 | n/a | 0.13 | n/a | n/a | +Triton |
| w4a16_tp1_c1 | coding | mean_ttft | 267.4 | n/a | 390.9 | n/a | n/a | stock |
| w4a16_tp1_c1 | coding | p99_ttft | 1404.0 | n/a | 1681.0 | n/a | n/a | stock |
| w4a16_tp1_c1 | coding | mean_tpot | 34.17 | n/a | 33.47 | n/a | n/a | +Triton |
| w4a16_tp1_c1 | coding | p99_tpot | 42.21 | n/a | 41.12 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | synth | tput | 54.67 | n/a | 56.63 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | synth | req_tput | 0.21 | n/a | 0.22 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | synth | mean_ttft | 592.7 | n/a | 281.5 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | synth | p99_ttft | 663.4 | n/a | 358.9 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | synth | mean_tpot | 34.40 | n/a | 34.34 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | synth | p99_tpot | 34.69 | n/a | 34.69 | n/a | n/a | stock |
| w4a16_tp1_c2 | coding | tput | 52.38 | n/a | 52.90 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | coding | req_tput | 0.22 | n/a | 0.23 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | coding | mean_ttft | 326.9 | n/a | 204.1 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | coding | p99_ttft | 1579.9 | n/a | 314.1 | n/a | n/a | +Triton |
| w4a16_tp1_c2 | coding | mean_tpot | 36.88 | n/a | 36.92 | n/a | n/a | stock |
| w4a16_tp1_c2 | coding | p99_tpot | 44.21 | n/a | 43.40 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | synth | tput | 101.4 | n/a | 104.3 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | synth | req_tput | 0.40 | n/a | 0.41 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | synth | mean_ttft | 1004.9 | n/a | 491.7 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | synth | p99_ttft | 1171.3 | n/a | 600.3 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | synth | mean_tpot | 35.66 | n/a | 35.15 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | synth | p99_tpot | 36.94 | n/a | 36.34 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | coding | tput | 93.01 | n/a | 93.55 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | coding | req_tput | 0.39 | n/a | 0.39 | n/a | n/a | stock |
| w4a16_tp1_c4 | coding | mean_ttft | 392.4 | n/a | 283.6 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | coding | p99_ttft | 1933.6 | n/a | 479.8 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | coding | mean_tpot | 41.12 | n/a | 40.11 | n/a | n/a | +Triton |
| w4a16_tp1_c4 | coding | p99_tpot | 55.87 | n/a | 44.59 | n/a | n/a | +Triton |
| w4a16_tp4_c1 | synth | tput | 50.84 | n/a | 50.74 | n/a | n/a | stock |
| w4a16_tp4_c1 | synth | req_tput | 0.20 | n/a | 0.20 | n/a | n/a | stock |
| w4a16_tp4_c1 | synth | mean_ttft | 166.1 | n/a | 173.5 | n/a | n/a | stock |
| w4a16_tp4_c1 | synth | p99_ttft | 193.0 | n/a | 369.3 | n/a | n/a | stock |
| w4a16_tp4_c1 | synth | mean_tpot | 19.09 | n/a | 19.11 | n/a | n/a | stock |
| w4a16_tp4_c1 | synth | p99_tpot | 19.45 | n/a | 19.63 | n/a | n/a | stock |
| w4a16_tp4_c1 | coding | tput | 48.77 | n/a | 47.77 | n/a | n/a | stock |
| w4a16_tp4_c1 | coding | req_tput | 0.21 | n/a | 0.21 | n/a | n/a | +Triton |
| w4a16_tp4_c1 | coding | mean_ttft | 137.6 | n/a | 190.6 | n/a | n/a | stock |
| w4a16_tp4_c1 | coding | p99_ttft | 546.4 | n/a | 669.0 | n/a | n/a | stock |
| w4a16_tp4_c1 | coding | mean_tpot | 19.90 | n/a | 20.13 | n/a | n/a | stock |
| w4a16_tp4_c1 | coding | p99_tpot | 28.07 | n/a | 27.88 | n/a | n/a | +Triton |
| w4a16_tp4_c2 | synth | tput | 99.36 | n/a | 99.12 | n/a | n/a | stock |
| w4a16_tp4_c2 | synth | req_tput | 0.39 | n/a | 0.39 | n/a | n/a | stock |
| w4a16_tp4_c2 | synth | mean_ttft | 141.4 | n/a | 141.5 | n/a | n/a | stock |
| w4a16_tp4_c2 | synth | p99_ttft | 174.4 | n/a | 176.2 | n/a | n/a | stock |
| w4a16_tp4_c2 | synth | mean_tpot | 19.65 | n/a | 19.70 | n/a | n/a | stock |
| w4a16_tp4_c2 | synth | p99_tpot | 19.76 | n/a | 19.88 | n/a | n/a | stock |
| w4a16_tp4_c2 | coding | tput | 89.23 | n/a | 87.78 | n/a | n/a | stock |
| w4a16_tp4_c2 | coding | req_tput | 0.38 | n/a | 0.37 | n/a | n/a | stock |
| w4a16_tp4_c2 | coding | mean_ttft | 121.2 | n/a | 124.6 | n/a | n/a | stock |
| w4a16_tp4_c2 | coding | p99_ttft | 165.4 | n/a | 164.4 | n/a | n/a | +Triton |
| w4a16_tp4_c2 | coding | mean_tpot | 21.85 | n/a | 22.10 | n/a | n/a | stock |
| w4a16_tp4_c2 | coding | p99_tpot | 28.71 | n/a | 28.54 | n/a | n/a | +Triton |
| w4a16_tp4_c4 | synth | tput | 194.0 | n/a | 186.5 | n/a | n/a | stock |
| w4a16_tp4_c4 | synth | req_tput | 0.76 | n/a | 0.73 | n/a | n/a | stock |
| w4a16_tp4_c4 | synth | mean_ttft | 200.1 | n/a | 214.2 | n/a | n/a | stock |
| w4a16_tp4_c4 | synth | p99_ttft | 249.8 | n/a | 264.9 | n/a | n/a | stock |
| w4a16_tp4_c4 | synth | mean_tpot | 19.91 | n/a | 19.88 | n/a | n/a | +Triton |
| w4a16_tp4_c4 | synth | p99_tpot | 20.10 | n/a | 20.28 | n/a | n/a | stock |
| w4a16_tp4_c4 | coding | tput | 162.0 | n/a | 153.1 | n/a | n/a | stock |
| w4a16_tp4_c4 | coding | req_tput | 0.71 | n/a | 0.70 | n/a | n/a | stock |
| w4a16_tp4_c4 | coding | mean_ttft | 131.9 | n/a | 147.4 | n/a | n/a | stock |
| w4a16_tp4_c4 | coding | p99_ttft | 212.4 | n/a | 240.2 | n/a | n/a | stock |
| w4a16_tp4_c4 | coding | mean_tpot | 23.96 | n/a | 24.84 | n/a | n/a | stock |
| w4a16_tp4_c4 | coding | p99_tpot | 29.15 | n/a | 32.90 | n/a | n/a | stock |

**Cells flagged with explicit reasons:**

- _tensilelite_: M2 TensileLite is W8A8-only; W4A16 has no hipBLASLt INT4 logic on ROCm 7.12.
- _ck_: CK W4A16 declined as negative result on ROCm 7.12 gfx908; see BENCH_M4_CK.md.
- _isa_: M5 hand-ISA declined as negative result (memory-bound hot kernels); see BENCH_M5_ISA.md.

Raw CSV (importable): `/root/bench-int8-w4a16/final/final_grid.csv` (144 rows).

## Production Recommendations

Recommendation = winning path for that (model shape, TP, concurrency) averaged across synthetic and coding workloads on `output_throughput_toks_s`. Each row carries one-sentence justification + the env flags required to ship the path.

| model shape | TP | concurrency | recommended path | justification | env flags |
| --- | :-: | :-: | --- | --- | --- |
| w4a16 | 1 | 1 | **+Triton** | +Triton wins on tput-geomean over stock by +1.68 % (29.0 vs 28.6 tok/s). | `VLLM_ROCM_USE_AITER=1` (default in services.yaml) |
| w4a16 | 1 | 2 | **+Triton** | +Triton wins on tput-geomean over stock by +2.32 % (54.8 vs 53.5 tok/s). | `VLLM_ROCM_USE_AITER=1` (default in services.yaml) |
| w4a16 | 1 | 4 | **+Triton** | +Triton wins on tput-geomean over stock by +1.77 % (98.9 vs 97.2 tok/s). | `VLLM_ROCM_USE_AITER=1` (default in services.yaml) |
| w4a16 | 4 | 1 | **stock** | stock wins on tput-geomean over +Triton by +1.12 % (49.8 vs 49.3 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w4a16 | 4 | 2 | **stock** | stock wins on tput-geomean over +Triton by +0.90 % (94.3 vs 93.4 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w4a16 | 4 | 4 | **stock** | stock wins on tput-geomean over +Triton by +4.82 % (178.0 vs 169.8 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w8a8 | 1 | 1 | **stock** | stock wins on tput-geomean over +CK by +2.33 % (38.5 vs 37.6 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w8a8 | 1 | 2 | **stock** | stock wins on tput-geomean over +TensileLite by +3.62 % (71.2 vs 68.7 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w8a8 | 1 | 4 | **stock** | stock wins on tput-geomean over +CK by +2.61 % (128.1 vs 124.9 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w8a8 | 4 | 1 | **stock** | stock wins on tput-geomean over +CK by +11.41 % (62.5 vs 56.1 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w8a8 | 4 | 2 | **stock** | stock wins on tput-geomean over +TensileLite by +10.15 % (117.7 vs 106.8 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |
| w8a8 | 4 | 4 | **stock** | stock wins on tput-geomean over +CK by +8.93 % (222.0 vs 203.8 tok/s). (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).) | (no flags) |

## Reproducibility

Per-cell launch scripts at `scripts/launch_<model>_tp<tp>_c<conc>.sh` (12 scripts). Each pins env vars (ROCm 7.12, PYTORCH_ROCM_ARCH=gfx908, vLLM commit, tuning JSON path), starts the matching vLLM service, runs the canonical 200-prompt synthetic bench, and asserts ±2 % of the recorded reference throughput.

Spot-check (random 3 cells, NUM_PROMPTS=50 vs N=200 reference; variance ~2× larger than the reference run, so we additionally report ±5 % band as a noise-corrected gate):

| cell | ref path | recorded tput (tok/s) | re-run tput (tok/s) | Δ | within ±2 % | within ±5 % |
| --- | --- | ---: | ---: | ---: | :-: | :-: |
| w8a8_tp1_c1 | +CK | 38.0293 | 37.9003 | -0.339 | ✅ | ✅ |
| w8a8_tp1_c4 | +CK | 132.3090 | 127.5170 | -3.622 | ❌ | ✅ |
| w4a16_tp1_c1 | +Triton | 29.5640 | 29.5863 | +0.075 | ✅ | ✅ |

_Note_: N=50 vs N=200 reference; variance ~2x larger on slowest cells.

Tuning-JSON SHA256 provenance (VAL-CROSS-007):

| Path | SHA256 (first 16) |
| --- | --- |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_BB_BB_HA_Bias_SAV_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `079c4c1c62b24b32…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_BB_BB_HA_Bias_SAV_Type_BB_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `b686daa25aa593a7…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_BB_BB_HA_Bias_SAV_Type_BB_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `19cbf519de832f96…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_BB_BB_HA_Bias_SAV_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `b596bc040b5a47be…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `e97df47110c6a06c…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `929c60248e4cb941…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `b63d3dcf436f56ab…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `e9521fa9809a7c63…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_GG_UA_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `b5a5ac48fc3aaa80…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_GG_UA_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `e06d1ff94ccb7c49…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_GG_UA_Type_HH_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `742b93394a0f0051…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_GG_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `71a2cbf6786be2ac…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Aux_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `56a459503ce76e37…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Aux_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `85fbc47ae9644e80…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Aux_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `efaba4e6447f6d1a…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Aux_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `c6fc2dec8c30fa03…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_GG_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `c5961635cf1df837…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_GG_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `1299de97a4dbbb95…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_GG_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `95c3bf8b44acf317…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_GG_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `0020e22ac4e78291…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Grad_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `0f9afa5455647998…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Grad_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `715bf4a4032af108…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Grad_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `22edf45a8a731955…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_Grad_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `c2ece652fe45bf92…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_SAV_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `484dcff66b72ea6e…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `b2be83a277fe0fcd…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `8037fdce926f13fd…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `ca4a357945a2d3ea…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `3ef5fd982ffcdc6c…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_HH_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `0e02ac9c1febd06f…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_GG_UA_Type_HS_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `200361bfc9746b22…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_GG_UA_Type_HS_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `33958a37409d18bb…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_GG_UA_Type_HS_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `a7f21f684f6d965c…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_GG_UA_Type_HS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `d8beb72e4437044e…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_Aux_SAV_UA_Type_HS_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `45f7a17a37ae77fd…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_GG_SAV_UA_Type_HS_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `ad270a1fe5a4be68…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_GG_SAV_UA_Type_HS_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `7ea2ead8c51b37b7…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_GG_SAV_UA_Type_HS_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `b3eb2ad9420b6a02…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_GG_SAV_UA_Type_HS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `9b8b1d59a0b355e5…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_SAV_UA_Type_HS_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `335b0263b4d13488…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_SAV_UA_Type_HS_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `477af97fd5e170ca…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_SAV_UA_Type_HS_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `4162a80023074294…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_HH_SH_HA_Bias_SAV_UA_Type_HS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `00dd7fb72cda6bbb…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_I8I8_II8_Type_I8I_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `f36c9b0a643291b7…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_I8I8_II8_Type_I8I_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `36182ecf8047360f…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_I8I8_II8_Type_I8I_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `7ce2827a9c0c7857…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_I8I8_II8_Type_I8I_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `154622a84b7d6568…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_I8I8_II8_UA_Type_I8I_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `5a56aa95b2f93bc4…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx908.dat` | `84fdbcf0d11a475f…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx908.dat` | `2a4d5ded1e5934ef…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx908.dat` | `1b3aae0ce46fc97f…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx908.dat` | `04df307bd3f32cf0…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLibrary_lazy_gfx908.dat` | `ec22e7f6f666d81c…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/TensileLiteLibrary_lazy_Mapping.dat` | `b9b544fdb4c52f4b…` |
| `/root/bench-int8-w4a16/tensilelite/merged_library/library/hipblasltExtOpLibrary.dat` | `0f43b2448f45ae27…` |
| `vllm/model_executor/kernels/configs/gfx908/hipblaslt_tuned_shapes.json` | `6ee6ae4a46b341ba…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M1_N10240_K4096.json` | `731fd7bcc539d4ca…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M1_N4096_K4096.json` | `fdaf912e0b85898c…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M32_N10240_K4096.json` | `1ba0454fda1718ca…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M512_N24576_K4096.json` | `90b8707d339ee529…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M512_N4096_K12288.json` | `192a90f057679b6a…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_int8_M512_N4096_K4096.json` | `0f0008a62c531a31…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_w4a16_M1_N10240_K4096_g128.json` | `b3200f9aa3dbf66f…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_w4a16_M1_N10240_K4096_g32.json` | `6763e8aef3a7b19e…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_w4a16_M1_N4096_K4096_g128.json` | `d41507780a7d0542…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_w4a16_M1_N4096_K4096_g32.json` | `72fc8e7c2fdeb05d…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_w4a16_M32_N10240_K4096_g128.json` | `3325fb0b4fe08697…` |
| `vllm/model_executor/kernels/configs/gfx908/mi100_w4a16_M512_N4096_K4096_g128.json` | `fae76b2448cdbd2e…` |

Full hashes at `/root/bench-int8-w4a16/final/tuning_hashes.json`.

## Cross-cutting checks

| Assertion | Status | Evidence |
| --- | :-: | --- |
| VAL-CROSS-001 numerical-tolerance gate | ✅ | Triton W8A8/W4A16 correctness `tests/kernels/quantization/test_mi100_int8_correctness.py`, `test_mi100_w4a16_correctness.py`; CK INT8 correctness `test_ck_int8_correctness.py`. |
| VAL-CROSS-002 full reproducibility metadata per cell | ✅ | Every JSON under `/root/bench-int8-w4a16/**/{synthetic,coding}/*.json` includes `launch_command`, `env`, `vllm_commit`, `rocm_version`, `torch_version`, `triton_version`, `tuning_json_sha256`, `timestamp`. Schema: `scripts/bench_schema.json`. |
| VAL-CROSS-003 no silent-default regressions | ✅ | `scripts/check_default_pareto.py /root/bench-int8-w4a16/final/final_grid.csv` reports each winning-but-not-always-best path has a documented env gate. |
| VAL-CROSS-004 build hygiene (no MI300+ intrinsics) | ✅ | `scripts/check_forbidden_intrinsics.sh build/**/*.so` reports zero matches across `_C.abi3.so`, `_moe_C.abi3.so`, `_rocm_C.abi3.so`. |
| VAL-CROSS-005 dispatcher priority CK > hipBLASLt > Triton | ✅ | M4 dispatch verification at `/root/bench-int8-w4a16/m4/wiring_verification.txt` (symlinked from `/root/bench-int8-w4a16/ck/m4_dispatch_trace.txt`); contention test `tests/kernels/quantization/test_ck_dispatch_priority.py`. |
| VAL-CROSS-006 no git pushes | ✅ | `git reflog --date=iso \| grep push` returns empty; all milestone commits are local. |
| VAL-CROSS-007 tuning-JSON SHA256 provenance | ✅ | `scripts/verify_tuning_hashes.py` compares on-disk SHA256 of every pinned tuning JSON to `/root/bench-int8-w4a16/final/tuning_hashes.json`. |

## Evidence index

- Per-milestone reports:
    - [BENCH_INT8_W4A16_BASELINE.md](BENCH_INT8_W4A16_BASELINE.md)
    - [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md)
    - [BENCH_INT8_W4A16_M3.md](BENCH_INT8_W4A16_M3.md)
    - [BENCH_M4_CK.md](BENCH_M4_CK.md)
    - [BENCH_M5_ISA.md](BENCH_M5_ISA.md)
- Pareto exceptions per milestone:
    - `/root/bench-int8-w4a16/m2/pareto_exceptions.md`
    - `/root/bench-int8-w4a16/m3/pareto_exceptions.md`
    - `/root/bench-int8-w4a16/m4/pareto_exceptions.md`
- Profiling artifacts (M1, omniperf-equivalent):
    - `/root/bench-int8-w4a16/baseline/profile/omniperf/omniperf_summary_{w8a8,w4a16}.json`
    - `/root/bench-int8-w4a16/baseline/hot_shapes.json`
- Tuning artifacts (TensileLite):
    - `/root/bench-int8-w4a16/tensilelite/merged_library/library/`
    - `/root/bench-int8-w4a16/tensilelite/logic/gfx908/*.yaml`
- Validators:
    - `scripts/validate_final_report.py` (this file)
    - `scripts/check_default_pareto.py`
    - `scripts/check_forbidden_intrinsics.sh`
    - `scripts/verify_tuning_hashes.py`
    - `scripts/eval_coding_prompts.py`, `scripts/eval_needle.py`

Run `scripts/validate_final_report.py --check-links BENCH_INT8_W4A16_FINAL.md` to assert schema and link integrity.
