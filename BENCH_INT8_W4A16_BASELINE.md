# BENCH — INT8 / W4A16 Baseline (gfx908 / MI100)

This document aggregates Milestone 0 + Milestone 1 baseline numbers for the
`vllm-project/vllm` MI100 quant-kernel optimization mission. Milestone 1
content will be added by the M1 worker once Milestone 0's GO/NO-GO decision
is resolved by the user/orchestrator.

## Hardware / software manifest

| | |
|---|---|
| Hosts | 1× node, 4× AMD Instinct MI100 (gfx908), 32 GB VRAM each |
| Driver | amdgpu-dkms 6.19.0+, perf=high, 250 W cap |
| ROCm | 7.12 (`/opt/rocm/core-7.12`) |
| PyTorch | 2.11.0+rocm7.2 |
| pytorch-triton-rocm | 3.5.1 |
| vLLM | 0.20.2rc1.dev93+g85994b2e3 (mission worktree, editable install) |
| Models | `/models/Qwen3.5-9B-{w8a8,w4a16}`, FP16 ref `/models/Qwen3.5-9B` |

## Milestone 0 — Quality Gate (hard gate)

All evidence in `/root/bench-int8-w4a16/baseline/`. See `M0_DECISION.md` for
the full per-assertion table.

| Gate | w8a8 | w4a16 |
|------|------|-------|
| Artifact present + valid `quantization_config` | PASS | PASS |
| vLLM smoke-load TP=1 (`/v1/models` 200) | PASS | PASS |
| vLLM smoke-load TP=4 (4 GPUs at 96+% VRAM) | PASS | PASS (eager) |
| Coherence ≥8/10 syntax, no >50% n-gram repetition (TP=1, TP=4) | 8/10, 8/10 | 8/10, 8/10 |
| NIAH 5/5 verbatim @ 8k | 5/5 | 5/5 |
| Determinism + no NaN/Inf | PASS | PASS |
| Wikitext-2 perplexity Δ ≤ +1% vs FP16 | **FAIL +1.37 %** | **FAIL +2.91 %** |

### Wikitext-2 perplexity (50 × 512 tokens, seed 0)

| Model | ppl |
|-------|------|
| FP16 (`Qwen/Qwen3.5-9B`) | 9.5254 |
| W8A8 (`RedHatAI/Qwen3.5-9B-quantized.w8a8`) | 9.6561 (Δ +1.37 %) |
| W4A16 (`apolo13x/Qwen3.5-9B-quantized.w4a16`) | 9.8030 (Δ +2.91 %) |

### M0 decision

**`DECISION: NO-GO`** — Wikitext-2 perplexity gate failed for both artifacts.

Generation quality (coherence, needle retrieval, determinism) PASSES on both
artifacts. The prior-round W8A8 GatedDeltaNet failure mode (gibberish output)
**is NOT reproduced** by these RedHatAI / apolo13x artifacts.

The Δ values (+1.37 %, +2.91 %) are within published norms for INT8 channel
+ INT4 group-128 PTQ. They merely exceed the contract's strict +1 % gate.

Mission paused at M0; orchestrator returned the decision to the user for a
pivot-or-relax-gate choice. See `/root/bench-int8-w4a16/baseline/M0_DECISION.md`.

## Milestone 1 — Baseline numbers

*(Pending M0 GO; will be filled in by the M1 worker.)*
