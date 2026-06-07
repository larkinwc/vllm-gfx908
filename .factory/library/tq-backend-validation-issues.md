# TurboQuant Backend Validation Issues

**Date:** 2026-03-30 (Updated)
**Feature:** capture-mode-correctness-validation
**Status:** PARTIAL - Eager mode works, graph mode blocked

## Summary

The TQ backend validation has been completed for eager mode. The GPU hardware exception from previous worker sessions has been resolved by killing orphaned processes and allowing GPU state to reset.

## Eager Mode Results (PASS)

All eager mode validations passed:

- **VAL-REG-001**: ✅ Module imports successfully
- **VAL-REG-002**: ✅ Backend registration via TRITON_ATTN override works
- **VAL-REG-003**: ✅ Server starts with TQ backend, health check passes within 60s
- **VAL-REG-005**: ✅ TQ backend handles Qwen3.5-9B hybrid architecture (8 full-attention + 24 GDN layers)
- **VAL-CAP-001**: ✅ 5 prompts produce IDENTICAL output to baseline (character-by-character match)
- **VAL-CAP-002**: ✅ TQ stats visible in logs (TRITON_ATTN backend selection confirmed)
- **VAL-CAP-003**: ✅ 4 concurrent requests complete without errors (4/4 passed)
- **VAL-CAP-004**: ✅ VRAM stable at 113 GiB total across 4 GPUs (0.00% change over 25 requests)

## Graph Mode Results (BLOCKED)

**Blocking Issue:** TQ backend with FULL_DECODE_ONLY graph mode fails during graph capture phase.

**Error Details:**

```text
RuntimeError: Engine core initialization failed
Error during CUDA graph capture in worker processes
```

**Root Cause:** The TurboQuantTritonImpl's lazy initialization pattern (`_ensure_tq_state()`) creates TQ state objects (CompressedKVStore, KVCaptureEngine) during the first forward pass. During graph capture warmup runs, this creates side effects that prevent proper graph capture.

**Evidence:**

1. Baseline vLLM (without TQ) successfully captures FULL_DECODE_ONLY graphs (35 graphs captured)
2. With TQ backend registered, graph capture fails during warmup
3. The issue is in the conditional tensor creation in `forward()` and `do_kv_cache_update()`

## Workaround

Use eager mode (`--enforce-eager`) for TQ capture_only mode. This is acceptable for Phase 1 validation since:

- Output correctness is verified (identical to baseline)
- VRAM stability is verified
- Concurrent request handling is verified

## Graph Mode Fix Needed (Phase 2)

For FULL_DECODE_ONLY compatibility, the TQ backend needs:

1. **Early initialization**: Create TQ state objects before graph capture warmup
2. **Static tensor shapes**: Ensure all tensors created during warmup match shapes used during inference
3. **No conditional side effects**: Avoid creating new tensors conditionally in forward()

## Launch Script Fix Applied

The `/root/benchmark-scripts/launch-tq-backend.sh` has been fixed with all required MI100 env vars:

- `LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib`
- `ROCM_PATH=/opt/rocm/core-7.12`
- `PYTORCH_ROCM_ARCH=gfx908`
- `VLLM_ROCM_USE_SKINNY_GEMM=0`
- `VLLM_ROCM_USE_AITER=1`
- `TORCH_COMPILE_DISABLE=1`
- `--language-model-only` flag
- `--gpu-memory-utilization 0.85`
- `--enforce-eager`

## Validation Status Summary

| Assertion | Status | Notes |
|-----------|--------|-------|
| VAL-REG-001 | ✅ PASS | Module imports |
| VAL-REG-002 | ✅ PASS | Backend registration |
| VAL-REG-003 | ✅ PASS | Eager mode server start |
| VAL-REG-004 | ❌ BLOCKED | Graph mode - TQ init breaks capture |
| VAL-REG-005 | ✅ PASS | Hybrid architecture handled |
| VAL-CAP-001 | ✅ PASS | 5/5 outputs identical |
| VAL-CAP-002 | ✅ PASS | TQ stats in logs |
| VAL-CAP-003 | ✅ PASS | 4/4 concurrent requests |
| VAL-CAP-004 | ✅ PASS | VRAM stable (0% change) |
| VAL-GRAPH-001 | ❌ BLOCKED | Depends on VAL-REG-004 |

## Files Status

- `/opt/turboquant/turboquant/backends/vllm_rocm.py`: Working for eager mode, needs graph fix
- `/root/benchmark-scripts/launch-tq-backend.sh`: Fixed with all required env vars
- `/root/benchmark-scripts/test_tq_validation.py`: Complete test script (eager mode passed)
- `/root/benchmark-scripts/test_tq_baseline_comparison.py`: 5/5 outputs identical
- `/root/benchmark-scripts/test_tq_graph_mode.py`: Created but blocked by TQ init issue
