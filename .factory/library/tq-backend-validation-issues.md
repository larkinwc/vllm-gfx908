# TurboQuant Backend Validation Issues

**Date:** 2026-03-29
**Feature:** capture-mode-correctness-validation
**Status:** FIX APPLIED - GPU environment issue blocking validation

## Summary

The TQ backend registration fix has been successfully implemented:
- Changed from `AttentionBackendEnum.CUSTOM` to `AttentionBackendEnum.TRITON_ATTN` override
- This allows full-attention layers to use TQ while GDN layers continue using their own backend

## The Fix

**Problem:** Previous worker used `register_backend(AttentionBackendEnum.CUSTOM)` + `--attention-backend CUSTOM` which forced ALL layers to use TQ backend, crashing GDN layers.

**Solution:** Use `register_backend(AttentionBackendEnum.TRITON_ATTN, 'turboquant.backends.vllm_rocm.TurboQuantTritonBackend')` to OVERRIDE the TRITON_ATTN backend (which is the default on ROCm).

**Why this works:**
- vLLM's `_get_backend_priorities()` on ROCm selects TRITON_ATTN as the default for standard attention
- ROCM_ATTN is only included if `use_prefill_decode_attention` is True
- Full-attention layers (8) use TRITON_ATTN -> get TQ backend
- GDN/linear-attention layers (24) use MambaAttentionBackendEnum.GDN_ATTN routing -> untouched
- No `--attention-backend` CLI flag needed

## Files Modified

1. `/opt/turboquant/turboquant/backends/vllm_rocm.py`:
   - Changed to extend `TritonAttentionBackend` instead of `RocmAttentionBackend`
   - `get_name()` returns "TRITON_ATTN" to map to the overridden enum
   - Class renamed to `TurboQuantTritonBackend` with `TurboQuantTritonImpl`
   - Backward-compatible aliases `TurboQuantRocmBackend` and `TurboQuantRocmImpl`

2. `/root/benchmark-scripts/launch-tq-backend.sh`:
   - Updated sitecustomize.py to register via `TRITON_ATTN` override
   - Removed `--attention-backend CUSTOM` flag (not needed)

3. `/root/benchmark-scripts/test_tq_backend_registration.py`:
   - Updated to test `TRITON_ATTN` override instead of `CUSTOM`

## Validation Status

- VAL-REG-001: ✅ PASSED (import works)
- VAL-REG-002: ✅ PASSED (registration with TRITON_ATTN works)
- Backend selection: ✅ CONFIRMED (logs show "Using TRITON_ATTN attention backend")
- Server startup: ✅ PASSED (health check returns 200 within 80s)

## Blocking Issue

GPU environment issue affecting both TQ and non-TQ inference:
- `hipErrorLaunchFailure` during GDN layer's `causal_conv1d_fn` Triton kernel
- This is a pre-existing issue unrelated to TQ backend changes
- Occurs in both TQ-enabled and baseline vLLM servers
- May be related to GPU thermal state, driver version, or model weights

## Next Steps

1. Wait for GPU environment to stabilize or reboot if needed
2. Re-run validation tests:
   - VAL-CAP-001: 5 prompts produce identical output
   - VAL-CAP-002: TQ stats collected
   - VAL-CAP-003: 4 concurrent requests
   - VAL-CAP-004: VRAM stability
   - VAL-REG-004: FULL_DECODE_ONLY graph mode
   - VAL-GRAPH-001: TQ capture mode with graphs
