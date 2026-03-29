# TurboQuant vLLM Integration Attempt - Technical Report

**Date**: 2026-03-29
**vLLM Version**: 0.18.1.dev4+g5c0b8b38d
**TurboQuant Version**: 0.2.0
**Hardware**: 4x AMD MI100 (gfx908), ROCm 7.12

## Executive Summary

**TurboQuant Triton kernels work on ROCm/gfx908**, but the vLLM integration via monkey-patching has **architectural incompatibilities** with vLLM v0.18.1's multi-process execution model.

**Status**: Integration NOT ACHIEVED due to vLLM architecture, not TurboQuant kernel issues.

## Validation Contract Status

| Assertion | Status | Notes |
|-----------|--------|-------|
| VAL-TQ-001 | PASS | TurboQuant installed, import succeeds |
| VAL-TQ-002 | PASS | Triton kernels compile on ROCm/gfx908 |
| VAL-TQ-003 | FAIL | Cannot install hooks on multi-process workers |
| VAL-TQ-004 | N/A | KV cache savings not measurable without hooks |
| VAL-TQ-005 | PASS* | 10/10 coding prompts coherent (baseline, not TQ) |
| VAL-TQ-006 | PASS* | Needle-in-haystack passes (baseline, not TQ) |
| VAL-TQ-007 | PASS | System falls back gracefully, no crashes |

*Quality tests pass on baseline vLLM without TurboQuant active.

## Technical Findings

### 1. TurboQuant Installation - SUCCESS

```bash
/opt/vllm-env/bin/pip install -e /opt/turboquant
# TurboQuant 0.2.0 installed
# All Triton kernels compile on ROCm/gfx908
```

**Triton Kernel Tests**:
- `_turboquant_mse_score_kernel`: PASS
- `_turboquant_qjl_score_kernel`: PASS
- `_turboquant_fused_decode_kernel`: PASS

### 2. vLLM Architecture Analysis

vLLM v0.18.1 uses a **multi-process architecture**:

```
APIServer (main process)
    └── EngineCore (separate process)
            └── MultiprocExecutor
                    ├── Worker_TP0 (separate process) -> GPUModelRunner
                    ├── Worker_TP1 (separate process)
                    ├── Worker_TP2 (separate process)
                    └── Worker_TP3 (separate process)
```

**The Problem**: TurboQuant's `install_hooks()` requires direct access to `GPUModelRunner` which lives in separate `Worker_TP*` processes. When hooks are installed from the main process, they don't affect the worker processes.

### 3. Attempted Integration Approaches

#### Approach A: Direct LLM Class with Hooks

```python
from vllm import LLM
from turboquant.integration.vllm import install_hooks

llm = LLM(model="/models/Qwen3.5-9B", ...)
executor = llm.llm_engine.model_executor
workers = executor.workers  # Access worker processes
model_runner = workers[0].model_runner  # FAILS: model_runner in different process
install_hooks(model_runner)  # Never reaches this point
```

**Result**: Engine initialization hangs during profile_run due to multimodal encoder issues.

#### Approach B: API Server with collective_rpc

```python
# Patch Executor to install hooks via collective_rpc
def _worker_install_tq(worker):
    model_runner = worker.model_runner
    install_hooks(model_runner, ...)
    
executor.collective_rpc(_worker_install_tq)
```

**Result**: Hooks install but are lost after worker process restarts or between requests.

#### Approach C: enable_no_alloc() Early Patching

```python
from turboquant.vllm_attn_backend import enable_no_alloc
enable_no_alloc(key_bits=3, value_bits=2, ...)
# Then create LLM engine
```

**Result**: Patches `Executor.get_kv_cache_specs` but workers spawn fresh processes that don't inherit the patches.

### 4. Root Cause Analysis

**TurboQuant's Integration Design**:
- Designed for single-process or thread-based vLLM
- Assumes model_runner is in same process as caller
- Uses monkey-patching on instance methods

**vLLM v0.18.1 Reality**:
- Workers are separate OS processes (spawn, not fork)
- Model runners are in GPU worker processes
- No IPC mechanism for method patching
- collective_rpc sends functions but patches don't persist

### 5. Quality Tests Results (Baseline vLLM)

Without TurboQuant hooks, the baseline vLLM server passes all quality tests:

**10 Coding Prompts**: 10/10 PASS
- Python function generation
- Algorithm explanations
- TypeScript interfaces
- SQL queries
- React components

**Needle-in-Haystack (8k context)**: PASS
- Needle found correctly at 8000 token context

**Throughput**: 87.5 tok/s (eager mode, no graphs)

## Recommendations

### Option 1: Wait for vLLM Native Integration
TurboQuant integration requires changes to vLLM core to support:
- Worker process initialization hooks
- Persistent attention backend configuration
- IPC-based configuration propagation

### Option 2: Custom vLLM Fork
Create a vLLM fork that:
- Bakes TurboQuant hooks into worker initialization
- Uses a custom attention backend
- Requires maintaining fork in sync with upstream

### Option 3: Use TurboQuant Standalone
For non-vLLM inference:
- Use TurboQuant with HuggingFace Transformers directly
- Manual KV cache management
- Not suitable for production serving

### Option 4: Alternative Optimizations
For MI100, continue with proven optimizations:
- FULL_DECODE_ONLY graph mode (+68% TPOT improvement)
- Prefix caching (TTFT reduction on cache hits)
- Config tuning (max-model-len, gpu-memory-utilization)

## Files Created

- `/root/benchmark-scripts/launch-turboquant.sh` - Integration launch script (non-working)
- `/root/benchmark-scripts/run_turboquant_test.py` - Integration test script
- `/root/benchmark-scripts/test_turboquant_direct.py` - Direct LLM test script
- `/root/benchmark-scripts/run_quality_tests.py` - Quality test script
- `/root/benchmark-results/turboquant_quality_20260329_150825.json` - Quality test results

## Conclusion

TurboQuant's Triton kernels work correctly on ROCm/gfx908, confirming the core technology is portable to AMD hardware. However, the vLLM integration layer assumes a single-process architecture incompatible with vLLM v0.18.1's multi-process design.

**Recommendation**: Mark TurboQuant integration as BLOCKED pending either:
1. TurboQuant updates for vLLM v0.18.x architecture
2. vLLM adding official extension hooks for attention backends
3. Custom fork development (not recommended for production)

For the MI100 optimization mission, continue with graph mode + prefix caching + config tuning which provides significant improvements without TurboQuant.
