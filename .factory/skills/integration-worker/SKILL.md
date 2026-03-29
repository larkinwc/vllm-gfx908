---
name: integration-worker
description: Creates TurboQuant vLLM attention backend, registration, and launch infrastructure
---

# Integration Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that involve:
- Creating the TurboQuantRocmBackend and TurboQuantRocmImpl classes
- Registering the backend with vLLM's registry API
- Creating launch scripts that set up and start vLLM with the TQ backend
- Implementing capture_only and hybrid decode modes
- Testing HIP graph compatibility with TQ kernels
- Quality validation (coding prompts, needle-in-haystack)

## Required Skills

None.

## Work Procedure

### Step 1: Read Context

Read these files before starting implementation:
- `.factory/library/architecture.md` -- system architecture
- `.factory/library/environment.md` -- env vars, paths
- `AGENTS.md` -- boundaries, critical technical context
- `/opt/turboquant/turboquant/integration/vllm.py` -- existing TQ integration (reference for what hooks exist)
- `/opt/turboquant/turboquant/triton_kernels.py` -- TQ Triton kernels
- `/opt/turboquant/turboquant/store.py`, `capture.py`, `quantizer.py` -- core TQ components

Also read vLLM's backend interface:
- The vLLM worktree's `vllm/v1/attention/backends/rocm_attn.py` -- RocmAttentionBackend and RocmAttentionImpl to extend
- `vllm/v1/attention/backends/registry.py` -- register_backend API
- `vllm/v1/attention/backend.py` -- AttentionBackend base class

### Step 2: Write Tests First

Before implementing the backend, write test scripts that will validate it:
- Import test: `turboquant.backends.vllm_rocm` imports cleanly
- Registration test: `register_backend()` succeeds, CUSTOM resolves to TQ backend
- Functional test: server starts, health check passes, requests return valid responses
- If hybrid mode: quality test with 10 coding prompts and needle-in-haystack

Place test scripts at `/root/benchmark-scripts/test_tq_backend_*.py`.

Run each test to confirm it FAILS before implementation (since the backend doesn't exist yet).

### Step 3: Implement the Backend

Create `/opt/turboquant/turboquant/backends/__init__.py` and `/opt/turboquant/turboquant/backends/vllm_rocm.py`.

**TurboQuantRocmBackend** must:
- Extend `RocmAttentionBackend` from `vllm.v1.attention.backends.rocm_attn`
- Override `get_name()` to return "TURBOQUANT_ROCM"
- Override `get_impl_cls()` to return `TurboQuantRocmImpl`
- Keep all other class methods delegating to super (kv_cache_shape, head_sizes, etc.)

**TurboQuantRocmImpl** must:
- Extend `RocmAttentionImpl`
- In `__init__`: create per-layer TQ state (CompressedKVStore, KVCaptureEngine)
- Override `do_kv_cache_update()`: call super() for standard paged cache, then capture K/V into TQ store
- Override `forward()`:
  - In capture_only mode: always delegate to super().forward()
  - In hybrid mode: use TQ hybrid decode for decode tokens (when compressed store has enough history), fall back to super() for prefill

**Critical implementation details:**
- Layer index tracking: use a class-level counter in `__init__` (increment per instance)
- Mode control: use environment variable `TURBOQUANT_MODE` (capture_only | hybrid) read at init time
- Head dim for Qwen3.5-9B: 256 (from `self.head_size` in RocmAttentionImpl)
- num_kv_heads: from `self.num_kv_heads` in RocmAttentionImpl
- Device: from the tensors passed to forward/do_kv_cache_update

### Step 4: Create Launch Script

Create `/root/benchmark-scripts/launch-tq-backend.sh` that:
1. Registers the TQ backend before starting vLLM
2. Starts vLLM with `--attention-backend CUSTOM` (or equivalent)
3. Configures TQ mode via environment variable
4. Handles all MI100-specific settings (TORCH_COMPILE_DISABLE=1, etc.)

The launch script should be a Python wrapper that:
```python
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
register_backend(AttentionBackendEnum.CUSTOM, "turboquant.backends.vllm_rocm.TurboQuantRocmBackend")
# Then start vLLM engine
```

### Step 5: Run Tests and Verify

1. Run all test scripts to verify they PASS
2. Start server with TQ backend, send test requests via curl
3. For capture_only: verify output matches baseline
4. For hybrid: verify output quality (coherent, syntactically valid)
5. Check server logs for TQ-specific messages (layer initialization, compression stats)
6. Verify no VRAM leaks: check rocm-smi before and after serving 20 requests

### Step 6: Manual Verification

- Start the server manually, send 3 diverse prompts via curl, verify responses make sense
- Check TQ stats in logs or via diagnostic output
- If graph mode: verify graph capture count in logs
- Stop server cleanly, verify no orphaned processes

## Example Handoff

```json
{
  "salientSummary": "Created TurboQuantRocmBackend extending RocmAttentionBackend with capture_only mode. Backend registers via register_backend(CUSTOM), server starts on port 8000, all 5 test requests return identical output to baseline. 8 TQ layers initialized for Qwen3.5-9B full-attention layers, 24 linear layers use standard GDN backend.",
  "whatWasImplemented": "Created /opt/turboquant/turboquant/backends/vllm_rocm.py with TurboQuantRocmBackend and TurboQuantRocmImpl. Created /root/benchmark-scripts/launch-tq-backend.sh for server startup with TQ backend registration. Created test scripts for import, registration, and functional validation.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "/opt/vllm-env/bin/python3 -c 'from turboquant.backends.vllm_rocm import TurboQuantRocmBackend; print(\"OK\")'",
        "exitCode": 0,
        "observation": "Import succeeds, TurboQuantRocmBackend class available"
      },
      {
        "command": "/opt/vllm-env/bin/python3 /root/benchmark-scripts/test_tq_backend_registration.py",
        "exitCode": 0,
        "observation": "register_backend succeeds, CUSTOM resolves to TurboQuantRocmBackend"
      },
      {
        "command": "curl -sf http://localhost:8000/health",
        "exitCode": 0,
        "observation": "Health check 200 after 85s startup with TQ backend"
      },
      {
        "command": "/opt/vllm-env/bin/python3 /root/benchmark-scripts/test_tq_backend_functional.py",
        "exitCode": 0,
        "observation": "5/5 requests return identical output to baseline, TQ stats show 8 layers capturing"
      },
      {
        "command": "rocm-smi --showmeminfo vram --json",
        "exitCode": 0,
        "observation": "VRAM stable at 93% after 20 requests, no leak detected"
      }
    ],
    "interactiveChecks": [
      {
        "action": "Sent 3 curl requests with different coding prompts",
        "observed": "All 3 responses coherent and correct, server logs show TQ capture on full-attention layers"
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "/root/benchmark-scripts/test_tq_backend_registration.py",
        "cases": [
          {"name": "test_import", "verifies": "turboquant.backends.vllm_rocm imports without error"},
          {"name": "test_register", "verifies": "register_backend succeeds with CUSTOM enum"}
        ]
      },
      {
        "file": "/root/benchmark-scripts/test_tq_backend_functional.py",
        "cases": [
          {"name": "test_server_health", "verifies": "vLLM starts with TQ backend"},
          {"name": "test_identical_output", "verifies": "capture_only output matches baseline"},
          {"name": "test_tq_stats", "verifies": "compression stats show active capture"}
        ]
      }
    ],
    "coverage": "Import, registration, server startup, output correctness, compression stats"
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- vLLM's `register_backend` API doesn't work as documented (missing, different signature)
- RocmAttentionImpl interface has changed and cannot be extended as planned
- TQ Triton kernels crash during graph capture (return with evidence for graph compatibility assessment)
- VRAM exhaustion prevents server startup with TQ backend
- Backend selection doesn't route to CUSTOM even with correct config
