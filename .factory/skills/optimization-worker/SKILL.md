---
name: optimization-worker
description: Applies vLLM serving optimizations (graph modes, config tuning, MTP), tests them, and benchmarks improvements
---

# Optimization Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that involve:

- Enabling/testing CUDA/HIP graph modes on MI100
- Configuring MTP speculative decoding
- Tuning serving parameters (max-model-len, batched-tokens, prefix caching)
- Creating optimized launch scripts
- Measuring performance impact of config changes

## Required Skills

None

## Work Procedure

1. **Read feature requirements** from the assigned feature. Understand exactly which optimization to apply and what success looks like.

2. **Prepare environment**:
   ```bash
   export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
   export ROCM_PATH=/opt/rocm/core-7.12
   export PYTORCH_ROCM_ARCH=gfx908
   export VLLM_ROCM_USE_SKINNY_GEMM=0
   export VLLM_ROCM_USE_AITER=1
   ```

3. **Stop any running vLLM server**: `lsof -ti :8000 | xargs kill -9 2>/dev/null; sleep 5`

4. **Read baseline results** from `/root/benchmark-results/` to know the comparison target.

5. **Apply the optimization**:
   - For config changes: Create a new launch script in `/root/benchmark-scripts/launch-<optimization>.sh`
   - For code changes: Modify vLLM source at `/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/chatty-books-share-1774757984587/` (the working tree) and rebuild if needed
   - For MTP: Add `--speculative_config '{"method":"mtp","num_speculative_tokens":N}'` to launch args
   - For graph modes: Remove `--enforce-eager`, potentially remove `TORCH_COMPILE_DISABLE=1`, add `--compilation-config '{"cudagraph_mode":"<MODE>"}'`

6. **Test the optimization**:
   - Start server with new config as background process
   - Verify health check passes within 120 seconds
   - Send 5 test requests to verify correctness
   - Check server logs for errors, warnings, graph activation messages
   - If server crashes or produces errors, diagnose and try fallback configurations

7. **Benchmark the optimization**:
   - Run the same benchmark scripts used for baseline (from `/root/benchmark-scripts/`)
   - Save results to `/root/benchmark-results/<optimization>-<model>-<timestamp>.json`
   - Compare against baseline results

8. **Stability test**: Send 50-100 requests at target concurrency (4 users) to verify no crashes.

9. **Document results**: Record what worked, what didn't, and performance delta vs baseline.

### Critical Notes

- **Graph modes on MI100**: Start with `FULL_DECODE_ONLY` (safest). If it works, try `PIECEWISE`. `FULL_AND_PIECEWISE` is unlikely to work without torch.compile.
- **If torch.compile crashes**: Keep `TORCH_COMPILE_DISABLE=1` and use `FULL_DECODE_ONLY` which doesn't require compilation.
- **MTP compatibility**: MTP may not work with all graph modes. Test MTP with enforce-eager first, then with graph modes.
- **Prefix caching**: Just add `--enable-prefix-caching` flag. No code changes needed.
- **Always kill previous server** before starting a new one.
- **Capture server logs**: Use `2>&1 | tee /root/benchmark-results/<name>-server.log` or redirect stderr.

## Example Handoff

```json
{
  "salientSummary": "Enabled FULL_DECODE_ONLY graph mode by removing --enforce-eager and adding --compilation-config. PIECEWISE failed (torch.compile cluster_dims error) but FULL_DECODE_ONLY worked. Decode tok/s improved 15% (85→98 at 1 user). Also enabled prefix caching which reduced TTFT by 40% on repeated prompts. MTP with num_speculative_tokens=1 added another 25% decode improvement (98→123 tok/s).",
  "whatWasImplemented": "Created /root/benchmark-scripts/launch-optimized.sh with FULL_DECODE_ONLY + prefix caching + MTP config. Ran benchmarks at 1/2/4 users, results in /root/benchmark-results/optimized-qwen35-9b.json. Stability test: 100/100 requests passed at 4 concurrent users.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "Launch with FULL_DECODE_ONLY mode", "exitCode": 0, "observation": "Server started, logs show 'CUDAGraphMode.FULL_DECODE_ONLY'"},
      {"command": "curl health check", "exitCode": 0, "observation": "Healthy after 55s (graph warmup added 10s)"},
      {"command": "vllm bench serve at 4 concurrent", "exitCode": 0, "observation": "98 tok/s decode (vs 85 baseline = +15%)"},
      {"command": "100-request stability test", "exitCode": 0, "observation": "100/100 completed, no errors, VRAM stable at 88%"}
    ],
    "interactiveChecks": [
      {"action": "Tested PIECEWISE mode", "observed": "Crashed with KernelMetadata.cluster_dims error, as expected for gfx908"},
      {"action": "Tested MTP with coding prompt", "observed": "Generated correct Python code, acceptance rate ~70%"}
    ]
  },
  "tests": {
    "added": []
  },
  "discoveredIssues": [
    {"severity": "info", "description": "PIECEWISE mode requires torch.compile which doesn't work on gfx908. FULL_DECODE_ONLY is the viable alternative."}
  ]
}
```

## When to Return to Orchestrator

- All graph modes crash on MI100 (NONE remains only option)
- MTP causes server crashes that can't be resolved
- Optimization causes quality regression (garbled outputs)
- VRAM overflow prevents running at target concurrency
- Need to modify vLLM core code in ways that affect other features
