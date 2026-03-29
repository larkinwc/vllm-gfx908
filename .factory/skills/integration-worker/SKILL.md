---
name: integration-worker
description: Integrates external libraries (TurboQuant) into vLLM on MI100, handles ROCm compatibility issues
---

# Integration Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that involve:
- Installing and integrating external packages with vLLM
- Porting CUDA/NVIDIA-specific code to ROCm/MI100
- Debugging Triton kernel compatibility on gfx908
- TurboQuant KV cache compression integration
- Creating combined optimization configurations

## Required Skills

None

## Work Procedure

1. **Read feature requirements** from the assigned feature. Understand the integration target and success criteria.

2. **Prepare environment**:
   ```bash
   export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
   export ROCM_PATH=/opt/rocm/core-7.12
   export PYTORCH_ROCM_ARCH=gfx908
   export VLLM_ROCM_USE_SKINNY_GEMM=0
   export VLLM_ROCM_USE_AITER=1
   ```

3. **Clone and install the external package**:
   - For TurboQuant: `cd /opt && git clone https://github.com/0xSero/turboquant.git && cd turboquant && /opt/vllm-env/bin/pip install -e .`
   - Verify import: `/opt/vllm-env/bin/python3 -c "import turboquant; print('OK')"`

4. **Test standalone functionality first**:
   - Run any included tests (e.g., `python validate_paper.py` for TurboQuant)
   - Check Triton kernel compilation on ROCm: look for compilation errors specific to gfx908
   - If Triton kernels fail, analyze the error and attempt fixes:
     a. Check for CUDA-specific Triton features not available on ROCm
     b. Check for hardcoded GPU architecture assumptions
     c. Check for unsupported Triton operations on gfx908

5. **Integrate with vLLM**:
   - Follow the package's integration guide (monkey-patching, config flags, etc.)
   - Start vLLM server with integration enabled
   - Verify health check and basic functionality
   - If integration fails, document the failure mode and attempt workarounds

6. **Verify quality**:
   - Send 10 test prompts covering code generation, reasoning, and factual recall
   - Compare output quality against baseline (no integration)
   - Run needle-in-haystack test if applicable (context-dependent features)

7. **Benchmark the integration**:
   - Run standard benchmark suite from `/root/benchmark-scripts/`
   - Measure KV cache savings, throughput changes, memory usage changes
   - Save results to `/root/benchmark-results/`

8. **Handle failures gracefully**:
   - If Triton kernels won't compile on ROCm, document the specific errors
   - If integration causes crashes, isolate the cause
   - If quality degrades, quantify the degradation
   - Always ensure the system can fall back to baseline operation

### Critical Notes for TurboQuant on MI100
- TurboQuant uses Triton kernels that were tested on NVIDIA only
- pytorch-triton-rocm 3.5.1 may not support all Triton features used
- Key Triton operations to check: tl.dot, tl.load/store with masks, atomic operations
- If kernels fail, check if there's a pure-PyTorch fallback path
- TurboQuant monkey-patches vLLM attention -- ensure the patch targets exist in our vLLM version (0.18.1)
- Qwen3.5-9B has only 8/32 full-attention layers -- TurboQuant savings will be ~25% max
- head_dim=256 is supported by TurboQuant (they tested with this dimension)

## Example Handoff

```json
{
  "salientSummary": "Installed TurboQuant from 0xSero/turboquant. Paper validation tests passed (9/9). Triton kernels compiled on ROCm gfx908 after fixing one tl.atomic_add incompatibility. vLLM integration via monkey-patch works -- KV cache compressed on 8 full-attention layers. Measured 22% KV cache reduction, needle-in-haystack passes at 8k context. Decode tok/s unchanged (within 2% of baseline).",
  "whatWasImplemented": "Cloned turboquant to /opt/turboquant, installed in vllm-env. Fixed triton_kernels.py line 142: replaced tl.atomic_add with tl.store for ROCm compatibility. Created /root/benchmark-scripts/launch-turboquant.sh with TurboQuant integration. Results in /root/benchmark-results/turboquant-qwen35-9b.json.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "pip install -e /opt/turboquant", "exitCode": 0, "observation": "Installed successfully"},
      {"command": "python validate_paper.py", "exitCode": 0, "observation": "9/9 paper validation tests passed"},
      {"command": "Launch vLLM with TurboQuant", "exitCode": 0, "observation": "Server started, logs show TurboQuant KV compression active"},
      {"command": "Needle-in-haystack at 8k context", "exitCode": 0, "observation": "Needle found correctly"}
    ],
    "interactiveChecks": [
      {"action": "Sent 10 coding prompts with TurboQuant active", "observed": "All 10 produced coherent code, no quality degradation visible"},
      {"action": "Checked rocm-smi VRAM with TurboQuant", "observed": "VRAM usage 68% vs 85% baseline (KV cache savings visible)"}
    ]
  },
  "tests": {
    "added": []
  },
  "discoveredIssues": [
    {"severity": "medium", "description": "TurboQuant hybrid decode dequantizes all history to float32 per decode step -- this may limit speedup at very long contexts", "suggestedFix": "Use TurboQuant's fused Triton decode kernels instead of hybrid path if they work on ROCm"}
  ]
}
```

## When to Return to Orchestrator

- Triton kernels fundamentally incompatible with ROCm gfx908 (not fixable with minor patches)
- Integration requires modifying vLLM core in ways that break other features
- Package version incompatible with vLLM 0.18.1 API
- Quality degradation is severe (>20% coherence loss)
- Package license concerns (GPL-3.0 interaction with Apache-2.0 vLLM)
