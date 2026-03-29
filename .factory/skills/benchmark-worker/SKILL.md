---
name: benchmark-worker
description: Downloads models, creates benchmark scripts, runs benchmarks, and produces comparison reports
---

# Benchmark Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that involve:

- Downloading and configuring models for benchmarking
- Creating benchmark scripts (synthetic and realistic workloads)
- Running vLLM benchmarks with specific configurations
- Producing structured benchmark reports
- Comparing results across configurations

## Required Skills

None

## Work Procedure

1. **Read feature requirements** from the assigned feature in features.json. Understand exactly which models, configurations, and metrics are needed.

2. **Prepare environment**:
   - Source environment: `export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib && export ROCM_PATH=/opt/rocm/core-7.12 && export PYTORCH_ROCM_ARCH=gfx908 && export VLLM_ROCM_USE_SKINNY_GEMM=0 && export VLLM_ROCM_USE_AITER=1`
   - All vLLM commands must use `/opt/vllm-env/bin/python3`
   - All model downloads use: `/opt/vllm-env/bin/huggingface-cli download <model> --local-dir /models/<model-name>`

3. **Download models** if not already present in `/models/`. Check first with `ls /models/`.

4. **Stop any running vLLM server**: `lsof -ti :8000 | xargs kill -9 2>/dev/null; sleep 5`

5. **Create benchmark scripts** in `/root/benchmark-scripts/`:
   - Scripts must be self-contained Python files that can be run directly
   - For synthetic benchmarks, use `vllm bench serve` or `vllm bench throughput` CLI
   - For realistic coding agent benchmarks, create a Python script using aiohttp/httpx to send concurrent requests
   - Scripts must output results in JSON format to `/root/benchmark-results/`

6. **Run benchmarks**:
   - Start vLLM server with the specific config as a background process
   - Wait for health check: `for i in $(seq 1 120); do curl -sf http://localhost:8000/health && break; sleep 1; done`
   - Run benchmark script
   - Save results to `/root/benchmark-results/<model>-<config>-<timestamp>.json`
   - Stop vLLM server after benchmark completes

7. **Produce reports**: Create a comparison report in `/root/benchmark-results/` combining results.

8. **Run validators**: The vLLM test suite is large; only run specific tests if the feature modifies vLLM code. For pure benchmark features, validation is the benchmark results themselves.

### Critical Notes

- **Always stop the vLLM server before starting a new one** (port 8000)
- **Wait for model loading** - large models take 30-60 seconds to load
- **Use --enforce-eager** for baseline configs (as specified in feature)
- **Record GPU stats**: `rocm-smi` output before/during/after benchmark
- **INT4 models** require `--dtype float16` (ExllamaLinearKernel requirement)
- **Do not use port 8080** (GPU dashboard)
- **Background the server**: Use `nohup ... &` or similar, capture PID for later kill

## Example Handoff

```json
{
  "salientSummary": "Downloaded Qwen3.5-9B FP16 and INT4 models, created synthetic and coding-agent benchmark scripts, ran baselines at 1/2/4 concurrent users. FP16 baseline: 85 tok/s decode at 1 user, 72 tok/s at 4 users. INT4: 110 tok/s at 1 user, 95 tok/s at 4 users. Results saved to /root/benchmark-results/.",
  "whatWasImplemented": "Downloaded Qwen3.5-9B FP16 (18GB) and AWQ-INT4 (5GB) to /models/. Created /root/benchmark-scripts/synthetic_bench.sh and /root/benchmark-scripts/coding_agent_bench.py. Ran all benchmarks, results in /root/benchmark-results/baseline-report.json with per-model, per-concurrency metrics.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "huggingface-cli download Qwen/Qwen3.5-9B --local-dir /models/Qwen3.5-9B", "exitCode": 0, "observation": "18GB downloaded in 4 minutes"},
      {"command": "curl -sf http://localhost:8000/health", "exitCode": 0, "observation": "Server healthy after 45s startup"},
      {"command": "python3 /root/benchmark-scripts/coding_agent_bench.py --concurrency 4", "exitCode": 0, "observation": "All 100 requests completed, results saved"},
      {"command": "rocm-smi", "exitCode": 0, "observation": "All 4 GPUs at 31-33°C idle, 45-55°C under load, VRAM 85%"}
    ],
    "interactiveChecks": [
      {"action": "Sent coding prompt to FP16 model", "observed": "Generated valid Python code for fibonacci function, 85 tok/s decode"},
      {"action": "Ran 4-user concurrent benchmark on INT4", "observed": "All 4 users got responses, aggregate 380 tok/s, no errors"}
    ]
  },
  "tests": {
    "added": []
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Model download fails (HuggingFace access issues, disk space)
- vLLM crashes during model loading (OOM, architecture incompatibility)
- Benchmark produces no results or all-zero results
- GPU hardware issues (rocm-smi shows errors)
- Required INT4 quantized model doesn't exist on HuggingFace
