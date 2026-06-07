# User Testing

## Validation Surface

- **Surface:** API endpoints on localhost:8000 (vLLM OpenAI-compatible API)
- **Tools:** curl, python scripts, vllm bench serve
- **Auth:** None required (local server)
- **Startup:** vLLM server must be started with TQ backend before testing

### Testing Approach

All validation is CLI/API-based:

1. Start vLLM with TQ backend (launch script)
2. Wait for health check (curl -sf <http://localhost:8000/health>)
3. Send requests via curl or python benchmark scripts
4. Collect results from JSON output files and server logs
5. Stop server, compare results

### Key Test Scripts (from previous mission, reusable)

- `/root/benchmark-scripts/run_synthetic_bench.sh` -- vllm bench serve wrapper
- `/root/benchmark-scripts/coding_agent_bench.py` -- concurrent coding workload
- `/root/benchmark-scripts/run_sustained_load_test.py` -- sustained load test
- `/root/benchmark-scripts/compare_results.py` -- comparison table generator

## Validation Concurrency

**Max concurrent validators:** 1

**Rationale:** GPU-bound workload. Only one vLLM instance can run at a time on 4x MI100 (93% VRAM used). Starting a second instance would fail due to insufficient VRAM. All validation must be sequential.

**Machine specs:** 64 CPU cores, 62 GB RAM, 4x MI100 32GB. CPU/RAM is abundant but GPU is the bottleneck.

## Flow Validator Guidance: API

This section covers how to safely test vLLM API endpoints on this machine.

### Pre-flight Checks

- Always kill any existing vLLM processes before starting a new test run:
  ```bash
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null; pkill -9 -f 'VLLM::' 2>/dev/null; sleep 5
  ```
- Verify GPU is free (base VRAM should be ~6-7MB):
  ```bash
  rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used"
  ```
- If GPU shows >1GB VRAM used with no vLLM processes, wait 10-15s for GPU to fully release memory.

### Launch Commands

**Eager mode (capture_only):**

```bash
# Use the launch script directly
/root/benchmark-scripts/launch-tq-backend.sh capture_only &
```

**Graph mode (FULL_DECODE_ONLY):**

```bash
/root/benchmark-scripts/launch-tq-graph-mode.sh capture_only &
```

**Baseline (no TQ backend, for comparison):**

```bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_ROCM_USE_AITER=1
export TORCH_COMPILE_DISABLE=1
/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.5-9B \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --max-model-len 32768 \
  --port 8000 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --language-model-only &
```

### Health Check

```bash
# Wait up to 120s
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health; then echo "UP"; break; fi
  sleep 2
done
```

### Stop Server

```bash
pkill -9 -f 'vllm.entrypoints' 2>/dev/null; pkill -9 -f 'VLLM::' 2>/dev/null; sleep 5
```

### Known Quirks

- **Startup time:** ~65-80s for eager mode, ~100-120s for graph mode
- **GPU state after kill:** Sometimes GPU takes 5-10s to release memory after process kill
- **hipErrorLaunchFailure:** If inference fails with this error, GPU is in bad state from a crash. Kill all vLLM processes, wait 10s, try again.
- **--language-model-only is MANDATORY** for Qwen3.5-9B. Without it, vision encoder allocation causes OOM on 32GB MI100.
- **Server log location:** Capture server output by redirecting stdout to a file when backgrounding.

### Shared State

- All tests share the same GPU (no isolation possible)
- Only ONE validator may run at a time
- Log file for server: redirect to /tmp/vllm_server_$(date +%s).log when starting

### Making API Requests

```bash
# Simple completion request
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3.5-9B",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0
  }'

# Models endpoint to verify server is ready
curl -s http://localhost:8000/v1/models
```
