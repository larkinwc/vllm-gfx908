# User Testing

Testing surface, required tools, and resource cost classification.

## Validation Surface

**Primary surface**: vLLM OpenAI-compatible API on localhost:8000
**Tools**: curl, custom Python benchmark scripts, `vllm bench` CLI
**No browser UI** -- all testing is API/CLI based

### Testing Approaches
1. **Health check**: `curl http://localhost:8000/health`
2. **Chat completion**: `curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '...'`
3. **Synthetic benchmark**: `vllm bench serve --model <model> --dataset-name sonnet ...`
4. **Throughput benchmark**: `vllm bench throughput --model <model> ...`
5. **Custom coding agent benchmark**: Python script sending concurrent coding prompts
6. **GPU monitoring**: `rocm-smi` for VRAM, temp, power, clocks
7. **Server log inspection**: grep for error messages, graph mode activation, MTP status

### Testing Workflow
1. Stop current vLLM server if running
2. Start vLLM with desired config
3. Wait for health check to pass (up to 120s for model loading)
4. Run benchmarks
5. Collect metrics
6. Stop server
7. Repeat for next config

## Gotchas and Known Issues

### Orphaned vLLM Worker Processes After Server Stop
When stopping the vLLM API server (e.g., via `lsof -ti :8000 | xargs kill -9`), the VLLM worker processes (VLLM::Worker_TP) are NOT automatically killed and continue to hold GPU VRAM. This prevents starting a new vLLM server.

**Fix**: After stopping the API server, explicitly kill all worker PIDs:
```bash
# Stop API server
lsof -ti :8000 | xargs kill -9 2>/dev/null; sleep 5

# Kill remaining vLLM worker processes
/opt/rocm/core-7.12/bin/rocm-smi --showpids 2>/dev/null | grep VLLM | awk '{print $1}' | xargs kill -9 2>/dev/null
sleep 10

# Verify VRAM is freed (should show ~6.5MB used, not 30GB+)
/opt/rocm/core-7.12/bin/rocm-smi --showmeminfo vram | grep "Used Memory"
```

### Startup Times
- Qwen3.5-9B FP16: ~180s to health check
- Llama-2-7b-hf FP16: ~60s to health check

## Flow Validator Guidance: CLI/API

All testing is via curl and Python scripts against the vLLM API at localhost:8000.
- Only one vLLM server at a time (max concurrent validators: 1)
- Must kill orphaned workers after stopping server (see Gotchas section above)
- Use /v1/completions for completion models; /v1/chat/completions for chat models with templates
- Llama-2-7b-hf requires --chat-template flag pointing to template file

## Validation Concurrency

**Max concurrent validators**: 1

Rationale: Testing involves starting/stopping vLLM servers which consume all 4 GPUs. Only one vLLM instance can run at a time (TP=4 uses all GPUs). Sequential validation is required.

**Resource constraints**:
- 4x MI100 @ 32GB each = 128GB total VRAM (all consumed by one TP=4 instance)
- 64GB system RAM (vLLM workers use ~16GB total)
- Benchmark scripts are lightweight (curl/Python)
- GPU monitoring via rocm-smi is negligible overhead
