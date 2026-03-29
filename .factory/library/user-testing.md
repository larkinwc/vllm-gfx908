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

## Validation Concurrency

**Max concurrent validators**: 1

Rationale: Testing involves starting/stopping vLLM servers which consume all 4 GPUs. Only one vLLM instance can run at a time (TP=4 uses all GPUs). Sequential validation is required.

**Resource constraints**:
- 4x MI100 @ 32GB each = 128GB total VRAM (all consumed by one TP=4 instance)
- 64GB system RAM (vLLM workers use ~16GB total)
- Benchmark scripts are lightweight (curl/Python)
- GPU monitoring via rocm-smi is negligible overhead
