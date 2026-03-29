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

### MTP Speculative Decoding on MI100

MTP speculative decoding (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`) is **incompatible with HIP graph mode on MI100/gfx908**. When combining `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` with `--speculative-config`, the server crashes with `RuntimeError: cancelled during graph capture/warmup`.

**MTP works correctly in enforce-eager mode** (`--enforce-eager`), but results in 9.3% TPOT regression vs non-MTP eager baseline (graph mode without MTP is 68% faster).

MTP startup time in eager mode: ~180s (same as non-MTP for Qwen3.5-9B FP16).

Use `launch-mtp.sh --enforce-eager` or add `--enforce-eager` flag when testing MTP.

MTP log verification: look for `Detected MTP model. Sharing target model embedding weights` and `SpecDecoding metrics: Mean acceptance length:` in server logs.

## Validation Concurrency

**Max concurrent validators**: 1

Rationale: Testing involves starting/stopping vLLM servers which consume all 4 GPUs. Only one vLLM instance can run at a time (TP=4 uses all GPUs). Sequential validation is required.

**Resource constraints**:

- 4x MI100 @ 32GB each = 128GB total VRAM (all consumed by one TP=4 instance)
- 64GB system RAM (vLLM workers use ~16GB total)
- Benchmark scripts are lightweight (curl/Python)
- GPU monitoring via rocm-smi is negligible overhead

## Flow Validator Guidance: combined

All combined milestone assertions are tested sequentially by a single flow validator (max concurrency=1).

### Assertions Covered

1. **VAL-COMBO-001**: Start vLLM with full optimized config (graph + prefix-caching + tuned params), check health. Use `/root/launch-vllm-optimized.sh`.
2. **VAL-COMBO-002**: Run `vllm bench serve` at c=2 vs baseline decode tok/s from `baseline-report.json`. Optimized must exceed baseline.
3. **VAL-COMBO-003**: Run 200-request sustained load test at 4 concurrent users; all must succeed, VRAM stable. Script: `/root/benchmark-scripts/run_sustained_load_test.py`
4. **VAL-COMBO-004**: Verify `/root/launch-vllm-optimized.sh` exists, starts server, passes health check within 120s.
5. **VAL-COMBO-005**: Verify final benchmark report exists at `/root/benchmark-results/final-report.json` with all required fields.
6. **VAL-COMBO-006**: Run benchmark on Llama-2-7b-hf with optimized config; compare to Llama-2-7B baseline.
7. **VAL-CROSS-001**: MTP + HIP graphs: Start with BOTH enabled, expect crash or incompatibility; document it.
8. **VAL-CROSS-002**: Prefix caching + MTP: Both enabled in eager mode; second request TTFT <= 80% of first.
9. **VAL-CROSS-003**: TurboQuant + MTP: Both disabled on MI100; document incompatibility.
10. **VAL-CROSS-004**: Server restart: clean shutdown, then restart within 120s.
11. **VAL-CROSS-005**: Thermal stability: GPU temps < 85°C during 5-min sustained load.

### Key References

- Production launch script: `/root/launch-vllm-optimized.sh`
- Benchmark scripts: `/root/benchmark-scripts/`
- Results directory: `/root/benchmark-results/`
- Baseline report: `/root/benchmark-results/baseline-report.json`
- Final report (if exists): `/root/benchmark-results/final-report.json`
- Required env vars: `LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib ROCM_PATH=/opt/rocm/core-7.12 PYTORCH_ROCM_ARCH=gfx908 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER=1 TORCH_COMPILE_DISABLE=1`
- Python: `/opt/vllm-env/bin/python3`
- rocm-smi: `/opt/rocm/core-7.12/bin/rocm-smi`

### CROSS-001 Known Behavior

MTP is documented as incompatible with HIP graph mode on gfx908. Evidence from prior test:
- Server crashes with `RuntimeError: cancelled during graph capture/warmup`
- Log file: `/root/benchmark-results/server_mtp_graph_n1_20260329_135239.log`
- This is expected behavior. CROSS-001 PASSES if: (a) combined start crashes with documented error, OR (b) incompatibility is confirmed documented.

### CROSS-003 Known Behavior

TurboQuant integration is blocked by vLLM v0.18.1 multi-process architecture. Evidence from prior test.
CROSS-003 PASSES if: The known incompatibility is documented. MTP + TurboQuant both fail on gfx908/vLLM v0.18.1.

### VRAM Monitoring

```bash
/opt/rocm/core-7.12/bin/rocm-smi --showmeminfo vram
/opt/rocm/core-7.12/bin/rocm-smi --showtemp --showclocks
```
