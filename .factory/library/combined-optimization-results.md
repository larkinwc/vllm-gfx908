# Combined Optimization Results for MI100

## Summary

The combined optimization configuration (FULL_DECODE_ONLY graph mode + prefix caching + tuned serving params) provides significant performance improvements on MI100/gfx908:

- **Throughput**: +16% at c2 (478 vs 412 tok/s synthetic)
- **TPOT**: -68% reduction (15.7ms vs 49.6ms)
- **Stability**: 200/200 requests passed sustained load test
- **Thermals**: GPU temps 43-51°C under load (< 85°C threshold)

## Excluded Optimizations

### MTP Speculative Decoding
- **Status**: NOT RECOMMENDED on MI100
- **Reason**: Incompatible with HIP graph mode on gfx908
- **Impact**: When forced to eager mode, causes 25% regression
- **Library Reference**: `.factory/library/mtp-results.md`

### TurboQuant KV Cache
- **Status**: BLOCKED
- **Reason**: vLLM v0.18.1 multi-process architecture incompatibility
- **Library Reference**: `.factory/library/turboquant.md`

## Production Configuration

```bash
/root/launch-vllm-optimized.sh --model /models/Qwen3.5-9B
```

Key parameters:
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
- `--enable-prefix-caching`
- `--max-model-len 32768` (Qwen3.5-9B) or `4096` (Llama-2-7B)
- `--max-num-batched-tokens 8192`
- `TORCH_COMPILE_DISABLE=1` (required for gfx908)

## Benchmark Results

| Config | Model | Concurrency | Output Tok/s | TPOT (ms) | TTFT (ms) |
|--------|-------|-------------|--------------|-----------|-----------|
| Optimized | Qwen3.5-9B | 1 | 248.18 | 13.62 | 740 |
| Optimized | Qwen3.5-9B | 2 | 478.60 | 15.73 | 137 |
| Optimized | Qwen3.5-9B | 4 | 882.99 | 21.08 | 166 |
| Optimized | Llama-2-7B | 2 | 482.74 | 15.52 | 255 |

## Validation Contract Status

All assertions PASS:
- VAL-COMBO-001 through VAL-COMBO-006
- VAL-CROSS-004, VAL-CROSS-005

## Files

- Launch script: `/root/launch-vllm-optimized.sh`
- Final report: `/root/benchmark-results/final-report.json`
- Synthetic benchmarks: `/root/benchmark-results/optimized_qwen_synthetic_c*.json`
- Coding agent benchmarks: `/root/benchmark-results/optimized_qwen_coding_c*.json`
- Sustained load test: `/root/benchmark-results/optimized_qwen_sustained_load.json`
