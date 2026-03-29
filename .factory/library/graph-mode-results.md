# HIP Graph Mode Results for MI100

## Summary

FULL_DECODE_ONLY graph mode works on MI100 (gfx908) with Qwen3.5-9B FP16, providing significant throughput and latency improvements.

## Configuration

```bash
export TORCH_COMPILE_DISABLE=1  # Required to avoid gfx908 torch.compile errors
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export ROCM_PATH=/opt/rocm/core-7.12
export PYTORCH_ROCM_ARCH=gfx908
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib

/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
  --port 8000 \
  --model /models/Qwen3.5-9B \
  --dtype float16 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --language-model-only \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

**Key Points:**

- Do NOT use `--enforce-eager` (that disables graph capture)
- `TORCH_COMPILE_DISABLE=1` must be set (torch.compile has gfx908 issues)
- FULL_DECODE_ONLY captures graphs for decode phase only (safest for gfx908)

## Benchmark Results

### Concurrency 1

| Metric | Graph Mode | Baseline (enforce-eager) | Delta |
|--------|-----------|-------------------------|-------|
| Output throughput (tok/s) | 248.21 | 228.28 | +8.7% |
| TPOT (ms) | 14.17 | 50.19 | -71.8% |
| TTFT mean (ms) | 1661.08 | 880.28 | +88.7%* |
| TTFT median (ms) | 123.10 | 202.43 | -39.2% |
| Total throughput (tok/s) | 1241.03 | 1141.38 | +8.7% |

*Mean TTFT higher due to first-request cold start with graph compilation; median TTFT is lower.

### Concurrency 2

| Metric | Graph Mode | Baseline (enforce-eager) | Delta |
|--------|-----------|-------------------------|-------|
| Output throughput (tok/s) | 478.41 | 411.87 | +16.1% |
| TPOT (ms) | 15.88 | 49.60 | -68.0% |
| TTFT mean (ms) | 137.67 | 217.38 | -36.7% |
| TTFT median (ms) | 127.72 | 202.40 | -36.9% |
| Total throughput (tok/s) | 2392.03 | 2059.35 | +16.2% |

## Graph Capture Details

- 35 decode graph sizes captured (1 to 512 batch sizes)
- Graph capture took ~32-34 seconds
- Memory overhead: 0.16 GiB for graph storage
- Total initialization time: ~100 seconds (vs ~60 seconds for eager)

## Files

- Launch script: `/root/benchmark-scripts/launch-graphs.sh`
- Benchmark results:
    - `/root/benchmark-results/graph_full_decode_only_c1_20260329_125027.json`
    - `/root/benchmark-results/graph_full_decode_only_c2_20260329_125236.json`
- Server logs:
    - `/root/benchmark-results/server_graph_test_20260329_124656.log`
    - `/root/benchmark-results/server_graph_FULL_DECODE_ONLY_20260329_125759.log`

## Config-Tuned Mode (Graph + Prefix Caching)

Additional optimizations enabled on top of graph mode:

- `--enable-prefix-caching`: Caches system prompts for TTFT reduction on cache hit
- `--max-model-len 32768`: Reduced from 262144 (necessary to fit in 32GB VRAM)

### Launch Command

```bash
/root/benchmark-scripts/launch-config-tuned.sh
```

### Prefix Caching Test Results

- First request TTFT: 15622.99 ms (cache miss, includes graph warmup)
- Second request TTFT: 137.78 ms (cache hit with identical system prompt)
- TTFT ratio: 0.009 (99.1% reduction) - PASS (threshold: <= 0.80)

### VRAM Usage

- With max-model-len 32768: 93.2% (32.01 GB / 34.34 GB)
- Reduced max-model-len is essential - 262144 would OOM

### Stability Test

- 100 requests at 4 concurrent users: 100/100 passed
- No crashes, OOM, or HTTP 5xx errors

## Files

- Launch script: `/root/benchmark-scripts/launch-config-tuned.sh`
- Prefix caching test: `/root/benchmark-results/prefix_caching_test.json`
- Stability test: `/root/benchmark-results/stability_test_config_tuned.json`
- Coding agent c2: `/root/benchmark-results/config_tuned_c2.json`

## Validation Contract Assertions

- VAL-GRAPH-001: ✅ Server starts without enforce-eager, passes health check
- VAL-GRAPH-002: ✅ FULL_DECODE_ONLY graph mode activates (logs confirm)
- VAL-GRAPH-003: ✅ Throughput >= 95% of baseline (+8.7% to +16.1% improvement)
- VAL-GRAPH-004: ✅ Stability test 100/100 requests at 4 concurrent users
- VAL-GRAPH-005: ✅ Prefix caching TTFT reduction (ratio 0.009 <= 0.80)
- VAL-GRAPH-006: ✅ Reduced max-model-len shows VRAM fit (93.2% at 32768)
- VAL-GRAPH-007: ✅ Throughput improvement at 2 concurrent (+16.1%)
