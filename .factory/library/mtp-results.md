# MTP Speculative Decoding Results for MI100

## Summary

**MTP speculative decoding is NOT RECOMMENDED on MI100 (gfx908)** due to incompatibility with HIP graph mode and resulting performance regression.

## Key Findings

### 1. MTP + Graph Mode Incompatibility

MTP speculative decoding crashes during graph capture/warmup on MI100:

```text
RuntimeError: cancelled
```

The error occurs in the engine core initialization during graph compilation. This is a fundamental incompatibility between MTP's speculative execution flow and HIP graph capture on gfx908.

### 2. MTP with Eager Mode Underperforms

Forcing eager mode to enable MTP results in significant performance regression:

| Config | TPOT | Aggregate tok/s (c1) | vs Baseline Eager |
|--------|------|---------------------|-------------------|
| Baseline Eager | 50.19ms | 21.39 | - |
| MTP n=1 Eager | 54.84ms | 15.90 | -25.7% |
| MTP n=2 Eager | 59.19ms | 12.37 | -42.2% |
| MTP n=3 Eager | 61.76ms | 11.80 | -44.9% |
| **Graph Mode (No MTP)** | **14.17ms** | **~90** | **+320%** |

### 3. Acceptance Rates

| num_speculative_tokens | Acceptance Rate |
|------------------------|-----------------|
| 1 | ~85% |
| 2 | ~70% |
| 3 | ~56% |

Higher speculation depths show rapidly diminishing acceptance rates, and even n=1 with 85% acceptance cannot compensate for the eager mode overhead.

### 4. Stability

MTP with eager mode is stable:

- 52/52 requests completed at 4 concurrent users
- VRAM at 90.3% (within 97% limit)
- No crashes or errors during operation

## Recommendations

1. **Primary**: Use FULL_DECODE_ONLY graph mode + prefix caching WITHOUT MTP
   - Launch: `/root/benchmark-scripts/launch-config-tuned.sh`
   - Expected performance: ~90 tok/s aggregate at 4 users, 11-14ms TPOT

2. **If MTP is required**: Use n=1 with enforce-eager
   - Launch: `/root/benchmark-scripts/launch-mtp.sh --enforce-eager`
   - Expected performance: ~56 tok/s aggregate at 4 users, 55ms TPOT
   - Accept that this is 37% slower than graph mode

## Files

- Launch script: `/root/benchmark-scripts/launch-mtp.sh`
- Results: `/root/benchmark-results/mtp_results.json`
- Benchmark outputs:
    - `/root/benchmark-results/mtp_eager_n1_c1.json`
    - `/root/benchmark-results/mtp_eager_n2_c1.json`
    - `/root/benchmark-results/mtp_eager_n3_c1.json`
    - `/root/benchmark-results/mtp_eager_n1_stability_c4.json`
- Server logs:
    - `/root/benchmark-results/server_mtp_graph_n1_20260329_135239.log` (failed graph mode attempt)
    - `/root/benchmark-results/server_mtp_eager_n1_20260329_140933.log` (working eager mode)
