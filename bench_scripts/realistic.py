#!/usr/bin/env python3
# Convert measured serving throughput -> effective realized TFLOPS / GB/s,
# and compare to the parallel-aggregate ceiling.
PARAMS = 9.653e9
BYTES  = PARAMS * 2          # fp16 weights actually streamed per decode step
# parallel-aggregate ceilings measured in PERF_GFX900.md
AGG_TFLOPS = {4: 4.5*4, 8: 4.5*8}      # ~per-GPU 4.5 * N
AGG_BW     = {4: 365*4,  8: 365*8}     # GB/s

def prefill_tflops(prompt_tok, ttft_s, batch=1):
    # FLOPs ~ 2 * params * tokens (whole batch's prompt processed in the TTFT window)
    flops = 2 * PARAMS * prompt_tok * batch
    return flops / ttft_s / 1e12

def decode_bw(out_tok_s, batch):
    # one forward step streams all weights once and emits `batch` tokens
    steps_s = out_tok_s / batch
    return steps_s * BYTES / 1e9     # GB/s realized

print("=== DECODE (bandwidth-bound): effective aggregate GB/s ===")
print(f"{'layout':<12}{'GPUs':>5}{'c':>3}{'out tok/s':>11}{'eff GB/s':>10}{'vs agg':>9}{'vs/GPU peak':>12}")
# (layout, gpus, c, out_tok_s)
decode = [
 ("TP8",      8,1, 7.94),("TP8",     8,4,19.54),
 ("TP4",      4,1, 4.58),("TP4",     4,4,12.23),
 ("TP2xPP4",  8,1, 6.95),("TP2xPP4", 8,4,22.87),
 ("TP2xPP2",  4,1, 6.81),("TP2xPP2", 4,4,18.14),
 ("TP2xPP4-coding", 8,4,25.69),
]
for name,g,c,o in decode:
    bw = decode_bw(o,c)
    agg = AGG_BW[g]
    print(f"{name:<12}{g:>5}{c:>3}{o:>11.2f}{bw:>10.1f}{100*bw/agg:>8.1f}%{100*bw/(365*g):>11.1f}%")

print("\n=== PREFILL (compute-bound): effective aggregate TFLOPS ===")
print(f"{'layout':<12}{'GPUs':>5}{'c':>3}{'TTFT p50 s':>11}{'eff TFLOPS':>12}{'vs agg':>9}")
# (layout, gpus, c, ttft_ms, prompt_tok, batch_in_flight)
prefill = [
 ("TP8",     8,1, 823, 1024,1),("TP8",    8,4,3394,1024,4),
 ("TP4",     4,1,1953, 1024,1),("TP4",    4,4,5762,1024,4),
 ("TP2xPP4", 8,1,2181, 1024,1),("TP2xPP4",8,4,4639,1024,4),
 ("TP2xPP2", 4,1,2289, 1024,1),("TP2xPP2",4,4,6533,1024,4),
]
for name,g,c,ttft,ptok,b in prefill:
    tf = prefill_tflops(ptok, ttft/1000.0, batch=b)
    agg = AGG_TFLOPS[g]
    print(f"{name:<12}{g:>5}{c:>3}{ttft/1000:>11.3f}{tf:>12.2f}{100*tf/agg:>8.1f}%")
