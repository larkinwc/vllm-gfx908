#!/usr/bin/env python3
"""Per-GPU FP16 TFLOPS + HBM bandwidth microbenchmark.
Run with HIP_VISIBLE_DEVICES=<i> so the target GPU is always cuda:0.
Launch one process per physical GPU concurrently to measure the aggregate
under real power/thermal/memory contention. Prints a single JSON line."""
import argparse, json, os, time
import torch

def time_loop(fn, iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0

def bench_flops(dtype, n, warmup, iters):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    c = torch.empty(n, n, device="cuda", dtype=dtype)
    f = lambda: torch.matmul(a, b, out=c)
    time_loop(f, warmup)
    dt = time_loop(f, iters)
    flops = 2.0 * n * n * n * iters
    return flops / dt / 1e12  # TFLOPS

def bench_bw(dtype, mb, warmup, iters):
    # copy_ reads src + writes dst => 2 * nbytes moved per iteration
    nelem = (mb * 1024 * 1024) // torch.tensor([], dtype=dtype).element_size()
    src = torch.randn(nelem, device="cuda", dtype=dtype)
    dst = torch.empty(nelem, device="cuda", dtype=dtype)
    f = lambda: dst.copy_(src)
    time_loop(f, warmup)
    dt = time_loop(f, iters)
    bytes_moved = 2.0 * src.numel() * src.element_size() * iters
    return bytes_moved / dt / 1e9  # GB/s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--n", type=int, default=8192)         # matmul dim
    p.add_argument("--mb", type=int, default=1024)        # bw buffer MB
    p.add_argument("--flops-iters", type=int, default=50)
    p.add_argument("--bw-iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--label", default="")
    a = p.parse_args()
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    name = torch.cuda.get_device_name(0)
    tflops = bench_flops(dt, a.n, a.warmup, a.flops_iters)
    bw = bench_bw(dt, a.mb, a.warmup, a.bw_iters)
    print(json.dumps({
        "label": a.label,
        "hip_visible": os.environ.get("HIP_VISIBLE_DEVICES", "?"),
        "device": name,
        "dtype": a.dtype,
        "matmul_n": a.n,
        "tflops": round(tflops, 2),
        "bw_gbps": round(bw, 1),
    }))

if __name__ == "__main__":
    main()
