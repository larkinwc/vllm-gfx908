#!/usr/bin/env python3
"""Sustained FP16 matmul load on cuda:0 while sampling power draw.
Run with HIP_VISIBLE_DEVICES=<i>. Reports TFLOPS sustained over the window
plus mean/peak power (read via rocm-smi for the same GPU index passed in)."""
import argparse, json, subprocess, threading, time, os
import torch

def read_power_w(power_path):
    try:
        with open(power_path) as f:
            return int(f.read().strip()) / 1e6  # uW -> W
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--smi-idx", type=int, default=-1, help="(unused, kept for compat)")
    p.add_argument("--power-path", required=True, help="sysfs power1_input/average path for this GPU")
    p.add_argument("--label", default="")
    a = p.parse_args()

    dt = torch.float16
    A = torch.randn(a.n, a.n, device="cuda", dtype=dt)
    B = torch.randn(a.n, a.n, device="cuda", dtype=dt)
    C = torch.empty(a.n, a.n, device="cuda", dtype=dt)
    # warmup
    for _ in range(10):
        torch.matmul(A, B, out=C)
    torch.cuda.synchronize()

    # Run in chunks: time each chunk with a sync (full throughput within the
    # chunk), then take one quick power sample between chunks. sysfs read is
    # ~microseconds so it barely dents the duty cycle.
    samples = []
    chunk = 10
    nchunks = max(1, a.iters // chunk)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(nchunks):
        for _ in range(chunk):
            torch.matmul(A, B, out=C)
        torch.cuda.synchronize()
        w = read_power_w(a.power_path)
        if w:
            samples.append(w)
    dt_s = time.perf_counter() - t0
    done = nchunks * chunk

    if len(samples) > 4:
        samples = samples[2:]              # drop clock-ramp samples
    tflops = (2.0 * a.n**3 * done) / dt_s / 1e12
    mean_w = sum(samples)/len(samples) if samples else None
    peak_w = max(samples) if samples else None
    print(json.dumps({
        "label": a.label,
        "tflops": round(tflops, 2),
        "mean_w": round(mean_w, 1) if mean_w else None,
        "peak_w": round(peak_w, 1) if peak_w else None,
        "tflops_per_w": round(tflops/mean_w, 4) if mean_w else None,
        "n_samples": len(samples),
    }))

if __name__ == "__main__":
    main()
