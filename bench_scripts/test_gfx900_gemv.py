"""Validate gfx900_w4a16_gemv against the existing triton_w4a16_gemm (same
GPTQ-sequential packing) for correctness across M, and time M=1 decode."""
import torch, time, sys
sys.path.insert(0, "/home/larkinwc/vllm-gfx908")
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm
from vllm.model_executor.kernels.linear.mixed_precision.gfx900_w4a16_gemv import w4a16_gemv_gfx900

DEV = "cuda"

def make(K, N, G):
    qw = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device=DEV)
    qz = torch.randint(0, 2**31, (K // G, N // 8), dtype=torch.int32, device=DEV)
    scales = (torch.randn(K // G, N, device=DEV) * 0.01).to(torch.float16)
    return qw, scales, qz

def bench(fn, it=50, wm=10):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/it*1e3

def main():
    G = 128
    for name, K, N in [("gate/up",4096,24576),("down",12288,4096)]:
        qw, scales, qz = make(K, N, G)
        for M in [1, 4, 8]:
            a = (torch.randn(M, K, device=DEV)*0.1).to(torch.float16).contiguous()
            ref = triton_w4a16_gemm(a=a, b_q=qw, scales=scales, qzeros=qz, group_size=G, zp_bias=0)
            got = w4a16_gemv_gfx900(a, qw, scales, qz, G, zp_bias=0)
            err = (ref.float()-got.float()).abs().max().item()
            rel = err/(ref.float().abs().max().item()+1e-6)
            tag = "OK" if rel < 0.02 else "FAIL"
            print(f"[{name}] M={M} K={K} N={N}  rel_err={rel:.4f}  {tag}")
        # M=1 timing
        a = (torch.randn(1, K, device=DEV)*0.1).to(torch.float16).contiguous()
        t_ref = bench(lambda: triton_w4a16_gemm(a=a, b_q=qw, scales=scales, qzeros=qz, group_size=G, zp_bias=0))
        t_gemv = bench(lambda: w4a16_gemv_gfx900(a, qw, scales, qz, G, zp_bias=0))
        print(f"   M=1: stock tl.dot {t_ref:.3f}ms  gemv {t_gemv:.3f}ms  speedup {t_ref/t_gemv:.2f}x")

if __name__ == "__main__":
    main()
