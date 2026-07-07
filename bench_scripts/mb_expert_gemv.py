"""Single-die signal test for the MoE custom kernel: does the #57 int4 GEMV beat
the padded tl.dot path on Qwen3-Next expert shapes (M=1 decode, per active expert)?
Expert GEMMs: gate_up [K=2048 -> N=1024], down [K=512 -> N=2048], int4 sym gs32.
Baseline = F.linear on dequantized fp16 weights (approximates the tl.dot cost after
dequant; both pad M=1). GEMV = existing w4a16_gemv_gfx900 (#57).
Run: HIP_VISIBLE_DEVICES=<free die> python mb_expert_gemv.py
"""
import torch, torch.nn.functional as F, time, sys
sys.path.insert(0, "/home/larkinwc/src/vllm-gfx908")
from vllm.model_executor.kernels.linear.mixed_precision.gfx900_w4a16_gemv import w4a16_gemv_gfx900
dev = "cuda"; torch.manual_seed(0)
G = 32


def make_int4(K, N):
    # symmetric int4, packed GPTQ-sequential: b_q [K, N//8] int32, scales [K//G, N]
    w = torch.randint(0, 15, (K, N), dtype=torch.int32, device=dev)
    packed = torch.zeros((K, N // 8), dtype=torch.int32, device=dev)
    for j in range(8):
        packed |= (w[:, j::8] & 0xF) << (j * 4)
    scales = (torch.rand((K // G, N), device=dev, dtype=torch.float16) * 0.02 + 0.01)
    # dequant reference (sym, zp_bias=8): (q - 8) * scale
    wq = (w.float() - 8.0)
    sc = scales.float().repeat_interleave(G, dim=0)
    w_deq = (wq * sc).to(torch.float16)   # [K, N]
    return packed, scales, w_deq


def bench(fn, it=100, wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / it * 1000


print(f"{'expert GEMM':16} {'K':>5} {'N':>5} | {'linear(deq)':>11} {'int4 GEMV':>10} | {'speedup':>8} {'relerr':>8}")
for name, K, N in [("gate_up", 2048, 1024), ("down", 512, 2048)]:
    a = torch.randn(1, K, dtype=torch.float16, device=dev)
    b_q, scales, w_deq = make_int4(K, N)
    ref = F.linear(a, w_deq.t().contiguous())          # [1,N], padded rocBLAS
    out = w4a16_gemv_gfx900(a, b_q, scales, None, G, zp_bias=8)
    relerr = ((out - ref).abs() / (ref.abs() + 1e-2)).max().item()
    t_lin = bench(lambda: F.linear(a, w_deq.t().contiguous()))
    t_gemv = bench(lambda: w4a16_gemv_gfx900(a, b_q, scales, None, G, zp_bias=8))
    print(f"{name:16} {K:>5} {N:>5} | {t_lin:11.4f} {t_gemv:10.4f} | {t_lin/t_gemv:7.2f}x {relerr:8.4f}")
# 10 active experts per token per layer: aggregate per-token cost of 10 sequential GEMV calls
print("\nNote: MoE decode routes M=1 token to 10 experts -> 10x these per layer (x48 layers).")
