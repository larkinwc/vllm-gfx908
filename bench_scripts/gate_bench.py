"""Microbench the shared_expert_gate GEMV [n=1,k=2048] @ [m,2048]^T -> [1,m]
for m in {1}, the decode shape that falls to padded rocBLAS MT64x64x8.
Compare: rocBLAS (F.linear), mul+sum, LLMM1-with-pad-to-4, tiny.
Correctness vs torch ref (rel-err), then ms over many iters. Issue #65 option A.
Single GPU (cuda:0)."""
import torch, time
import vllm._custom_ops as ops

dev = "cuda:0"
torch.manual_seed(0)
K = 2048
iters = 2000


def bench(fn, ref, name):
    # correctness
    out = fn()
    rel = (out.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-9)
    torch.cuda.synchronize()
    # time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print("  %-22s rel_err=%.2e  %.1f us/call" % (name, rel, us), flush=True)
    return us


for m in [1, 2]:
    print("=== m=%d k=%d (n=1) ===" % (m, K), flush=True)
    x = torch.randn(1, K, device=dev, dtype=torch.float16)
    w = torch.randn(m, K, device=dev, dtype=torch.float16)
    ref = torch.nn.functional.linear(x.float(), w.float())  # fp32 ref

    bench(lambda: torch.nn.functional.linear(x, w), ref, "F.linear (rocBLAS)")
    bench(lambda: (x * w).sum(dim=-1, keepdim=True).t() if m > 1 else (x * w).sum(dim=-1, keepdim=True), ref, "mul+sum")
    # matmul variant
    bench(lambda: x @ w.t(), ref, "x @ w.t()")

    # LLMM1 with weight padded to multiple of 4 rows
    mp = ((m + 3) // 4) * 4
    wpad = torch.zeros(mp, K, device=dev, dtype=torch.float16)
    wpad[:m] = w

    def llmm1_pad():
        o = ops.LLMM1(wpad, x, 4)
        return o[:, :m]
    try:
        bench(llmm1_pad, ref, "LLMM1(pad m->%d)" % mp)
    except Exception as e:
        print("  LLMM1 pad FAILED:", str(e)[:60], flush=True)

print("GATE_BENCH_DONE", flush=True)
