"""Microbenchmark: hand-written int4 AWQ GEMV (M=1) for gfx900 vs FP16 and vs
the existing padded tl.dot W4A16 path. Issue #57.

AWQ layout:
  qweight [K, N//8] int32  (8 int4 packed per int32 along N, reverse-AWQ order)
  scales  [K//G, N] fp16
  qzeros  [K//G, N//8] int32 (same packing as qweight)
reverse-AWQ order = [0,4,1,5,2,6,3,7]; shift = order*4.
"""
import torch, time
import triton
import triton.language as tl

DEV = "cuda"


# ----------------------- int4 AWQ GEMV (M=1) -----------------------
@triton.jit
def awq_gemv_kernel(
    a_ptr,        # [K] fp16   (single token activation vector)
    qw_ptr,       # [K, N//8] int32
    scales_ptr,   # [K//G, N] fp16
    qz_ptr,       # [K//G, N//8] int32
    c_ptr,        # [N] fp16
    K, N,
    G: tl.constexpr,
    BLOCK_N: tl.constexpr,   # output cols per program (multiple of 8)
):
    # One program owns BLOCK_N output columns and loops over all K.
    # Scales/zeros are loaded ONCE PER GROUP (not per K-row) -> tiny.
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BN] output cols
    mask_n = offs_n < N
    offs_n8 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    mask_n8 = offs_n8 < (N // 8)

    # reverse-AWQ shifts: order[j]=(j%2)*4+(j//2); shift=order*4
    j = tl.arange(0, BLOCK_N) % 8
    shifts = ((j % 2) * 4 + (j // 2)) * 4              # [BN]
    j8 = tl.arange(0, BLOCK_N // 8)                    # unused but documents packing

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    shifts2d = shifts[None, :]                          # [1,BN]

    # BLOCK_K rows per step; BLOCK_K <= G so each tile is within ONE group.
    n_groups = K // G
    for g in range(0, n_groups):
        # group scales/zeros loaded ONCE per group as [BN]
        s_ptrs = scales_ptr + g * N + offs_n
        scales = tl.load(s_ptrs, mask=mask_n, other=0.0).to(tl.float32)   # [BN]
        qz_ptrs = qz_ptr + g * (N // 8) + offs_n8
        qzv = tl.load(qz_ptrs, mask=mask_n8, other=0)                     # [BN//8]
        ze = tl.interleave(qzv, qzv); ze = tl.interleave(ze, ze); ze = tl.interleave(ze, ze)
        zf = ((ze >> shifts) & 0xF).to(tl.float32)                        # [BN]

        # one 2D tile of G rows: [G, BN]
        offs_k = g * G + tl.arange(0, G)                                  # [G]
        a = tl.load(a_ptr + offs_k).to(tl.float32)                        # [G]
        qw_ptrs = qw_ptr + offs_k[:, None] * (N // 8) + offs_n8[None, :]   # [G, BN//8]
        m_w = (offs_k[:, None] < K) & mask_n8[None, :]
        qwt = tl.load(qw_ptrs, mask=m_w, other=0)                         # [G, BN//8]
        bt = tl.interleave(qwt, qwt); bt = tl.interleave(bt, bt); bt = tl.interleave(bt, bt)  # [G,BN]
        bt = (bt >> shifts2d) & 0xF
        wt = (bt.to(tl.float32) - zf[None, :]) * scales[None, :]          # [G, BN]
        acc += tl.sum(a[:, None] * wt, axis=0)                            # [BN]

    c = acc.to(c_ptr.type.element_ty)
    tl.store(c_ptr + offs_n, c, mask=mask_n)


def awq_gemv(a, qw, scales, qz, G, BLOCK_N=64, num_warps=4):
    K = a.shape[-1]
    N = scales.shape[1]
    c = torch.empty((N,), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(N, BLOCK_N),)
    awq_gemv_kernel[grid](a, qw, scales, qz, c, K, N, G,
                          BLOCK_N=BLOCK_N, num_warps=num_warps)
    return c


@triton.jit
def awq_gemv_splitk_kernel(
    a_ptr, qw_ptr, scales_ptr, qz_ptr, c_ptr,
    K, N,
    G: tl.constexpr, BLOCK_N: tl.constexpr, GROUPS_PER_PID: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n8 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    mask_n8 = offs_n8 < (N // 8)
    j = tl.arange(0, BLOCK_N) % 8
    shifts = ((j % 2) * 4 + (j // 2)) * 4
    shifts2d = shifts[None, :]
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    g0 = pid_k * GROUPS_PER_PID
    for gi in range(0, GROUPS_PER_PID):
        g = g0 + gi
        s_ptrs = scales_ptr + g * N + offs_n
        scales = tl.load(s_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        qz_ptrs = qz_ptr + g * (N // 8) + offs_n8
        qzv = tl.load(qz_ptrs, mask=mask_n8, other=0)
        ze = tl.interleave(qzv, qzv); ze = tl.interleave(ze, ze); ze = tl.interleave(ze, ze)
        zf = ((ze >> shifts) & 0xF).to(tl.float32)
        offs_k = g * G + tl.arange(0, G)
        a = tl.load(a_ptr + offs_k).to(tl.float32)
        qw_ptrs = qw_ptr + offs_k[:, None] * (N // 8) + offs_n8[None, :]
        m_w = (offs_k[:, None] < K) & mask_n8[None, :]
        qwt = tl.load(qw_ptrs, mask=m_w, other=0)
        bt = tl.interleave(qwt, qwt); bt = tl.interleave(bt, bt); bt = tl.interleave(bt, bt)
        bt = (bt >> shifts2d) & 0xF
        wt = (bt.to(tl.float32) - zf[None, :]) * scales[None, :]
        acc += tl.sum(a[:, None] * wt, axis=0)
    tl.atomic_add(c_ptr + offs_n, acc, mask=mask_n)


def awq_gemv_splitk(a, qw, scales, qz, G, BLOCK_N=64, num_warps=4, split_k=4, num_stages=2):
    K = a.shape[-1]; N = scales.shape[1]
    c = torch.zeros((N,), dtype=torch.float32, device=a.device)
    n_groups = K // G
    gpp = triton.cdiv(n_groups, split_k)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(n_groups, gpp))
    awq_gemv_splitk_kernel[grid](a, qw, scales, qz, c, K, N,
                                 G=G, BLOCK_N=BLOCK_N, GROUPS_PER_PID=gpp,
                                 num_warps=num_warps, num_stages=num_stages)
    return c.to(a.dtype)


# reference dequant in torch (for correctness)
def awq_dequant_ref(qw, scales, qz, G):
    K, N8 = qw.shape
    N = N8 * 8
    shifts = torch.tensor([(((j % 8) % 2) * 4 + ((j % 8) // 2)) * 4 for j in range(N)],
                          device=qw.device, dtype=torch.int32)
    gk = (torch.arange(K, device=qw.device) // G)
    # weights packed per-K: [K,N8]->[K,N]
    b = qw.repeat_interleave(8, dim=1)            # [K, N]
    b = (b >> shifts[None, :]) & 0xF
    # zeros packed per-group: [K//G,N8]->[K//G,N]->index by gk ->[K,N]
    zg = qz.repeat_interleave(8, dim=1)           # [K//G, N]
    zg = (zg >> shifts[None, :]) & 0xF
    z = zg[gk]                                     # [K, N]
    s = scales[gk]                                 # [K, N]
    w = (b.to(torch.float32) - z.to(torch.float32)) * s.to(torch.float32)
    return w                                        # [K, N] fp32


def bench(fn, *a, iters=50, warm=10):
    for _ in range(warm): fn(*a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn(*a)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def make_awq(K, N, G):
    N8 = N // 8
    qw = torch.randint(0, 2**31, (K, N8), dtype=torch.int32, device=DEV)
    qz = torch.randint(0, 2**31, (K // G, N8), dtype=torch.int32, device=DEV)
    scales = (torch.randn(K // G, N, device=DEV) * 0.01).to(torch.float16)
    return qw, scales, qz


def main():
    torch.manual_seed(0)
    G = 128
    shapes = [
        ("gate/up", 4096, 12288 * 2),
        ("down",    12288, 4096),
    ]
    for name, K, N in shapes:
        a = (torch.randn(K, device=DEV) * 0.1).to(torch.float16)
        qw, scales, qz = make_awq(K, N, G)
        # correctness vs ref (dequant @ a)
        w_ref = awq_dequant_ref(qw, scales, qz, G)        # [K,N] fp32
        c_ref = (a.to(torch.float32) @ w_ref).to(torch.float16)
        c_gemv = awq_gemv(a, qw, scales, qz, G)
        err = (c_ref.float() - c_gemv.float()).abs().max().item()
        rel = err / (c_ref.float().abs().max().item() + 1e-6)
        # FP16 baseline GEMV (full dense weight)
        w16 = w_ref.to(torch.float16)
        def fp16_gemv(): return (a @ w16)
        t_fp16 = bench(fp16_gemv)
        gb_i4 = K * N * 0.5 / 1e9
        gb_16 = K * N * 2 / 1e9
        print(f"[{name}] K={K} N={N}  maxabs_err={err:.4f} rel={rel:.4f}")
        print(f"   fp16 GEMV : {t_fp16:.3f} ms  ({gb_16/(t_fp16/1e3):.0f} GB/s)")
        best = None
        for BN in [32, 64, 128, 256]:
            for nw in [1, 2, 4, 8]:
                try:
                    c2 = awq_gemv(a, qw, scales, qz, G, BLOCK_N=BN, num_warps=nw)
                    if (c_ref.float() - c2.float()).abs().max().item() > 0.5: continue
                    t = bench(lambda: awq_gemv(a, qw, scales, qz, G, BLOCK_N=BN, num_warps=nw))
                    if best is None or t < best[0]: best = (t, BN, nw)
                except Exception:
                    pass
        t, BN, nw = best
        print(f"   int4 GEMV*: {t:.3f} ms  ({gb_i4/(t/1e3):.0f} GB/s)  [BLOCK_N={BN} warps={nw}]  speedup {t_fp16/t:.2f}x")
        # split-K variant (wider sweep)
        bestk = None
        n_groups = K // G
        for BN in [32, 64, 128, 256]:
            for nw in [1, 2, 4]:
                for sk in [4, 8, 16, 32, 48, n_groups]:
                    for ns in [1, 2, 3]:
                        try:
                            c2 = awq_gemv_splitk(a, qw, scales, qz, G, BLOCK_N=BN, num_warps=nw, split_k=sk, num_stages=ns)
                            if (c_ref.float() - c2.float()).abs().max().item() > 0.5: continue
                            tt = bench(lambda: awq_gemv_splitk(a, qw, scales, qz, G, BLOCK_N=BN, num_warps=nw, split_k=sk, num_stages=ns))
                            if bestk is None or tt < bestk[0]: bestk = (tt, BN, nw, sk, ns)
                        except Exception:
                            pass
        if bestk:
            tt, BN, nw, sk, ns = bestk
            print(f"   int4 splitK: {tt:.3f} ms  ({gb_i4/(tt/1e3):.0f} GB/s)  [BLOCK_N={BN} warps={nw} splitK={sk} stages={ns}]  speedup {t_fp16/tt:.2f}x")


if __name__ == "__main__":
    main()
