import gzip, json, glob, sys
from collections import defaultdict
d_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/larkinwc/moe35_prof"
f = sorted(glob.glob(d_dir + "/*rank0.*.pt.trace.json.gz"))[0]
d = json.load(gzip.open(f))
t = defaultdict(float); c = defaultdict(int); tot = 0.0


def bucket(n):
    l = n.lower()
    if "llgemm" in l: return "LLMM1 skinny GEMV (FP16 attn/shared)"
    if "fused_moe" in l or "moe" in l: return "fused_moe expert GEMM (tl.dot)"
    if "w4a16_gemv" in l or "gemv_splitk" in l: return "int4 GEMV (#57)"
    if n.startswith("Cijk_") or "gemm" in l or "hgemm" in l: return "rocBLAS GEMM (padded)"
    if "nccl" in l or "all_reduce" in l or "allreduce" in l: return "RCCL all-reduce"
    if "gated_delta" in l or "causal_conv" in l or "chunk" in l or "recurrent" in l: return "GDN linear-attn"
    if "topk" in l or "softmax" in l or "moe_align" in l or "sort" in l or "argsort" in l: return "MoE routing (topk/sort/align)"
    if "rms_norm" in l: return "RMSNorm"
    if "silu" in l or "act_and_mul" in l: return "SiLU/act"
    if "elementwise" in l or "copy" in l or "fill" in l or "vectorized" in l: return "elementwise/copy"
    return "other:" + n[:34]


for e in d["traceEvents"]:
    if e.get("cat") == "kernel" and "dur" in e:
        b = bucket(e["name"]); t[b] += e["dur"]; c[b] += 1; tot += e["dur"]
print("Total GPU kernel time (rank0): %.1f ms" % (tot / 1000))
print("%-40s %10s %7s %8s" % ("bucket", "ms", "%", "calls"))
for k in sorted(t, key=lambda x: -t[x])[:18]:
    print("%-40s %10.1f %6.1f%% %8d" % (k, t[k] / 1000, 100 * t[k] / tot, c[k]))
