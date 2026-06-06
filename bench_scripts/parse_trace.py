import gzip, json, glob, re
from collections import defaultdict
f=sorted(glob.glob("/home/larkinwc/decode_prof/rank0.*.pt.trace.json.gz"))[0]
d=json.load(gzip.open(f))
ev=d["traceEvents"]
# GPU kernel events: have "cat":"kernel" and dur, on a device stream
cat=defaultdict(float); cnt=defaultdict(int); tot=0.0
def bucket(n):
    if n.startswith("Cijk_") or "gemm" in n.lower(): return "GEMM (rocBLAS hgemm, M=1 padded)"
    if "nccl" in n.lower() or "all_reduce" in n.lower() or "AllReduce" in n: return "RCCL all-reduce/comm"
    if "gated_delta" in n or "gdn" in n.lower() or "causal_conv" in n or "ChunkGated" in n or "reduce_segments" in n or "layer_norm_fwd" in n: return "GDN linear-attn"
    if "unified_attention" in n or "kernel_unified" in n or "paged" in n or "flash" in n: return "Full attention"
    if "rms_norm" in n: return "RMSNorm"
    if "act_and_mul" in n or "silu" in n: return "SiLU/act"
    if "elementwise" in n or "reduce_kernel" in n or "vectorized" in n or "copy" in n.lower() or "fill" in n or "Copy" in n: return "elementwise/copy"
    if "mrope" in n or "rope" in n.lower(): return "RoPE"
    if "reshape_and_cache" in n or "kv_cache" in n: return "KV cache write"
    if "argmax" in n or "embedding" in n.lower() or "index" in n: return "sampling/embed/index"
    return "other:"+n[:30]
for e in ev:
    if e.get("cat")=="kernel" and "dur" in e:
        n=e.get("name","")
        b=bucket(n); cat[b]+=e["dur"]; cnt[b]+=1; tot+=e["dur"]
print("Total GPU kernel time in window: %.1f ms (rank0)"%(tot/1000))
print("%-42s %10s %8s %8s"%("category","ms","%","calls"))
for k in sorted(cat,key=lambda x:-cat[x]):
    print("%-42s %10.1f %7.1f%% %8d"%(k, cat[k]/1000, 100*cat[k]/tot, cnt[k]))
