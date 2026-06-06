import torch, time
import vllm._custom_ops as ops
from vllm.model_executor.layers.utils import num_compute_units
torch.manual_seed(0)
dev="cuda"
# lm_head shapes: vocab shard per TP rank. Full vocab 248320, TP4 -> 62080. Also test TP1 full.
H=4096
cu=num_compute_units()
print("CUs=",cu)
def bench(fn,*a,iters=50):
    for _ in range(5): fn(*a)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(iters): r=fn(*a)
    torch.cuda.synchronize(); return (time.time()-t)/iters*1000, r
for V in [62080, 124160, 248320]:   # m = vocab shard (TP4, TP2, TP1)
    # ensure m%4==0 for LLMM1, n==1
    w=torch.randn(V,H,device=dev,dtype=torch.float16)*0.02
    x=torch.randn(1,H,device=dev,dtype=torch.float16)*0.02
    ref=torch.nn.functional.linear(x,w)
    # rocBLAS baseline
    tb,_=bench(lambda x,w: torch.nn.functional.linear(x,w), x,w)
    # LLMM1: signature ops.LLMM1(weight, x_view, rows_per_block=4)
    try:
        tl,rl=bench(lambda w,x: ops.LLMM1(w, x, 4), w, x)
        err=(rl.reshape(ref.shape)-ref).abs().max().item()/ (ref.abs().max().item()+1e-6)
        okl="OK" if err<2e-2 else "BAD"
    except Exception as e:
        tl,err,okl=-1,-1,"ERR:"+str(e)[:40]
    print("V=%6d  rocBLAS=%.3fms  LLMM1=%.3fms (%.2fx) relerr=%.1e %s"%(V,tb,tl,(tb/tl if tl>0 else 0),err,okl), flush=True)
print("DONE")
