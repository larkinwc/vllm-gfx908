import os, sys, time
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode
def main():
    layout=sys.argv[1]   # tp4 | tp2pp2 | tp8 | tp2pp4
    graphs=(sys.argv[2]=="graph") if len(sys.argv)>2 else True
    cfg=dict(tp4=(4,1), tp2pp2=(2,2), tp8=(8,1), tp2pp4=(2,4))[layout]
    tp,pp=cfg
    kw=dict(model="Qwen/Qwen3.5-9B", tensor_parallel_size=tp, pipeline_parallel_size=pp,
            dtype="float16", gpu_memory_utilization=0.90, max_model_len=2048,
            language_model_only=True, max_num_seqs=32)
    if graphs:
        kw["enforce_eager"]=False
        kw["compilation_config"]=CompilationConfig(mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1])
    else:
        kw["enforce_eager"]=True
    t0=time.time(); llm=LLM(**kw); print("INIT[%s graphs=%s] %.0fs"%(layout,graphs,time.time()-t0),flush=True)
    # decode sweep at several concurrencies
    base="The history of computing began long ago and "
    for c in [1, 8, 32]:
        prompts=[base+str(i) for i in range(c)]
        sp=SamplingParams(max_tokens=96, temperature=0, ignore_eos=True)
        llm.generate(prompts[:1], SamplingParams(max_tokens=4,temperature=0,ignore_eos=True))
        t0=time.time(); o=llm.generate(prompts, sp); dt=time.time()-t0
        tot=sum(len(x.outputs[0].token_ids) for x in o)
        print("[%s g=%s] c=%2d  out_tok/s=%.1f  per-stream=%.2f  tpot_ms=%.1f"%(layout,graphs,c,tot/dt,tot/dt/c,1000*dt/96),flush=True)
    print("PP_DONE_%s"%layout,flush=True)
if __name__=="__main__": main()
