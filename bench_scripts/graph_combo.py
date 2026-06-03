import os, sys, time
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode
def main():
    mode=sys.argv[1]; preset=sys.argv[2] if len(sys.argv)>2 else "turboquant_k8v4"
    kw=dict(model="QuantTrio/Qwen3.5-9B-AWQ", tensor_parallel_size=2, dtype="float16",
            gpu_memory_utilization=0.85, max_model_len=4096, language_model_only=True, max_num_seqs=32,
            kv_cache_dtype=preset)
    if mode=="eager":
        kw["enforce_eager"]=True
    else:
        kw["enforce_eager"]=False
        kw["compilation_config"]=CompilationConfig(mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1])
    llm=LLM(**kw)
    sp=SamplingParams(max_tokens=160, temperature=0, ignore_eos=True)
    p=["The capital of France is"]
    llm.generate(p, SamplingParams(max_tokens=8,temperature=0,ignore_eos=True))
    best=1e9
    for _ in range(3):
        t0=time.time(); o=llm.generate(p, sp); dt=time.time()-t0
        n=len(o[0].outputs[0].token_ids); best=min(best,1000*dt/n)
    print("COMBO MODE=%s preset=%s best %.1f ms/tok (%.2f tok/s)"%(mode,preset,best,1000/best),flush=True)
    print("SAMPLE:", repr(o[0].outputs[0].text[:60]),flush=True)
    print("CB_DONE",flush=True)
if __name__=="__main__": main()
