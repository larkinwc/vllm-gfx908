import os, time, sys
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    preset = sys.argv[1]
    kw = dict(model="QuantTrio/Qwen3.5-9B-AWQ", tensor_parallel_size=2, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.85, max_model_len=4096,
              **{"language_model_only": True})
    if preset!="auto": kw["kv_cache_dtype"]=preset
    llm=LLM(**kw)
    sp=SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
    p=["The capital of France is"]
    llm.generate(p, SamplingParams(max_tokens=8,temperature=0,ignore_eos=True))  # warm
    t0=time.time(); o=llm.generate(p, sp); dt=time.time()-t0
    n=len(o[0].outputs[0].token_ids)
    print("DECODE[%s] short-ctx B=1: %.2f tok/s  %.1f ms/tok (n=%d)"%(preset,n/dt,1000*dt/n,n), flush=True)
    print("SAMPLE:", repr(o[0].outputs[0].text[:80]), flush=True)
    print("DONE_%s"%preset, flush=True)
if __name__=="__main__": main()
