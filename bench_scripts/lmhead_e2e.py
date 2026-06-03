import os, time
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    llm=LLM(model="Qwen/Qwen3.5-9B", tensor_parallel_size=4, dtype="float16",
            enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=2048,
            language_model_only=True)
    sp=SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
    p=["The capital of France is"]
    llm.generate(p, SamplingParams(max_tokens=8,temperature=0,ignore_eos=True))
    t0=time.time(); o=llm.generate(p, sp); dt=time.time()-t0
    n=len(o[0].outputs[0].token_ids)
    print("DECODE B=1: %.2f tok/s  %.1f ms/tok (n=%d)"%(n/dt,1000*dt/n,n), flush=True)
    print("SAMPLE:", repr(o[0].outputs[0].text[:70]), flush=True)
    print("E2E_DONE", flush=True)
if __name__=="__main__": main()
