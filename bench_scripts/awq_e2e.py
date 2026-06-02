import os, time
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    llm = LLM(model="QuantTrio/Qwen3.5-9B-AWQ", tensor_parallel_size=2, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=4096,
              max_num_seqs=64, **{"language_model_only": True})
    # correctness
    o=llm.generate(["The capital of France is"], SamplingParams(max_tokens=24,temperature=0))
    print("CORRECT:", repr(o[0].outputs[0].text), flush=True)
    # decode latency bs=1
    prompt="Write a long detailed essay about the history of computing."
    sp=SamplingParams(max_tokens=128,temperature=0,ignore_eos=True)
    llm.generate([prompt], SamplingParams(max_tokens=8,temperature=0,ignore_eos=True))
    for B in [1,8,32]:
        t0=time.time(); oo=llm.generate([prompt]*B, sp); dt=time.time()-t0
        ot=sum(len(x.outputs[0].token_ids) for x in oo)
        print("B=%2d agg_tok_s=%.2f ms_per_tok=%.1f"%(B, ot/dt, 1000*dt/128), flush=True)
if __name__=="__main__": main()
