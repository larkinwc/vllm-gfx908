import os, time, sys
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    TP=int(sys.argv[1])
    llm = LLM(model="QuantTrio/Qwen3.5-9B-AWQ", tensor_parallel_size=TP, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=4096,
              max_num_seqs=160, **{"language_model_only": True})
    prompt="Write a long detailed essay about the history of computing."
    GEN=96
    llm.generate([prompt], SamplingParams(max_tokens=8,temperature=0,ignore_eos=True))
    print("== AWQ INT4 decode sweep (TP%d, gen=%d) =="%(TP,GEN), flush=True)
    for B in [1,8,32,64,96]:
        sp=SamplingParams(max_tokens=GEN,temperature=0,ignore_eos=True)
        t0=time.time(); o=llm.generate([prompt]*B, sp); dt=time.time()-t0
        ot=sum(len(x.outputs[0].token_ids) for x in o)
        print("B=%3d out=%5d wall=%6.2fs agg_tok_s=%7.2f per_step_ms=%6.1f"%(B,ot,dt,ot/dt,1000*dt/GEN), flush=True)
if __name__=="__main__": main()
