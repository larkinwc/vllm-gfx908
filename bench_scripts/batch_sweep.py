import os, time
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    llm = LLM(model="Qwen/Qwen3.5-9B", tensor_parallel_size=4, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=4096,
              max_num_seqs=160, **{"language_model_only": True})
    prompt = "Write a long detailed essay about the history of computing."
    GEN=96
    # warmup
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
    print("== decode batch sweep (TP4, FP16, gen=%d tok) =="%GEN, flush=True)
    for B in [32,48,64,96,128]:
        sp = SamplingParams(max_tokens=GEN, temperature=0, ignore_eos=True)
        prompts=[prompt]*B
        t0=time.time(); o=llm.generate(prompts, sp); dt=time.time()-t0
        outtok=sum(len(x.outputs[0].token_ids) for x in o)
        # subtract approx prefill by measuring step rate: total decode tok / wall
        print("B=%2d  out_tok=%4d  wall=%6.2fs  agg_tok_s=%6.2f  per_step_ms=%6.1f"%(
              B, outtok, dt, outtok/dt, 1000*dt/GEN), flush=True)
if __name__=="__main__": main()
