import os, time, sys
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    TP=int(sys.argv[1]) if len(sys.argv)>1 else 2
    t0=time.time()
    llm = LLM(model="QuantTrio/Qwen3.5-9B-AWQ", tensor_parallel_size=TP,
              dtype="float16", enforce_eager=True,
              gpu_memory_utilization=0.90, max_model_len=4096,
              max_num_seqs=64, **{"language_model_only": True})
    print("AWQ_INIT TP=%d in %.1fs"%(TP,time.time()-t0), flush=True)
    sp=SamplingParams(max_tokens=32, temperature=0)
    o=llm.generate(["The capital of France is"], sp)
    print("RESULT:", repr(o[0].outputs[0].text), flush=True)
    print("ALL_DONE", flush=True)
if __name__=="__main__": main()
