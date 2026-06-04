import os, sys, time
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode
MODEL="bullpoint/Qwen3-Coder-Next-AWQ-4bit"
def run(tp, pp, label):
    t0=time.time()
    llm=LLM(model=MODEL, tensor_parallel_size=tp, pipeline_parallel_size=pp, dtype="float16",
            gpu_memory_utilization=0.92, max_model_len=4096, max_num_seqs=8,
            trust_remote_code=True, enforce_eager=False,
            compilation_config=CompilationConfig(mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=[1,8]))
    print("INIT[%s] %.0fs"%(label, time.time()-t0), flush=True)
    base="Write a short note about topic number "
    for c in [1,8]:
        prompts=[base+str(i) for i in range(c)]
        sp=SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
        llm.generate(prompts[:1], SamplingParams(max_tokens=4,temperature=0,ignore_eos=True))
        t0=time.time(); o=llm.generate(prompts, sp); dt=time.time()-t0
        tot=sum(len(x.outputs[0].token_ids) for x in o)
        print("[%s] c=%2d agg=%.1f per-stream=%.2f"%(label,c,tot/dt,tot/dt/c), flush=True)
    print("DONE[%s]"%label, flush=True)
if __name__=="__main__":
    run(8,1,"TP8") if sys.argv[1]=="tp8" else run(4,2,"TP4xPP2")
