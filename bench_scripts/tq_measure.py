import os, time, sys
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    preset = sys.argv[1]   # "auto" (fp16) or turboquant_*
    kw = dict(model="Qwen/Qwen3.5-9B", tensor_parallel_size=4, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=16384,
              **{"language_model_only": True})
    if preset != "auto":
        kw["kv_cache_dtype"] = preset
    llm = LLM(**kw)
    # report KV cache blocks available (proxy for KV mem efficiency)
    try:
        nb = llm.llm_engine.cache_config.num_gpu_blocks
        bs = llm.llm_engine.cache_config.block_size
        print("KV_BLOCKS[%s] num_gpu_blocks=%s block_size=%s -> tokens=%s"%(preset, nb, bs, (nb*bs if nb else "?")), flush=True)
    except Exception as e:
        print("kvinfo err", e, flush=True)
    # long-context decode: prefill ~8k, decode 64
    longp = "The history of computing began long ago. " * 400  # ~ a few k tokens
    sp=SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
    llm.generate([longp], SamplingParams(max_tokens=4,temperature=0,ignore_eos=True))
    for B in [1, 8]:
        t0=time.time(); o=llm.generate([longp]*B, sp); dt=time.time()-t0
        ot=sum(len(x.outputs[0].token_ids) for x in o)
        print("[%s] B=%d decode_tok_s=%.2f ms_per_tok=%.1f"%(preset, B, ot/dt, 1000*dt/64), flush=True)
    print("DONE_%s"%preset, flush=True)
if __name__=="__main__": main()
