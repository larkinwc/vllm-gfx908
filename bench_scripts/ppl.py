import os, sys, math
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams, TokensPrompt
def main():
    preset=sys.argv[1]
    import pandas as pd
    ds=pd.read_parquet("/home/larkinwc/wikitext2_test.parquet")
    text="\n\n".join(t for t in ds["text"].tolist() if t.strip())
    kw=dict(model="Qwen/Qwen3.5-9B", tensor_parallel_size=4, dtype="float16",
            enforce_eager=True, gpu_memory_utilization=0.70, max_model_len=4096,
            language_model_only=True)
    if preset!="auto": kw["kv_cache_dtype"]=preset
    llm=LLM(**kw)
    tok=llm.get_tokenizer()
    ids=tok(text, add_special_tokens=False)["input_ids"]
    W=512; stride=512; N=24  # 20 windows of 2048 = ~40k tokens evaluated
    prompts=[]
    for i in range(N):
        s=i*stride
        if s+W>len(ids): break
        prompts.append(TokensPrompt(prompt_token_ids=ids[s:s+W]))
    sp=SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)
    nll=0.0; ntok=0
    for pr in prompts:
        out=llm.generate([pr], sampling_params=sp)
        o=out[0]
        pl=o.prompt_logprobs
        for d in pl[1:]:
            lp=next(iter(d.values())).logprob
            nll+=-lp; ntok+=1
    ppl=math.exp(nll/ntok)
    print("PPL[%s] = %.4f  (tokens=%d, windows=%d)"%(preset, ppl, ntok, len(prompts)), flush=True)
    print("PPL_DONE_%s"%preset, flush=True)
if __name__=="__main__": main()
