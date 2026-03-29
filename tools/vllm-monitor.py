#!/usr/bin/env python3
import subprocess, urllib.request, time, sys, os, shutil

METRICS_URL = "http://localhost:8000/metrics"
REFRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

BOLD = "\033[1m"
DIM = "\033[2m"
RST = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
WHITE = "\033[37m"

prev_prompt = prev_gen = prev_time = None

def color_temp(t):
    if t < 60: return f"{GREEN}{t:5.0f}{RST}"
    if t < 80: return f"{YELLOW}{t:5.0f}{RST}"
    return f"{RED}{t:5.0f}{RST}"

def color_util(u):
    if u < 30: return f"{DIM}{u:3.0f}{RST}"
    if u < 80: return f"{GREEN}{u:3.0f}{RST}"
    return f"{YELLOW}{u:3.0f}{RST}"

def color_power(w):
    if w < 100: return f"{DIM}{w:5.0f}{RST}"
    if w < 200: return f"{YELLOW}{w:5.0f}{RST}"
    return f"{RED}{w:5.0f}{RST}"

def get_gpu_stats():
    r = subprocess.run(
        ["rocm-smi", "--showtemp", "--showuse", "--showpower", "--showmemuse", "--csv"],
        capture_output=True, text=True, timeout=5
    )
    gpus = []
    for line in r.stdout.strip().split("\n")[1:]:
        if not line.strip():
            continue
        f = line.split(",")
        gpus.append({
            "id": f[0].replace("card", ""),
            "edge_c": float(f[1]), "junc_c": float(f[2]), "mem_c": float(f[3]),
            "power_w": float(f[4]),
            "gpu_pct": float(f[5]), "vram_pct": float(f[6]),
        })
    return gpus

def get_vllm_metrics():
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=3) as resp:
            body = resp.read().decode()
    except Exception:
        return None
    m = {}
    for line in body.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            m[parts[0].split("{")[0]] = float(parts[1])
    return m

def find_metric(m, key):
    if not m:
        return 0.0
    for k, v in m.items():
        if key in k:
            return v
    return 0.0

try:
    while True:
        cols = shutil.get_terminal_size().columns
        gpus = get_gpu_stats()
        metrics = get_vllm_metrics()
        now = time.time()

        prompt_tok = find_metric(metrics, "vllm:prompt_tokens_total")
        gen_tok = find_metric(metrics, "vllm:generation_tokens_total")
        running = find_metric(metrics, "vllm:num_requests_running")
        waiting = find_metric(metrics, "vllm:num_requests_waiting")
        kv_pct = find_metric(metrics, "vllm:kv_cache_usage_perc") * 100

        ttft_sum = find_metric(metrics, "vllm:time_to_first_token_seconds_sum")
        ttft_cnt = find_metric(metrics, "vllm:time_to_first_token_seconds_count")
        e2e_sum = find_metric(metrics, "vllm:e2e_request_latency_seconds_sum")
        e2e_cnt = find_metric(metrics, "vllm:e2e_request_latency_seconds_count")

        prompt_tps = gen_tps = 0.0
        if prev_time and (now - prev_time) > 0.1:
            dt = now - prev_time
            prompt_tps = (prompt_tok - prev_prompt) / dt
            gen_tps = (gen_tok - prev_gen) / dt
        prev_prompt, prev_gen, prev_time = prompt_tok, gen_tok, now

        avg_ttft = (ttft_sum / ttft_cnt) if ttft_cnt > 0 else 0
        avg_e2e = (e2e_sum / e2e_cnt) if e2e_cnt > 0 else 0

        os.system("clear")
        w = min(cols, 100)
        bar = "=" * w
        print(f"{BOLD}{CYAN}  vLLM MI100 Status Monitor{RST}")
        print(f"{DIM}{bar}{RST}")

        print(f"{BOLD}  GPU  Junc*C  Mem*C  Power(W)  Util%  VRAM%{RST}")
        print(f"  {'-'*46}")
        for g in gpus:
            print(
                f"   {WHITE}{g['id']:>2}{RST}"
                f"   {color_temp(g['junc_c'])}"
                f"   {color_temp(g['mem_c'])}"
                f"    {color_power(g['power_w'])}"
                f"    {color_util(g['gpu_pct'])}"
                f"    {color_util(g['vram_pct'])}"
            )

        print(f"  {'-'*46}")
        total_w = sum(g["power_w"] for g in gpus)
        avg_j = sum(g["junc_c"] for g in gpus) / max(len(gpus), 1)
        print(f"  {DIM}Total power: {total_w:.0f}W   Avg junction: {avg_j:.0f}*C{RST}")
        print()

        if metrics is None:
            print(f"  {RED}vLLM metrics unavailable (server down?){RST}")
        else:
            print(f"{BOLD}  Inference{RST}")
            print(f"  {'-'*46}")

            gen_color = GREEN if gen_tps > 0 else DIM
            print(f"   Prompt tok/s:  {gen_color}{prompt_tps:>8.1f}{RST}")
            print(f"   Gen tok/s:     {gen_color}{gen_tps:>8.1f}{RST}")
            print(f"   Avg TTFT:      {avg_ttft:>7.2f}s")
            print(f"   Avg E2E:       {avg_e2e:>7.2f}s")
            print()
            print(f"   Requests:      {CYAN}{running:.0f}{RST} running  {YELLOW}{waiting:.0f}{RST} waiting")
            print(f"   KV cache:      {kv_pct:>5.1f}%")
            print(f"   Total tokens:  {prompt_tok:.0f} prompt  {gen_tok:.0f} gen")

        print()
        print(f"  {DIM}Refresh: {REFRESH}s | Ctrl+C to quit{RST}")
        time.sleep(REFRESH)

except KeyboardInterrupt:
    print(f"\n{DIM}Exiting.{RST}")
