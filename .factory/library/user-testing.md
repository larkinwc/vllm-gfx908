# User Testing

## Validation Surface

- **Surface:** API endpoints on localhost:8000 (vLLM OpenAI-compatible API)
- **Tools:** curl, python scripts, vllm bench serve
- **Auth:** None required (local server)
- **Startup:** vLLM server must be started with TQ backend before testing

### Testing Approach

All validation is CLI/API-based:
1. Start vLLM with TQ backend (launch script)
2. Wait for health check (curl -sf http://localhost:8000/health)
3. Send requests via curl or python benchmark scripts
4. Collect results from JSON output files and server logs
5. Stop server, compare results

### Key Test Scripts (from previous mission, reusable)
- `/root/benchmark-scripts/run_synthetic_bench.sh` -- vllm bench serve wrapper
- `/root/benchmark-scripts/coding_agent_bench.py` -- concurrent coding workload
- `/root/benchmark-scripts/run_sustained_load_test.py` -- sustained load test
- `/root/benchmark-scripts/compare_results.py` -- comparison table generator

## Validation Concurrency

**Max concurrent validators:** 1

**Rationale:** GPU-bound workload. Only one vLLM instance can run at a time on 4x MI100 (93% VRAM used). Starting a second instance would fail due to insufficient VRAM. All validation must be sequential.

**Machine specs:** 64 CPU cores, 62 GB RAM, 4x MI100 32GB. CPU/RAM is abundant but GPU is the bottleneck.
