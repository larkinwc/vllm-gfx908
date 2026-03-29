---
name: benchmark-worker
description: Runs benchmarks, creates comparison reports, and documents production recommendations
---

# Benchmark Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that involve:
- Running synthetic and coding agent benchmarks
- Comparing TQ backend performance vs baseline/optimized configurations
- Measuring VRAM usage and compression ratios
- Creating comparison reports
- Writing production recommendations

## Required Skills

None.

## Work Procedure

### Step 1: Read Context

Read these files:
- `.factory/library/architecture.md` -- system architecture
- `.factory/library/environment.md` -- paths, previous baseline results
- `AGENTS.md` -- boundaries, server management
- `.factory/services.yaml` -- how to start/stop vLLM

Also read previous benchmark results for comparison baselines:
- `/root/benchmark-results/final-report.json` -- previous mission's final report
- `/root/benchmark-results/baseline-report.json` -- original baselines

### Step 2: Plan Benchmark Matrix

Define the configurations to test:
- **Baseline optimized**: FULL_DECODE_ONLY + prefix caching (reference from previous mission)
- **TQ capture_only**: TQ backend in capture_only mode (overhead measurement)
- **TQ hybrid**: TQ backend in hybrid mode (actual TQ decode)
- Each at c=1, c=2, c=4 concurrent users

For each configuration, measure:
- Throughput (tok/s) via vllm bench serve
- TPOT (time per output token)
- TTFT (time to first token)
- VRAM usage via rocm-smi

### Step 3: Run Benchmarks

For each configuration:
1. Stop any running vLLM instance
2. Start vLLM with the target configuration
3. Wait for health check
4. Run synthetic benchmark: `vllm bench serve --model /models/Qwen3.5-9B --num-prompts 100 --request-rate 2 --dataset-name random --save-result`
5. Run coding agent benchmark: `/opt/vllm-env/bin/python3 /root/benchmark-scripts/coding_agent_bench.py` at c=1,2,4
6. Record VRAM usage: `rocm-smi --showmeminfo vram --json`
7. Save results to `/root/benchmark-results/` with descriptive filenames including timestamp

**IMPORTANT:** Save raw benchmark output JSON files. Do not just report summary numbers.

### Step 4: Create Comparison Report

Create a structured report comparing all configurations:
- Save as `/root/benchmark-results/tq-comparison-report.json` (machine-readable)
- Include percentage changes vs baseline optimized
- Include VRAM comparison
- Include quality assessment (if applicable)

### Step 5: Write Production Recommendation

Based on benchmark evidence, write a clear recommendation:
- Should TQ be enabled in production for Qwen3.5-9B on MI100?
- If yes: create a production launch script with TQ
- If no: explain why with data (e.g., overhead exceeds savings, quality regression)
- Document what would need to change for TQ to be beneficial (e.g., model with more full-attention layers)

### Step 6: Verify Results

- Cross-check that all benchmark JSON files are valid and complete
- Verify comparison report numbers match raw data
- Ensure recommendation is supported by the data

## Example Handoff

```json
{
  "salientSummary": "Ran full benchmark suite: TQ capture_only has 2.3% overhead vs optimized baseline, TQ hybrid shows 5% throughput improvement at c=2 with 15% VRAM savings on full-attention layers. Recommendation: enable TQ hybrid for workloads with >8k context where VRAM savings matter.",
  "whatWasImplemented": "Ran synthetic + coding agent benchmarks at c=1,2,4 for 3 configurations. Created tq-comparison-report.json with full metrics. Created production recommendation with data-backed conclusion.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "/opt/vllm-env/bin/python3 -m vllm.entrypoints.cli.main bench serve --model /models/Qwen3.5-9B --num-prompts 100 --request-rate 2 --dataset-name random --save-result",
        "exitCode": 0,
        "observation": "TQ hybrid: 465 tok/s at c=2 (vs 478 baseline optimized = -2.7%)"
      },
      {
        "command": "rocm-smi --showmeminfo vram --json",
        "exitCode": 0,
        "observation": "TQ hybrid VRAM: 28.8 GB avg vs 29.5 GB baseline = 2.4% savings"
      },
      {
        "command": "/opt/vllm-env/bin/python3 /root/benchmark-scripts/coding_agent_bench.py --concurrency 4",
        "exitCode": 0,
        "observation": "4/4 concurrent requests successful, aggregate 310 tok/s"
      }
    ],
    "interactiveChecks": [
      {
        "action": "Verified tq-comparison-report.json has all required fields",
        "observed": "Report contains throughput, TPOT, TTFT, VRAM for all 3 configs at all concurrency levels"
      }
    ]
  },
  "tests": {
    "added": [],
    "coverage": "Benchmark results validated via cross-checking raw JSON files against report"
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- TQ backend server fails to start (backend implementation issue, not benchmark issue)
- Benchmark scripts from previous mission are broken or incompatible
- Results show catastrophic regression (>50% throughput loss) suggesting a backend bug
- VRAM exhaustion prevents completing benchmark suite
