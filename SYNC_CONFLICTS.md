# Upstream Sync 2026-05-28 — Conflict Surface

**Status:** in-progress (M1-F2 dry-run capture; resolution deferred to M1-F3+)
**Sync branch:** `upstream-sync-2026-05-28`
**Base (ours):** `3d9de886d` (`origin/mi100-fixes` HEAD at mission start)
**Theirs:** `upstream/main` = `5b115bb8a33d72820075450ecefcd292607bfe57`
  - "[Attention][AMD] Standardize kv layout to blocks first for AMD (#43660)"
**Commits behind upstream/main:** 770

## Dry-run merge command

```bash
git merge --no-commit --no-ff upstream/main
git diff --name-only --diff-filter=U > /tmp/conflicts.txt
```

After capture, the merge was aborted (`git merge --abort`) per M1-F2 scope.

## Conflict surface (raw, from `git diff --name-only --diff-filter=U`)

Total unmerged paths: **2**

| # | Path | Conflict kind | File lines (working tree) | `<<<<<<<` hunks | Notes / planned resolution |
|---|------|---------------|---------------------------|-----------------|----------------------------|
| 1 | `examples/deployment/quantize_w8a8_mi100.py` | rename/location (AU) | 170 | 0 | HEAD added `examples/offline_inference/quantize_w8a8_mi100.py`; upstream renamed the parent `examples/offline_inference/` → `examples/deployment/`. Resolution: accept the relocation — `git add examples/deployment/quantize_w8a8_mi100.py` and `git rm examples/offline_inference/quantize_w8a8_mi100.py` if still present. Content is ours unchanged. |
| 2 | `vllm/v1/worker/gpu_model_runner.py` | content (UU) | 7448 | 1 | Single 3-marker conflict hunk. Must hand-merge per integration-worker playbook (blocks-first KV layout #43660 / FP8 Q-scale handling #42080 / GPU↔CPU sync removal #41434 cluster). Resolution deferred to M1-F3. |

## Auto-merged files (informational; resolved by git, no manual action required)

The following files had textual overlap that `git merge` resolved automatically — they
are NOT conflicts but are listed so the resolver in M1-F3 knows where upstream
touched MI100-adjacent surface:

- `.buildkite/test_areas/lm_eval.yaml`
- `.gitignore`
- `AGENTS.md`
- `CMakeLists.txt`
- `csrc/rocm/skinny_gemms.cu` *(integration-worker playbook calls out wvSplitK N=5 #40687 here — verify post-merge content is correct)*
- `vllm/distributed/device_communicators/quick_all_reduce.py`
- `vllm/model_executor/kernels/linear/__init__.py`
- `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py` *(playbook calls out Marlin-fallback #43731 + MI100 branch preservation — verify)*
- `vllm/model_executor/kernels/linear/scaled_mm/__init__.py`
- `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/quantization/gguf.py`
- `vllm/model_executor/model_loader/gguf_loader.py`
- `vllm/model_executor/model_loader/weight_utils.py`
- `vllm/model_executor/models/minimax_m2.py`
- `vllm/platforms/rocm.py` *(playbook calls out blocks-first KV #43660 + libtorch-stable ABI activation #42663 + auto_gptq rename #38288 — verify all 166 MI100 dispatch lines survived auto-merge)*
- `vllm/tokenizers/registry.py`
- `vllm/transformers_utils/config.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backends/triton_attn.py` *(playbook calls out #42095 / #42080 / #41434 + ROCM_CK_FA dispatch hook — verify)*
- `vllm/v1/attention/ops/triton_unified_attention.py`

## Discrepancy vs. mission plan

Mission `architecture.md` and `integration-worker` skill anticipated a
**"22-file conflict surface."** Actual conflict count is **2 files** (plus the
~22 auto-merged files listed above). The mission's "22" appears to have been
the count of overlapping-touch files, of which all but 2 auto-merged cleanly.

**Implication for M1-F3:** The hand-merge surface is much smaller than budgeted.
However, the integration-worker playbook's verification guidance on the
auto-merged files (`skinny_gemms.cu`, `rocm.py`, `triton_attn.py`,
`triton_w4a16.py`) must still be carried out as a post-merge correctness audit
even though git did not flag them as conflicts.

## Next step (M1-F3)

Restart the merge (`git merge --no-commit --no-ff upstream/main`), resolve the
two conflicts above per the playbook, audit the four MI100-critical
auto-merged files, then commit.
