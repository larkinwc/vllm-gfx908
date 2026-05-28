# Upstream Sync 2026-05-28 — Conflict Surface

**Status:** resolved (M1-F3 hand-merge + audit complete; merge commit pending in worktree)
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

## M1-F3 resolution record (per-file decision matrix)

### Conflicts resolved

| file | upstream_pr(s) | our_ref | decision | rationale |
|------|---------------|---------|----------|-----------|
| `examples/deployment/quantize_w8a8_mi100.py` | upstream rename `examples/offline_inference/` → `examples/deployment/` | HEAD added `examples/offline_inference/quantize_w8a8_mi100.py` (170 LOC) at `3d9de886d` | **rename-accept**: adopt upstream's new path, content is `ours` verbatim (170 LOC identical). `git add examples/deployment/quantize_w8a8_mi100.py`; the `offline_inference/` copy was implicitly removed by the rename. | Upstream relocation is the new canonical location for deployment examples; preserving our MI100 example at the new path keeps it discoverable while honoring upstream's directory restructure. |
| `vllm/v1/worker/gpu_model_runner.py` | #41745 (Gemma4 MTP) + #39949 / #33736 (ExtractHiddenStatesProposer hidden-states extraction) | our +5 LOC defensive `getattr(self, "drafter", None)` guard for non-last PP ranks | **hand-merge**: kept our `drafter = getattr(...); if drafter is None: pass; elif isinstance(drafter, ...)` guard structure, expanded the `isinstance` tuple to upstream's 4-class set `(EagleProposer, DFlashProposer, Gemma4Proposer, ExtractHiddenStatesProposer)`. Single 3-marker hunk at lines 2417-2437; resolved to lines 2415-2435 post-merge. | Both intents preserved: our PP-rank-safety guard (defensive against non-last ranks where `self.drafter` is never assigned) AND upstream's expanded proposer family. |

### MI100-critical auto-merged files — post-merge audit (clean)

For each file: ran `git diff --cached pre-sync-baseline -- <file>` (proves our MI100 changes survived) and `git diff --cached upstream/main -- <file>` (proves upstream's intent is integrated). MI100-marker preservation verified by `grep -c` against baseline.

| file | upstream PR(s) integrated | MI100 marker count (baseline → current) | audit verdict |
|------|--------------------------|-----------------------------------------|---------------|
| `csrc/rocm/skinny_gemms.cu` | wvSplitK N=5 (#40687) adopted | 1 → 1 | clean: 27 `wvSplitK` references; gfx908 patches intact |
| `vllm/platforms/rocm.py` | blocks-first KV (#43660), libtorch-stable ABI activation (#42663), auto_gptq rename (#38288) | 30 → 30 | clean: all MI100 dispatch markers preserved |
| `vllm/v1/attention/backends/triton_attn.py` | num_blocks-first layout (#42095), FP8 per-tensor Q-scale (#42080), GPU↔CPU sync removal (#41434) | MI100 helpers present at lines 53-99 (matches baseline lines 48-94) | clean: `_get_mi100_tuned_constants`, MI100 Flash-Decoding constants preserved |
| `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py` | Marlin-fallback dispatch (#43731) | 4 MI100 markers → 4 | clean: `is_mi100()` branch + `_mi100_fused_mm_cache` producer/consumer tie-ins preserved ahead of upstream's Marlin dispatch |
| `vllm/v1/attention/backends/registry.py` | upstream backend additions | `ROCM_CK_FA` registration at L57-58 | clean |
| `vllm/model_executor/kernels/linear/__init__.py` | upstream refactor | 7 MI100 references | clean |
| `vllm/model_executor/kernels/linear/scaled_mm/__init__.py` | upstream refactor | `MI100FP8ScaledMMLinearKernel`, `MI100Int8ScaledMMLinearKernel` exports preserved (L24-28, 65-66) | clean |
| `vllm/model_executor/model_loader/weight_utils.py` | upstream weight-utils refactor | 54 → 54 (gguf/safetensors/MI100 markers) | clean |
| `CMakeLists.txt` | upstream cmake updates | `gfx908` arch flag (L44), M4 CK glue (L1290-1341) preserved | clean |

### AGENTS.md / CLAUDE.md guard (hard constraint #1)

`git merge` auto-merged a +2 line addition into `AGENTS.md` (upstream PR added a line-length note). Per mission AGENTS.md hard constraint #1, both files MUST stay `ours` unconditionally. Restored with `git checkout HEAD -- AGENTS.md CLAUDE.md`; `git diff pre-sync-baseline -- AGENTS.md CLAUDE.md` is now empty.

### Next step

Merge commit pending: `Merge upstream/main into mi100-fixes (770 commits, 2 conflicts resolved, MI100 audit clean)` with `Co-authored-by: Claude` + `Signed-off-by:` trailers per project AGENTS.md §2.
