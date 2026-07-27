# gfx900 AWQ compile dispatch fix handoff

## Goal and status

Fix the gfx900 AWQ TP4 production `VLLM_COMPILE` decode collapse. The source fix
is commit `643fc1735d56b705bf26a5e9400ee81e346a379a`; production `vllm serve`
consumption is **not yet verified**.

The accepted platform digest is
`61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`.
The acceptance clone is
`/home/larkinwc/src/vllm-gfx900-acceptance-3973e0ec9`, rooted at
`3973e0ec9cd10b95f4663096237c025806efdfdb`. Its last-known HEAD before the
outage was `bc7681153de2b00a2ed40abe1aba8943a825ab73`, after the
session-ephemeral digest fix.

## Confirmed root cause and causal proof

vLLM's no-guards compile wrapper drops all Torch compile guards by default. The
plain-Python `M <= 8` dispatch in `triton_w4a16_gemm` was first traced during
the M=512 profile run. That decision was frozen, so later M=1 decode calls
reused the generic GEMM instead of the gfx900 GEMV fast path.

The isolated M=512-then-M=1 three-arm test changed only guard handling:

1. Guards dropped: M=1 silently reused `triton_w4a16_gemm_kernel`; the call was
   about 14.3 ms and no GEMV kernel ran.
2. Guards kept with `fail_on_recompile`: PyTorch raised the expected guard
   failure (`65 <= a.size()[0]`), proving the M change was detectable.
3. Guards kept with recompilation allowed: M=1 retraced (`compile_id=0/1`) and
   ran `_w4a16_gemv_splitk_kernel` in 156 µs; the generic kernel was absent.

The full profiler measured the frozen generic kernel at **14.634 ms/call**
(1,488 calls, 97.86% of GPU time) versus the correct GEMV at **81.468 µs/call**
(1,922 calls), about a 180x kernel-level slowdown. This explains the observed
compiled decode collapse from the fresh eager/graph control of 44.8 tok/s to
1.06 tok/s (offline profiler reproduction: 1.043 tok/s).

## Implemented fix and completed verification

Commit `643fc1735d56b705bf26a5e9400ee81e346a379a` registers
`triton_w4a16_gemm` through the existing `direct_register_custom_op` helper.
The custom-op boundary keeps Dynamo from tracing and freezing the Python
shape-dependent dispatch; the dispatch now executes for each real call.

Source and regression test files:

- `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`
- `tests/kernels/quantization/test_triton_w4a16.py`

The custom-op identity test and gfx900 guard-dropping regression test pass with
the fix and were confirmed to fail against the pre-fix source. The following
focused suites passed with the fix:

- `test_triton_w4a16`
- `test_w4a16_kernel_selection`
- `test_mi100_w4a16`
- `test_mi100_w4a16_correctness`
- `test_rocm_compressed_tensors_w4a16`
- `test_marlin_utils_compile_guard`

Offline verification recovered the exact compile/decode repro from 1.043 tok/s
to 14.208 tok/s. Its profiler showed GEMV for all 1,426 real M<=8 decode calls
and the generic kernel only for the 62 M>8 prefill calls. Cold-cache warmup was
803.6 seconds, matching the prior roughly 14-minute unrelated GDN/FLA Triton
autotune. Eager behavior is unchanged because the registered op directly calls
the same implementation; non-gfx900 platforms continue through their existing
dispatch paths under the same CUDA platform dispatch registration.

## Critical pending production verification

Initial `vllm serve` tests accidentally imported the **unfixed** acceptance
clone. A console script does not put its shell CWD on `sys.path`, whereas the
offline scripts explicitly injected the scratch-worktree path. Therefore the
roughly 1 tok/s serve readings neither invalidate the source fix nor validate
that a real server consumes it.

The real `c4130-2` is blocked by an SSH/NetBird outage. A network scan found
`uno-poweredge-c4130` at `192.168.1.134` and its iDRAC at `192.168.1.136`, but
positively identified that host as a different NVIDIA machine. **Never mutate
or use that machine for this campaign.** Processes 284134 and 279717 may still
be unattended on the real host; check and clean them after reconnecting.

Resume in this exact order:

1. Reconnect to the real host. Prove identity from
   `/home/larkinwc/src/vllm-gfx900-acceptance-3973e0ec9`,
   `/home/larkinwc/venvs/triton`, `/dev/kfd`, and the `gfx900-runs` tree.
2. Check/clean PIDs 284134 and 279717 and ensure the GPUs are clean.
3. Sync fix commit `643fc1735d56b705bf26a5e9400ee81e346a379a` with
   `git format-patch`, `scp`, and `git am`.
4. In the live `vllm serve` worker, prove `vllm.__file__` and the
   `triton_w4a16` module path both resolve to the patched acceptance clone.
5. Run a real **unprofiled** `vllm serve` plus `curl` 128-token test. Compare
   against the broken 1.06 tok/s and eager baseline 44.8 tok/s. Only after that
   succeeds may the root-cause-only docs be updated to claim the defect fixed.

Current `GFX900_RECOMMENDED.md` and `PERF_GFX900.md` intentionally document the
root cause only. They **must not** claim a production fix before the live serve
validation above.

Raw evidence on the real host:

- `/home/larkinwc/gfx900-runs/awq-compile-profile-20260723/`
- `/home/larkinwc/gfx900-runs/awq-compile-hang-20260723/`
- `/home/larkinwc/gfx900-runs/awq-compile-fix-verify-20260723/`

## Unrelated work preserved separately

The Marlin guard work is not part of this branch and remains untouched:

- branch: `upstream-marlin-atomic-add-dynamo-guard` at
  `46f01a50acd6862806ed67b88176c96c2b161142`
- stash: `stash@{0}` / `407ffb3dd14a3ce4696be1427f36f728d8bbec40`
- stash subject: `preserve gfx900-upstream-check WIP (is_dynamo_compiling guard
    - test + runbook) before restoring branch to gfx900-support`
- preserved files: `vllm/model_executor/layers/quantization/utils/marlin_utils.py`,
  `tests/kernels/quantization/test_marlin_utils_compile_guard.py`, and
  `TURBOQUANT_QUALITY_RECONFIRM_RUNBOOK.md`

Do not pop, drop, modify, or include that stash or its runbook in this branch.
