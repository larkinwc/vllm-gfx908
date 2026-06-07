# M1-F6: /opt/vllm-env Editable Install Repoint

## Problem

M3-F1 finding: `/opt/vllm-env`'s editable install of vllm pointed at the WRONG
sibling worktree (`fuzzy-hornets-see-szfl4`, branch
`mi100/fused-act-quant-m26-m33-m11` @ `79f24b48d`), not the upstream-sync
worktree (`loose-rats-clean-phm2k`, branch `upstream-sync-2026-05-28` @
`fc437f708`). All prior M1-F5 / M2-F1 / M2-F2 measurements were therefore made
against the wrong code:

- Python sources resolved correctly via `vllm.__file__` because the loose-rats
  worktree happens earlier in `sys.path`/CWD lookups, BUT
- The compiled extensions (`vllm._C`, `vllm._rocm_C`, `vllm._moe_C`) loaded
  through the editable `MAPPING` were the fuzzy-hornets `.abi3.so` artifacts.

## Before

```text
MAPPING: {'vllm': '/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4/vllm'}
direct_url.json: file:///home/aimeme/.../fuzzy-hornets-see-szfl4

vllm._C       -> .../fuzzy-hornets-see-szfl4/vllm/_C.abi3.so
vllm._rocm_C  -> .../fuzzy-hornets-see-szfl4/vllm/_rocm_C.abi3.so
(no .abi3.so files under loose-rats-clean-phm2k/vllm/)
```

## Procedure Executed

1. Verified worktree HEAD: `upstream-sync-2026-05-28` @ `fc437f708a563d7be08d6636fa2cac7234966d1b`, working tree clean.
2. `/opt/vllm-env/bin/pip uninstall -y vllm` → `library/uninstall.log`.
3. Verified ROCm torch intact post-uninstall:
   `torch: 2.11.0+rocm7.2 hip: 7.2.26015` (no recovery needed).
4. Installed missing build dependency `setuptools-rust>=1.9.0` (and
   `semantic_version`) into `/opt/vllm-env`. (First build attempt failed in
   `prepare_metadata_for_build_editable` with `ModuleNotFoundError: setuptools_rust`,
   logged in `library/install-fromsrc.log`.)
5. From-source editable install (NO `VLLM_USE_PRECOMPILED` — per M3-F1, the
   x86_64 wheel is not on pypi.amd.com for this version):

   ```bash
   cd /home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k
   export PYTORCH_ROCM_ARCH=gfx908 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2
   /opt/vllm-env/bin/pip install -e . --no-build-isolation 2>&1 | tee library/install-fromsrc.log
   ```

   - Build wall time: 06:47:05 → 06:57:40 CDT 2026 = ~10 m 35 s (gfx908-only
     arch, `MAX_JOBS=8` to stay within 17 GiB free RAM on this 64-core / 62-GiB
     host). Faster than the orchestrator's "multi-hour" expectation because
     the build was scoped to a single arch and parallelism was sized to fit
     memory. EXIT=0.
6. Verified finder MAPPING now references THIS worktree (see "After").
7. Verified `.abi3.so` artifacts exist under THIS worktree (see "After").
8. Re-ran smoke imports — all green (`library/env-repoint-smoke.log`).

## After

```text
MAPPING: {'vllm': '/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k/vllm'}
direct_url.json: file:///home/aimeme/.../loose-rats-clean-phm2k

vllm.__file__:    .../loose-rats-clean-phm2k/vllm/__init__.py
vllm.__version__: 0.21.1rc1.dev583+gfc437f708
vllm._C:          .../loose-rats-clean-phm2k/vllm/_C.abi3.so          (64 972 464 B, 2026-05-28 18:51)
vllm._rocm_C:     .../loose-rats-clean-phm2k/vllm/_rocm_C.abi3.so     (84 608 312 B, 2026-05-28 18:57)
vllm._moe_C:      .../loose-rats-clean-phm2k/vllm/_moe_C.abi3.so      (24 812 432 B, 2026-05-28 18:51)
(also built: vllm/_C_stable_libtorch.abi3.so — 34 392 744 B, 2026-05-28 18:50)

Smoke imports (rocm_ck_fa, triton_attn, registry, RocmPlatform): imports ok
```

## Branch / commit context

- Worktree HEAD: `fc437f708a563d7be08d6636fa2cac7234966d1b` on `upstream-sync-2026-05-28`.
- Merged history vs pre-sync-baseline: 775 commits.

## Implication for prior gates

M1-F5 imports, M2-F1 correctness gates (ppl 9.6551, coding 9/10, needle 5/5),
and M2-F2 CK-FA smoke (73k dispatches) ran against the fuzzy-hornets compiled
extensions. The Python source path was correct, but compiled kernels were
from a different branch. Those gates must be re-run against this re-pointed
install before the sync PR is opened.

## Files

- `library/install-fromsrc.log` — full pip install transcript (both attempts).
- `library/uninstall.log` — pip uninstall transcript.
- `library/env-repoint-smoke.log` — smoke-import transcript.
