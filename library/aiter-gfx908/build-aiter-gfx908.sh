#!/usr/bin/env bash
# Build AMD AITER cleanly for MI100 (gfx908 / CDNA1).
#
# Why this is needed
# ------------------
# AITER is tuned/validated for CDNA2/CDNA3 (MI200/MI300, gfx90a/gfx942). On
# MI100 (gfx908 / CDNA1) its runtime-JIT path is fragile:
#
#   1. The installed `aiter_meta/3rdparty/` ships only `ck_helper`, NOT the full
#      Composable Kernel (CK) tree.  AITER's default
#        CK_DIR = {AITER_META_DIR}/3rdparty/composable_kernel
#      therefore does not exist, so the rmsnorm CK blob-gen fails with
#        fatal error: 'rmsnorm2d_fwd.hpp' file not found
#      (vllm-gfx908 issue #74, failure-mode 1).
#
#   2. A few kernels hand-write CDNA2+ inline asm (`v_pk_mul_f32`,
#      `v_cvt_pk_fp8_f32`/`v_cvt_pk_bf8_f32`) that does NOT exist on gfx908,
#      so `module_rmsnorm_quant` / `module_activation` fail to compile with
#        error: instruction not supported on this GPU
#      (issue #74, failure-mode 2).
#
# What this script does (idempotent)
# ----------------------------------
#   A. Vendors the exact CK revision AITER pins (composable_kernel @ CK_PIN)
#      into {AITER_META}/3rdparty/composable_kernel.
#   B. Applies the gfx908 ISA fallback patches under ./patches/ so the
#      CDNA2+-only kernels compile on CDNA1 (functionally-equivalent scalar
#      paths; FP8 packs degrade gracefully — MI100 has no native FP8).
#   C. AOT-builds every buildable AITER module to a prebuilt .so under
#      aiter/jit/, so there is NO runtime JIT at serve time.
#
# Re-running is safe: the CK fetch, patches, and per-module builds all skip
# work that is already done (unless FORCE_REBUILD=1).
#
# Companion to library/fix-flash-attn-editable-pointer.sh.
#
# Usage:
#   library/aiter-gfx908/build-aiter-gfx908.sh            # vendor CK + patch + build all
#   FORCE_REBUILD=1 library/aiter-gfx908/build-aiter-gfx908.sh
#   MODULES="module_rmsnorm module_rmsnorm_quant" library/aiter-gfx908/build-aiter-gfx908.sh
set -euo pipefail

# --- config ----------------------------------------------------------------
PYBIN="${PYBIN:-/opt/vllm-env/bin/python}"
CK_PIN="${CK_PIN:-b0c13f312443332c7c13a8cd26b3662582c8d3d4}"
CK_URL="${CK_URL:-https://github.com/ROCm/composable_kernel.git}"
GFX_ARCH="${GFX_ARCH:-gfx908}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/patches"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# gfx908 build environment (issue #75, task 3)
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx908}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.0.8}"
export GPU_ARCHS="${GPU_ARCHS:-gfx908}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-7.12}"

echo "==> AITER gfx908 build"
echo "    PYBIN=${PYBIN}"
echo "    CK_PIN=${CK_PIN}"
echo "    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}  GPU_ARCHS=${GPU_ARCHS}  ROCM_PATH=${ROCM_PATH}"

# --- resolve aiter_meta/csrc paths -----------------------------------------
read -r AITER_META AITER_JIT < <("${PYBIN}" - <<'PY'
import os, aiter
meta = os.path.join(os.path.dirname(os.path.dirname(aiter.__file__)), "aiter_meta")
# aiter_meta is a sibling top-level package
import importlib.util
spec = importlib.util.find_spec("aiter_meta")
if spec and spec.submodule_search_locations:
    meta = list(spec.submodule_search_locations)[0]
jit = os.path.join(os.path.dirname(aiter.__file__), "jit")
print(meta, jit)
PY
)
CSRC="${AITER_META}/csrc"
CK_DST="${AITER_META}/3rdparty/composable_kernel"
echo "    AITER_META=${AITER_META}"
echo "    CK_DST=${CK_DST}"
export CK_DIR="${CK_DST}"

# --- A. vendor pinned CK ----------------------------------------------------
need_ck=1
if [[ -f "${CK_DST}/example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp" ]]; then
  have_pin="$(git -C "${CK_DST}" rev-parse HEAD 2>/dev/null || echo none)"
  if [[ "${have_pin}" == "${CK_PIN}" ]]; then
    echo "==> CK already vendored @ ${CK_PIN} (skip)"
    need_ck=0
  else
    echo "==> CK present but at ${have_pin}, want ${CK_PIN}"
  fi
fi
if [[ "${need_ck}" == "1" ]]; then
  echo "==> Vendoring CK @ ${CK_PIN} (shallow) into ${CK_DST}"
  mkdir -p "${CK_DST}"
  if [[ ! -d "${CK_DST}/.git" ]]; then
    git -C "${CK_DST}" init -q
    git -C "${CK_DST}" remote add origin "${CK_URL}" 2>/dev/null || true
  fi
  git -C "${CK_DST}" fetch --depth 1 origin "${CK_PIN}"
  git -C "${CK_DST}" checkout -q FETCH_HEAD
  test -f "${CK_DST}/example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp" \
    || { echo "ERROR: CK checkout missing rmsnorm2d_fwd.hpp" >&2; exit 1; }
fi

# --- B. apply gfx908 ISA fallback patches ----------------------------------
# apply_patch <patch> <target_file> <root_dir> <marker>
# Idempotent: skips if <marker> already present in <target_file>.
apply_patch() {
  local patch="$1" target="$2" root="$3" marker="$4"
  if grep -q "${marker}" "${target}" 2>/dev/null; then
    echo "    patched already: $(basename "${target}")"
    return 0
  fi
  ( cd "${root}" && patch -p1 --forward < "${patch}" )
  echo "    applied: $(basename "${patch}")"
}
AITER_PKG_DIR="$(dirname "${AITER_JIT}")"           # .../site-packages/aiter
AITER_ROOT="$(dirname "${AITER_PKG_DIR}")"          # .../site-packages
echo "==> Applying gfx908 ISA fallback patches (aiter_meta/csrc)"
apply_patch "${PATCH_DIR}/vec_convert.h.gfx908.patch"            "${CSRC}/include/ck_tile/vec_convert.h"      "${AITER_META}" "__gfx908__"
apply_patch "${PATCH_DIR}/rmsnorm_quant_kernels.cu.gfx908.patch" "${CSRC}/kernels/rmsnorm_quant_kernels.cu"   "${AITER_META}" "__gfx908__"
apply_patch "${PATCH_DIR}/activation_kernels.cu.gfx908.patch"    "${CSRC}/kernels/activation_kernels.cu"      "${AITER_META}" "__gfx908__"
echo "==> Applying gfx908 arch-allowlist patch (aiter/jit/core.py)"
apply_patch "${PATCH_DIR}/core.py.gfx908-allowlist.patch"        "${AITER_JIT}/core.py"                       "${AITER_ROOT}" '"gfx908",  # MI100'

# --- C. AOT-build all buildable modules ------------------------------------
# Module set: every torch-extension module in optCompilerConfig.json, minus
# *_tune autotune helpers and gfx950/MI350-only asm modules.  Pass MODULES=...
# to override.
# Per-module wall-clock cap (s).  A few FP8 a8w8 CK GEMM instances (e.g.
# a8w8_rowwise_256x224x256x128 intrawave_v3) blow up the gfx908 backend and
# compile for 30+ min, so we cap each module and mark over-runs TIMEOUT.
PER_MODULE_TIMEOUT="${PER_MODULE_TIMEOUT:-600}"
# FP8 a8w8 GEMM family: pathological/non-viable on gfx908 (MI100 has no native
# FP8; vLLM uses the int8 gemm_a8w8_CK path or Triton).  Skipped by default;
# opt in with INCLUDE_FP8_GEMM=1.
INCLUDE_FP8_GEMM="${INCLUDE_FP8_GEMM:-0}"
# CK MHA/FMHA family: huge CK kernels that time out on gfx908; MI100 serves
# attention via the CK-FA / Triton backends, not AITER MHA.  Skipped by
# default; opt in with INCLUDE_MHA=1.
INCLUDE_MHA="${INCLUDE_MHA:-0}"

echo "==> AOT-building AITER modules for ${GFX_ARCH} (no runtime JIT at serve)"
echo "    PER_MODULE_TIMEOUT=${PER_MODULE_TIMEOUT}s  INCLUDE_FP8_GEMM=${INCLUDE_FP8_GEMM}  INCLUDE_MHA=${INCLUDE_MHA}"
MODULES="${MODULES:-}" FORCE_REBUILD="${FORCE_REBUILD}" \
PER_MODULE_TIMEOUT="${PER_MODULE_TIMEOUT}" INCLUDE_FP8_GEMM="${INCLUDE_FP8_GEMM}" \
INCLUDE_MHA="${INCLUDE_MHA}" \
PYBIN="${PYBIN}" "${PYBIN}" - <<'PY'
import json, os, signal, subprocess, sys, tempfile, time
# Run from a scratch dir: aiter/hipcc capability probes drop stray "-.o"
# objects into CWD, and we must not pollute the repo root.
os.chdir(tempfile.mkdtemp(prefix="aiter_gfx908_build_"))
import aiter.jit.core as core

this_dir = os.path.dirname(core.__file__)
cfg = json.load(open(os.path.join(this_dir, "optCompilerConfig.json")))

override = os.environ.get("MODULES", "").split()
force = os.environ.get("FORCE_REBUILD", "0") == "1"
per_to = int(os.environ.get("PER_MODULE_TIMEOUT", "600"))
include_fp8_gemm = os.environ.get("INCLUDE_FP8_GEMM", "0") == "1"
pybin = os.environ.get("PYBIN", sys.executable)

# gfx950 / MI350-only asm modules — skip on gfx908.
SKIP_SUBSTR = ("mi350", "gfx950", "fp4", "a4w4")
# FP8 a8w8 CK GEMM family — pathological compile on gfx908 (see header).
FP8_GEMM = {
    "module_gemm_a8w8",
    "module_gemm_a8w8_blockscale",
    "module_gemm_a8w8_blockscale_cktile",
    "module_gemm_a8w8_blockscale_bpreshuffle",
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile",
    "module_gemm_a8w8_bpreshuffle",
    "module_gemm_a8w8_bpreshuffle_cktile",
    "module_deepgemm",
}
# CK flash-attention modules (mha/fmha) — huge CK kernels that time out on
# gfx908, and unnecessary: this fork serves attention via the ROCM_CK_FA /
# Triton backends on MI100, not AITER MHA.  Opt in with INCLUDE_MHA=1.
MHA = {
    "module_mha_fwd",
    "module_mha_varlen_fwd",
    "module_mha_batch_prefill",
    "module_mha_bwd",
    "module_mha_varlen_bwd",
    "module_fmha_v3_fwd",
    "module_fmha_v3_bwd",
    "module_fmha_v3_varlen_fwd",
    "module_fmha_v3_varlen_bwd",
    "libmha_fwd",
    "libmha_bwd",
}

def is_target(name):
    if name.endswith("_tune"):
        return False
    low = name.lower()
    if any(s in low for s in SKIP_SUBSTR):
        return False
    if not include_fp8_gemm and name in FP8_GEMM:
        return False
    if os.environ.get("INCLUDE_MHA", "0") != "1" and name in MHA:
        return False
    return True

if override:
    modules = override
else:
    modules = [m for m in cfg if is_target(m)]

# One-module build worker (run as a fresh subprocess so we can hard-kill its
# whole ninja process group on timeout).
WORKER = r'''
import os, sys, aiter.jit.core as core
md = sys.argv[1]
d = core.get_args_of_build(md)
if not d.get("srcs"):
    sys.exit(3)  # no-srcs
core.build_module(
    md, d["srcs"], d["flags_extra_cc"], d["flags_extra_hip"],
    d["blob_gen_cmd"], d["extra_include"], d["extra_ldflags"],
    d["verbose"], d["is_python_module"], d["is_standalone"],
    d["torch_exclude"], d.get("hipify", False),
)
'''

jit_dir = this_dir
results = []
for md in modules:
    so = os.path.join(jit_dir, f"{md}.so")
    if os.path.exists(so) and not force:
        results.append((md, "prebuilt", 0.0))
        print(f"  [skip ] {md} (.so exists)", flush=True)
        continue
    t = time.time()
    # Clear any stale baton lock for this module before building.
    for lk in (os.path.join(jit_dir, "build", f"lock_{md}"),):
        try:
            os.remove(lk)
        except OSError:
            pass
    proc = subprocess.Popen(
        [pybin, "-c", WORKER, md],
        start_new_session=True,  # own process group => clean kill on timeout
    )
    try:
        rc = proc.wait(timeout=per_to)
        dt = time.time() - t
        if rc == 0 and os.path.exists(so):
            results.append((md, "BUILT", dt))
            print(f"  [build] {md}  ({dt:.1f}s)", flush=True)
        elif rc == 3:
            results.append((md, "no-srcs", dt))
            print(f"  [n/a  ] {md} (no srcs in config)", flush=True)
        else:
            results.append((md, f"FAIL: rc={rc}", dt))
            print(f"  [FAIL ] {md}  ({dt:.1f}s): rc={rc}", flush=True)
    except subprocess.TimeoutExpired:
        # Kill the whole process group (ninja + all hipcc/clang children).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        proc.wait()
        dt = time.time() - t
        # remove stale lock left by the killed worker
        try:
            os.remove(os.path.join(jit_dir, "build", f"lock_{md}"))
        except OSError:
            pass
        results.append((md, "TIMEOUT", dt))
        print(f"  [TMOUT] {md}  ({dt:.1f}s > {per_to}s cap)", flush=True)

print("\n==> Build summary")
ok = [r for r in results if r[1] in ("BUILT", "prebuilt")]
bad = [r for r in results if r[1].startswith("FAIL")]
to = [r for r in results if r[1] == "TIMEOUT"]
na = [r for r in results if r[1] in ("no-srcs", "n/a")]
for md, st, dt in results:
    print(f"    {md:48s} {st}")
print(
    f"\n    {len(ok)} ok, {len(bad)} failed, {len(to)} timeout, "
    f"{len(na)} n/a, of {len(results)} attempted"
)
# Persist machine-readable status next to this script.
out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "")
status_path = os.environ.get("AITER_BUILD_STATUS")
if status_path:
    with open(status_path, "w") as f:
        json.dump([{"module": m, "status": s, "secs": round(d, 1)} for m, s, d in results], f, indent=2)
    print(f"    wrote status -> {status_path}")
# Non-zero exit only on hard build failures (TIMEOUT/n/a are expected on
# gfx908 for the FP8 GEMM family and asm-only modules).
sys.exit(1 if bad else 0)
PY

echo "==> Done."
