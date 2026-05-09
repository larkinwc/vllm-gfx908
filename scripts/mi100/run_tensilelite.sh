#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/run_tensilelite.sh -- M2 TensileLite tuning driver
# =============================================================================
#
# Runs the TensileLite tuning pipeline (`Tensile <yaml> <out>`) against
# every per-shape YAML emitted by gen_tensilelite_configs.py.
#
# Each invocation:
#   - Generates candidate kernels for the configured (M, N, K).
#   - Compiles them.
#   - Benchmarks them on real gfx908 hardware via tensilelite-client.
#   - Writes the winning solution as `3_LibraryLogic/arcturus_*.yaml`
#     plus a matching code-object .co.
#
# We stage the resulting per-shape logic YAMLs into
# $HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/logic/gfx908/
# so a follow-on hipBLASLt smoke can verify runtime selection.
#
# Pre-requisites (validated by m2-hipblaslt-build):
#   - libhipblaslt.so.1.2 at /root/hipblaslt-src/build/release/library/
#   - tensilelite-client at /root/hipblaslt-src/tensilelite/build_tmp/...
#   - Tensile python pipeline importable per PYTHONPATH below
#   - /shared -> /root/shared symlink present (for origami)
#
# Per shape rough wall-time on this host: 2–10 min.
# =============================================================================
set -uo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
TENSILE_DIR=/root/hipblaslt-src/tensilelite
TENSILE_BIN=$TENSILE_DIR/Tensile/bin/Tensile
PY=/opt/vllm-env/bin/python3

CONFIGS_DIR=${CONFIGS_DIR:-/root/bench-int8-w4a16/tensilelite/configs}
RUNS_DIR=${RUNS_DIR:-/root/bench-int8-w4a16/tensilelite/runs}
LOGIC_DIR=${LOGIC_DIR:-/root/bench-int8-w4a16/tensilelite/logic/gfx908}
LOGS_DIR=${LOGS_DIR:-/root/bench-int8-w4a16/tensilelite/logs}
SUMMARY_JSON=${SUMMARY_JSON:-/root/bench-int8-w4a16/tensilelite/tuning_summary.json}

mkdir -p "$RUNS_DIR" "$LOGIC_DIR" "$LOGS_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

if [[ ! -x "$TENSILE_DIR/build_tmp/tensilelite/client/tensilelite-client" ]]; then
  echo "FATAL: tensilelite-client missing — run m2-hipblaslt-build first." >&2
  exit 2
fi
if [[ ! -f "$TENSILE_BIN" ]]; then
  echo "FATAL: Tensile binary missing at $TENSILE_BIN" >&2
  exit 2
fi
if [[ ! -L /shared ]]; then
  echo "FATAL: /shared symlink missing; CMake reconfigure will fail." >&2
  exit 2
fi

# Tensile invocation environment per library/m2-build-artifacts.md.
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export ROCM_PATH=/opt/rocm/core-7.12
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export PYTHONPATH=/root/hipblaslt-src/build/release/tensilelite/rocisa/lib:/root/hipblaslt-src/tensilelite

declare -a YAMLS
if [[ "${1:-}" == "" ]]; then
  mapfile -t YAMLS < <(ls -1 "$CONFIGS_DIR"/tune_M*_N*_K*.yaml 2>/dev/null)
else
  YAMLS=("$@")
fi

if [[ ${#YAMLS[@]} -eq 0 ]]; then
  echo "FATAL: no tuning YAMLs at $CONFIGS_DIR (matching tune_M*_N*_K*.yaml)" >&2
  exit 2
fi

log "TensileLite tuning campaign: ${#YAMLS[@]} shape(s)"
log "  configs dir:  $CONFIGS_DIR"
log "  runs dir:     $RUNS_DIR"
log "  logic dir:    $LOGIC_DIR"
log "  logs dir:     $LOGS_DIR"

successes=()
failures=()
logic_paths=()
for cfg in "${YAMLS[@]}"; do
  name=$(basename "$cfg" .yaml)
  out_dir="$RUNS_DIR/$name"
  cfg_log="$LOGS_DIR/${name}.log"
  log ">>> tune $name"
  log "    cfg=$cfg"
  log "    out_dir=$out_dir"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  set +e
  ( cd "$TENSILE_DIR" && "$PY" "$TENSILE_BIN" "$cfg" "$out_dir" ) \
      > "$cfg_log" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    # The canonical per-shape logic YAML lives at
    # $out_dir/3_LibraryLogic/arcturus_<contraction>.yaml
    logic_yaml=$(ls "$out_dir"/3_LibraryLogic/arcturus_*.yaml 2>/dev/null \
        | head -1 || true)
    if [[ -n "$logic_yaml" ]]; then
      cp "$logic_yaml" "$LOGIC_DIR/${name}.yaml"
      logic_paths+=("$LOGIC_DIR/${name}.yaml")
      successes+=("$name")
      log "    OK -> $LOGIC_DIR/${name}.yaml"
    else
      failures+=("$name (no logic yaml)")
      log "    FAIL: tuning ran but no 3_LibraryLogic/*.yaml produced"
    fi
  else
    failures+=("$name (exit=$rc)")
    log "    FAIL: Tensile exit=$rc — see $cfg_log"
    tail -n 20 "$cfg_log" | sed 's/^/        | /'
  fi
done

log "=== Summary ==="
n_succ=${#successes[@]}
n_fail=${#failures[@]}
log "  succeeded: $n_succ / ${#YAMLS[@]}"
if [[ $n_succ -gt 0 ]]; then
  for s in "${successes[@]}"; do log "    + $s"; done
fi
if [[ $n_fail -gt 0 ]]; then
  for f in "${failures[@]}"; do log "    - $f"; done
fi

# Also stage code-objects from each successful run alongside the yaml so
# hipBLASLt can find them at runtime.
if [[ $n_succ -gt 0 ]]; then
  for s in "${successes[@]}"; do
    src=$(ls "$RUNS_DIR/$s"/3_LibraryLogic/*.co 2>/dev/null | head -1 || true)
    if [[ -n "$src" ]]; then
      cp "$src" "$LOGIC_DIR/${s}.co" 2>/dev/null || true
    fi
  done
fi

# Persist a JSON summary for later validators.
"$PY" - "$SUMMARY_JSON" "$LOGIC_DIR" "${successes[@]:+${successes[@]}}" <<'EOF'
import json, os, sys
out_path = sys.argv[1]
logic_dir = sys.argv[2]
successes = sys.argv[3:]
yamls = sorted(os.path.join(logic_dir, f"{s}.yaml") for s in successes)
cos = sorted(os.path.join(logic_dir, f"{s}.co") for s in successes
             if os.path.exists(os.path.join(logic_dir, f"{s}.co")))
manifest = {
    "logic_dir": logic_dir,
    "n_shapes_tuned": len(successes),
    "shape_yamls": yamls,
    "shape_cos": cos,
    "successes": successes,
}
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"summary -> {out_path}")
EOF

if [[ ${#failures[@]} -gt 0 ]]; then
  exit 1
fi
exit 0
