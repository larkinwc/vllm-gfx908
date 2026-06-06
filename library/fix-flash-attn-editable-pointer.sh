#!/usr/bin/env bash
# Repoint the /opt/vllm-env editable install of `flash_attn` (the ROCm CK
# flash-attention fork used by the ROCM_CK_FA attention backend) at the
# persistent source tree.
#
# Why this is needed
# ------------------
# `flash_attn` 2.8.4 is installed into /opt/vllm-env as an *editable* package.
# Its setuptools finder originally mapped the package at
#   /tmp/rocm-flash-attention-test/...
# That /tmp build dir is wiped on reboot, so `import flash_attn` then fails
# with `ModuleNotFoundError`, and any run with `--attention-backend ROCM_CK_FA`
# crashes during attention `forward()` (rocm_ck_fa.py `_resolve_flash_attn`).
#
# The real CK flash-attention tree (with the built
# `flash_attn_2_cuda.cpython-312-*.so`) persists at:
#   /home/aimeme/Desktop/rocm-flash-attention
#
# This script rewrites the editable finder's MAPPING to point there. It is
# idempotent and only touches the /opt/vllm-env site-packages finder file.
set -euo pipefail

SITE="/opt/vllm-env/lib/python3.12/site-packages"
FINDER="${SITE}/__editable___flash_attn_2_8_4_finder.py"
SRC="/home/aimeme/Desktop/rocm-flash-attention"

if [[ ! -f "${FINDER}" ]]; then
  echo "ERROR: editable finder not found: ${FINDER}" >&2
  echo "Is flash_attn installed editable into /opt/vllm-env?" >&2
  exit 1
fi

if [[ ! -f "${SRC}/flash_attn/__init__.py" ]]; then
  echo "ERROR: flash_attn source tree missing at ${SRC}" >&2
  exit 1
fi

# Repoint any stale MAPPING target (e.g. the wiped /tmp dir) to ${SRC}.
python3 - "$FINDER" "$SRC" <<'PY'
import re
import sys

finder, src = sys.argv[1], sys.argv[2]
text = open(finder).read()
new_mapping = (
    "MAPPING: dict[str, str] = {"
    f"'flash_attn': '{src}/flash_attn', "
    f"'flash_attn_2_cuda': '{src}/flash_attn_2_cuda', "
    f"'hopper': '{src}/hopper'"
    "}"
)
text2 = re.sub(r"^MAPPING: dict\[str, str\] = \{.*\}$", new_mapping, text, flags=re.M)
if text2 == text:
    print("MAPPING already points at the expected tree (no change).")
else:
    open(finder, "w").write(text2)
    print(f"Repointed flash_attn editable MAPPING -> {src}")
PY

echo "Verifying import..."
HSA_OVERRIDE_GFX_VERSION=9.0.8 /opt/vllm-env/bin/python - <<'PY'
import flash_attn
from flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: F401
import flash_attn_2_cuda  # noqa: F401
print(f"OK: flash_attn {flash_attn.__version__} from {flash_attn.__file__}")
PY
