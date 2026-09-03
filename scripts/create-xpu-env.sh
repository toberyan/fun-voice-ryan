#!/usr/bin/env bash
# Developer/POC compatibility wrapper for the repository-local XPU runtime.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FUN_VOICE_VENV_DIR:-${ROOT_DIR}/.venv}"
MODELS_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/models"

"${ROOT_DIR}/scripts/create-runtime-env.sh" \
    --backend xpu \
    --runtime-dir "${VENV_DIR}" \
    --models-root "${MODELS_ROOT}" \
    --allow-project-venv

"${VENV_DIR}/bin/python" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch.xpu.is_available={torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"torch.xpu.device_count={torch.xpu.device_count()}")
    print(f"torch.xpu.device_name={torch.xpu.get_device_name(0)}")
    try:
        properties = torch.xpu.get_device_properties(0)
        print(f"torch.xpu.total_memory={properties.total_memory}")
    except Exception as exc:
        print(f"torch.xpu.total_memory=unavailable:{type(exc).__name__}")
PY
