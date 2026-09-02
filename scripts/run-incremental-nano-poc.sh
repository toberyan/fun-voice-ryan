#!/usr/bin/env bash
#
# Explicit local acceptance gate for Nano incremental windows.  It never runs
# at login and writes only aggregate metrics under the owner-only runtime dir.

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
RUNTIME_DIR="${XDG_RUNTIME_DIR:?XDG_RUNTIME_DIR is required}/fun-voice-ryan"
REPORT="${RUNTIME_DIR}/incremental-poc-report.json"
MODEL_REVISION="master"
MODELS_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/models"

if [[ ! -x "${PYTHON}" ]]; then
    printf '%s\n' '[run-incremental-nano-poc] ERROR: XPU virtual environment is missing' >&2
    exit 1
fi

mkdir -p -m 700 "${RUNTIME_DIR}"
chmod 700 "${RUNTIME_DIR}"
rm -f "${REPORT}"

MODELSCOPE_CACHE="${MODELS_ROOT}" \
FUN_VOICE_MODELS_ROOT="${MODELS_ROOT}" \
HF_HUB_OFFLINE=1 \
MODELSCOPE_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="${ROOT_DIR}/src" "${PYTHON}" -m fun_voice.incremental_poc \
    --report "${REPORT}" \
    --corpus "${1:?pass a user-owned local corpus directory}" \
    --revision "${MODEL_REVISION}"
