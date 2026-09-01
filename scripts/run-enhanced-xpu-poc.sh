#!/usr/bin/env bash
#
# run-enhanced-xpu-poc.sh — enhanced local voice pipeline XPU hard gate.
#
# This command is an explicit installation-time operation. It downloads the
# required ModelScope snapshots when absent, then switches every hub client to
# offline mode before it loads any model. A report is written only after every
# gate succeeds; it contains device/version/memory evidence but never audio,
# transcript, profile or result data.

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
MODELS_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/models"
MODEL_REVISION="master"

NANO_MODEL_ID="FunAudioLLM/Fun-ASR-Nano-2512"
SENSEVOICE_MODEL_ID="iic/SenseVoiceSmall"
VAD_MODEL_ID="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
CAMPLUS_MODEL_ID="iic/speech_campplus_sv_zh-cn_16k-common"
QWEN_MODEL_ID="Qwen/Qwen3.5-0.8B"

NANO_DIR="${MODELS_ROOT}/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/${MODEL_REVISION}"
SENSEVOICE_DIR="${MODELS_ROOT}/models/iic--SenseVoiceSmall/snapshots/${MODEL_REVISION}"
VAD_DIR="${MODELS_ROOT}/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/${MODEL_REVISION}"
CAMPLUS_DIR="${MODELS_ROOT}/models/iic--speech_campplus_sv_zh-cn_16k-common/snapshots/${MODEL_REVISION}"
QWEN_DIR="${MODELS_ROOT}/models/Qwen--Qwen3.5-0.8B/snapshots/${MODEL_REVISION}"
SNAPSHOT_READY_TIMEOUT_SECONDS=15

log() { printf '[run-enhanced-xpu-poc] %s\n' "$*"; }
die() { printf '[run-enhanced-xpu-poc] ERROR: %s\n' "$*" >&2; exit 1; }

snapshot_has_metadata() {
    local model_dir="$1"
    [[ -d "${model_dir}" ]] && {
        [[ -f "${model_dir}/config.json" \
            || -f "${model_dir}/config.yaml" \
            || -f "${model_dir}/configuration.json" ]]
    }
}

wait_for_snapshot_metadata() {
    local model_dir="$1"
    local deadline=$((SECONDS + SNAPSHOT_READY_TIMEOUT_SECONDS))
    until snapshot_has_metadata "${model_dir}"; do
        if (( SECONDS >= deadline )); then
            return 1
        fi
        sleep 1
    done
}

if [[ ! -x "${PYTHON}" ]]; then
    die "XPU virtual environment is missing; run scripts/create-xpu-env.sh first"
fi
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
    die "XDG_RUNTIME_DIR is unset or unavailable"
fi

RUNTIME_DIR="${XDG_RUNTIME_DIR}/fun-voice-ryan"
REPORT="${RUNTIME_DIR}/enhanced-poc-report.json"
mkdir -p -m 700 "${RUNTIME_DIR}"
chmod 700 "${RUNTIME_DIR}"
rm -f "${REPORT}"

log "resolving explicitly requested local model snapshots"
MODELSCOPE_CACHE="${MODELS_ROOT}" "${PYTHON}" - \
    "${NANO_MODEL_ID}" "${SENSEVOICE_MODEL_ID}" "${VAD_MODEL_ID}" "${CAMPLUS_MODEL_ID}" "${QWEN_MODEL_ID}" \
    "${MODEL_REVISION}" <<'PY'
from __future__ import annotations

import sys

from modelscope.hub.snapshot_download import snapshot_download

*model_ids, revision = sys.argv[1:]
for model_id in model_ids:
    snapshot_download(model_id, revision=revision)
PY

for model_dir in "${NANO_DIR}" "${SENSEVOICE_DIR}" "${VAD_DIR}" "${CAMPLUS_DIR}" "${QWEN_DIR}"; do
    wait_for_snapshot_metadata "${model_dir}" \
        || die "model snapshot is incomplete"
done

log "running enhanced XPU gates"
MODELSCOPE_CACHE="${MODELS_ROOT}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MODELSCOPE_OFFLINE=1 \
PYTHONPATH="${ROOT_DIR}/src" \
"${PYTHON}" -m fun_voice.enhanced_poc \
    --report "${REPORT}" \
    --nano-dir "${NANO_DIR}" \
    --sensevoice-dir "${SENSEVOICE_DIR}" \
    --vad-dir "${VAD_DIR}" \
    --camplus-dir "${CAMPLUS_DIR}" \
    --qwen-dir "${QWEN_DIR}" \
    --revision "${MODEL_REVISION}"

chmod 600 "${REPORT}"
log "report written after all gates passed"
PY
