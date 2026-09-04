#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FCITX_OUTPUT="${ROOT_DIR}/build/fcitx/fcitx5-fun-voice.so"
OVERLAY_OUTPUT="${ROOT_DIR}/build/dtk-overlay/fun-voice-overlay"

valid_file() {
    local path="$1"
    [[ -f "${path}" && ! -L "${path}" && "$(stat -c '%u' -- "${path}")" == "$(id -u)" ]]
}

if valid_file "${FCITX_OUTPUT}" && valid_file "${OVERLAY_OUTPUT}" \
    && [[ -x "${OVERLAY_OUTPUT}" ]]; then
    exit 0
fi

if ! cmake -S "${ROOT_DIR}/native/fcitx5-fun-voice" -B "${ROOT_DIR}/build/fcitx" \
    || ! cmake --build "${ROOT_DIR}/build/fcitx"; then
    printf '%s\n' 'native_prerequisite: Fcitx configure/build failed; install the documented Fcitx development packages and retry' >&2
    exit 1
fi
valid_file "${FCITX_OUTPUT}" || {
    printf '%s\n' 'native_prerequisite: Fcitx build did not produce an owned shared library' >&2
    exit 1
}

if ! cmake -S "${ROOT_DIR}/native/dtk-overlay" -B "${ROOT_DIR}/build/dtk-overlay" \
    || ! cmake --build "${ROOT_DIR}/build/dtk-overlay"; then
    printf '%s\n' 'native_prerequisite: DTK configure/build failed; install the documented DTK development packages and retry' >&2
    exit 1
fi
valid_file "${OVERLAY_OUTPUT}" && [[ -x "${OVERLAY_OUTPUT}" ]] || {
    printf '%s\n' 'native_prerequisite: DTK build did not produce an owned executable overlay' >&2
    exit 1
}
