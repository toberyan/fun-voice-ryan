#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIMES_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/runtimes"
MODELS_BASE="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/models"
UV="${UV:-$(command -v uv || true)}"
FUNASR_COMMIT="8cd758c0ced576516b05a749194e6a94cdd38f99"
FUNASR_TARBALL_SHA256="f8b2c9b9954c463b5c0e433bd1f2706b5c6c28f16f755f55ec66365960c06da0"

BACKEND=""
RUNTIME_DIR=""
MODELS_ROOT=""
ALLOW_PROJECT_VENV=0
FUNASR_DOWNLOAD=""
FUNASR_STAGE=""

die() { printf '[create-runtime-env] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[create-runtime-env] %s\n' "$*"; }
cleanup() {
    if [[ -n "${FUNASR_DOWNLOAD}" && -f "${FUNASR_DOWNLOAD}" ]]; then
        rm -f -- "${FUNASR_DOWNLOAD}"
    fi
    if [[ -n "${FUNASR_STAGE}" && -d "${FUNASR_STAGE}" ]]; then
        rm -rf -- "${FUNASR_STAGE}"
    fi
}
trap cleanup EXIT

while (($#)); do
    case "$1" in
        --backend|--runtime-dir|--models-root)
            (($# >= 2)) || die "missing option value"
            case "$1" in
                --backend) BACKEND="$2" ;;
                --runtime-dir) RUNTIME_DIR="$2" ;;
                --models-root) MODELS_ROOT="$2" ;;
            esac
            shift 2
            ;;
        --allow-project-venv)
            ALLOW_PROJECT_VENV=1
            shift
            ;;
        *) die "unknown option" ;;
    esac
done

[[ -n "${BACKEND}" && -n "${RUNTIME_DIR}" && -n "${MODELS_ROOT}" ]] \
    || die "--backend, --runtime-dir and --models-root are required"
case "${BACKEND}" in
    cuda) EXTRA_INDEX="https://download.pytorch.org/whl/cu130" ;;
    xpu) EXTRA_INDEX="https://download.pytorch.org/whl/xpu" ;;
    cpu) EXTRA_INDEX="https://download.pytorch.org/whl/cpu" ;;
    *) die "unsupported backend" ;;
esac
LOCK_FILE="${ROOT_DIR}/requirements-${BACKEND}.lock"
[[ -n "${UV}" && -x "${UV}" ]] || die "uv is unavailable"
[[ -f "${LOCK_FILE}" ]] || die "runtime lock is unavailable"

RUNTIMES_RESOLVED="$(realpath -m "${RUNTIMES_ROOT}")"
RUNTIME_RESOLVED="$(realpath -m "${RUNTIME_DIR}")"
MODELS_RESOLVED="$(realpath -m "${MODELS_ROOT}")"
PROJECT_VENV_RESOLVED="$(realpath -m "${ROOT_DIR}/.venv")"
if [[ "${ALLOW_PROJECT_VENV}" -eq 0 ]]; then
    case "${RUNTIME_RESOLVED}" in
        "${RUNTIMES_RESOLVED}"/*) ;;
        *) die "runtime directory is outside the application runtimes root" ;;
    esac
    [[ "${RUNTIME_RESOLVED}" != "${PROJECT_VENV_RESOLVED}" ]] \
        || die "repository .venv is not an isolated runtime"
fi
[[ "${MODELS_RESOLVED}" == "$(realpath -m "${MODELS_BASE}")" ]] \
    || die "models root is outside the application data root"
[[ ! -L "${RUNTIME_DIR}" ]] || die "runtime directory must not be a symlink"

mkdir -p "${RUNTIMES_ROOT}" "${RUNTIME_DIR}" "${MODELS_ROOT}"
chmod 700 "${RUNTIMES_ROOT}" "${RUNTIME_DIR}" "${MODELS_ROOT}"

if [[ ! -x "${RUNTIME_DIR}/bin/python" ]]; then
    log "creating ${BACKEND} Python 3.12 runtime"
    "${UV}" venv "${RUNTIME_DIR}" --python 3.12
fi
PYTHON="${RUNTIME_DIR}/bin/python"
log "syncing hash-locked ${BACKEND} dependencies"
"${UV}" pip sync \
    --python "${PYTHON}" \
    --require-hashes \
    --index-url "https://pypi.tuna.tsinghua.edu.cn/simple" \
    --extra-index-url "${EXTRA_INDEX}" \
    --index-strategy unsafe-best-match \
    "${LOCK_FILE}"

FUNASR_TARBALL="${RUNTIME_DIR}/.funasr-src.tar.gz"
FUNASR_SRC="${RUNTIME_DIR}/.funasr-src"
archive_matches_hash() {
    printf '%s  %s\n' "${FUNASR_TARBALL_SHA256}" "$1" \
        | sha256sum --check --status
}
if [[ ! -f "${FUNASR_TARBALL}" ]] || ! archive_matches_hash "${FUNASR_TARBALL}"; then
    FUNASR_DOWNLOAD="$(mktemp "${RUNTIME_DIR}/.funasr.download.XXXXXX")"
    curl -sSL --retry 8 --retry-delay 3 --retry-all-errors \
        -o "${FUNASR_DOWNLOAD}" \
        "https://codeload.github.com/modelscope/FunASR/tar.gz/${FUNASR_COMMIT}"
    archive_matches_hash "${FUNASR_DOWNLOAD}" || die "FunASR archive hash mismatch"
    mv "${FUNASR_DOWNLOAD}" "${FUNASR_TARBALL}"
    FUNASR_DOWNLOAD=""
fi

FUNASR_STAGE="$(mktemp -d "${RUNTIME_DIR}/.funasr-stage.XXXXXX")"
tar xzf "${FUNASR_TARBALL}" -C "${FUNASR_STAGE}" --strip-components=1
find "${FUNASR_STAGE}" -type l -lname '/*' -delete
if [[ -e "${FUNASR_SRC}" ]]; then
    rm -rf -- "${FUNASR_SRC}"
fi
mv "${FUNASR_STAGE}" "${FUNASR_SRC}"
FUNASR_STAGE=""
"${UV}" pip install --python "${PYTHON}" --no-deps "${FUNASR_SRC}"

MODELSCOPE_CACHE="${MODELS_ROOT}" "${PYTHON}" - <<'PY'
import importlib
import importlib.metadata

for name in ("torch", "funasr", "modelscope", "transformers", "Xlib"):
    try:
        importlib.import_module(name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        print(f"{name}={version}")
    except Exception as exc:
        print(f"{name}=import_failed:{type(exc).__name__}")
        raise SystemExit(1) from None
PY
