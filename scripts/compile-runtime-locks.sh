#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-$(command -v uv || true)}"
FUNASR_COMMIT="8cd758c0ced576516b05a749194e6a94cdd38f99"
FUNASR_TARBALL_SHA256="f8b2c9b9954c463b5c0e433bd1f2706b5c6c28f16f755f55ec66365960c06da0"
WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

[[ -n "${UV}" && -x "${UV}" ]] || { printf 'uv is unavailable\n' >&2; exit 1; }
ARCHIVE="${WORK_DIR}/funasr.tar.gz"
FUNASR_STAGE="${WORK_DIR}/funasr"
curl -sSL --retry 8 --retry-delay 3 --retry-all-errors \
    -o "${ARCHIVE}" \
    "https://codeload.github.com/modelscope/FunASR/tar.gz/${FUNASR_COMMIT}"
printf '%s  %s\n' "${FUNASR_TARBALL_SHA256}" "${ARCHIVE}" \
    | sha256sum --check --status
mkdir -m 700 "${FUNASR_STAGE}"
tar xzf "${ARCHIVE}" -C "${FUNASR_STAGE}" --strip-components=1
find "${FUNASR_STAGE}" -type l -lname '/*' -delete

for BACKEND in cuda xpu cpu; do
    OUTPUT="${WORK_DIR}/requirements-${BACKEND}.lock"
    "${UV}" pip compile \
        --generate-hashes \
        --quiet \
        --no-header \
        --no-annotate \
        --python-version 3.12 \
        --index-strategy unsafe-best-match \
        --no-emit-package funasr \
        "${ROOT_DIR}/requirements-${BACKEND}.in" \
        "${FUNASR_STAGE}/pyproject.toml" \
        --output-file "${OUTPUT}"
    grep -q -- '--hash=sha256:' "${OUTPUT}"
    case "${BACKEND}" in
        cuda)
            TORCH_SUFFIX='cu130'
            EXTRA_INDEX='https://download.pytorch.org/whl/cu130'
            ;;
        xpu)
            TORCH_SUFFIX='xpu'
            EXTRA_INDEX='https://download.pytorch.org/whl/xpu'
            ;;
        cpu)
            TORCH_SUFFIX='cpu'
            EXTRA_INDEX='https://download.pytorch.org/whl/cpu'
            ;;
    esac
    grep -Fq "torch==2.13.0+${TORCH_SUFFIX}" "${OUTPUT}"
    grep -Fq 'modelscope==1.39.1' "${OUTPUT}"
    grep -Fq 'transformers==5.16.1' "${OUTPUT}"
    if grep -Eq '^(vllm|vllm-xpu-kernels|cuda-python|flashinfer-python)==' "${OUTPUT}"; then
        printf 'unapproved runtime dependency in %s lock\n' "${BACKEND}" >&2
        exit 1
    fi

    SMOKE_VENV="${WORK_DIR}/smoke-${BACKEND}"
    "${UV}" venv "${SMOKE_VENV}" --python 3.12
    "${UV}" pip sync \
        --python "${SMOKE_VENV}/bin/python" \
        --require-hashes \
        --index-url 'https://pypi.tuna.tsinghua.edu.cn/simple' \
        --extra-index-url "${EXTRA_INDEX}" \
        --index-strategy unsafe-best-match \
        "${OUTPUT}"
    "${SMOKE_VENV}/bin/python" - <<'PY'
import torch

assert torch.ones(1, dtype=torch.float32).sum().item() == 1.0
PY
    rm -rf -- "${SMOKE_VENV}"
done

for BACKEND in cuda xpu cpu; do
    mv "${WORK_DIR}/requirements-${BACKEND}.lock" \
        "${ROOT_DIR}/requirements-${BACKEND}.lock"
done
