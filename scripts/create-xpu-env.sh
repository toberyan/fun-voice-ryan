#!/usr/bin/env bash
#
# create-xpu-env.sh — 创建项目 XPU 推理环境(幂等)。
#
# - 用 uv 在项目内创建 .venv(Python 3.12)。测试时可通过 FUN_VOICE_VENV_DIR
#   指向独立临时环境，不会改动现有已验证 .venv。
# - 从带 SHA-256 的 requirements-xpu.lock 同步原生 PyTorch XPU 和全部第三方依赖
#   (不混用清华镜像,显式 --index-strategy unsafe-best-match 解析 +xpu 变体)。
# - FunASR 固定到提交 8cd758c0ced576516b05a749194e6a94cdd38f99(经 codeload
#   tarball 下载并校验 SHA-256，清理上游绝对符号链接后无依赖安装)。
# - 打印安装版本与设备信息;不打印任何音频路径或转写文本。
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FUN_VOICE_VENV_DIR:-${ROOT_DIR}/.venv}"
UV="${UV:-/home/toberyan/.local/bin/uv}"
LOCK_FILE="${ROOT_DIR}/requirements-xpu.lock"

FUNASR_COMMIT="8cd758c0ced576516b05a749194e6a94cdd38f99"
FUNASR_TARBALL="${ROOT_DIR}/.funasr-src.tar.gz"
FUNASR_TARBALL_SHA256="f8b2c9b9954c463b5c0e433bd1f2706b5c6c28f16f755f55ec66365960c06da0"

PYPI_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
XPU_INDEX="https://download.pytorch.org/whl/xpu"

log() { printf '[create-xpu-env] %s\n' "$*"; }
die() { printf '[create-xpu-env] ERROR: %s\n' "$*" >&2; exit 1; }

archive_matches_hash() {
    local archive="$1"
    printf '%s  %s\n' "${FUNASR_TARBALL_SHA256}" "${archive}" \
        | sha256sum --check --status
}

# --- 1. venv ---------------------------------------------------------------
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    log ".venv 已存在,跳过创建"
else
    log "创建 .venv (Python 3.12)..."
    "${UV}" venv "${VENV_DIR}" --python 3.12
fi

PYTHON="${VENV_DIR}/bin/python"

sync_xpu_lock() {
    "${UV}" pip sync \
        --python "${PYTHON}" \
        --require-hashes \
        --index-url "${PYPI_INDEX}" \
        --extra-index-url "${XPU_INDEX}" \
        --index-strategy unsafe-best-match \
        "${LOCK_FILE}"
}

# --- 2. Hash-locked third-party XPU runtime --------------------------------
[[ -f "${LOCK_FILE}" ]] || die "requirements lock missing: ${LOCK_FILE}"
log "从带哈希的 requirements-xpu.lock 同步 XPU 运行时..."
sync_xpu_lock

# --- 3. FunASR @ pinned commit + modelscope ---------------------------------
FUNASR_SRC="${ROOT_DIR}/.funasr-src"
if ! [[ -f "${FUNASR_TARBALL}" ]] || ! archive_matches_hash "${FUNASR_TARBALL}"; then
    log "下载 FunASR @ ${FUNASR_COMMIT} ..."
    FUNASR_DOWNLOAD="$(mktemp "${ROOT_DIR}/.funasr-src.tar.gz.download.XXXXXX")"
    curl -sSL --retry 8 --retry-delay 3 --retry-all-errors \
        -o "${FUNASR_DOWNLOAD}" \
        "https://codeload.github.com/modelscope/FunASR/tar.gz/${FUNASR_COMMIT}" \
        || die "could not download FunASR @ ${FUNASR_COMMIT}"
    archive_matches_hash "${FUNASR_DOWNLOAD}" \
        || die "downloaded FunASR tarball SHA-256 mismatch"
    mv "${FUNASR_DOWNLOAD}" "${FUNASR_TARBALL}"
fi
archive_matches_hash "${FUNASR_TARBALL}" \
    || die "FunASR tarball SHA-256 mismatch: ${FUNASR_TARBALL}"
log "解包 FunASR 源码并移除指向绝对路径的损坏符号链接..."
FUNASR_STAGE="$(mktemp -d "${ROOT_DIR}/.funasr-src.stage.XXXXXX")"
tar xzf "${FUNASR_TARBALL}" -C "${FUNASR_STAGE}" --strip-components=1
find "${FUNASR_STAGE}" -type l -lname '/*' -delete
FUNASR_PREVIOUS="${ROOT_DIR}/.funasr-src.previous"
rm -rf "${FUNASR_PREVIOUS}"
if [[ -e "${FUNASR_SRC}" || -L "${FUNASR_SRC}" ]]; then
    mv "${FUNASR_SRC}" "${FUNASR_PREVIOUS}"
fi
mv "${FUNASR_STAGE}" "${FUNASR_SRC}"
rm -rf "${FUNASR_PREVIOUS}"
log "安装已校验的 FunASR @ ${FUNASR_COMMIT}（依赖已由 lock 同步）..."
"${UV}" pip install --python "${PYTHON}" --no-deps "${FUNASR_SRC}"

# --- 4. 版本与设备信息 ------------------------------------------------------
log "版本与设备信息:"
"${PYTHON}" - <<'PY'
import sys
print(f"python={sys.version.split()[0]}")

for name in ("torch", "torchaudio", "funasr", "modelscope", "transformers"):
    try:
        mod = __import__(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}=import_failed:{type(exc).__name__}")

try:
    import torch

    print(f"torch.xpu.is_available={torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"torch.xpu.device_count={torch.xpu.device_count()}")
        print(f"torch.xpu.device_name={torch.xpu.get_device_name(0)}")
        try:
            props = torch.xpu.get_device_properties(0)
            print(f"torch.xpu.total_memory={props.total_memory}")
        except Exception as exc:  # noqa: BLE001
            print(f"torch.xpu.total_memory=unavailable:{type(exc).__name__}")
except Exception as exc:  # noqa: BLE001
    print(f"torch.xpu=check_failed:{type(exc).__name__}")
PY
