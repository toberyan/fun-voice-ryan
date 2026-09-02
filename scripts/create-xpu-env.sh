#!/usr/bin/env bash
#
# create-xpu-env.sh — 创建项目 XPU 推理环境(幂等)。
#
# - 用 uv 在项目内创建 .venv(Python 3.12)。
# - 用 PyTorch XPU wheel 索引安装原生 FunASR/PyTorch 所需的 torch 与 torchaudio
#   (不混用清华镜像,显式 --index-strategy unsafe-best-match 解析 +xpu 变体)。
# - FunASR 固定到提交 8cd758c0ced576516b05a749194e6a94cdd38f99(经 codeload
#   tarball 下载,避免 git clone 在弱网下 TLS 抖动)。
# - 打印安装版本与设备信息;不打印任何音频路径或转写文本。
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
UV="${UV:-/home/toberyan/.local/bin/uv}"

FUNASR_COMMIT="8cd758c0ced576516b05a749194e6a94cdd38f99"
FUNASR_TARBALL="${ROOT_DIR}/.funasr-src.tar.gz"

PYPI_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
XPU_INDEX="https://download.pytorch.org/whl/xpu"

log() { printf '[create-xpu-env] %s\n' "$*"; }

# --- 1. venv ---------------------------------------------------------------
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    log ".venv 已存在,跳过创建"
else
    log "创建 .venv (Python 3.12)..."
    "${UV}" venv "${VENV_DIR}" --python 3.12
fi

PYTHON="${VENV_DIR}/bin/python"

install_xpu() {
    "${UV}" pip install \
        --python "${PYTHON}" \
        --index-url "${PYPI_INDEX}" \
        --extra-index-url "${XPU_INDEX}" \
        --index-strategy unsafe-best-match \
        "$@"
}

# --- 2. PyTorch + torchaudio XPU --------------------------------------------
log "安装 torch + torchaudio (XPU)..."
install_xpu torch torchaudio

# --- 3. FunASR @ pinned commit + modelscope ---------------------------------
FUNASR_SRC="${ROOT_DIR}/.funasr-src"
if [[ ! -f "${FUNASR_TARBALL}" ]]; then
    log "下载 FunASR @ ${FUNASR_COMMIT} ..."
    curl -sSL --retry 8 --retry-delay 3 --retry-all-errors \
        -o "${FUNASR_TARBALL}" \
        "https://codeload.github.com/modelscope/FunASR/tar.gz/${FUNASR_COMMIT}"
fi
if [[ ! -f "${FUNASR_SRC}/funasr/__init__.py" ]]; then
    log "解包 FunASR 源码并移除指向绝对路径的损坏符号链接..."
    rm -rf "${FUNASR_SRC}"
    mkdir -p "${FUNASR_SRC}"
    tar xzf "${FUNASR_TARBALL}" -C "${FUNASR_SRC}" --strip-components=1
    find "${FUNASR_SRC}" -type l -lname '/*' -delete
fi
log "安装 FunASR @ ${FUNASR_COMMIT} + modelscope..."
install_xpu "${FUNASR_SRC}" modelscope

# The enhanced identity store depends on an authenticated Secret Service key
# and AES-GCM. They are ordinary host-side dependencies, but installing them
# here keeps the XPU environment setup reproducible.
log "安装 enhanced identity dependencies (cryptography + secretstorage)..."
install_xpu cryptography secretstorage

# --- 4. 版本与设备信息 ------------------------------------------------------
log "版本与设备信息:"
"${PYTHON}" - <<'PY'
import sys
print(f"python={sys.version.split()[0]}")

for name in ("torch", "torchaudio", "funasr", "modelscope"):
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
