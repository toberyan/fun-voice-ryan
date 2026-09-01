#!/usr/bin/env bash
#
# create-xpu-env.sh — 创建项目 XPU 推理环境(幂等)。
#
# - 用 uv 在项目内创建 .venv(Python 3.12,vllm-xpu-kernels 要求 3.12)。
# - 用 vLLM 官方 XPU wheel 索引 + PyTorch XPU 索引安装 vllm 与 torch(不混用
#   清华镜像,显式 --index-strategy unsafe-best-match 解析 +xpu 变体)。
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
VLLM_INDEX="https://wheels.vllm.ai/nightly/xpu"
TRITON_SHIM_INDEX="https://wheels.vllm.ai/xpu"

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
        --extra-index-url "${VLLM_INDEX}" \
        --extra-index-url "${XPU_INDEX}" \
        --index-strategy unsafe-best-match \
        "$@"
}

# --- 2. vLLM + torch/torchaudio + XPU kernels -------------------------------
log "安装 vllm + vllm-xpu-kernels + torch/torchaudio (XPU)..."
install_xpu vllm==0.28.0 vllm-xpu-kernels==0.1.14.1 torchaudio

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

# --- 3.4 triton XPU shim 修复 ------------------------------------------------
# xgrammar 0.2.3 的 Requires-Dist: triton 会把 PyPI 的 NVIDIA-only triton 3.8.0
# 拉进环境,其 libtriton.so 不含 intel 后端,覆盖 triton-xpu 3.7.2 的实现,导致
# vLLM kernel warm-up 在 vllm/v1/worker/block_table.py:195 报
# "function is not subscriptable"。按 vLLM XPU 文档固定 wheels.vllm.ai/xpu 的
# triton==3.7.2+xpu shim(纯元数据,Requires-Dist: triton-xpu==3.7.2),透明解析
# 到真正的 Intel XPU 实现。
log "应用 triton XPU shim 修复 (triton==3.7.2+xpu)..."
TRITON_VER="$("${PYTHON}" -c "import importlib.metadata as m; print(m.version('triton'))" 2>/dev/null || true)"
if [[ "${TRITON_VER}" != "3.7.2+xpu" ]]; then
    "${UV}" pip uninstall triton --python "${PYTHON}" >/dev/null 2>&1 || true
    "${UV}" pip install --reinstall-package triton-xpu --no-deps \
        --python "${PYTHON}" \
        --index-url "${XPU_INDEX}" \
        triton-xpu==3.7.2
    "${UV}" pip install --no-deps --python "${PYTHON}" \
        --index-url "${TRITON_SHIM_INDEX}" \
        triton==3.7.2+xpu
else
    log "triton shim 已就绪,跳过"
fi

# --- 3.5 vLLM oneCCL 单卡 warm-up 修复 --------------------------------------
# vLLM 0.28.0 的 XPUWorker.init_device() 在 world_size==1 时仍做一次 oneCCL
# warm-up all_reduce;本机 Intel Arc(compute-runtime 25.18 < 26.18)上 oneCCL
# 无法枚举 GPU 拓扑,导致引擎启动失败。上游已修复(vllm#52386 / PR#52389,含于
# v0.28.1),此处对 0.28.0 应用等价的单行修复:仅多卡才 warm-up。
log "应用 vLLM oneCCL 单卡 warm-up 修复 (vllm#52386)..."
"${PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm.v1.worker.xpu_worker")
assert spec is not None and spec.origin is not None, "vllm xpu_worker not found"
path = Path(spec.origin)
text = path.read_text(encoding="utf-8")

old = (
    "        # global all_reduce needed for overall oneccl warm up\n"
    "        if torch.distributed.is_xccl_available():\n"
    "            torch.distributed.all_reduce(torch.zeros(1).xpu())\n"
)
new = (
    "        # oneCCL warm-up; only meaningful for multi-device runs. Requiring\n"
    "        # it with a single worker breaks platforms where oneCCL cannot\n"
    "        # enumerate device topology (e.g. paravirtualized GPUs).\n"
    "        if (\n"
    "            self.parallel_config.world_size > 1\n"
    "            and torch.distributed.is_xccl_available()\n"
    "        ):\n"
    "            torch.distributed.all_reduce(torch.zeros(1).xpu())\n"
)

if "world_size > 1" in text:
    print("[create-xpu-env] oneCCL warm-up 修复已存在,跳过")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[create-xpu-env] 已修复: {path}")
else:
    raise SystemExit("oneCCL warm-up 代码模式未匹配,请检查 vllm 版本")
PY

# --- 3.6 vLLM XPU 显存探测回退 ---------------------------------------------
# Intel Arc(compute-runtime 25.18 < 26.18)上 vllm-xpu-kernels 的
# getMemoryInfo 返回 0 可用显存(驱动不支持查询),导致 vLLM 启动时误判"显存不足"。
# 回退为 total - reserved,避免启动检查失败。
log "应用 vLLM XPU 显存探测回退 (compute-runtime < 26.18)..."
"${PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm.platforms.xpu")
assert spec is not None and spec.origin is not None, "vllm xpu platform not found"
path = Path(spec.origin)
text = path.read_text(encoding="utf-8")

old = (
    "    # Call the underlying C++ implementation\n"
    "    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)\n"
    "\n"
    "    return free, total\n"
)
new = (
    "    # Call the underlying C++ implementation\n"
    "    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)\n"
    "\n"
    "    # Intel Arc with compute-runtime < 26.18 cannot query free memory and\n"
    "    # returns 0. Fall back to total-minus-reserved so vLLM's startup memory\n"
    "    # check does not see the device as having 0 free bytes.\n"
    "    if free == 0 and total > 0:\n"
    "        free = total - torch.accelerator.memory_reserved(device)\n"
    "\n"
    "    return free, total\n"
)

if "compute-runtime < 26.18" in text:
    print("[create-xpu-env] 显存探测回退已存在,跳过")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[create-xpu-env] 已修复: {path}")
else:
    raise SystemExit("显存探测代码模式未匹配,请检查 vllm 版本")
PY

# --- 3.7 triton Intel 后端 Level Zero 头文件与库链接修复 -----------------------
# triton-xpu 的 Intel 后端首次 JIT 编译 driver.c(spirv_utils)时需要
# <level_zero/ze_api.h> 与 libze_loader.so(通常由 level-zero-dev 提供)。本机只
# 装了 libze1/libze-intel-gpu1 运行时(无头文件),且不允许 sudo 装 level-zero-dev。
# 此处从 oneapi-src/level-zero(v1.21.9,匹配 libze1 1.21.9)下载头文件并入
# .venv/include/level_zero/,并创建 .venv/lib/libze_loader.so 软链。
LZ_VER="1.21.9"
if [[ ! -f "${VENV_DIR}/include/level_zero/ze_api.h" ]]; then
    log "部署 Level Zero 头文件 (v${LZ_VER})..."
    LZ_TARBALL="${ROOT_DIR}/.level-zero-headers.tar.gz"
    curl -fsSL --retry 6 --retry-delay 3 --retry-all-errors \
        "https://codeload.github.com/oneapi-src/level-zero/tar.gz/refs/tags/v${LZ_VER}" \
        -o "${LZ_TARBALL}"
    mkdir -p "${VENV_DIR}/include/level_zero"
    tar xzf "${LZ_TARBALL}" -C "${VENV_DIR}/include/level_zero" --strip-components=2 \
        "level-zero-${LZ_VER}/include"
    rm -f "${LZ_TARBALL}"
fi
if [[ ! -e "${VENV_DIR}/lib/libze_loader.so" ]]; then
    log "创建 libze_loader.so 软链..."
    ZE_LOADER_SO="$(ldconfig -p 2>/dev/null | awk '$1 == "libze_loader.so.1" { print $NF; exit }' || true)"
    if [[ -z "${ZE_LOADER_SO}" ]]; then
        for cand in /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu /usr/lib; do
            if [[ -f "${cand}/libze_loader.so.1" ]]; then
                ZE_LOADER_SO="${cand}/libze_loader.so.1"
                break
            fi
        done
    fi
    if [[ -n "${ZE_LOADER_SO}" && -f "${ZE_LOADER_SO}" ]]; then
        ln -sf "${ZE_LOADER_SO}" "${VENV_DIR}/lib/libze_loader.so"
    else
        log "警告:未找到系统 libze_loader.so.1,跳过软链(triton Intel 后端 JIT 编译可能失败)"
    fi
fi

# --- 4. 版本与设备信息 ------------------------------------------------------
log "版本与设备信息:"
"${PYTHON}" - <<'PY'
import sys
print(f"python={sys.version.split()[0]}")

for name in ("torch", "torchaudio", "vllm", "funasr", "modelscope"):
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
