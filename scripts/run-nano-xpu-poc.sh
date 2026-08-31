#!/usr/bin/env bash
#
# run-nano-xpu-poc.sh — Fun-ASR-Nano Intel XPU 阻断 POC。
#
# 流程:
#   1. 确保 XDG_RUNTIME_DIR 可用,所有临时产物只写在其下的
#      fun-voice-ryan/poc-samples/ 目录,trap 清理。
#   2. (幂等)从 ModelScope 下载 Fun-ASR-Nano-2512 与 FSMN-VAD 模型到
#      ${XDG_DATA_HOME:-~/.local/share}/fun-voice-ryan/models。
#   3. 自动下载开源示例音频(普通话+英文),ffmpeg 重采样 16kHz 单声道并拼接成
#      10s 与 60s 两个样本(可用 --short/--long 覆盖)。
#   4. 运行 fun-voice-preflight 九项硬门,输出 poc-report.json。
#   5. 把样本构成(仅来源 URL 与时长,不含音频路径/转写文本)并入报告。
#
# 退出码:0=全部硬门通过;非 0=任一失败或环境/样本缺失(绝不静默退回 CPU)。
MODELS_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/models"
# modelscope 1.39.x 缓存布局:${MODELSCOPE_CACHE}/models/<owner>--<name>/snapshots/<revision>
NANO_MODEL_DIR="${MODELS_ROOT}/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"

NANO_MODEL_ID="FunAudioLLM/Fun-ASR-Nano-2512"
VAD_MODEL_ID="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# 开源示例音频(普通话/英文),ModelScope 官方测试音频 + Qwen3-ASR 示例。
SAMPLE_SOURCES=(
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav|mandarin"
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav|english"
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav|mandarin"
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav|english"
)

SHORT_SAMPLE=""
LONG_SAMPLE=""
MODEL_DIR_ARG=""
SKIP_MODEL=0

log() { printf '[run-nano-xpu-poc] %s\n' "$*"; }
die() { printf '[run-nano-xpu-poc] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --short) SHORT_SAMPLE="$2"; shift 2 ;;
        --long) LONG_SAMPLE="$2"; shift 2 ;;
        --model-dir) MODEL_DIR_ARG="$2"; shift 2 ;;
        --skip-model-download) SKIP_MODEL=1; shift ;;
        --samples-missing) shift ;;  # 样本由脚本自动生成,保留兼容旧参数
        *) die "unknown argument: $1" ;;
    esac
done

# --- 1. XDG_RUNTIME_DIR 校验 ------------------------------------------------
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
    die "XDG_RUNTIME_DIR 未设置或不存在,拒绝运行"
fi
RUNTIME_BASE="${XDG_RUNTIME_DIR}/fun-voice-ryan"
REPORT_DIR="${RUNTIME_BASE}"
REPORT="${REPORT_DIR}/poc-report.json"
SAMPLES_DIR="${RUNTIME_BASE}/poc-samples"
mkdir -p "${REPORT_DIR}" "${SAMPLES_DIR}" || die "无法创建 ${REPORT_DIR}"

cleanup() {
    rm -rf "${SAMPLES_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# --- 2. 模型下载(幂等)------------------------------------------------------
MODEL_DIR="${MODEL_DIR_ARG:-${NANO_MODEL_DIR}}"
if [[ "${SKIP_MODEL}" -ne 1 && ! -f "${MODEL_DIR}/model.pt" ]]; then
    log "下载模型到 ${MODELS_ROOT} ..."
    MODELSCOPE_CACHE="${MODELS_ROOT}" "${PYTHON}" - "${NANO_MODEL_ID}" "${VAD_MODEL_ID}" <<'PY'
import sys

from modelscope.hub.snapshot_download import snapshot_download

for model_id in sys.argv[1:]:
    print(f"[run-nano-xpu-poc] snapshot_download {model_id}", flush=True)
    snapshot_download(model_id, revision="master")
PY
fi
if [[ ! -f "${MODEL_DIR}/model.pt" ]]; then
    die "模型目录缺少 model.pt: ${MODEL_DIR}"
fi

# --- 3. 样本生成 ------------------------------------------------------------
if [[ -n "${SHORT_SAMPLE}" && -n "${LONG_SAMPLE}" ]]; then
    log "使用显式样本 --short/--long"
    [[ -f "${SHORT_SAMPLE}" ]] || die "--short 样本不存在"
    [[ -f "${LONG_SAMPLE}" ]] || die "--long 样本不存在"
else
    log "自动生成 10s/60s 样本(开源示例音频拼接)"
    SRC_DIR="${SAMPLES_DIR}/src"
    mkdir -p "${SRC_DIR}"

    comp=()
    concat_list="${SAMPLES_DIR}/concat.txt"
    : > "${concat_list}"

    # 下载 + 归一化每个源片段(16kHz 单声道 s16le)
    norm_files=()
    idx=0
    for entry in "${SAMPLE_SOURCES[@]}"; do
        url="${entry%%|*}"
        lang="${entry##*|}"
        raw="${SRC_DIR}/raw_${idx}.wav"
        norm="${SRC_DIR}/norm_${idx}.wav"
        curl -sSL --retry 6 --retry-delay 3 --retry-all-errors -o "${raw}" "${url}" \
            || die "样本下载失败: ${url}"
        ffmpeg -y -v error -i "${raw}" -ar 16000 -ac 1 -c:a pcm_s16le "${norm}" \
            || die "样本重采样失败: ${url}"
        dur="$(ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "${norm}")"
        comp+=("{\"source\":\"${url}\",\"language\":\"${lang}\",\"duration_s\":${dur}}")
        norm_files+=("${norm}")
        idx=$((idx + 1))
    done

    # 中英交替拼接列表,循环足够多次覆盖 60s
    for _ in 0 1 2 3 4 5; do
        for f in "${norm_files[@]}"; do
            printf 'file %s\n' "${f}" >> "${concat_list}"
        done
    done

    SHORT_SAMPLE="${SAMPLES_DIR}/poc-10s.wav"
    LONG_SAMPLE="${SAMPLES_DIR}/poc-60s.wav"
    ffmpeg -y -v error -f concat -safe 0 -i "${concat_list}" \
        -t 10 -c:a pcm_s16le "${SHORT_SAMPLE}" \
        || die "10s 样本拼接失败"
    ffmpeg -y -v error -f concat -safe 0 -i "${concat_list}" \
        -t 60 -c:a pcm_s16le "${LONG_SAMPLE}" \
        || die "60s 样本拼接失败"

    short_dur="$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "${SHORT_SAMPLE}")"
    long_dur="$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "${LONG_SAMPLE}")"
    sources_json="[$(IFS=,; echo "${comp[*]}")]"
    printf '{"short":{"duration_s":%s,"sources":%s},"long":{"duration_s":%s,"sources":%s}}\n' \
        "${short_dur}" "${sources_json}" "${long_dur}" "${sources_json}" \
        > "${SAMPLES_DIR}/sample-composition.json"
fi

log "样本就绪 (short=$(basename "${SHORT_SAMPLE}"), long=$(basename "${LONG_SAMPLE}"))"

# --- 4. 运行九项硬门 preflight ----------------------------------------------
log "运行 fun-voice-preflight ..."
set +e
PYTHONPATH="${ROOT_DIR}/src" "${PYTHON}" -m fun_voice.preflight \
    --short "${SHORT_SAMPLE}" \
    --long "${LONG_SAMPLE}" \
    --model-dir "${MODEL_DIR}" \
    --report "${REPORT}"
preflight_rc=$?
set -e

# --- 5. 并入样本构成(仅来源与时长)------------------------------------------
if [[ -f "${SAMPLES_DIR}/sample-composition.json" ]]; then
    "${PYTHON}" - "${REPORT}" "${SAMPLES_DIR}/sample-composition.json" <<'PY'
import json
import sys

report_path, comp_path = sys.argv[1], sys.argv[2]
with open(report_path, encoding="utf-8") as fh:
    report = json.load(fh)
with open(comp_path, encoding="utf-8") as fh:
    composition = json.load(fh)
report["sample_composition"] = composition
with open(report_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, sort_keys=True)
    fh.write("\n")
print(f"[run-nano-xpu-poc] report: {report_path}")
PY
fi

# --- 6. 汇总(只打印状态,不打印音频路径/转写文本)----------------------------
"${PYTHON}" - "${REPORT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    report = json.load(fh)
ready = bool(report.get("ready"))
print(f"[run-nano-xpu-poc] ready={ready}")
for check in report.get("checks", []):
    print(f"  {check['name']}: {check['status']}")
sys.exit(0 if ready else 1)
PY

exit "${preflight_rc}"
