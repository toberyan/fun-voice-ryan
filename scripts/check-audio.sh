#!/usr/bin/env bash
#
# check-audio.sh — 快速检查录音链路:PipeWire/pw-record 可用性、默认 source 存在、
# 以及一次 1 秒的 raw s16le 16 kHz 单声道试录。
#
# 只读检查 + 临时文件自清理;不打印任何音频内容或转写文本。
#
set -euo pipefail

TMP_PCM="$(mktemp --suffix=.pcm)"
trap 'rm -f "${TMP_PCM}"' EXIT

fail() { printf '[check-audio] FAIL: %s\n' "$*" >&2; exit 1; }

# 1. PipeWire / pw-record 可用性
if ! command -v pw-record >/dev/null 2>&1; then
    fail "pw-record not found on PATH"
fi

# 2. 默认 source 存在(强制英文输出,避免本地化差异)
DEFAULT_SOURCE="$(LC_ALL=C pactl info 2>/dev/null \
    | awk -F': ' '/^Default Source:/{print $2; exit}')"
if [[ -z "${DEFAULT_SOURCE}" ]]; then
    fail "cannot determine default source via 'pactl info'"
fi
printf '[check-audio] default source: %s\n' "${DEFAULT_SOURCE}"

# 3. 快速 1 秒试录:16000 采样 = 1 秒 @16 kHz = 32000 字节。
#    pw-record 1.6.4 即使成功也返回 1,故以写入字节数作为成功判据。
timeout 10 pw-record --rate 16000 --channels 1 --format s16 \
    --media-type Audio --raw -n 16000 "${TMP_PCM}" >/dev/null 2>&1 || true

SIZE="$(stat -c%s "${TMP_PCM}" 2>/dev/null || echo 0)"
if (( SIZE < 32000 )); then
    fail "trial recording produced ${SIZE} bytes (expected 32000)"
fi

printf '[check-audio] OK: captured %s bytes of raw s16le 16 kHz mono PCM\n' "${SIZE}"
