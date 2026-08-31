#!/usr/bin/env bash
# Register the Fun Voice Ryan push-to-talk shortcut (Super+C) with DDE Keybinding1.
#
# DDE executes the *bridge* command on trigger; the model is never run directly.
# This script checks for a conflict, registers the shortcut, and persists the
# returned custom shortcut id so unregister-dde-shortcut.sh can remove it later.
# It does not load the model and never reads /dev/input.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# The bridge command DDE executes. Prefer the installed console script, then
# the project virtualenv. Override with FUN_VOICE_BRIDGE_ACTION.
if [[ -n "${FUN_VOICE_BRIDGE_ACTION:-}" ]]; then
  BRIDGE_ACTION="$FUN_VOICE_BRIDGE_ACTION"
elif command -v fun-voice-bridge >/dev/null 2>&1; then
  BRIDGE_ACTION="$(command -v fun-voice-bridge)"
else
  BRIDGE_ACTION="$ROOT/.venv/bin/fun-voice-bridge"
fi

cd "$ROOT"
exec uv run python -m fun_voice.desktop register --action "$BRIDGE_ACTION" "$@"
