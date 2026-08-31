#!/usr/bin/env bash
# Remove the Fun Voice Ryan custom shortcut registered by register-dde-shortcut.sh.
#
# Reads the persisted custom shortcut id and calls DeleteCustomShortcut. A no-op
# when nothing was ever registered.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

cd "$ROOT"
exec uv run python -m fun_voice.desktop unregister "$@"
