#!/usr/bin/env bash
# uninstall-user.sh — remove the user-scoped Fun Voice Ryan installation.
#
# Reverses exactly what install-user.sh wrote, in order:
#   1. systemctl --user disable --now worker + daemon
#   2. Unregister the DDE Super+C shortcut (only if its id file exists)
#   3. Remove the systemd units, autostart desktop entry, and Fcitx addon files
#   4. Remove the five console scripts from ~/.local/bin
#   5. Remove the runtime sockets and capture shards
#
# The model cache and user config are always preserved unless --purge is given,
# in which case they are deleted after a second confirmation.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

BIN_DIR="${HOME}/.local/bin"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
FCITX_LIB_DIR="${HOME}/.local/lib/fcitx5"
FCITX_ADDON_DIR="${HOME}/.local/share/fcitx5/addon"
AUTOSTART_DIR="${HOME}/.config/autostart"
CONFIG_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/fun-voice-ryan"
MODELS_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/models"

CONSOLE_SCRIPTS=(fun-voice-daemon fun-voice-worker fun-voice-bridge fun-voice-preflight fun-voice-selftest)
SYSTEMD_SERVICES=(fun-voice-worker.service fun-voice-daemon.service)

SHORTCUT_ID_FILE="${CONFIG_DIR}/dde-shortcut-id"
RUNTIME_DIR="${XDG_RUNTIME_DIR:-}/fun-voice-ryan"
FCITX_SOCKET="${XDG_RUNTIME_DIR:-}/fun-voice-ryan-fcitx.sock"

log() { printf '[uninstall-user] %s\n' "$*"; }
die() { printf '[uninstall-user] ERROR(%s): %s\n' "$1" "$2" >&2; exit 1; }

PURGE=0
if [[ "${1:-}" == "--purge" ]]; then
    PURGE=1
elif [[ -n "${1:-}" ]]; then
    die "usage" "unknown argument: $1 (only --purge is supported)"
fi

# remove_file PATH — best-effort removal of a regular file or symlink.
remove_file() {
    local path="$1"
    if [[ -e "${path}" || -L "${path}" ]]; then
        rm -f "${path}" && log "removed ${path}"
    fi
}

# --- 1. Stop and disable the systemd services ------------------------------
if command -v systemctl >/dev/null 2>&1; then
    for service in "${SYSTEMD_SERVICES[@]}"; do
        systemctl --user disable --now "${service}" 2>/dev/null || true
    done
    systemctl --user daemon-reload 2>/dev/null || true
    log "disabled and stopped ${SYSTEMD_SERVICES[*]}"
fi

# --- 2. Unregister the DDE Super+C shortcut ---------------------------------
if [[ -f "${SHORTCUT_ID_FILE}" ]]; then
    bash "${ROOT}/scripts/unregister-dde-shortcut.sh" \
        || die "dde" "unregister-dde-shortcut.sh failed"
    log "unregistered DDE Super+C shortcut"
else
    log "no DDE shortcut id file; nothing to unregister"
fi

# --- 3. Remove unit / desktop / addon files --------------------------------
for unit in fun-voice-worker.service fun-voice-daemon.service; do
    remove_file "${SYSTEMD_USER_DIR}/${unit}"
done
remove_file "${AUTOSTART_DIR}/fun-voice-session.desktop"
remove_file "${FCITX_LIB_DIR}/fcitx5-fun-voice.so"
remove_file "${FCITX_ADDON_DIR}/fcitx5-fun-voice.conf"

# --- 4. Remove the console scripts ------------------------------------------
for script in "${CONSOLE_SCRIPTS[@]}"; do
    remove_file "${BIN_DIR}/${script}"
done

# --- 5. Remove runtime sockets and capture shards ----------------------------
for socket in "${RUNTIME_DIR}/daemon.sock" "${RUNTIME_DIR}/worker.sock"; do
    remove_file "${socket}"
done
remove_file "${FCITX_SOCKET}"
if [[ -d "${RUNTIME_DIR}/capture" ]]; then
    rm -rf "${RUNTIME_DIR}/capture" && log "removed capture shards under ${RUNTIME_DIR}/capture"
fi

# --- 6. Optional purge (model cache + user config) --------------------------
if [[ "${PURGE}" -eq 1 ]]; then
    printf '[uninstall-user] WARNING: --purge will permanently delete:\n'
    printf '  model cache: %s\n' "${MODELS_DIR}"
    printf '  user config: %s\n' "${CONFIG_DIR}"
    printf 'Type "DELETE" (without quotes) to confirm: '
    read -r answer
    if [[ "${answer}" != "DELETE" ]]; then
        die "abort" "purge not confirmed; model cache and config preserved"
    fi
    rm -rf "${MODELS_DIR}" && log "purged model cache ${MODELS_DIR}"
    rm -rf "${CONFIG_DIR}" && log "purged user config ${CONFIG_DIR}"
else
    log "model cache and user config preserved (use --purge to delete)"
fi

log "uninstall complete"
