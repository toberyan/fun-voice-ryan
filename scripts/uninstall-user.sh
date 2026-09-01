#!/usr/bin/env bash
# uninstall-user.sh — remove the user-scoped Fun Voice Ryan installation.
#
# Reverses exactly what install-user.sh wrote, in order:
#   1. systemctl --user disable --now worker + daemon
#   2. Remove the systemd units, autostart desktop entry, and Fcitx addon files
#   3. Remove the six console scripts from ~/.local/bin
#   4. Remove the runtime sockets and capture shards
#
# The model cache and user config are always preserved unless --purge is given,
# in which case they are deleted after a second confirmation.

set -euo pipefail

BIN_DIR="${HOME}/.local/bin"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
FCITX_LIB_DIR="${HOME}/.local/lib/fcitx5"
FCITX_ADDON_DIR="${HOME}/.local/share/fcitx5/addon"
AUTOSTART_DIR="${HOME}/.config/autostart"
CONFIG_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/fun-voice-ryan"
MODELS_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/models"

CONSOLE_SCRIPTS=(
  fun-voice-daemon fun-voice-worker fun-voice-preflight fun-voice-selftest
  fun-voice-corrector fun-voice-benchmark
)
SYSTEMD_SERVICES=(fun-voice-worker.service fun-voice-worker@nano.service fun-voice-worker@sensevoice.service fun-voice-daemon.service)

log() { printf '[uninstall-user] %s\n' "$*"; }
die() { printf '[uninstall-user] ERROR(%s): %s\n' "$1" "$2" >&2; exit 1; }

# Guard the runtime paths: with XDG_RUNTIME_DIR unset, ${VAR:-}/fun-voice-ryan
# would collapse to the root-level "/fun-voice-ryan". Never touch those paths
# unless the runtime dir is valid (mirrors the install script's precondition).
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
    log "XDG_RUNTIME_DIR is not set or does not exist; skipping runtime cleanup"
    RUNTIME_DIR=""
    FCITX_SOCKET=""
else
    RUNTIME_DIR="${XDG_RUNTIME_DIR}/fun-voice-ryan"
    FCITX_SOCKET="${XDG_RUNTIME_DIR}/fun-voice-ryan-fcitx.sock"
fi

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

# --- 2. Remove unit / desktop / addon files --------------------------------
for unit in fun-voice-worker.service fun-voice-worker@.service fun-voice-daemon.service; do
    remove_file "${SYSTEMD_USER_DIR}/${unit}"
done
remove_file "${AUTOSTART_DIR}/fun-voice-session.desktop"
remove_file "${FCITX_LIB_DIR}/fcitx5-fun-voice.so"
remove_file "${FCITX_ADDON_DIR}/fcitx5-fun-voice.conf"

# --- 3. Remove the console scripts ------------------------------------------
for script in "${CONSOLE_SCRIPTS[@]}"; do
    remove_file "${BIN_DIR}/${script}"
done

# --- 4. Remove runtime sockets and capture shards ----------------------------
if [[ -n "${RUNTIME_DIR}" ]]; then
    for socket in "${RUNTIME_DIR}/daemon.sock" "${RUNTIME_DIR}/worker.sock" "${RUNTIME_DIR}/worker-sensevoice.sock"; do
        remove_file "${socket}"
    done
    remove_file "${FCITX_SOCKET}"
    if [[ -d "${RUNTIME_DIR}/capture" ]]; then
        rm -rf "${RUNTIME_DIR}/capture" && log "removed capture shards under ${RUNTIME_DIR}/capture"
    fi
fi

# --- 5. Optional purge (model cache + user config) --------------------------
if [[ "${PURGE}" -eq 1 ]]; then
    printf '[uninstall-user] WARNING: --purge will permanently delete:\n'
    printf '  model cache: %s\n' "${MODELS_DIR}"
    printf '  user config: %s\n' "${CONFIG_DIR}"
    printf 'Type "DELETE" (without quotes) to confirm: '
    if ! read -r answer; then
        printf '\n[uninstall-user] ERROR(abort): purge not confirmed (stdin closed); model cache and config preserved\n' >&2
        exit 1
    fi
    if [[ "${answer}" != "DELETE" ]]; then
        die "abort" "purge not confirmed; model cache and config preserved"
    fi
    rm -rf "${MODELS_DIR}" && log "purged model cache ${MODELS_DIR}"
    rm -rf "${CONFIG_DIR}" && log "purged user config ${CONFIG_DIR}"
else
    log "model cache and user config preserved (use --purge to delete)"
fi

log "uninstall complete"
