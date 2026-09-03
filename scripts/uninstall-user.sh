#!/usr/bin/env bash
# uninstall-user.sh — remove the user-scoped Fun Voice Ryan installation.
#
# Reverses exactly what install-user.sh wrote, in order:
#   1. systemctl --user disable --now worker + daemon
#   2. Remove the systemd units, autostart desktop entry, and Fcitx addon files
#   3. Remove the six console scripts from ~/.local/bin
#   4. Remove the runtime sockets and capture shards
#
# Model snapshots, portable runtimes, selection state, and user config are
# always preserved so reinstalling never needs an implicit model download.

set -euo pipefail

BIN_DIR="${HOME}/.local/bin"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
FCITX_LIB_DIR="${HOME}/.local/lib/fcitx5"
FCITX_ADDON_DIR="${HOME}/.local/share/fcitx5/addon"
OVERLAY_INSTALL_DIR="${HOME}/.local/lib/fun-voice-ryan"
AUTOSTART_DIR="${HOME}/.config/autostart"

CONSOLE_SCRIPTS=(
  fun-voice-daemon fun-voice-worker fun-voice-preflight fun-voice-selftest
  fun-voice-corrector fun-voice-benchmark
)
SYSTEMD_SERVICES=(fun-voice-worker.service fun-voice-worker@nano.service fun-voice-worker@sensevoice.service fun-voice-daemon.service)

log() { printf '[uninstall-user] %s\n' "$*"; }
die() { printf '[uninstall-user] ERROR(%s): %s\n' "$1" "$2" >&2; exit 1; }

if [[ "$#" -ne 0 ]]; then
    die "usage" "no arguments are supported"
fi

private_owned_directory() {
    local path="$1" mode
    if [[ ! -d "${path}" || -L "${path}" || ! -O "${path}" ]]; then
        return 1
    fi
    mode="$(stat -c '%a' -- "${path}" 2>/dev/null)" || return 1
    [[ "${mode}" == "700" ]]
}

# Validate the entire recursive-cleanup boundary before removing any installed
# artifact. An unset/nonexistent session root has nothing to clean; a present
# but shared or redirected tree is an error, because its contents are not
# demonstrably application-owned.
RUNTIME_DIR=""
FCITX_SOCKET=""
if [[ -z "${XDG_RUNTIME_DIR:-}" \
    || ( ! -e "${XDG_RUNTIME_DIR}" && ! -L "${XDG_RUNTIME_DIR}" ) ]]; then
    log "XDG_RUNTIME_DIR is not set or does not exist; skipping runtime cleanup"
elif [[ "${XDG_RUNTIME_DIR}" != /* ]] \
    || ! private_owned_directory "${XDG_RUNTIME_DIR}"; then
    die "runtime-safety" "XDG_RUNTIME_DIR is not a private owned directory"
else
    RUNTIME_DIR="${XDG_RUNTIME_DIR}/fun-voice-ryan"
    FCITX_SOCKET="${XDG_RUNTIME_DIR}/fun-voice-ryan-fcitx.sock"
    if [[ -e "${RUNTIME_DIR}" || -L "${RUNTIME_DIR}" ]]; then
        private_owned_directory "${RUNTIME_DIR}" \
            || die "runtime-safety" "application runtime directory is unsafe"
        CAPTURE_DIR="${RUNTIME_DIR}/capture"
        if [[ -e "${CAPTURE_DIR}" || -L "${CAPTURE_DIR}" ]]; then
            private_owned_directory "${CAPTURE_DIR}" \
                || die "runtime-safety" "capture directory is unsafe"
        fi
    fi
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
remove_file "${OVERLAY_INSTALL_DIR}/fun-voice-overlay"

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
        rm -rf -- "${RUNTIME_DIR}/capture" \
            && log "removed capture shards under ${RUNTIME_DIR}/capture"
    fi
fi

log "model snapshots, portable runtimes, selection state, and user config preserved"

log "uninstall complete"
