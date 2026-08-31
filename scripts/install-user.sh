#!/usr/bin/env bash
# install-user.sh — install Fun Voice Ryan into the current user's home.
#
# Hard gate: the Intel XPU POC report must exist and say ready=true before
# anything is written (the plan forbids installing/starting desktop services
# before every POC hard gate passes).
#
# Every write target is user-scoped (home or XDG_RUNTIME_DIR); the script never
# uses sudo and never touches system-wide paths. Each step is idempotent (safe
# to re-run) and reversible via uninstall-user.sh.
#
# Steps:
#   1. Console scripts            -> ~/.local/bin
#   2. systemd units              -> ~/.config/systemd/user  (replaces symlinks)
#   3. Fcitx addon .so + .conf    -> ~/.local/lib/fcitx5 and ~/.local/share/fcitx5/addon
#   4. Autostart desktop entry    -> ~/.config/autostart
#   5. systemd daemon-reload + enable --now worker + daemon
#   6. Register the Super+C DDE shortcut

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# --- User-scoped target paths ----------------------------------------------
BIN_DIR="${HOME}/.local/bin"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
FCITX_LIB_DIR="${HOME}/.local/lib/fcitx5"
FCITX_ADDON_DIR="${HOME}/.local/share/fcitx5/addon"
AUTOSTART_DIR="${HOME}/.config/autostart"
SHORTCUT_ID_FILE="${XDG_CONFIG_HOME:-${HOME}/.config}/fun-voice-ryan/dde-shortcut-id"
CONSOLE_SCRIPTS=(fun-voice-daemon fun-voice-worker fun-voice-bridge fun-voice-preflight fun-voice-selftest)
SYSTEMD_UNITS=(fun-voice-worker.service fun-voice-daemon.service)
SYSTEMD_SERVICES=(fun-voice-worker.service fun-voice-daemon.service)

POC_REPORT="${XDG_RUNTIME_DIR:-}/fun-voice-ryan/poc-report.json"

# Source artifacts (validated up front so a missing file fails before any write).
FCITX_SO="${ROOT}/build/fcitx/fcitx5-fun-voice.so"
FCITX_CONF="${ROOT}/native/fcitx5-fun-voice/fcitx5-fun-voice.conf"
DESKTOP_SRC="${ROOT}/systemd/fun-voice-session.desktop"
FCITX_LIB_ABS="${FCITX_LIB_DIR}/fcitx5-fun-voice"  # no ".so"; fcitx5 appends it
log() { printf '[install-user] %s\n' "$*"; }
die() { printf '[install-user] ERROR(%s): %s\n' "$1" "$2" >&2; exit 1; }

# install_file SRC DEST MODE — copy with an explicit mode, creating the parent
# dir and replacing a pre-existing symlink with a real file (Task 6 may have
# left unit symlinks pointing back into the repo).
install_file() {
    local src="$1" dest="$2" mode="$3" dir
    dir="$(dirname "${dest}")"
    mkdir -p "${dir}" || die "mkdir" "cannot create ${dir}"
    if [[ -L "${dest}" ]]; then
        rm -f "${dest}" || die "rm" "cannot remove stale symlink ${dest}"
    fi
    install -m "${mode}" "${src}" "${dest}" \
        || die "copy" "cannot install ${src} -> ${dest}"
}

# --- 0. Hard gate: XPU POC must be ready -----------------------------------
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
    die "precondition" "XDG_RUNTIME_DIR is not set or does not exist"
fi
if [[ ! -f "${POC_REPORT}" ]]; then
    die "precondition" \
        "XPU POC report missing: ${POC_REPORT}. Run scripts/run-nano-xpu-poc.sh first."
fi
POC_READY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ready"))' "${POC_REPORT}" 2>/dev/null || true)"
if [[ "${POC_READY}" != "True" ]]; then
    die "precondition" \
        "XPU POC report not ready (ready=${POC_READY:-missing}); refusing to install."
fi
log "XPU POC report ready=true — proceeding"

# --- 0b. Source validation (fail fast, before any write) -------------------
for script in "${CONSOLE_SCRIPTS[@]}"; do
    [[ -f "${ROOT}/.venv/bin/${script}" ]] \
        || die "source" "console script missing: ${ROOT}/.venv/bin/${script}"
done
for unit in "${SYSTEMD_UNITS[@]}"; do
    [[ -f "${ROOT}/systemd/${unit}" ]] \
        || die "source" "systemd unit missing: ${ROOT}/systemd/${unit}"
done
[[ -f "${FCITX_SO}" ]] \
    || die "source" "fcitx addon .so missing: ${FCITX_SO} (build it first)"
[[ -f "${FCITX_CONF}" ]] || die "source" "fcitx addon conf missing: ${FCITX_CONF}"
[[ -f "${DESKTOP_SRC}" ]] || die "source" "desktop entry missing: ${DESKTOP_SRC}"
log "all source artifacts present"

# --- 1. Console scripts -> ~/.local/bin ------------------------------------
mkdir -p "${BIN_DIR}" || die "mkdir" "cannot create ${BIN_DIR}"
for script in "${CONSOLE_SCRIPTS[@]}"; do
    src="${ROOT}/.venv/bin/${script}"
    install_file "${src}" "${BIN_DIR}/${script}" 755
done
log "installed console scripts into ${BIN_DIR}"

# --- 2. systemd units -> ~/.config/systemd/user ----------------------------
for unit in "${SYSTEMD_UNITS[@]}"; do
    src="${ROOT}/systemd/${unit}"
    install_file "${src}" "${SYSTEMD_USER_DIR}/${unit}" 644
done
log "installed systemd units into ${SYSTEMD_USER_DIR}"

# --- 3. Fcitx addon (.so + .conf) ------------------------------------------
install_file "${FCITX_SO}" "${FCITX_LIB_DIR}/fcitx5-fun-voice.so" 644
FCITX_CONF_TMP="$(mktemp)"
sed "s|^Library=.*|Library=${FCITX_LIB_ABS}|" "${FCITX_CONF}" > "${FCITX_CONF_TMP}"
install_file "${FCITX_CONF_TMP}" "${FCITX_ADDON_DIR}/fcitx5-fun-voice.conf" 644
rm -f "${FCITX_CONF_TMP}"
log "installed fcitx addon (${FCITX_LIB_DIR}, ${FCITX_ADDON_DIR})"

# --- 4. Autostart desktop entry --------------------------------------------
DESKTOP_TMP="$(mktemp)"
sed "s|@REPO_ROOT@|${ROOT}|g" "${DESKTOP_SRC}" > "${DESKTOP_TMP}"
install_file "${DESKTOP_TMP}" "${AUTOSTART_DIR}/fun-voice-session.desktop" 644
rm -f "${DESKTOP_TMP}"
log "installed autostart desktop entry into ${AUTOSTART_DIR}"

# --- 5. systemd daemon-reload + enable --now -------------------------------
if ! command -v systemctl >/dev/null 2>&1; then
    die "systemd" "systemctl not found on PATH"
fi
systemctl --user daemon-reload || die "systemd" "daemon-reload failed"
for service in "${SYSTEMD_SERVICES[@]}"; do
    systemctl --user enable --now "${service}" \
        || die "systemd" "enable --now ${service} failed"
done
log "enabled and started ${SYSTEMD_SERVICES[*]}"

# --- 6. Register the Super+C DDE shortcut ----------------------------------
if [[ -f "${SHORTCUT_ID_FILE}" ]]; then
    log "DDE Super+C shortcut already registered; skipping"
else
    FUN_VOICE_BRIDGE_ACTION="${BIN_DIR}/fun-voice-bridge" \
        bash "${ROOT}/scripts/register-dde-shortcut.sh" \
        || die "dde" "register-dde-shortcut.sh failed (Super+C registration)"
    log "registered DDE Super+C shortcut"
fi

log "installation complete. Verify with: fun-voice-selftest --format json"
