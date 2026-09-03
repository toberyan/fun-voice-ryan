#!/usr/bin/env bash
# install-user.sh — install Fun Voice Ryan into the current user's home.
#
# Hard gate: a private, validated portable runtime selection must identify the
# interpreter used by every installed launcher before anything is written.
#
# Every write target is user-scoped (home or XDG_RUNTIME_DIR); the script never
# uses sudo and never touches system-wide paths. Each step is idempotent (safe
# to re-run) and reversible via uninstall-user.sh.
#
# Steps:
#   1. Selection-aware launchers  -> ~/.local/bin
#   2. systemd units              -> ~/.config/systemd/user  (replaces symlinks)
#   3. Fcitx addon .so + .conf    -> ~/.local/lib/fcitx5 and ~/.local/share/fcitx5/addon
#   4. DTK overlay executable     -> ~/.local/lib/fun-voice-ryan
#   5. Autostart desktop entry    -> ~/.config/autostart
#   6. Safely retire a verified legacy DDE bridge shortcut (upgrade only)
#   7. Retire the warm worker unit, then enable/restart only the lightweight daemon

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# --- User-scoped target paths ----------------------------------------------
BIN_DIR="${HOME}/.local/bin"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
FCITX_LIB_DIR="${HOME}/.local/lib/fcitx5"
FCITX_ADDON_DIR="${HOME}/.local/share/fcitx5/addon"
OVERLAY_INSTALL_DIR="${HOME}/.local/lib/fun-voice-ryan"
AUTOSTART_DIR="${HOME}/.config/autostart"
LEGACY_SHORTCUT_ID_FILE="${XDG_CONFIG_HOME:-${HOME}/.config}/fun-voice-ryan/dde-shortcut-id"
CONSOLE_SCRIPTS=(
    fun-voice-daemon fun-voice-worker fun-voice-preflight fun-voice-selftest
    fun-voice-corrector fun-voice-benchmark
)
SYSTEMD_UNITS=(fun-voice-worker@.service fun-voice-daemon.service)

RUNTIME_SELECTION="${XDG_DATA_HOME:-${HOME}/.local/share}/fun-voice-ryan/runtime/selection.json"

# Source artifacts (validated up front so a missing file fails before any write).
FCITX_SO="${ROOT}/build/fcitx/fcitx5-fun-voice.so"
FCITX_CONF="${ROOT}/native/fcitx5-fun-voice/fcitx5-fun-voice.conf"
OVERLAY_BIN="${ROOT}/build/dtk-overlay/fun-voice-overlay"
DESKTOP_SRC="${ROOT}/systemd/fun-voice-session.desktop"
FCITX_LIB_ABS="${FCITX_LIB_DIR}/fcitx5-fun-voice"  # no ".so"; fcitx5 appends it
log() { printf '[install-user] %s\n' "$*"; }
die() { printf '[install-user] ERROR(%s): %s\n' "$1" "$2" >&2; exit 1; }

if [[ "$#" -eq 0 ]]; then
    :
elif [[ "$#" -eq 2 && "$1" == "--runtime-selection" && -n "$2" ]]; then
    RUNTIME_SELECTION="$2"
else
    die "usage" "expected --runtime-selection PATH"
fi

# Retire only the DDE shortcut created by an older Fun Voice Ryan release.
# A persisted id alone is not authority to remove a global shortcut: verify
# that DDE still maps it to the old bridge wrapper before deleting it.
retire_legacy_dde_shortcut() {
    if [[ ! -e "${LEGACY_SHORTCUT_ID_FILE}" ]]; then
        return
    fi
    if [[ ! -f "${LEGACY_SHORTCUT_ID_FILE}" ]]; then
        die "legacy-dde" "legacy shortcut id is not a regular file; remove it manually"
    fi

    local -a legacy_ids
    mapfile -t legacy_ids < "${LEGACY_SHORTCUT_ID_FILE}"
    if [[ "${#legacy_ids[@]}" -ne 1 || -z "${legacy_ids[0]}" ]]; then
        die "legacy-dde" "legacy shortcut id file is malformed; remove its shortcut manually"
    fi
    local shortcut_id="${legacy_ids[0]}"
    if ! command -v busctl >/dev/null 2>&1; then
        die "legacy-dde" "busctl is required to verify the legacy shortcut"
    fi

    local legacy_reply legacy_action
    if ! legacy_reply="$(busctl --user call \
        org.deepin.dde.Keybinding1 /org/deepin/dde/Keybinding1 \
        org.deepin.dde.Keybinding1 GetShortcutCommand s "${shortcut_id}")"; then
        die "legacy-dde" "cannot read the legacy shortcut command; remove it manually"
    fi
    if [[ ! "${legacy_reply}" =~ ^s[[:space:]]+\".*\"$ ]]; then
        die "legacy-dde" "legacy shortcut command is not recognizable; remove it manually"
    fi
    legacy_action="${legacy_reply#s }"
    legacy_action="${legacy_action#\"}"
    legacy_action="${legacy_action%\"}"
    if [[ -z "${legacy_action}" || "${legacy_action}" == *\"* || "${legacy_action}" == *\\* \
        || ( "${legacy_action}" != "fun-voice-bridge" \
            && "${legacy_action}" != */fun-voice-bridge ) ]]; then
        die "legacy-dde" "legacy shortcut is not owned by Fun Voice Ryan; remove it manually"
    fi

    busctl --user call org.deepin.dde.Keybinding1 /org/deepin/dde/Keybinding1 \
        org.deepin.dde.Keybinding1 DeleteCustomShortcut s "${shortcut_id}" >/dev/null \
        || die "legacy-dde" "cannot delete verified legacy shortcut"
    rm -f "${LEGACY_SHORTCUT_ID_FILE}" \
        || die "legacy-dde" "cannot clear verified legacy shortcut id"
    log "retired verified legacy DDE shortcut"
}

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

# --- 0. Hard gate: validate selection and selected environment -------------
if [[ ! -e "${RUNTIME_SELECTION}" && ! -L "${RUNTIME_SELECTION}" ]]; then
    die "runtime_selection_invalid" "portable runtime selection is unavailable"
fi
SELECTION_PYTHON="$(PYTHONPATH="${ROOT}/src" python3 -P -c '
from pathlib import Path
import sys
from fun_voice.runtime_selection import RuntimeSelectionError, load_runtime_selection, selection_path
manifest = Path(sys.argv[1])
if not manifest.is_absolute() or manifest.name != "selection.json" or manifest.parent.name != "runtime":
    raise SystemExit(2)
root = manifest.parent.parent
if selection_path(root) != manifest:
    raise SystemExit(2)
try:
    selection = load_runtime_selection(root)
except RuntimeSelectionError:
    raise SystemExit(2) from None
print(selection.python)
' "${RUNTIME_SELECTION}" 2>/dev/null)" \
    || die "runtime_selection_invalid" "portable runtime selection is invalid"

RUNTIME_IMPORT_CHECK='from pathlib import Path; import sys; from fun_voice.runtime_selection import load_runtime_selection; manifest = Path(sys.argv[1]); selection = load_runtime_selection(manifest.parent.parent); assert Path(sys.executable).resolve() == selection.python.resolve(); import torch, funasr, modelscope, transformers, Xlib'
if ! PYTHONPATH="${ROOT}/src" "${SELECTION_PYTHON}" -P -c \
    "${RUNTIME_IMPORT_CHECK}" "${RUNTIME_SELECTION}" >/dev/null 2>&1; then
    die "runtime_import_failed" "selected runtime imports are unavailable"
fi
log "portable runtime selection and imports verified"

if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
    die "precondition" "XDG_RUNTIME_DIR is not set or does not exist"
fi

# --- 0b. Source validation (fail fast, before any write) -------------------
[[ -f "${ROOT}/scripts/run-selected-runtime.sh" ]] \
    || die "source" "selected runtime adapter is missing"
for unit in "${SYSTEMD_UNITS[@]}"; do
    [[ -f "${ROOT}/systemd/${unit}" ]] \
        || die "source" "systemd unit missing: ${ROOT}/systemd/${unit}"
done
[[ -f "${FCITX_SO}" ]] \
    || die "source" "fcitx addon .so missing: ${FCITX_SO} (build it first)"
[[ -f "${FCITX_CONF}" ]] || die "source" "fcitx addon conf missing: ${FCITX_CONF}"
[[ -f "${OVERLAY_BIN}" ]] \
    || die "source" "DTK overlay binary missing: ${OVERLAY_BIN} (build it first)"
[[ -f "${DESKTOP_SRC}" ]] || die "source" "desktop entry missing: ${DESKTOP_SRC}"
log "all source artifacts present"

# --- 1. Selection-aware launchers -> ~/.local/bin --------------------------
install_launcher() {
    local name="$1"
    local target="${BIN_DIR}/${name}"
    install -d -m 700 "${BIN_DIR}"
    umask 077
    rm -f "${target}.tmp"
    printf '%s\n' '#!/usr/bin/env bash' \
        "exec \"${ROOT}/scripts/run-selected-runtime.sh\" \"${name}\" \"\$@\"" \
        > "${target}.tmp"
    chmod 700 "${target}.tmp"
    mv -f "${target}.tmp" "${target}"
}

for script in "${CONSOLE_SCRIPTS[@]}"; do
    install_launcher "${script}"
done
log "installed selected-runtime launchers into ${BIN_DIR}"

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

# --- 4. Private DTK overlay executable -------------------------------------
install_file "${OVERLAY_BIN}" "${OVERLAY_INSTALL_DIR}/fun-voice-overlay" 755
log "installed private DTK overlay binary into ${OVERLAY_INSTALL_DIR}"

# --- 5. Autostart desktop entry --------------------------------------------
DESKTOP_TMP="$(mktemp)"
sed "s|@REPO_ROOT@|${ROOT}|g" "${DESKTOP_SRC}" > "${DESKTOP_TMP}"
install_file "${DESKTOP_TMP}" "${AUTOSTART_DIR}/fun-voice-session.desktop" 644
rm -f "${DESKTOP_TMP}"
log "installed autostart desktop entry into ${AUTOSTART_DIR}"

# --- 6. Safely retire the old DDE bridge shortcut (upgrade only) -----------
retire_legacy_dde_shortcut
if [[ -f "${BIN_DIR}/fun-voice-bridge" || -L "${BIN_DIR}/fun-voice-bridge" ]]; then
    rm -f "${BIN_DIR}/fun-voice-bridge" \
        || die "legacy-bridge" "cannot remove obsolete bridge wrapper"
    log "removed obsolete bridge wrapper"
fi

# --- 7. Retire warm worker + defer daemon startup to the X11 session -------
if ! command -v systemctl >/dev/null 2>&1; then
    die "systemd" "systemctl not found on PATH"
fi
# ``fun-voice-worker.service`` belongs to the old warm architecture. It is an
# exact application-owned path, not a glob or user-selected target.
systemctl --user disable --now fun-voice-worker.service 2>/dev/null || true
rm -f "${SYSTEMD_USER_DIR}/fun-voice-worker.service" \
    || die "systemd" "cannot remove retired warm worker unit"
systemctl --user daemon-reload || die "systemd" "daemon-reload failed"
# Starting here is too early: the user manager can run before the graphical
# session has supplied DISPLAY/XAUTHORITY.  The autostart session importer
# performs the explicit start only after it has imported those values.
systemctl --user disable --now fun-voice-daemon.service 2>/dev/null || true
log "retired warm worker; daemon startup is deferred to the X11 session importer"

log "installation complete. Verify with: fun-voice-selftest --format json"
