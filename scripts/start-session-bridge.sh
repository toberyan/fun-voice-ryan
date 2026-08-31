#!/usr/bin/env bash
# start-session-bridge.sh — Fun Voice Ryan graphical-session bridge.
#
# Autostarted once per login from fun-voice-session.desktop. The DDE session
# provides DISPLAY/XAUTHORITY/DBUS_SESSION_BUS_ADDRESS; this script imports them
# into the systemd user manager (so the daemon unit can reach X11), ensures the
# daemon unit is running, then runs the bridge once to translate the current
# C-key state into a daemon request.
#
# A bridge connection failure exits non-zero but never retries (no restart
# storm); the autostart entry is one-shot and the daemon is the idempotency
# boundary.

set -euo pipefail

log()  { printf '[fun-voice-session] %s\n' "$*"; }
warn() { printf '[fun-voice-session] WARNING: %s\n' "$*" >&2; }

# --- 1. Import the graphical session environment into systemd --user --------
# The daemon unit needs DISPLAY/XAUTHORITY for X11 focus and C-key queries; the
# user manager runs detached from the session and must be told about them.
if command -v systemctl >/dev/null 2>&1; then
    systemctl --user import-environment DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS \
        || warn "could not import session environment into systemd --user"
fi

# --- 2. Fallback check: refuse to proceed without a display ------------------
if [[ -z "${DISPLAY:-}" ]]; then
    warn "DISPLAY is not set; cannot reach the X11 server"
    exit 1
fi

# --- 3. Ensure the daemon unit is running (idempotent) -----------------------
if command -v systemctl >/dev/null 2>&1; then
    systemctl --user start fun-voice-daemon.service \
        || warn "could not start fun-voice-daemon.service"
else
    warn "systemctl unavailable; the daemon unit will not be started"
fi

# --- 4. Run the bridge once --------------------------------------------------
# fun-voice-bridge reads the live C-key state and sends start_if_idle/stop to
# the daemon. It exits non-zero on any connection/X11 failure and never retries.
if command -v fun-voice-bridge >/dev/null 2>&1; then
    exec fun-voice-bridge
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/fun-voice-bridge" ]]; then
    exec "${ROOT}/.venv/bin/fun-voice-bridge"
fi
warn "fun-voice-bridge not found; is the installation complete?"
exit 1
