#!/usr/bin/env bash
# Import the graphical X11 session into systemd --user, then restart daemon.
#
# This is a login-time environment handoff only. Hotkey press/release events
# are owned by the daemon's X11 listener; this script never sends daemon IPC.

set -euo pipefail

log()  { printf '[fun-voice-session] %s\n' "$*"; }
warn() { printf '[fun-voice-session] WARNING: %s\n' "$*" >&2; }

if [[ -z "${DISPLAY:-}" ]]; then
    warn "DISPLAY is not set; cannot reach the X11 server"
    exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl unavailable; cannot restart fun-voice-daemon.service"
    exit 1
fi

systemctl --user import-environment DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS \
    || { warn "could not import session environment into systemd --user"; exit 1; }
systemctl --user restart fun-voice-daemon.service \
    || { warn "could not restart fun-voice-daemon.service"; exit 1; }
log "imported X11 session environment and restarted daemon"
