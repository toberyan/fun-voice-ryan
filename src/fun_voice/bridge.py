"""Hotkey bridge: translate a DDE trigger into a daemon request.

DDE invokes this console script once per trigger without a press/release event,
so the bridge reads the live X11 ``C`` key state and sends a single-line JSON
request to the daemon, then exits:

- ``C`` held  -> ``{"op": "start_if_idle"}``
- otherwise   -> ``{"op": "stop"}``

Repeated triggers are idempotent because the daemon ignores ``start_if_idle``
while already recording and ignores ``stop`` outside ``RECORDING``. A failed
connection exits non-zero and never retries (no restart storm).
"""

from __future__ import annotations

import argparse
import socket
from collections.abc import Sequence
from pathlib import Path

from fun_voice import config
from fun_voice.desktop import HotkeyBridge, X11Error, X11FocusGuard


def _default_socket_path() -> Path:
    return config.build_runtime_paths(config.resolve_runtime_dir()).daemon_socket


def send_request(guard: X11FocusGuard, socket_path: Path) -> int:
    """Connect to ``socket_path`` and send the request implied by the C key state.

    Returns ``0`` on success and a non-zero exit status on any connection or X11
    failure. Never retries: the daemon is the idempotency boundary, not here.
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(str(socket_path))
            bridge = HotkeyBridge(
                guard, lambda payload: sock.sendall(payload + b"\n")
            )
            bridge.handle()
    except (OSError, X11Error):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fun-voice-bridge",
        description="Forward the current C-key state to the voice daemon.",
    )
    parser.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="override the daemon socket path (defaults to the runtime dir)",
    )
    args = parser.parse_args(argv)

    try:
        socket_path = args.socket if args.socket is not None else _default_socket_path()
    except config.ConfigError:
        return 1

    return send_request(X11FocusGuard(), socket_path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
