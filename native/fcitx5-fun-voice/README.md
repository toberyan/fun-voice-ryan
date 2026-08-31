# fcitx5-fun-voice

Fcitx5 addon that exposes a private Unix socket for **focus-safe text commit**.
The voice daemon obtains an unpredictable focus token bound to the currently
focused input context, then commits transcription text only while that same
context still holds focus. Commits go through `InputContext::commitString`;
keyboard events are never simulated.

## Build

Requires the `Fcitx5Core` development package (pkg-config name `Fcitx5Core`).

```bash
cmake -S native/fcitx5-fun-voice -B build/fcitx
cmake --build build/fcitx
ctest --test-dir build/fcitx --output-on-failure
```

## Install

Real installation is out of scope for this task; the addon is built and tested
only. When installing manually, the expected locations are:

- `~/.local/lib/fcitx5/fcitx5-fun-voice.so`
- `~/.local/share/fcitx5/addon/fcitx5-fun-voice.conf`

The `Library=` field must be an **absolute** path (without the `.so` suffix):
Deepin 25's fcitx5 does not resolve a bare relative `Library=` against
`~/.local/lib/fcitx5`. `scripts/install-user.sh` substitutes `@FCITX_LIB@` with
the absolute path at install time.

## Runtime socket

On startup the addon creates `$XDG_RUNTIME_DIR/fun-voice-ryan-fcitx.sock`:

- An existing path is removed only after `lstat` confirms it is a socket owned
  by the current user; anything else (regular file, foreign owner) is refused.
- After `bind`, the socket is `chmod 0600`.

## Wire protocol

Every message in both directions is length-prefixed with a 4-byte big-endian
unsigned length, followed by that many payload bytes (a payload of 64 KiB or
more is rejected). The payload itself matches `fun_voice.contracts`:

```
daemon -> addon:  PING
addon  -> daemon:  PONG

daemon -> addon:  START_FOCUS
addon  -> daemon:  FOCUS <128-bit-hex-token>
                  | REJECT no-input-context

daemon -> addon:  COMMIT <focus-token> <sequence> <total>\n<utf8-text>
addon  -> daemon:  OK
                  | REJECT stale-focus
                  | REJECT no-input-context
                  | ERROR <code>
```

`<utf8-text>` may contain newlines and is preserved verbatim (the length prefix
makes framing unambiguous). A `COMMIT` is at most 64 KiB; longer text is split
by the daemon into ordered chunks of at most 8 KiB on Unicode boundaries, each
carrying its `sequence`/`total` and the same focus token. The addon enforces
strict ordering: any out-of-order, duplicated, oversized, or malformed frame is
rejected without committing, and any reject stops the daemon from sending the
remaining chunks. Unknown commands return `ERROR unsupported`.

## Privacy

The addon never logs a `COMMIT` payload, transcription text, or focus tokens.
Diagnostic logs are limited to socket/connection lifecycle and rejection
reasons. Tokens are held only in short-lived memory and are cleared on focus
change, input-context destruction, and daemon disconnect.
