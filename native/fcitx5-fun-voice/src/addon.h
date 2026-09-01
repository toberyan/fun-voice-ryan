// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Focus-safe protocol engine for the Fun Voice Ryan Fcitx5 addon.
//
// This header is deliberately free of any fcitx dependency so the protocol
// state machine can be unit-tested without a running fcitx instance.
#pragma once

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fun_voice {

/// Maximum wire size (bytes) of a single Fcitx frame, matching
/// ``fun_voice.contracts.MAX_MESSAGE_BYTES``.
constexpr std::size_t kMaxFrameBytes = 64 * 1024;

/// Maximum total buffered text (bytes) of a multi-chunk COMMIT, matching
/// ``fun_voice.contracts.WORKER_RESPONSE_MAX_BYTES``. A single session holds
/// at most one buffer, so this bound keeps memory use predictable.
constexpr std::size_t kMaxBufferedBytes = 4 * 1024 * 1024;

/// Abstraction over the live fcitx input-context state so the protocol engine
/// can be tested with a mock instead of a real fcitx instance.
class FocusBridge {
public:
    virtual ~FocusBridge() = default;

    /// Hex-encoded UUID of the currently focused input context, or "" if none.
    virtual std::string currentFocusedUuid() const = 0;

    /// Whether the context identified by ``uuid`` still exists and has focus.
    virtual bool isFocused(const std::string &uuid) const = 0;

    /// Commit ``text`` into the context identified by ``uuid``.
    /// Returns false when the context no longer exists.
    virtual bool commit(const std::string &uuid, const std::string &text) = 0;
};

/// Stateful implementation of the daemon <-> addon protocol.
///
/// Handles ``PING``, ``START_FOCUS`` and ``COMMIT`` frames and returns the
/// single-line reply to send back. A multi-chunk ``COMMIT`` is buffered in
/// memory and committed atomically only once the final chunk of a complete,
/// in-order sequence arrives, so a partial transcription never reaches
/// ``commitString``. The token -> input-context mapping is kept short-lived:
/// a token (and any buffered text) is dropped when its context loses focus
/// (``dropContext``), when the daemon disconnects (``clear``), as soon as its
/// final chunk is committed, or when a new token supersedes it.
class ProtocolEngine {
public:
    explicit ProtocolEngine(FocusBridge *bridge) : bridge_(bridge) {}

    /// Process one complete request frame (payload, without transport framing)
    /// and return the reply line.
    std::string handle(const std::string &frame) {
        if (frame.size() > kMaxFrameBytes) {
            return "ERROR too-large";
        }
        const auto newline = frame.find('\n');
        const std::string header =
            newline == std::string::npos ? frame : frame.substr(0, newline);
        if (header == "PING") {
            return "PONG";
        }
        if (header == "START_FOCUS") {
            return handleStartFocus();
        }
        if (header.rfind("COMMIT ", 0) == 0) {
            if (newline == std::string::npos) {
                return "ERROR malformed";
            }
            return handleCommit(header, frame.substr(newline + 1));
        }
        return "ERROR unsupported";
    }

    /// Drop every outstanding token (daemon disconnect / addon shutdown).
    void clear() { tokens_.clear(); }

    /// Drop tokens bound to a context that lost focus or was destroyed.
    void dropContext(const std::string &uuid) {
        for (auto it = tokens_.begin(); it != tokens_.end();) {
            if (it->second.uuid == uuid) {
                it = tokens_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::size_t tokenCount() const { return tokens_.size(); }

private:
    struct Session {
        std::string uuid;
        int nextSeq = 1;
        int total = 0;
        std::string buffer; // ordered text accumulated until the final chunk
    };

    std::string handleStartFocus() {
        const std::string uuid = bridge_->currentFocusedUuid();
        if (uuid.empty()) {
            return "REJECT no-input-context";
        }
        // A new recording session supersedes any earlier token for the same
        // context, so a leaked token can never commit into a later session.
        dropContext(uuid);
        const std::string token = generateToken();
        tokens_.emplace(token, Session{uuid, 1, 0});
        return "FOCUS " + token;
    }

    std::string handleCommit(const std::string &header,
                             const std::string &text) {
        const auto parts = splitWhitespace(header);
        if (parts.size() != 4) {
            return "ERROR malformed";
        }
        int sequence = 0;
        int total = 0;
        if (!parseUint(parts[2], sequence) || !parseUint(parts[3], total)) {
            return "ERROR malformed";
        }
        if (sequence < 1 || total < 1 || sequence > total) {
            return "ERROR bad-sequence";
        }
        auto it = tokens_.find(parts[1]);
        if (it == tokens_.end()) {
            return "REJECT stale-focus";
        }
        Session &session = it->second;
        if (sequence != session.nextSeq) {
            return "ERROR bad-sequence";
        }
        if (session.total != 0 && total != session.total) {
            return "ERROR bad-sequence";
        }
        // The token must still name the same input context that was focused
        // when it was issued, and that context must still hold focus.
        if (!bridge_->isFocused(session.uuid)) {
            tokens_.erase(it);
            return "REJECT stale-focus";
        }
        if (session.total == 0) {
            session.total = total;
        }

        if (total == 1) {
            // Single-chunk commit keeps its immediate behavior.
            if (!bridge_->commit(session.uuid, text)) {
                tokens_.erase(it);
                return "REJECT stale-focus";
            }
            tokens_.erase(it); // token is spent
            return "OK";
        }

        // Multi-chunk commit: buffer ordered chunks and commit only once the
        // final chunk arrives. A mid-stream failure (focus change, disconnect,
        // malformed frame) drops the buffer without committing, so a partial
        // transcription is never injected. The accumulated buffer is bounded
        // by the 4 MiB buffer limit to keep memory use predictable.
        if (session.buffer.size() + text.size() > kMaxBufferedBytes) {
            tokens_.erase(it);
            return "ERROR too-large";
        }
        session.buffer.append(text);
        session.nextSeq = sequence + 1;

        if (sequence == total) {
            // Final chunk with every chunk 1..total present in order: commit
            // the concatenated text atomically.
            if (!bridge_->commit(session.uuid, session.buffer)) {
                tokens_.erase(it);
                return "REJECT stale-focus";
            }
            tokens_.erase(it); // final chunk: token is spent, buffer released
            return "OK";
        }
        return "OK";
    }

    static bool parseUint(const std::string &text, int &out) {
        if (text.empty()) {
            return false;
        }
        int value = 0;
        for (char c : text) {
            if (c < '0' || c > '9') {
                return false;
            }
            const int digit = c - '0';
            if (value > (INT_MAX - digit) / 10) {
                return false;
            }
            value = value * 10 + digit;
        }
        out = value;
        return true;
    }

    static std::vector<std::string> splitWhitespace(const std::string &text) {
        std::vector<std::string> parts;
        std::istringstream stream(text);
        std::string part;
        while (stream >> part) {
            parts.push_back(part);
        }
        return parts;
    }

    /// A 128-bit random token, hex-encoded as 32 lowercase characters.
    std::string generateToken() const {
        std::random_device source;
        std::array<unsigned char, 16> bytes{};
        for (auto &byte : bytes) {
            byte = static_cast<unsigned char>(source() & 0xffu);
        }
        static const char hex[] = "0123456789abcdef";
        std::string token;
        token.reserve(32);
        for (unsigned char byte : bytes) {
            token.push_back(hex[byte >> 4]);
            token.push_back(hex[byte & 0x0f]);
        }
        return token;
    }

    FocusBridge *bridge_;
    std::unordered_map<std::string, Session> tokens_;
};

} // namespace fun_voice
