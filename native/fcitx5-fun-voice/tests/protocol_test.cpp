// Protocol tests for the Fun Voice Ryan Fcitx5 addon.
//
// Exercises the state machine in addon.h against a mock focus bridge, covering
// the acceptance criteria: PING -> PONG, unknown commands rejected, and stale
// focus tokens, missing input contexts, out-of-order chunks and over-64 KiB
// messages must never commit text.

#include "addon.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

class MockBridge : public fun_voice::FocusBridge {
public:
    std::string focusedUuid = "00112233445566778899aabbccddeeff";
    bool focused = true;
    std::vector<std::string> commits;

    std::string currentFocusedUuid() const override {
        return focused ? focusedUuid : std::string();
    }

    bool isFocused(const std::string &uuid) const override {
        return focused && uuid == focusedUuid;
    }

    bool commit(const std::string &uuid, const std::string &text) override {
        if (!isFocused(uuid)) {
            return false;
        }
        commits.push_back(text);
        return true;
    }
};

std::string commitFrame(const std::string &token, int sequence, int total,
                        const std::string &text) {
    return "COMMIT " + token + " " + std::to_string(sequence) + " " +
           std::to_string(total) + "\n" + text;
}

int failures = 0;

void check(bool condition, const char *what) {
    if (condition) {
        std::cout << "ok - " << what << '\n';
    } else {
        std::cerr << "FAIL - " << what << '\n';
        ++failures;
    }
}

void checkEqual(const std::string &actual, const std::string &expected,
                const char *what) {
    check(actual == expected, what);
    if (actual != expected) {
        std::cerr << "       expected: " << expected << "\n       actual:   "
                  << actual << '\n';
    }
}

} // namespace

int main() {
    // PING -> PONG
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        checkEqual(engine.handle("PING"), "PONG", "PING returns PONG");
    }

    // Unknown command rejected, nothing committed.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        checkEqual(engine.handle("HELLO"), "ERROR unsupported",
                   "unknown command rejected");
        check(bridge.commits.empty(), "unknown command commits nothing");
    }

    // START_FOCUS with no focused input context.
    {
        MockBridge bridge;
        bridge.focused = false;
        fun_voice::ProtocolEngine engine(&bridge);
        checkEqual(engine.handle("START_FOCUS"), "REJECT no-input-context",
                   "START_FOCUS without input context");
    }

    // START_FOCUS issues a 128-bit token; COMMIT commits exactly once.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string reply = engine.handle("START_FOCUS");
        check(reply.rfind("FOCUS ", 0) == 0, "START_FOCUS returns FOCUS");
        check(reply.size() == 6 + 32, "focus token is 128-bit hex");
        const std::string token = reply.substr(6);
        checkEqual(engine.handle(commitFrame(token, 1, 1, "你好 world")), "OK",
                   "single chunk commit accepted");
        check(bridge.commits == std::vector<std::string>{"你好 world"},
              "committed text preserved verbatim");
    }

    // Unknown focus token is rejected and commits nothing.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        checkEqual(engine.handle(commitFrame("deadbeef", 1, 1, "x")),
                   "REJECT stale-focus", "unknown token rejected");
        check(bridge.commits.empty(), "stale token commits nothing");
    }

    // A token whose context lost focus is rejected and commits nothing.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string token = engine.handle("START_FOCUS").substr(6);
        bridge.focused = false; // focus moved elsewhere
        checkEqual(engine.handle(commitFrame(token, 1, 1, "x")),
                   "REJECT stale-focus", "unfocused token rejected");
        check(bridge.commits.empty(), "unfocused token commits nothing");
    }

    // A token whose context was destroyed is rejected and commits nothing.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string token = engine.handle("START_FOCUS").substr(6);
        bridge.focusedUuid = ""; // context gone
        bridge.focused = false;
        checkEqual(engine.handle(commitFrame(token, 1, 1, "x")),
                   "REJECT stale-focus", "destroyed context token rejected");
        check(bridge.commits.empty(), "destroyed context commits nothing");
    }

    // Out-of-order chunk (2 before 1) is rejected and commits nothing.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string token = engine.handle("START_FOCUS").substr(6);
        checkEqual(engine.handle(commitFrame(token, 2, 2, "second")),
                   "ERROR bad-sequence", "out-of-order chunk rejected");
        check(bridge.commits.empty(), "out-of-order chunk commits nothing");
    }

    // Duplicate chunk (sequence replayed) is rejected.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string token = engine.handle("START_FOCUS").substr(6);
        checkEqual(engine.handle(commitFrame(token, 1, 2, "one")), "OK",
                   "first chunk accepted");
        checkEqual(engine.handle(commitFrame(token, 1, 2, "one")),
                   "ERROR bad-sequence", "replayed chunk rejected");
        check(bridge.commits == std::vector<std::string>{"one"},
              "replayed chunk not committed twice");
    }

    // A frame larger than 64 KiB is rejected and commits nothing.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string token = engine.handle("START_FOCUS").substr(6);
        const std::string big(64 * 1024 + 1, 'a');
        checkEqual(engine.handle(commitFrame(token, 1, 1, big)),
                   "ERROR too-large", "over-64 KiB frame rejected");
        check(bridge.commits.empty(), "oversized frame commits nothing");
    }

    // Ordered multi-chunk commit is committed chunk by chunk, in order.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string token = engine.handle("START_FOCUS").substr(6);
        checkEqual(engine.handle(commitFrame(token, 1, 2, "abc")), "OK",
                   "chunk 1 accepted");
        checkEqual(engine.handle(commitFrame(token, 2, 2, "def")), "OK",
                   "chunk 2 accepted");
        check(bridge.commits == std::vector<std::string>{"abc", "def"},
              "chunks committed in order");
        // Token is spent after the final chunk.
        checkEqual(engine.handle(commitFrame(token, 1, 1, "again")),
                   "REJECT stale-focus", "spent token rejected");
    }

    // START_FOCUS supersedes an earlier token for the same context.
    {
        MockBridge bridge;
        fun_voice::ProtocolEngine engine(&bridge);
        const std::string first = engine.handle("START_FOCUS").substr(6);
        const std::string second = engine.handle("START_FOCUS").substr(6);
        check(first != second, "tokens are distinct");
        checkEqual(engine.handle(commitFrame(first, 1, 1, "x")),
                   "REJECT stale-focus", "superseded token rejected");
        checkEqual(engine.handle(commitFrame(second, 1, 1, "y")), "OK",
                   "newest token accepted");
        check(bridge.commits == std::vector<std::string>{"y"},
              "only newest token commits");
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "all protocol tests passed\n";
    return 0;
}
