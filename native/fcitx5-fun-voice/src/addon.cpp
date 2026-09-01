// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Fcitx5 addon that exposes a private Unix socket for focus-safe text commit.
//
// The daemon obtains an unpredictable focus token (bound to the currently
// focused input context) and may then commit text only while that same
// context still holds focus. Commits go through InputContext::commitString;
// keyboard events are never simulated. Multi-chunk commits are buffered and
// committed atomically on the final chunk, so a partial transcription is
// never injected.
//
// Privacy: diagnostic logs never include commit payloads, transcription text,
// or focus tokens.

#include "addon.h"

#include <fcitx-utils/event.h>
#include <fcitx-utils/eventloopinterface.h>
#include <fcitx-utils/handlertable.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/unixfd.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputcontextmanager.h>
#include <fcitx/instance.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace fun_voice {

namespace {

constexpr char kSocketName[] = "fun-voice-ryan-fcitx.sock";
constexpr int kListenBacklog = 8;

std::string uuidToHex(const fcitx::ICUUID &uuid) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    for (unsigned char byte : uuid) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0x0f]);
    }
    return out;
}

bool hexToUuid(const std::string &hex, fcitx::ICUUID &out) {
    if (hex.size() != out.size() * 2) {
        return false;
    }
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return -1;
    };
    for (std::size_t i = 0; i < out.size(); ++i) {
        const int hi = nibble(hex[2 * i]);
        const int lo = nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return true;
}

std::string socketPath() {
    const char *runtime = std::getenv("XDG_RUNTIME_DIR");
    if (!runtime || !*runtime) {
        return {};
    }
    return std::string(runtime) + "/" + kSocketName;
}

bool setCloexec(int fd) {
    const int flags = ::fcntl(fd, F_GETFD);
    return flags >= 0 && ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

bool setNonBlocking(int fd) {
    const int flags = ::fcntl(fd, F_GETFL);
    return flags >= 0 && ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

class FcitxFocusBridge final : public FocusBridge {
public:
    explicit FcitxFocusBridge(fcitx::InputContextManager *manager)
        : manager_(manager) {}

    std::string currentFocusedUuid() const override {
        auto *context = manager_->lastFocusedInputContext();
        if (!context || !context->hasFocus()) {
            return {};
        }
        return uuidToHex(context->uuid());
    }

    bool isFocused(const std::string &uuid) const override {
        auto *context = findContext(uuid);
        return context && context->hasFocus();
    }

    bool commit(const std::string &uuid, const std::string &text) override {
        auto *context = findContext(uuid);
        if (!context) {
            return false;
        }
        context->commitString(text);
        return true;
    }

private:
    fcitx::InputContext *findContext(const std::string &uuid) const {
        fcitx::ICUUID parsed{};
        if (!hexToUuid(uuid, parsed)) {
            return nullptr;
        }
        return manager_->findByUUID(parsed);
    }

    fcitx::InputContextManager *manager_;
};

} // namespace

class FunVoiceAddon;

/// One accepted daemon connection, driven on the fcitx event loop thread so
/// that commitString and focus queries stay single-threaded.
class Connection {
public:
    Connection(fcitx::EventLoop *loop, fcitx::UnixFD fd, FunVoiceAddon *addon);
    ~Connection();

    bool isClosed() const { return closed_; }

    void sendReply(const std::string &line);

private:
    bool onEvent(fcitx::IOEventFlags revents);
    bool markClosed();
    bool readAvailable();
    bool drainFrames();
    bool flush();
    void updateEvents();

    fcitx::EventLoop *loop_;
    fcitx::UnixFD fd_;
    FunVoiceAddon *addon_;
    std::unique_ptr<fcitx::EventSourceIO> ioEvent_;
    std::string readBuffer_;
    std::string sendBuffer_;
    bool closed_ = false;
};

class FunVoiceAddon final : public fcitx::AddonInstance {
public:
    explicit FunVoiceAddon(fcitx::Instance *instance);
    ~FunVoiceAddon() override;

    std::string handleFrame(const std::string &frame);
    void onConnectionClosed();

private:
    void onContextLostFocus(fcitx::Event &event);
    void setupSocket();
    bool onAccept(int listenFd);

    fcitx::Instance *instance_;
    fcitx::InputContextManager &inputContextManager_;
    fcitx::EventLoop *loop_;
    std::unique_ptr<FcitxFocusBridge> bridge_;
    ProtocolEngine engine_;
    std::unique_ptr<fcitx::HandlerTableEntry<fcitx::EventHandler>>
        focusOutWatcher_;
    std::unique_ptr<fcitx::HandlerTableEntry<fcitx::EventHandler>>
        destroyedWatcher_;
    fcitx::UnixFD listenFd_;
    std::unique_ptr<fcitx::EventSourceIO> listenerEvent_;
    std::unique_ptr<Connection> connection_;
};

// --- Connection -------------------------------------------------------------

Connection::Connection(fcitx::EventLoop *loop, fcitx::UnixFD fd,
                       FunVoiceAddon *addon)
    : loop_(loop), fd_(std::move(fd)), addon_(addon) {
    ioEvent_ = loop_->addIOEvent(
        fd_.fd(), fcitx::IOEventFlag::In,
        [this](fcitx::EventSourceIO *, int, fcitx::IOEventFlags flags) {
            return onEvent(flags);
        });
}

Connection::~Connection() { ioEvent_.reset(); }

void Connection::sendReply(const std::string &line) {
    const std::uint32_t length = static_cast<std::uint32_t>(line.size());
    sendBuffer_.push_back(static_cast<char>((length >> 24) & 0xff));
    sendBuffer_.push_back(static_cast<char>((length >> 16) & 0xff));
    sendBuffer_.push_back(static_cast<char>((length >> 8) & 0xff));
    sendBuffer_.push_back(static_cast<char>(length & 0xff));
    sendBuffer_.append(line);
    updateEvents();
}

bool Connection::onEvent(fcitx::IOEventFlags revents) {
    if (revents.test(fcitx::IOEventFlag::Err) ||
        revents.test(fcitx::IOEventFlag::Hup)) {
        return markClosed();
    }
    if (revents.test(fcitx::IOEventFlag::In) && !readAvailable()) {
        return markClosed();
    }
    if (revents.test(fcitx::IOEventFlag::Out) && !flush()) {
        return markClosed();
    }
    updateEvents();
    return true;
}

bool Connection::markClosed() {
    closed_ = true;
    addon_->onConnectionClosed();
    return false;
}

bool Connection::readAvailable() {
    char buffer[4096];
    bool eof = false;
    for (;;) {
        const ssize_t n = ::read(fd_.fd(), buffer, sizeof(buffer));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            return false;
        }
        if (n == 0) {
            eof = true;
            break;
        }
        readBuffer_.append(buffer, static_cast<std::size_t>(n));
        if (static_cast<std::size_t>(n) < sizeof(buffer)) {
            break;
        }
    }
    if (!drainFrames()) {
        return false;
    }
    return !eof;
}

bool Connection::drainFrames() {
    for (;;) {
        if (readBuffer_.size() < 4) {
            return true;
        }
        const std::uint32_t length =
            (static_cast<std::uint32_t>(
                 static_cast<unsigned char>(readBuffer_[0]))
             << 24) |
            (static_cast<std::uint32_t>(
                 static_cast<unsigned char>(readBuffer_[1]))
             << 16) |
            (static_cast<std::uint32_t>(
                 static_cast<unsigned char>(readBuffer_[2]))
             << 8) |
            static_cast<std::uint32_t>(
                static_cast<unsigned char>(readBuffer_[3]));
        if (length > kMaxFrameBytes) {
            // The peer's declared frame size violates the protocol; drop the
            // connection without a reply (its framing cannot be trusted).
            return false;
        }
        if (readBuffer_.size() < 4 + length) {
            return true;
        }
        const std::string payload(readBuffer_, 4, length);
        readBuffer_.erase(0, 4 + length);
        sendReply(addon_->handleFrame(payload));
    }
}

bool Connection::flush() {
    while (!sendBuffer_.empty()) {
        const ssize_t n =
            ::write(fd_.fd(), sendBuffer_.data(), sendBuffer_.size());
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            return false;
        }
        sendBuffer_.erase(0, static_cast<std::size_t>(n));
    }
    return true;
}

void Connection::updateEvents() {
    fcitx::IOEventFlags flags = fcitx::IOEventFlag::In;
    if (!sendBuffer_.empty()) {
        flags |= fcitx::IOEventFlag::Out;
    }
    ioEvent_->setEvents(flags);
}

// --- FunVoiceAddon ----------------------------------------------------------

FunVoiceAddon::FunVoiceAddon(fcitx::Instance *instance)
    : instance_(instance),
      inputContextManager_(instance->inputContextManager()),
      loop_(&instance->eventLoop()),
      bridge_(std::make_unique<FcitxFocusBridge>(&inputContextManager_)),
      engine_(bridge_.get()) {
    focusOutWatcher_ = instance_->watchEvent(
        fcitx::EventType::InputContextFocusOut, fcitx::EventWatcherPhase::Default,
        [this](fcitx::Event &event) { onContextLostFocus(event); });
    destroyedWatcher_ = instance_->watchEvent(
        fcitx::EventType::InputContextDestroyed,
        fcitx::EventWatcherPhase::Default,
        [this](fcitx::Event &event) { onContextLostFocus(event); });
    setupSocket();
}

FunVoiceAddon::~FunVoiceAddon() {
    connection_.reset();
    engine_.clear();
    if (listenFd_.isValid()) {
        ::unlink(socketPath().c_str());
    }
}

std::string FunVoiceAddon::handleFrame(const std::string &frame) {
    return engine_.handle(frame);
}

void FunVoiceAddon::onConnectionClosed() { engine_.clear(); }

void FunVoiceAddon::onContextLostFocus(fcitx::Event &event) {
    auto *inputContextEvent = static_cast<fcitx::InputContextEvent *>(&event);
    engine_.dropContext(uuidToHex(inputContextEvent->inputContext()->uuid()));
}

void FunVoiceAddon::setupSocket() {
    const std::string path = socketPath();
    if (path.empty()) {
        FCITX_WARN()
            << "fun-voice: XDG_RUNTIME_DIR is unset; commit socket disabled";
        return;
    }

    struct stat st {};
    if (::lstat(path.c_str(), &st) == 0) {
        // Only remove an existing socket that we own; never touch a regular
        // file or something owned by another user.
        if (!S_ISSOCK(st.st_mode)) {
            FCITX_WARN() << "fun-voice: refusing to remove non-socket at "
                         << path;
            return;
        }
        if (st.st_uid != ::geteuid()) {
            FCITX_WARN() << "fun-voice: refusing to remove foreign-owned socket "
                            "at "
                         << path;
            return;
        }
        if (::unlink(path.c_str()) != 0) {
            FCITX_WARN() << "fun-voice: failed to remove stale socket at "
                         << path;
            return;
        }
    }

    fcitx::UnixFD fd(
        fcitx::UnixFD::own(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)));
    if (!fd.isValid()) {
        FCITX_WARN() << "fun-voice: socket() failed";
        return;
    }
    if (!setNonBlocking(fd.fd())) {
        FCITX_WARN() << "fun-voice: failed to make socket non-blocking";
        return;
    }

    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        FCITX_WARN() << "fun-voice: socket path too long";
        return;
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(fd.fd(), reinterpret_cast<struct sockaddr *>(&addr),
               sizeof(addr)) != 0) {
        FCITX_WARN() << "fun-voice: bind failed";
        return;
    }
    if (::chmod(path.c_str(), 0600) != 0) {
        FCITX_WARN() << "fun-voice: chmod 0600 failed on " << path;
    }
    if (::listen(fd.fd(), kListenBacklog) != 0) {
        FCITX_WARN() << "fun-voice: listen failed";
        return;
    }

    listenFd_ = std::move(fd);
    listenerEvent_ = loop_->addIOEvent(
        listenFd_.fd(), fcitx::IOEventFlag::In,
        [this](fcitx::EventSourceIO *, int fd, fcitx::IOEventFlags) {
            return onAccept(fd);
        });
    FCITX_INFO() << "fun-voice: listening on " << path;
}

bool FunVoiceAddon::onAccept(int listenFd) {
    for (;;) {
        const int fd = ::accept(listenFd, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return true;
            }
            FCITX_WARN() << "fun-voice: accept failed";
            return true; // keep listening on transient errors
        }
        if (connection_ && !connection_->isClosed()) {
            ::close(fd); // single daemon: refuse extra clients
            continue;
        }
        if (!setCloexec(fd) || !setNonBlocking(fd)) {
            ::close(fd);
            continue;
        }
        // Replacing a previously closed connection is safe here: we are in the
        // listener callback, not in the old connection's callback.
        connection_.reset();
        connection_ =
            std::make_unique<Connection>(loop_, fcitx::UnixFD::own(fd), this);
        FCITX_INFO() << "fun-voice: daemon connected";
    }
}

class FunVoiceFactory : public fcitx::AddonFactory {
public:
    fcitx::AddonInstance *create(fcitx::AddonManager *manager) override {
        return new FunVoiceAddon(manager->instance());
    }
};

} // namespace fun_voice

FCITX_ADDON_FACTORY(fun_voice::FunVoiceFactory)
