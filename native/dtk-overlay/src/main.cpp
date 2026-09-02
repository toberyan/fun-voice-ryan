#include "overlay_window.h"
#include "protocol.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QSocketNotifier>
#include <QTimer>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <optional>

#include <unistd.h>

using namespace fun_voice_overlay;

namespace {

constexpr int kIdleExitMilliseconds = 5000;
constexpr double kMinimumVerticalCenterRatio = 0.50;
constexpr double kMaximumVerticalCenterRatio = 0.85;
constexpr int kMinimumWidthPx = 420;
constexpr int kMaximumWidthPx = 1000;
constexpr double kMinimumFontScale = 0.80;
constexpr double kMaximumFontScale = 1.80;

std::optional<OverlayLayout> parseLayout(const QStringList &arguments) {
    QCommandLineParser parser;
    parser.addOption(QCommandLineOption(
        QStringLiteral("vertical-center-ratio"),
        QStringLiteral("vertical center ratio"), QStringLiteral("ratio"),
        QStringLiteral("0.70")));
    parser.addOption(QCommandLineOption(
        QStringLiteral("width-px"), QStringLiteral("overlay width in pixels"),
        QStringLiteral("pixels"), QStringLiteral("680")));
    parser.addOption(QCommandLineOption(
        QStringLiteral("font-scale"), QStringLiteral("font scale"),
        QStringLiteral("scale"), QStringLiteral("1.0")));
    if (!parser.parse(arguments)) {
        return std::nullopt;
    }
    bool ratioOk = false;
    bool widthOk = false;
    bool scaleOk = false;
    const double ratio =
        parser.value(QStringLiteral("vertical-center-ratio")).toDouble(&ratioOk);
    const int width = parser.value(QStringLiteral("width-px")).toInt(&widthOk);
    const double scale = parser.value(QStringLiteral("font-scale")).toDouble(&scaleOk);
    if (!ratioOk || !widthOk || !scaleOk || !std::isfinite(ratio) ||
        !std::isfinite(scale) || ratio < kMinimumVerticalCenterRatio ||
        ratio > kMaximumVerticalCenterRatio || width < kMinimumWidthPx ||
        width > kMaximumWidthPx || scale < kMinimumFontScale ||
        scale > kMaximumFontScale) {
        return std::nullopt;
    }
    return OverlayLayout{ratio, width, scale};
}

class OverlayApplication final : public QObject {
public:
    explicit OverlayApplication(OverlayLayout layout) : window_(layout) {
        notifier_.setEnabled(true);
        QObject::connect(&notifier_, &QSocketNotifier::activated, this,
                         [this] { readStdin(); });
        idleExit_.setSingleShot(true);
        QObject::connect(&idleExit_, &QTimer::timeout, this, [] {
            QCoreApplication::quit();
        });
        QObject::connect(qApp, &QCoreApplication::aboutToQuit, this,
                         [this] { window_.clearModel(); });
        writeReply(ReplyCode::Ready);
    }

private:
    void readStdin() {
        char buffer[4096];
        const ssize_t size = ::read(STDIN_FILENO, buffer, sizeof(buffer));
        if (size == 0) {
            window_.clearModel();
            QCoreApplication::quit();
            return;
        }
        if (size < 0) {
            if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                window_.clearModel();
                QCoreApplication::quit();
            }
            return;
        }

        if (decoder_.append(QByteArray(buffer, static_cast<int>(size))) !=
            FrameError::None) {
            writeReply(ReplyCode::ErrorFrame);
            return;
        }
        for (const QByteArray &payload : decoder_.takeFrames()) {
            OverlayCommand command;
            QString error;
            if (!parseCommand(payload, &command, &error)) {
                writeReply(ReplyCode::ErrorCommand);
                continue;
            }
            handleCommand(command);
        }
    }

    void handleCommand(const OverlayCommand &command) {
        switch (command.kind) {
        case OverlayCommand::Kind::Show:
            idleExit_.stop();
            window_.showModel(command);
            break;
        case OverlayCommand::Kind::Clear:
            window_.clearModel();
            idleExit_.start(kIdleExitMilliseconds);
            break;
        case OverlayCommand::Kind::Shutdown:
            window_.clearModel();
            QCoreApplication::quit();
            break;
        }
    }

    static void writeReply(ReplyCode code) {
        const QByteArray reply = encodeReply(code);
        const char *data = reply.constData();
        qsizetype remaining = reply.size();
        while (remaining > 0) {
            const ssize_t written = ::write(STDOUT_FILENO, data,
                                            static_cast<size_t>(remaining));
            if (written > 0) {
                data += written;
                remaining -= written;
                continue;
            }
            if (written < 0 && errno == EINTR) {
                continue;
            }
            return;
        }
    }

    FrameDecoder decoder_;
    OverlayWindow window_;
    QSocketNotifier notifier_{STDIN_FILENO, QSocketNotifier::Read};
    QTimer idleExit_;
};

}  // namespace

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    app.setQuitOnLastWindowClosed(false);
    const std::optional<OverlayLayout> layout =
        parseLayout(QCoreApplication::arguments());
    if (!layout.has_value()) {
        return 2;
    }
    OverlayApplication controller(*layout);
    return app.exec();
}
