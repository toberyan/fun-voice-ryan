#pragma once

#include <QByteArray>
#include <QString>

#include <optional>
#include <vector>

namespace fun_voice_overlay {

constexpr int kMaxFrameBytes = 64 * 1024;

enum class FrameError {
    None,
    TooLarge,
    Malformed,
};

class FrameDecoder {
public:
    FrameError append(const QByteArray &bytes);
    std::vector<QByteArray> takeFrames();

private:
    QByteArray buffer_;
    std::vector<QByteArray> frames_;
};

enum class ReplyCode {
    Ready,
    ErrorFrame,
    ErrorCommand,
};

struct OverlayCommand {
    enum class Kind {
        Show,
        Clear,
        Shutdown,
    };

    Kind kind = Kind::Clear;
    QString phase;
    QString stableText;
    QString provisionalText;
    std::optional<int> level;
};

QByteArray encodeFrame(const QByteArray &payload);
QByteArray encodeReply(ReplyCode code);
bool parseCommand(const QByteArray &payload, OverlayCommand *command,
                  QString *error);

}  // namespace fun_voice_overlay
