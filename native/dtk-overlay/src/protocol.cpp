#include "protocol.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include <cmath>

namespace fun_voice_overlay {
namespace {

void setCommandError(QString *error) {
    if (error != nullptr) {
        *error = QStringLiteral("error_command");
    }
}

quint32 decodeLength(const QByteArray &bytes) {
    return (static_cast<quint32>(static_cast<unsigned char>(bytes.at(0))) << 24U) |
           (static_cast<quint32>(static_cast<unsigned char>(bytes.at(1))) << 16U) |
           (static_cast<quint32>(static_cast<unsigned char>(bytes.at(2))) << 8U) |
           static_cast<quint32>(static_cast<unsigned char>(bytes.at(3)));
}

bool requireText(const QJsonObject &object, const char *name, QString *out) {
    const QJsonValue value = object.value(QLatin1String(name));
    if (!value.isString()) {
        return false;
    }
    *out = value.toString();
    return true;
}

}  // namespace

FrameError FrameDecoder::append(const QByteArray &bytes) {
    buffer_.append(bytes);
    while (buffer_.size() >= 4) {
        const quint32 length = decodeLength(buffer_);
        if (length == 0 || length > static_cast<quint32>(kMaxFrameBytes)) {
            buffer_.clear();
            frames_.clear();
            return FrameError::TooLarge;
        }
        const int total = 4 + static_cast<int>(length);
        if (buffer_.size() < total) {
            return FrameError::None;
        }
        frames_.push_back(buffer_.mid(4, static_cast<int>(length)));
        buffer_.remove(0, total);
    }
    return FrameError::None;
}

std::vector<QByteArray> FrameDecoder::takeFrames() {
    std::vector<QByteArray> frames;
    frames.swap(frames_);
    return frames;
}

QByteArray encodeFrame(const QByteArray &payload) {
    const quint32 length = static_cast<quint32>(payload.size());
    QByteArray frame;
    frame.reserve(4 + payload.size());
    frame.append(static_cast<char>((length >> 24U) & 0xffU));
    frame.append(static_cast<char>((length >> 16U) & 0xffU));
    frame.append(static_cast<char>((length >> 8U) & 0xffU));
    frame.append(static_cast<char>(length & 0xffU));
    frame.append(payload);
    return frame;
}

QByteArray encodeReply(ReplyCode code) {
    const char *reply = "error_command";
    switch (code) {
    case ReplyCode::Ready:
        reply = "ready";
        break;
    case ReplyCode::ErrorFrame:
        reply = "error_frame";
        break;
    case ReplyCode::ErrorCommand:
        break;
    }
    return encodeFrame(QJsonDocument(QJsonObject{{"reply", reply}}).toJson(
        QJsonDocument::Compact));
}

bool parseCommand(const QByteArray &payload, OverlayCommand *command,
                  QString *error) {
    if (command == nullptr || payload.isEmpty() || payload.size() > kMaxFrameBytes) {
        setCommandError(error);
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(payload, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        setCommandError(error);
        return false;
    }

    const QJsonObject object = document.object();
    const QJsonValue commandValue = object.value(QStringLiteral("command"));
    if (!commandValue.isString()) {
        setCommandError(error);
        return false;
    }

    const QString commandName = commandValue.toString();
    if (commandName == QStringLiteral("show")) {
        OverlayCommand parsed;
        parsed.kind = OverlayCommand::Kind::Show;
        if (!requireText(object, "phase", &parsed.phase) ||
            !requireText(object, "stable_text", &parsed.stableText) ||
            !requireText(object, "provisional_text", &parsed.provisionalText) ||
            parsed.phase.isEmpty() || parsed.phase.toUtf8().size() > 128) {
            setCommandError(error);
            return false;
        }
        if (object.contains(QStringLiteral("level"))) {
            const QJsonValue levelValue = object.value(QStringLiteral("level"));
            if (!levelValue.isDouble()) {
                setCommandError(error);
                return false;
            }
            const double rawLevel = levelValue.toDouble();
            if (!std::isfinite(rawLevel) || std::floor(rawLevel) != rawLevel ||
                rawLevel < 0 || rawLevel > 100) {
                setCommandError(error);
                return false;
            }
            parsed.level = static_cast<int>(rawLevel);
        }
        *command = std::move(parsed);
        return true;
    }

    if ((commandName == QStringLiteral("clear") ||
         commandName == QStringLiteral("shutdown")) &&
        object.size() == 1) {
        command->kind = commandName == QStringLiteral("clear")
                            ? OverlayCommand::Kind::Clear
                            : OverlayCommand::Kind::Shutdown;
        command->phase.clear();
        command->stableText.clear();
        command->provisionalText.clear();
        command->level.reset();
        return true;
    }

    setCommandError(error);
    return false;
}

}  // namespace fun_voice_overlay
