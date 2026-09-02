#include "protocol.h"

#include <cassert>

using namespace fun_voice_overlay;

int main() {
    FrameDecoder decoder;
    const QByteArray payload =
        R"({"command":"show","phase":"recording","stable_text":"中文 git commit",)"
        R"("provisional_text":"pytest -q","level":42})";
    const QByteArray frame = encodeFrame(payload);

    assert(decoder.append(frame.left(3)) == FrameError::None);
    assert(decoder.takeFrames().empty());
    assert(decoder.append(frame.mid(3)) == FrameError::None);
    const auto frames = decoder.takeFrames();
    assert(frames.size() == 1 && frames.front() == payload);

    OverlayCommand command;
    QString error;
    assert(parseCommand(frames.front(), &command, &error));
    assert(command.kind == OverlayCommand::Kind::Show);
    assert(command.stableText == QString::fromUtf8("中文 git commit"));
    assert(command.level && *command.level == 42);

    assert(!parseCommand(R"({"command":"show","phase":3})", &command, &error));
    assert(error == QStringLiteral("error_command"));
    assert(decoder.append(QByteArray(4, '\xff')) == FrameError::TooLarge);
}
