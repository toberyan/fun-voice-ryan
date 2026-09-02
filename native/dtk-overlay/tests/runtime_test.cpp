#include "protocol.h"

#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>

#include <cassert>

using namespace fun_voice_overlay;

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    assert(argc == 2);

    QProcess overlay;
    overlay.setProgram(QString::fromLocal8Bit(argv[1]));
    overlay.start();
    assert(overlay.waitForStarted(2000));
    assert(overlay.waitForReadyRead(2000));

    FrameDecoder decoder;
    assert(decoder.append(overlay.readAllStandardOutput()) == FrameError::None);
    const auto replies = decoder.takeFrames();
    assert(replies.size() == 1);
    const QJsonObject ready = QJsonDocument::fromJson(replies.front()).object();
    assert(ready.value(QStringLiteral("reply")) == QStringLiteral("ready"));

    const QByteArray clear = encodeFrame(R"({"command":"clear"})");
    assert(overlay.write(clear) == clear.size());
    assert(overlay.waitForBytesWritten(1000));
    const QByteArray shutdown = encodeFrame(R"({"command":"shutdown"})");
    assert(overlay.write(shutdown) == shutdown.size());
    assert(overlay.waitForBytesWritten(1000));
    assert(overlay.waitForFinished(2000));
    assert(overlay.exitStatus() == QProcess::NormalExit);
    assert(overlay.exitCode() == 0);
}
