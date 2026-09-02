#include "overlay_window.h"

#include <QApplication>
#include <QLabel>

#include <cassert>

using namespace fun_voice_overlay;

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    const QRect rect = OverlayWindow::bottomCenteredRect(
        QRect(0, 0, 1920, 1080), QSize(420, 112));
    assert(rect == QRect(750, 932, 420, 112));

    OverlayWindow window;
    OverlayCommand command;
    command.kind = OverlayCommand::Kind::Show;
    command.phase = QStringLiteral("recording");
    command.stableText = QString::fromUtf8("今天下午三点执行 git commit");
    command.provisionalText = QStringLiteral("然后运行 pytest -q");
    command.level = 42;
    window.showModel(command);

    const auto *stable = window.findChild<QLabel *>(QStringLiteral("stable-text"));
    const auto *provisional =
        window.findChild<QLabel *>(QStringLiteral("provisional-text"));
    assert(stable != nullptr && stable->text() == command.stableText);
    assert(provisional != nullptr && provisional->text() == command.provisionalText);
    assert(window.isVisible());
    assert(window.focusPolicy() == Qt::NoFocus);
    assert(window.windowFlags().testFlag(Qt::FramelessWindowHint));
    assert(window.windowFlags().testFlag(Qt::WindowStaysOnTopHint));
    assert(window.windowFlags().testFlag(Qt::WindowDoesNotAcceptFocus));
    assert(window.windowFlags().testFlag(Qt::WindowTransparentForInput));

    window.clearModel();
    assert(stable->text().isEmpty());
    assert(provisional->text().isEmpty());
    assert(!window.isVisible());
}
