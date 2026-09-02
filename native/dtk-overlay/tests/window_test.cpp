#include "overlay_window.h"

#include <QApplication>
#include <QLabel>

#include <cassert>
#include <cmath>

using namespace fun_voice_overlay;

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    const OverlayLayout defaultLayout{};
    const QRect middleLower = OverlayWindow::centeredRect(
        QRect(0, 0, 1920, 1080), QSize(680, 112), defaultLayout);
    assert(middleLower == QRect(620, 700, 680, 112));

    const QRect narrowed = OverlayWindow::centeredRect(
        QRect(0, 0, 440, 800), QSize(680, 112), defaultLayout);
    assert(narrowed == QRect(24, 504, 392, 112));

    const OverlayLayout lowEdge{0.85, 680, 1.0};
    const QRect clamped = OverlayWindow::centeredRect(
        QRect(0, 0, 1920, 1080), QSize(680, 360), lowEdge);
    assert(clamped == QRect(620, 696, 680, 360));

    const OverlayLayout scaled{0.70, 680, 1.20};
    OverlayWindow window(scaled);
    OverlayCommand command;
    command.kind = OverlayCommand::Kind::Show;
    command.phase = QStringLiteral("recording");
    command.stableText = QString::fromUtf8("今天下午三点执行 git commit");
    command.provisionalText = QStringLiteral("然后运行 pytest -q");
    command.level = 42;
    window.showModel(command);

    const auto *status = window.findChild<QLabel *>(QStringLiteral("status-text"));
    const auto *level = window.findChild<QLabel *>(QStringLiteral("level-text"));
    const auto *stable = window.findChild<QLabel *>(QStringLiteral("stable-text"));
    const auto *provisional =
        window.findChild<QLabel *>(QStringLiteral("provisional-text"));
    assert(status != nullptr &&
           std::abs(status->font().pointSizeF() - 21.6) < 0.01);
    assert(level != nullptr && std::abs(level->font().pointSizeF() - 15.6) < 0.01);
    assert(stable != nullptr && std::abs(stable->font().pointSizeF() - 18.0) < 0.01);
    assert(provisional != nullptr &&
           std::abs(provisional->font().pointSizeF() - 18.0) < 0.01);
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
