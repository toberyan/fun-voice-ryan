#pragma once

#include "protocol.h"

#include <QRect>
#include <QSize>
#include <QWidget>

class QLabel;
class QVBoxLayout;

namespace Dtk::Widget {
class DFloatingWidget;
}

namespace fun_voice_overlay {

class OverlayWindow final : public QWidget {
public:
    explicit OverlayWindow(QWidget *parent = nullptr);

    void showModel(const OverlayCommand &command);
    void clearModel();

    static QRect bottomCenteredRect(const QRect &available, QSize requested);

private:
    void applyTheme();
    void placeOnActiveScreen();
    void resizeForContent(const QRect &available);

    Dtk::Widget::DFloatingWidget *card_;
    QLabel *statusLabel_;
    QLabel *levelLabel_;
    QLabel *stableLabel_;
    QLabel *provisionalLabel_;
    QVBoxLayout *contentLayout_;
};

}  // namespace fun_voice_overlay
