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

struct OverlayLayout final {
    double verticalCenterRatio = 0.70;
    int widthPx = 680;
    double fontScale = 1.0;
};

class OverlayWindow final : public QWidget {
public:
    explicit OverlayWindow(OverlayLayout layout = {}, QWidget *parent = nullptr);

    void showModel(const OverlayCommand &command);
    void clearModel();

    static QRect centeredRect(const QRect &available, QSize requested,
                              const OverlayLayout &layout);

private:
    void applyFonts();
    void applyTheme();
    void placeOnActiveScreen();
    void resizeForContent(const QRect &available);

    const OverlayLayout layout_;
    Dtk::Widget::DFloatingWidget *card_;
    QLabel *statusLabel_;
    QLabel *levelLabel_;
    QLabel *stableLabel_;
    QLabel *provisionalLabel_;
    QVBoxLayout *contentLayout_;
};

}  // namespace fun_voice_overlay
