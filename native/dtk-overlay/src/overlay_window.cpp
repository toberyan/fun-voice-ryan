#include "overlay_window.h"

#include <DBlurEffectWidget>
#include <DFloatingWidget>
#include <DGuiApplicationHelper>
#include <DWindowManagerHelper>

#include <QCursor>
#include <QFontMetrics>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QPalette>
#include <QScreen>
#include <QVBoxLayout>

#include <algorithm>

DWIDGET_USE_NAMESPACE
DGUI_USE_NAMESPACE

namespace fun_voice_overlay {
namespace {

constexpr int kBottomMargin = 36;
constexpr int kMinimumWidth = 320;
constexpr int kMaximumWidth = 720;
constexpr int kCardRadius = 16;
constexpr int kOuterMargin = 12;
constexpr int kContentMargin = 14;

QString phaseLabel(const QString &phase) {
    static const QHash<QString, QString> labels{
        {QStringLiteral("preparing"), QStringLiteral("正在准备本地模型")},
        {QStringLiteral("recording"), QStringLiteral("录音中")},
        {QStringLiteral("finalizing"), QStringLiteral("正在整理")},
        {QStringLiteral("correcting"), QStringLiteral("正在精修")},
        {QStringLiteral("committing"), QStringLiteral("正在输入")},
        {QStringLiteral("rehydrating"), QStringLiteral("正在恢复本地模型")},
        {QStringLiteral("enriching"), QStringLiteral("正在整理结果")},
        {QStringLiteral("active_idle"), QStringLiteral("本地模型就绪")},
    };
    return labels.value(phase, QStringLiteral("语音输入"));
}

void configureTextLabel(QLabel *label, const QString &objectName) {
    label->setObjectName(objectName);
    label->setTextFormat(Qt::PlainText);
    label->setTextInteractionFlags(Qt::NoTextInteraction);
    label->setWordWrap(false);
    label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
}

QString elideForWidth(const QLabel *label, const QString &text, int width) {
    const int available = std::max(1, width);
    return label->fontMetrics().elidedText(text, Qt::ElideRight, available);
}

}  // namespace

OverlayWindow::OverlayWindow(QWidget *parent) : QWidget(parent) {
    setWindowFlags(Qt::ToolTip | Qt::FramelessWindowHint |
                   Qt::WindowStaysOnTopHint | Qt::WindowDoesNotAcceptFocus |
                   Qt::WindowTransparentForInput);
    setAttribute(Qt::WA_TranslucentBackground, true);
    setAttribute(Qt::WA_ShowWithoutActivating, true);
    setAttribute(Qt::WA_TransparentForMouseEvents, true);
    setFocusPolicy(Qt::NoFocus);

    auto *outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(kOuterMargin, kOuterMargin, kOuterMargin,
                                    kOuterMargin);

    card_ = new DFloatingWidget(this);
    card_->setFramRadius(kCardRadius);
    outerLayout->addWidget(card_);

    auto *content = new QWidget(card_);
    contentLayout_ = new QVBoxLayout(content);
    contentLayout_->setContentsMargins(kContentMargin, kContentMargin,
                                       kContentMargin, kContentMargin);
    contentLayout_->setSpacing(5);

    statusLabel_ = new QLabel(content);
    statusLabel_->setObjectName(QStringLiteral("status-text"));
    statusLabel_->setTextFormat(Qt::PlainText);
    statusLabel_->setTextInteractionFlags(Qt::NoTextInteraction);
    statusLabel_->setFont(QFont(font().family(), font().pointSize(), QFont::DemiBold));
    levelLabel_ = new QLabel(content);
    levelLabel_->setObjectName(QStringLiteral("level-text"));
    levelLabel_->setTextFormat(Qt::PlainText);
    levelLabel_->setTextInteractionFlags(Qt::NoTextInteraction);
    stableLabel_ = new QLabel(content);
    provisionalLabel_ = new QLabel(content);
    configureTextLabel(stableLabel_, QStringLiteral("stable-text"));
    configureTextLabel(provisionalLabel_, QStringLiteral("provisional-text"));

    contentLayout_->addWidget(statusLabel_);
    contentLayout_->addWidget(levelLabel_);
    contentLayout_->addWidget(stableLabel_);
    contentLayout_->addWidget(provisionalLabel_);
    card_->setWidget(content);

    const auto *manager = DWindowManagerHelper::instance();
    card_->setBlurBackgroundEnabled(manager->hasComposite() &&
                                    manager->hasBlurWindow());
    if (auto *blur = card_->blurBackground(); blur != nullptr) {
        blur->setMaskAlpha(112);
    }

    auto *helper = DGuiApplicationHelper::instance();
    connect(helper, &DGuiApplicationHelper::themeTypeChanged, this,
            [this](DGuiApplicationHelper::ColorType) { applyTheme(); });
    connect(helper, &DGuiApplicationHelper::fontChanged, this,
            [this](const QFont &) { placeOnActiveScreen(); });
    connect(DWindowManagerHelper::instance(),
            &DWindowManagerHelper::hasCompositeChanged, this,
            [this] { applyTheme(); });
    connect(DWindowManagerHelper::instance(),
            &DWindowManagerHelper::hasBlurWindowChanged, this,
            [this] { applyTheme(); });
    applyTheme();
}

QRect OverlayWindow::bottomCenteredRect(const QRect &available, QSize requested) {
    const int maximumWidth = std::max(1, std::min(kMaximumWidth,
                                                   available.width() - 48));
    const int minimumWidth = std::min(kMinimumWidth, maximumWidth);
    const int width = std::clamp(requested.width(), minimumWidth, maximumWidth);
    const int maximumHeight = std::max(1, available.height() / 3);
    const int height = std::clamp(requested.height(), 1, maximumHeight);
    const int x = available.x() + (available.width() - width) / 2;
    const int y = std::max(available.y(), available.bottom() + 1 -
                                            kBottomMargin - height);
    return QRect(x, y, width, height);
}

void OverlayWindow::showModel(const OverlayCommand &command) {
    if (command.kind != OverlayCommand::Kind::Show) {
        return;
    }
    statusLabel_->setText(phaseLabel(command.phase));
    if (command.level) {
        levelLabel_->setText(QStringLiteral("音量 %1%").arg(*command.level));
        levelLabel_->show();
    } else {
        levelLabel_->clear();
        levelLabel_->hide();
    }
    stableLabel_->setText(command.stableText);
    provisionalLabel_->setText(command.provisionalText);
    stableLabel_->setVisible(!command.stableText.isEmpty());
    provisionalLabel_->setVisible(!command.provisionalText.isEmpty());
    applyTheme();
    placeOnActiveScreen();
    show();
    raise();
}

void OverlayWindow::clearModel() {
    statusLabel_->clear();
    levelLabel_->clear();
    stableLabel_->clear();
    provisionalLabel_->clear();
    levelLabel_->hide();
    stableLabel_->hide();
    provisionalLabel_->hide();
    hide();
}

void OverlayWindow::applyTheme() {
    const bool dark = DGuiApplicationHelper::instance()->themeType() ==
                      DGuiApplicationHelper::DarkType;
    const bool blur = DWindowManagerHelper::instance()->hasComposite() &&
                      DWindowManagerHelper::instance()->hasBlurWindow();
    card_->setBlurBackgroundEnabled(blur);
    const QColor background = dark ? QColor(32, 35, 42, blur ? 218 : 255)
                                   : QColor(247, 249, 252, blur ? 218 : 255);
    const QColor status = dark ? QColor(241, 244, 248) : QColor(28, 32, 38);
    const QColor stable = dark ? QColor(224, 229, 236) : QColor(35, 39, 45);
    const QColor provisional = dark ? QColor(165, 174, 188) : QColor(101, 110, 121);
    const QColor level = dark ? QColor(124, 188, 255) : QColor(0, 102, 204);
    card_->setStyleSheet(QStringLiteral(
        "DFloatingWidget { background-color: %1; border-radius: %2px; }")
                             .arg(background.name(QColor::HexArgb))
                             .arg(kCardRadius));
    statusLabel_->setStyleSheet(QStringLiteral("color: %1;").arg(status.name()));
    stableLabel_->setStyleSheet(QStringLiteral("color: %1;").arg(stable.name()));
    provisionalLabel_->setStyleSheet(
        QStringLiteral("color: %1;").arg(provisional.name()));
    levelLabel_->setStyleSheet(QStringLiteral("color: %1;").arg(level.name()));
}

void OverlayWindow::placeOnActiveScreen() {
    QScreen *screen = QGuiApplication::screenAt(QCursor::pos());
    if (screen == nullptr) {
        screen = QGuiApplication::primaryScreen();
    }
    if (screen == nullptr) {
        return;
    }
    resizeForContent(screen->availableGeometry());
}

void OverlayWindow::resizeForContent(const QRect &available) {
    const int maximumWidth = std::max(1, std::min(kMaximumWidth,
                                                   available.width() - 48));
    const int textWidth = std::max(1, maximumWidth - 2 * (kOuterMargin + kContentMargin));
    stableLabel_->setText(elideForWidth(stableLabel_, stableLabel_->text(), textWidth));
    provisionalLabel_->setText(
        elideForWidth(provisionalLabel_, provisionalLabel_->text(), textWidth));
    const QRect target = bottomCenteredRect(available, sizeHint());
    resize(target.size());
    move(target.topLeft());
}

}  // namespace fun_voice_overlay
