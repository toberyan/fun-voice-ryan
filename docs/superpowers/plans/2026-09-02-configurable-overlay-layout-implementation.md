# Configurable DTK Overlay Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users configure the DTK transient overlay's middle-lower vertical placement, fixed width, and text scale through `config.toml`, without changing any speech or input behavior.

**Architecture:** Python remains the sole TOML reader and validates an immutable `OverlayConfig`. The daemon supplies only three numeric command-line values when it lazily spawns the private native overlay. The Qt/DTK child parses and defensively validates those values, then applies an `OverlayLayout` to geometry and fonts while the text-bearing stdin/stdout protocol remains unchanged.

**Tech Stack:** Python 3.11 `tomllib` and pytest; C++17, Qt 6, DTK 6, CMake/CTest; user-level systemd on Deepin DDE X11.

## Global Constraints

- Target DDE **X11** only; keep the existing `Super+C` press-and-hold hotkey and target-window focus semantics unchanged.
- Default layout values are exactly `vertical_center_ratio = 0.70`, `width_px = 680`, and `font_scale = 1.0`.
- Accept only ratio `[0.50, 0.85]`, width `[420, 1000]`, and scale `[0.80, 1.80]`; reject invalid known `[overlay]` fields at daemon startup.
- Native effective width must be no larger than work-area width minus 48 px; all four edges retain a 24 px preferred safety margin when space permits.
- Base fonts are status `18 pt` DemiBold, stable/provisional transcript `15 pt`, and level `13 pt`, each multiplied by `font_scale`.
- The selected screen is the one under the pointer when `show` is received (primary-screen fallback); its window center sits at the configured ratio of `availableGeometry()` and remains horizontally centered.
- Preserve no-focus, mouse-through, transparent/rounded DTK effects, lazy child lifecycle, 5-second idle exit, and `NullOverlay` failure degradation.
- Do not add hot reload, a settings UI, new IPC commands, model changes, clipboard changes, logs containing text, or persistent transcript data.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/fun_voice/config.py` | Define, parse and validate immutable `OverlayConfig`; expose it as `Config.overlay`. |
| `tests/test_config.py` | Lock Python defaults, TOML overrides and invalid-overlay rejection. |
| `src/fun_voice/overlay.py` | Serialize only validated numeric layout options into the native child argv. |
| `tests/test_overlay.py` | Ensure lazy spawn argv is exact and carries no transcript text. |
| `src/fun_voice/daemon.py` | Pass `cfg.overlay` into the DTK controller at the one construction site. |
| `native/dtk-overlay/src/overlay_window.h/.cpp` | Own `OverlayLayout`, fixed-width center geometry, 24 px clamping and scaled label fonts. |
| `native/dtk-overlay/src/main.cpp` | Parse/reject three CLI values before creating the Qt event-loop controller. |
| `native/dtk-overlay/tests/window_test.cpp` | Verify geometry, small-screen clamp, text scale and existing noninteractive behavior offscreen. |
| `native/dtk-overlay/tests/runtime_test.cpp` | Verify valid CLI starts normally and invalid CLI exits before opening a protocol session. |
| `scripts/config.example.toml` | Publish the `[overlay]` keys with their defaults and restart semantics. |
| `docs/operations.md` | Document configuration scope and the precise rebuild/restart command. |
| `docs/acceptance-checklist.md` | Add a real-DDE manual check for all three configuration knobs. |

### Task 1: Make the native DTK window accept and render a layout

**Files:**

- Modify: `native/dtk-overlay/src/overlay_window.h`
- Modify: `native/dtk-overlay/src/overlay_window.cpp`
- Modify: `native/dtk-overlay/src/main.cpp`
- Modify: `native/dtk-overlay/tests/window_test.cpp`
- Modify: `native/dtk-overlay/tests/runtime_test.cpp`

**Interfaces:**

- Consumes: the existing length-framed `OverlayCommand` protocol and Qt `QApplication` event loop.
- Produces: `fun_voice_overlay::OverlayLayout`, `OverlayWindow(OverlayLayout, QWidget *)`, and `OverlayWindow::centeredRect(const QRect &, QSize, const OverlayLayout &)`, used by `main.cpp` and the native tests.
- Preserves: the runtime binary still writes one text-free `ready` reply, accepts `show`/`clear`/`shutdown`, and exits with `0` on normal shutdown.

- [ ] **Step 1: Write failing native geometry, scale, and CLI tests**

  Replace the first geometry assertion in `native/dtk-overlay/tests/window_test.cpp` with assertions that define the approved geometry contract. Use a 1920x1080 work area, a 680x112 requested card, and a 70% ratio. Add the small-screen and lower-clamp cases below; retain all existing Unicode, visibility, clear, and window-flag assertions.

  ```cpp
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
  // Build and show `command` exactly as the existing test does.
  const auto *status = window.findChild<QLabel *>(QStringLiteral("status-text"));
  const auto *level = window.findChild<QLabel *>(QStringLiteral("level-text"));
  const auto *stable = window.findChild<QLabel *>(QStringLiteral("stable-text"));
  const auto *provisional = window.findChild<QLabel *>(QStringLiteral("provisional-text"));
  assert(status != nullptr && std::abs(status->font().pointSizeF() - 21.6) < 0.01);
  assert(level != nullptr && std::abs(level->font().pointSizeF() - 15.6) < 0.01);
  assert(stable != nullptr && std::abs(stable->font().pointSizeF() - 18.0) < 0.01);
  assert(provisional != nullptr && std::abs(provisional->font().pointSizeF() - 18.0) < 0.01);
  ```

  Include `<cmath>` for `std::abs`. The expected small-screen rectangle follows the 24 px horizontal safe margin: `440 - 48 = 392`. The 360 px card with a 0.85 center must stop at `1080 - 24 - 360 = 696`.

  In `native/dtk-overlay/tests/runtime_test.cpp`, preserve the normal ready/clear/shutdown probe and add a separate process which must reject a width below the public lower bound:

  ```cpp
  QProcess invalid;
  invalid.setProgram(QString::fromLocal8Bit(argv[1]));
  invalid.setArguments({QStringLiteral("--width-px"), QStringLiteral("419")});
  invalid.start();
  assert(invalid.waitForFinished(2000));
  assert(invalid.exitStatus() == QProcess::NormalExit);
  assert(invalid.exitCode() != 0);
  ```

- [ ] **Step 2: Run the native tests and verify they fail for the missing API**

  Run:

  ```bash
  cmake -S native/dtk-overlay -B build/dtk-overlay
  cmake --build build/dtk-overlay
  ctest --test-dir build/dtk-overlay --output-on-failure
  ```

  Expected: compilation fails because `OverlayLayout` and `OverlayWindow::centeredRect` do not exist and the old constructor does not accept a layout.

- [ ] **Step 3: Define the native layout value object and geometry API**

  In `native/dtk-overlay/src/overlay_window.h`, add this public value type above `OverlayWindow` and replace the bottom-position constructor/static function declarations:

  ```cpp
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
      // Keep the existing widget and label members below this declaration.
  };
  ```

  Remove `bottomCenteredRect`; no compatibility alias is needed because it is private to this native executable/test target.

- [ ] **Step 4: Implement fixed-width, middle-lower placement and scaled fonts**

  In `native/dtk-overlay/src/overlay_window.cpp`, replace `kBottomMargin`, `kMinimumWidth`, and `kMaximumWidth` with these visual constants. Keep the existing card margins, radius, colors, and label object names.

  ```cpp
  constexpr int kScreenMargin = 24;
  constexpr int kMaximumHeightDivisor = 3;
  constexpr qreal kStatusPointSize = 18.0;
  constexpr qreal kTranscriptPointSize = 15.0;
  constexpr qreal kLevelPointSize = 13.0;
  ```

  Change the constructor to save `layout`, call `applyFonts()` after creating all four labels, and on `DGuiApplicationHelper::fontChanged` call both `applyFonts()` and `placeOnActiveScreen()`.

  ```cpp
  OverlayWindow::OverlayWindow(OverlayLayout layout, QWidget *parent)
      : QWidget(parent), layout_(layout) {
      // Retain the existing window flags, layouts, labels, theme hooks and blur setup.
      // After configureTextLabel(...) calls:
      applyFonts();
  }

  void OverlayWindow::applyFonts() {
      const QFont base = QGuiApplication::font();
      auto scaled = [this, &base](qreal points, QFont::Weight weight) {
          QFont result(base);
          result.setPointSizeF(points * layout_.fontScale);
          result.setWeight(weight);
          return result;
      };
      statusLabel_->setFont(scaled(kStatusPointSize, QFont::DemiBold));
      levelLabel_->setFont(scaled(kLevelPointSize, QFont::Normal));
      stableLabel_->setFont(scaled(kTranscriptPointSize, QFont::Normal));
      provisionalLabel_->setFont(scaled(kTranscriptPointSize, QFont::Normal));
  }
  ```

  Implement `centeredRect` as below. It uses an inset work area when it remains valid and degrades to the true work area only on a physically too-small screen.

  ```cpp
  QRect OverlayWindow::centeredRect(const QRect &available, QSize requested,
                                    const OverlayLayout &layout) {
      const QRect inset = available.adjusted(kScreenMargin, kScreenMargin,
                                             -kScreenMargin, -kScreenMargin);
      const QRect bounds = inset.isValid() ? inset : available;
      const int width = std::clamp(layout.widthPx, 1, std::max(1, bounds.width()));
      const int maximumHeight = std::max(
          1, std::min(bounds.height(), available.height() / kMaximumHeightDivisor));
      const int height = std::clamp(requested.height(), 1, maximumHeight);
      const int x = available.x() + (available.width() - width) / 2;
      const int centerY = available.y() +
          qRound(available.height() * layout.verticalCenterRatio);
      const int minimumY = bounds.top();
      const int maximumY = std::max(minimumY, bounds.bottom() + 1 - height);
      const int y = std::clamp(centerY - height / 2, minimumY, maximumY);
      return QRect(x, y, width, height);
  }
  ```

  Update `resizeForContent` to derive `textWidth` from the same fixed effective width and call `centeredRect(available, sizeHint(), layout_)`. The label elision remains right-side Unicode elision, but its budget must be `target.width() - 2 * (kOuterMargin + kContentMargin)` rather than an obsolete `kMaximumWidth`.

- [ ] **Step 5: Parse native options before constructing the overlay controller**

  In `native/dtk-overlay/src/main.cpp`, add `#include <QCommandLineOption>`, `#include <QCommandLineParser>`, `#include <optional>`, and `#include <cmath>`. Before `OverlayApplication`, add these bounds and parser. The parser deliberately handles only numeric layout fields and never reads stdin text or a config file.

  ```cpp
  constexpr double kMinimumVerticalCenterRatio = 0.50;
  constexpr double kMaximumVerticalCenterRatio = 0.85;
  constexpr int kMinimumWidthPx = 420;
  constexpr int kMaximumWidthPx = 1000;
  constexpr double kMinimumFontScale = 0.80;
  constexpr double kMaximumFontScale = 1.80;

  std::optional<OverlayLayout> parseLayout(const QStringList &arguments) {
      QCommandLineParser parser;
      parser.addOption({QStringLiteral("vertical-center-ratio"),
                        QStringLiteral("vertical center ratio"),
                        QStringLiteral("ratio"), QStringLiteral("0.70")});
      parser.addOption({QStringLiteral("width-px"),
                        QStringLiteral("overlay width in pixels"),
                        QStringLiteral("pixels"), QStringLiteral("680")});
      parser.addOption({QStringLiteral("font-scale"),
                        QStringLiteral("font scale"),
                        QStringLiteral("scale"), QStringLiteral("1.0")});
      if (!parser.parse(arguments)) {
          return std::nullopt;
      }
      bool ratioOk = false;
      bool widthOk = false;
      bool scaleOk = false;
      const double ratio = parser.value(QStringLiteral("vertical-center-ratio")).toDouble(&ratioOk);
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
  ```

  Give `OverlayApplication` an explicit `OverlayLayout layout` constructor and initialize `window_(layout)`. In `main`, parse after creating `QApplication`; return `2` when parsing fails, otherwise construct `OverlayApplication controller(*layout)` and run the event loop:

  ```cpp
  QApplication app(argc, argv);
  app.setQuitOnLastWindowClosed(false);
  const std::optional<OverlayLayout> layout = parseLayout(QCoreApplication::arguments());
  if (!layout.has_value()) {
      return 2;
  }
  OverlayApplication controller(*layout);
  return app.exec();
  ```

  Do not emit user text in parser failures. Existing stderr remains redirected to `/dev/null` by the Python parent in production.

- [ ] **Step 6: Run native tests and build the installed artifact**

  Run:

  ```bash
  cmake -S native/dtk-overlay -B build/dtk-overlay
  cmake --build build/dtk-overlay
  ctest --test-dir build/dtk-overlay --output-on-failure
  ```

  Expected: all three tests pass: `overlay_protocol_test`, `overlay_window_test`, and `overlay_runtime_test`.

- [ ] **Step 7: Commit the native layout feature**

  ```bash
  git add native/dtk-overlay/src/overlay_window.h \
      native/dtk-overlay/src/overlay_window.cpp \
      native/dtk-overlay/src/main.cpp \
      native/dtk-overlay/tests/window_test.cpp \
      native/dtk-overlay/tests/runtime_test.cpp
  git commit -m "feat: support configurable dtk overlay layout"
  ```

### Task 2: Parse the user configuration and pass only numeric options to DTK

**Files:**

- Modify: `src/fun_voice/config.py`
- Modify: `tests/test_config.py`
- Modify: `src/fun_voice/overlay.py`
- Modify: `tests/test_overlay.py`
- Modify: `src/fun_voice/daemon.py`

**Interfaces:**

- Consumes: the exact native flags `--vertical-center-ratio`, `--width-px`, and `--font-scale` from Task 1.
- Produces: `OverlayConfig(vertical_center_ratio: float = 0.70, width_px: int = 680, font_scale: float = 1.0)`, `validate_overlay_config(value)`, and `Config.overlay`.
- Preserves: `DtkOverlayController.show()` continues to encode text only in the bounded private stdin frame; `argv` contains executable path plus three fixed-name numeric pairs.

- [ ] **Step 1: Write failing Python configuration tests**

  In `tests/test_config.py`, import `OverlayConfig` and `validate_overlay_config`. Extend `test_config_defaults` with `assert cfg.overlay == OverlayConfig()`. Add the valid TOML test:

  ```python
  def test_load_config_parses_overlay_layout(tmp_path: Path) -> None:
      path = tmp_path / "config.toml"
      path.write_text(
          "[overlay]\nvertical_center_ratio = 0.85\nwidth_px = 900\nfont_scale = 1.2\n",
          encoding="utf-8",
      )

      assert load_config(path).overlay == OverlayConfig(
          vertical_center_ratio=0.85,
          width_px=900,
          font_scale=1.2,
      )
  ```

  Add the parameterized invalid cases; each must raise an error that names the offending field:

  ```python
  @pytest.mark.parametrize(
      ("value", "message"),
      [
          (OverlayConfig(vertical_center_ratio=0.49), "overlay.vertical_center_ratio"),
          (OverlayConfig(vertical_center_ratio=0.86), "overlay.vertical_center_ratio"),
          (OverlayConfig(width_px=419), "overlay.width_px"),
          (OverlayConfig(width_px=1001), "overlay.width_px"),
          (OverlayConfig(font_scale=0.79), "overlay.font_scale"),
          (OverlayConfig(font_scale=1.81), "overlay.font_scale"),
      ],
  )
  def test_overlay_config_rejects_out_of_range_values(
      value: OverlayConfig, message: str
  ) -> None:
      with pytest.raises(ConfigError, match=message):
          validate_overlay_config(value)
  ```

  Add TOML type/finite coverage for `width_px = true`, `vertical_center_ratio = "low"`, and `font_scale = nan`; each must raise `ConfigError` rather than fall back silently.

  In `tests/test_overlay.py`, capture the lazy-spawn argv and assert the layout is passed before the first in-memory frame:

  ```python
  def test_dtk_controller_passes_validated_layout_without_transcript_in_argv() -> None:
      argv: list[str] = []
      controller = DtkOverlayController(
          executable=Path("/native/fun-voice-overlay"),
          layout=OverlayConfig(0.70, 680, 1.25),
          popen=lambda value: argv.extend(value) or FakeProcess(),
      )

      controller.show(OverlayModel(phase=DaemonState.RECORDING, stable_text="私密文本"))

      assert argv == [
          "/native/fun-voice-overlay",
          "--vertical-center-ratio", "0.7",
          "--width-px", "680",
          "--font-scale", "1.25",
      ]
      assert "私密文本" not in argv
  ```

- [ ] **Step 2: Run focused Python tests and verify they fail**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_config.py tests/test_overlay.py -q
  ```

  Expected: collection/import failure for `OverlayConfig` and a `DtkOverlayController.__init__()` unexpected `layout` argument.

- [ ] **Step 3: Add immutable schema parsing and strict validation**

  In `src/fun_voice/config.py`, import `math` and add these definitions after `EnhancedInferenceConfig`:

  ```python
  @dataclass(frozen=True)
  class OverlayConfig:
      """Validated geometry and typography for the transient native overlay."""

      vertical_center_ratio: float = 0.70
      width_px: int = 680
      font_scale: float = 1.0


  def _finite_float(value: object, *, key: str, default: float) -> float:
      if value is None:
          return default
      if isinstance(value, bool) or not isinstance(value, (int, float)):
          raise ConfigError(f"{key} must be a finite number")
      parsed = float(value)
      if not math.isfinite(parsed):
          raise ConfigError(f"{key} must be a finite number")
      return parsed


  def validate_overlay_config(value: OverlayConfig) -> OverlayConfig:
      if not 0.50 <= value.vertical_center_ratio <= 0.85:
          raise ConfigError("overlay.vertical_center_ratio must be in [0.50, 0.85]")
      if not 420 <= value.width_px <= 1000:
          raise ConfigError("overlay.width_px must be in [420, 1000]")
      if not math.isfinite(value.font_scale) or not 0.80 <= value.font_scale <= 1.80:
          raise ConfigError("overlay.font_scale must be in [0.80, 1.80]")
      return value
  ```

  Add `overlay: OverlayConfig = field(default_factory=OverlayConfig)` to `Config`. In `load_config`, obtain `overlay = _table(raw.get("overlay"))`, build `OverlayConfig` with `_finite_float` for both float fields and `_positive_int(..., key="overlay.width_px", default=680)` for width, validate it, and return it as `overlay=overlay_config`.

  Keep `_float` unchanged for existing backward-compatible inference keys; its permissive fallback must not be reused for new overlay keys.

- [ ] **Step 4: Serialize layout only at process start and wire daemon configuration**

  In `src/fun_voice/overlay.py`, import `OverlayConfig`. Give `DtkOverlayController.__init__` this parameter and retain it in an immutable instance field:

  ```python
  def __init__(
      self,
      *,
      executable: Path | None = None,
      layout: OverlayConfig = OverlayConfig(),
      popen: OverlayPopen = _default_popen,
  ) -> None:
      self._executable = default_overlay_executable() if executable is None else executable
      self._layout = layout
      self._popen = popen
      # retain the existing process, closed, and lock initialization
  ```

  Add a private `_argv()` helper and use it at the only `self._popen(...)` call:

  ```python
  def _argv(self) -> list[str]:
      return [
          str(self._executable),
          "--vertical-center-ratio", str(self._layout.vertical_center_ratio),
          "--width-px", str(self._layout.width_px),
          "--font-scale", str(self._layout.font_scale),
      ]

  # in _ensure_process_locked:
  process = self._popen(self._argv())
  ```

  Do not include `OverlayModel`, frame bytes, phase, or any text in `_argv()`.

  In `src/fun_voice/daemon.py`, change the sole production constructor to:

  ```python
  overlay: OverlayController = DtkOverlayController(
      executable=default_overlay_executable(),
      layout=cfg.overlay,
  )
  ```

- [ ] **Step 5: Run focused tests, type checks, and commit**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_config.py tests/test_overlay.py -q
  .venv/bin/ruff check src/fun_voice/config.py src/fun_voice/overlay.py src/fun_voice/daemon.py tests/test_config.py tests/test_overlay.py
  .venv/bin/mypy src/fun_voice
  ```

  Expected: focused pytest passes; Ruff reports no diagnostics; mypy reports success for all checked source files.

  ```bash
  git add src/fun_voice/config.py src/fun_voice/overlay.py src/fun_voice/daemon.py \
      tests/test_config.py tests/test_overlay.py
  git commit -m "feat: configure dtk overlay layout"
  ```

### Task 3: Publish the configuration and perform end-to-end verification

**Files:**

- Modify: `scripts/config.example.toml`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`

**Interfaces:**

- Consumes: the exact `[overlay]` schema and bounds from Task 2, plus the systemd service/binary installation flow already documented by the project.
- Produces: copy-pasteable user configuration and an unambiguous restart/visual acceptance path.
- Preserves: no configuration is written automatically; users retain ownership of `~/.config/fun-voice-ryan/config.toml`.

- [ ] **Step 1: Add the documented `[overlay]` sample**

  Insert this section in `scripts/config.example.toml` after `[input_method]` and before model configuration:

  ```toml
  [overlay]
  # DTK 瞬态悬浮窗：中心位于鼠标所在屏幕工作区的 70% 高度；重启 daemon 后生效。
  # 允许范围：vertical_center_ratio 0.50--0.85，width_px 420--1000，font_scale 0.80--1.80。
  vertical_center_ratio = 0.70
  # 固定逻辑宽度；小屏自动收缩并保留边距。
  width_px = 680
  # 1.0 对应状态 18pt、转写 15pt、音量 13pt 的默认大字号。
  font_scale = 1.0
  ```

- [ ] **Step 2: Update operations and manual acceptance documentation**

  In `docs/operations.md` section 3, extend the list of active keys with `overlay.vertical_center_ratio`, `overlay.width_px`, and `overlay.font_scale`; state their bounds and that invalid known values stop daemon startup with a config error. Add this exact configuration application procedure to the DTK build/deploy section:

  ```bash
  cmake -S native/dtk-overlay -B build/dtk-overlay
  cmake --build build/dtk-overlay
  scripts/install-user.sh
  systemctl --user restart fun-voice-daemon.service
  ```

  Explain that the new layout is read only at daemon startup; no model worker or Qwen process starts because of a layout-only restart.

  Add an “悬浮窗布局配置” subsection to `docs/acceptance-checklist.md` with these checkboxed manual checks:

  ```markdown
  - [ ] 将 `vertical_center_ratio` 设为 `0.70`，重启 daemon 后按住 `Super+C`，确认卡片位于当前屏幕中下部、水平居中且未压到 Dock。
  - [ ] 将 `width_px` 改为 `420` 与 `1000` 分别重启测试，确认卡片宽度变化；在窄屏上确认它自动收缩且仍在工作区内。
  - [ ] 将 `font_scale` 改为 `0.80` 与 `1.80` 分别重启测试，确认四类文字同比缩放且中文、英文、代码混排无乱码。
  - [ ] 将任一字段设为越界值，重启后确认 daemon 明确拒绝启动；恢复合法值后确认语音与目标窗口焦点正常。
  ```

- [ ] **Step 3: Run complete automated verification**

  Run:

  ```bash
  cmake -S native/fcitx5-fun-voice -B build/fcitx
  cmake --build build/fcitx
  ctest --test-dir build/fcitx --output-on-failure
  cmake -S native/dtk-overlay -B build/dtk-overlay
  cmake --build build/dtk-overlay
  ctest --test-dir build/dtk-overlay --output-on-failure
  PYTHONPATH=src .venv/bin/pytest -q
  .venv/bin/ruff check src tests
  .venv/bin/mypy src/fun_voice
  bash -n scripts/install-user.sh scripts/uninstall-user.sh
  ```

  Expected: both native CTest suites pass, Python suite passes, Ruff/mypy are clean, and both shell scripts pass syntax checking.

- [ ] **Step 4: Install the native binary, restart only the daemon, and perform a privacy-safe live probe**

  Run:

  ```bash
  scripts/install-user.sh
  systemctl --user restart fun-voice-daemon.service
  systemctl --user is-active fun-voice-daemon.service
  ```

  Expected: `active`. Before a real `Super+C` hold, `pgrep -af 'fun-voice-(worker|corrector)'` must show no model process; the DTK process is also absent until the first overlay frame. Trigger one normal hold-to-talk test using speech chosen by the user, then visually confirm the manual checklist without copying its transcript to terminal output or logs.

- [ ] **Step 5: Commit documentation and verification artifacts**

  ```bash
  git add scripts/config.example.toml docs/operations.md docs/acceptance-checklist.md
  git commit -m "docs: document configurable overlay layout"
  ```

## Final Handoff Checks

- [ ] Inspect `git status --short`; it must be empty before claiming completion.
- [ ] Record the three feature commits and the full verification command results in the final handoff.
- [ ] Do not push or alter remote repository state unless the user explicitly asks.
