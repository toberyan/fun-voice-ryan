# DTK 原生瞬态悬浮窗 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以按需启动的 Qt 6 / DTK 6 原生窗口替换 Xlib 白色状态框，在 DDE X11 底部居中显示可读的中英混合临时状态与转写。

**Architecture:** Python daemon 保留 `OverlayController` 抽象，新的 `DtkOverlayController` 通过专有 stdin/stdout 长度前缀管道懒启动 `fun-voice-overlay`。C++ 子进程拥有 Qt 事件循环、DTK 卡片、主题适配、Unicode 文本排版和 5 秒空闲退出；任一 UI 故障只退化为无显示，绝不影响语音主链路。

**Tech Stack:** Python 3.12、`subprocess`、JSON、C++17、Qt 6 Core/Gui/Widgets/Test、DTK 6 Gui/Widget、CMake/CTest、DDE X11。

## Global Constraints

- 仅支持 DDE X11；不修改现有 X11 `Super+C` 热键实现，也不重新引入 DDE 快捷键注册。
- 窗口固定在指针所在屏幕的可用工作区底部居中，距底 36 px；最小宽 320 px，最大宽 `min(720 px, availableWidth - 48 px)`，最大高为工作区三分之一。
- 使用 Qt Unicode 文本布局和系统字体回退；禁止在悬浮窗调用 Xlib Core Font 的 `draw_text`。
- 使用 `DFloatingWidget`、`DBlurEffectWidget`、`DGuiApplicationHelper`、`DWindowManagerHelper`；无合成器或模糊时降级为主题适配的实色圆角卡片。
- 窗口无边框、始终置顶、不会激活、没有键盘焦点、鼠标穿透；不得访问剪贴板、输入法或 XTEST。
- 临时文本只经 daemon 与其子进程的内存管道传递；禁止记录 payload、写磁盘、写剪贴板或提交到目标输入框。
- 每帧最大 64 KiB；无效帧、未知命令、子进程崩溃与写入失败只能返回固定错误码或无 UI 退化，不能泄露文本。
- 登录空闲时不得启动 Qt/DTK 或模型进程；`clear` 后由 native 子进程保活 5 秒再退出，`shutdown` 立即退出。
- 不新增 PySide/PyQt、模型依赖、XPU 任务或后台服务。

---

## File Structure

| 路径 | 责任 |
| --- | --- |
| `src/fun_voice/overlay.py` | 共享 `OverlayModel`/协议常量；Python `DtkOverlayController` 的懒启动、帧写入、退出和无 UI 降级。 |
| `src/fun_voice/daemon.py` | 在生产入口创建 `DtkOverlayController`，保留状态机调用点不变。 |
| `tests/test_overlay.py` | Python 协议帧、懒启动、错误降级、清空和关闭的内存测试。 |
| `native/dtk-overlay/CMakeLists.txt` | 构建 `fun-voice-overlay` 与 native CTest 目标。 |
| `native/dtk-overlay/src/protocol.h/.cpp` | 长度帧增量解析、严格 JSON 命令验证与无文本 ACK 编码。 |
| `native/dtk-overlay/src/overlay_window.h/.cpp` | DTK 卡片、主题、布局、无焦点/穿透窗口属性与状态清空。 |
| `native/dtk-overlay/src/main.cpp` | Qt 应用、stdin `QSocketNotifier`、协议路由和 5 秒空闲退出。 |
| `native/dtk-overlay/tests/protocol_test.cpp` | native 协议边界、乱码防护和清空语义。 |
| `native/dtk-overlay/tests/window_test.cpp` | 底部居中、尺寸限制、Unicode 与无合成器降级的 headless 测试。 |
| `scripts/install-user.sh` / `scripts/uninstall-user.sh` | 验证、安装和移除用户范围的 native overlay 二进制。 |
| `tests/test_install_scripts.py` | 断言部署脚本不会遗漏或常驻启动 overlay。 |
| `README.md` / `docs/operations.md` / `docs/acceptance-checklist.md` | DTK 构建依赖、重新部署命令、故障回退与人工 DDE 验收。 |

## Task 1: 建立可测试的 native 协议与 CMake 目标

**Files:**

- Create: `native/dtk-overlay/CMakeLists.txt`
- Create: `native/dtk-overlay/src/protocol.h`
- Create: `native/dtk-overlay/src/protocol.cpp`
- Create: `native/dtk-overlay/tests/protocol_test.cpp`

**Interfaces:**

- Produces `fun_voice_overlay::FrameDecoder::append(QByteArray)`，返回完整 payload 队列或 `FrameError::{TooLarge,Malformed}`。
- Produces `fun_voice_overlay::parseCommand(const QByteArray &, OverlayCommand *, QString *) -> bool`。
- `OverlayCommand` 必须只表达 `Show { phase, stableText, provisionalText, std::optional<int> level }`、`Clear`、`Shutdown`；错误文本只限固定枚举。
- Produces `encodeReply(ReplyCode) -> QByteArray`，只允许 `ready`、`error_frame`、`error_command`。

- [ ] **Step 1: 写 native 协议失败测试**

在 `native/dtk-overlay/tests/protocol_test.cpp` 建立最小断言程序，覆盖分段帧、超限、错误命令与 UTF-8 文本不经日志转换：

```cpp
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
```

- [ ] **Step 2: 运行测试，确认其因缺少构建目标而失败**

Run: `cmake -S native/dtk-overlay -B build/dtk-overlay && cmake --build build/dtk-overlay && ctest --test-dir build/dtk-overlay --output-on-failure`

Expected: FAIL，原因是 `native/dtk-overlay/CMakeLists.txt` 不存在。

- [ ] **Step 3: 实现严格的协议层和 CMake 构建**

建立如下 CMake 依赖关系，显式使用系统 DTK 6 导出的 CMake targets，不安装任何 Python Qt binding：

```cmake
cmake_minimum_required(VERSION 3.16)
project(fun-voice-overlay LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Test)
find_package(Dtk6Gui REQUIRED)
find_package(Dtk6Widget REQUIRED)

add_library(fun-voice-overlay-protocol STATIC src/protocol.cpp src/protocol.h)
target_link_libraries(fun-voice-overlay-protocol PUBLIC Qt6::Core)
target_include_directories(fun-voice-overlay-protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()
add_executable(overlay_protocol_test tests/protocol_test.cpp)
target_link_libraries(overlay_protocol_test PRIVATE fun-voice-overlay-protocol Qt6::Core)
add_test(NAME overlay_protocol_test COMMAND overlay_protocol_test)
```

在 `protocol.cpp` 以 `quint32` 大端长度实现 `encodeFrame`，在 `FrameDecoder` 中保留不超过
64 KiB 的未完成帧；任何声明长度为零或大于 65536 的帧均清空缓冲并返回 `TooLarge`。使用
`QJsonDocument::fromJson` 与显式 `isString` / `isDouble` 检查；`show` 必须有非空、最多 128
字节的 `phase`，两个文本字段均为字符串，`level` 缺失或为 0--100 的整数。`clear` 和
`shutdown` 不接受文本字段。解析失败将 `QStringLiteral("error_command")` 写入错误参数，绝不
拼入 payload。

- [ ] **Step 4: 运行 native GREEN 测试**

Run: `cmake -S native/dtk-overlay -B build/dtk-overlay && cmake --build build/dtk-overlay && ctest --test-dir build/dtk-overlay --output-on-failure`

Expected: `overlay_protocol_test` PASS。

- [ ] **Step 5: 提交协议基础设施**

```bash
git add native/dtk-overlay/CMakeLists.txt native/dtk-overlay/src/protocol.h \
  native/dtk-overlay/src/protocol.cpp native/dtk-overlay/tests/protocol_test.cpp
git commit -m "feat: add private dtk overlay protocol"
```

## Task 2: 实现 DTK 底部居中、主题自适应窗口

**Files:**

- Create: `native/dtk-overlay/src/overlay_window.h`
- Create: `native/dtk-overlay/src/overlay_window.cpp`
- Create: `native/dtk-overlay/tests/window_test.cpp`
- Modify: `native/dtk-overlay/CMakeLists.txt`

**Interfaces:**

- Produces `OverlayWindow::showModel(const OverlayCommand &)`, `clearModel()`, `setBlurAvailableForTest(bool)` 和 `static QRect bottomCenteredRect(const QRect &, QSize)`。
- `showModel` 只接受已验证的 `OverlayCommand::Show`；`clearModel` 必须先清除全部 `QString`/`QLabel` 文本，再隐藏。
- `bottomCenteredRect(available, requested)` 输出位于 `available.bottom() - 36` 之上的、受 320/720/三分之一约束的矩形。

- [ ] **Step 1: 写窗口布局与 Unicode 的失败测试**

在 `native/dtk-overlay/tests/window_test.cpp` 写入以下行为测试，且 CMake 将测试设为 `QT_QPA_PLATFORM=offscreen`：

```cpp
#include "overlay_window.h"
#include <QApplication>
#include <cassert>

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
    window.showModel(command);
    assert(window.stableTextForTest() == command.stableText);
    assert(window.provisionalTextForTest() == command.provisionalText);
    window.clearModel();
    assert(window.stableTextForTest().isEmpty());
    assert(window.provisionalTextForTest().isEmpty());
    assert(!window.isVisible());
}
```

补充断言：窗口旗标包含 `Qt::FramelessWindowHint`、`Qt::WindowStaysOnTopHint`、
`Qt::WindowDoesNotAcceptFocus` 与 `Qt::WindowTransparentForInput`，且 `focusPolicy()` 为
`Qt::NoFocus`；`setBlurAvailableForTest(false)` 后卡片仍有不透明主题背景。

- [ ] **Step 2: 运行测试，确认其因缺少窗口实现而失败**

Run: `cmake -S native/dtk-overlay -B build/dtk-overlay && cmake --build build/dtk-overlay && ctest --test-dir build/dtk-overlay --output-on-failure`

Expected: FAIL，原因是 `overlay_window.h` / `overlay_window.cpp` 尚未定义。

- [ ] **Step 3: 实现 DTK 卡片与确定性布局**

在 `OverlayWindow` 构造器中使用以下窗口安全属性，并以 `DFloatingWidget` 容纳 `QVBoxLayout`
内的状态、音量、稳定和推测 `QLabel`：

```cpp
setWindowFlags(Qt::ToolTip | Qt::FramelessWindowHint |
               Qt::WindowStaysOnTopHint | Qt::WindowDoesNotAcceptFocus |
               Qt::WindowTransparentForInput);
setAttribute(Qt::WA_TranslucentBackground, true);
setAttribute(Qt::WA_ShowWithoutActivating, true);
setAttribute(Qt::WA_TransparentForMouseEvents, true);
setFocusPolicy(Qt::NoFocus);

card_->setFramRadius(16);
const bool blur = DWindowManagerHelper::instance()->hasComposite() &&
                  DWindowManagerHelper::instance()->hasBlurWindow();
card_->setBlurBackgroundEnabled(blur);
```

将 `DGuiApplicationHelper::themeTypeChanged` 和 `fontChanged` 连接到 `applyTheme()` / `relayout()`。
`applyTheme()` 必须为浅色与深色主题分别设置可读前景、稳定正文、较弱推测文本和卡片背景；当
`blur` 为 false 时使用 alpha 为 255 的背景色。状态文字必须由一个仅含固定值的函数生成：

```cpp
static QString phaseLabel(const QString &phase) {
    static const QHash<QString, QString> labels{
        {"preparing", QStringLiteral("正在准备本地模型")},
        {"recording", QStringLiteral("录音中")},
        {"finalizing", QStringLiteral("正在整理")},
        {"correcting", QStringLiteral("正在精修")},
        {"committing", QStringLiteral("正在输入")},
        {"rehydrating", QStringLiteral("正在恢复本地模型")},
        {"enriching", QStringLiteral("正在整理结果")},
        {"active_idle", QStringLiteral("本地模型就绪")},
    };
    return labels.value(phase, QStringLiteral("语音输入"));
}
```

通过 `QGuiApplication::screenAt(QCursor::pos())` 选择屏幕，缺失时使用 `primaryScreen()`；使用
`availableGeometry()` 和 `bottomCenteredRect()` 移动窗口。每个转写 label 禁止富文本，使用
`Qt::PlainText` 与 `QFontMetrics::elidedText(..., Qt::ElideRight, maxWidth)`，保证中英混排
可显示又不会溢出。`clearModel` 调用每个 label 的 `clear()`、`hide()`，然后 `hide()` 窗口。

在 CMake 中加入：

```cmake
add_executable(overlay_window_test tests/window_test.cpp src/overlay_window.cpp src/overlay_window.h)
target_link_libraries(overlay_window_test PRIVATE fun-voice-overlay-protocol Qt6::Gui Qt6::Widgets Dtk6::Gui Dtk6::Widget)
add_test(NAME overlay_window_test COMMAND overlay_window_test)
set_tests_properties(overlay_window_test PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
```

- [ ] **Step 4: 运行 native 窗口测试**

Run: `cmake --build build/dtk-overlay && ctest --test-dir build/dtk-overlay --output-on-failure`

Expected: `overlay_protocol_test` 与 `overlay_window_test` 全部 PASS。

- [ ] **Step 5: 提交原生视觉层**

```bash
git add native/dtk-overlay/CMakeLists.txt native/dtk-overlay/src/overlay_window.h \
  native/dtk-overlay/src/overlay_window.cpp native/dtk-overlay/tests/window_test.cpp
git commit -m "feat: render dtk bottom centered overlay"
```

## Task 3: 实现 native 事件循环与 Python 按需控制器

**Files:**

- Create: `native/dtk-overlay/src/main.cpp`
- Modify: `native/dtk-overlay/CMakeLists.txt`
- Modify: `src/fun_voice/overlay.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `tests/test_overlay.py`
- Modify: `tests/test_daemon.py`

**Interfaces:**

- `DtkOverlayController(executable: Path, *, popen: OverlayPopen = _default_popen)` 实现 `OverlayController` 的 `show(model)`, `clear()`, `close()`；其 public 方法不抛出子进程或管道异常。
- `default_overlay_executable() -> Path` 返回 `~/.local/lib/fun-voice-ryan/fun-voice-overlay`；测试可通过构造参数注入临时路径。
- native `OverlayApplication::onStdinReadable()` 从 stdin 读长度帧、调用 `parseCommand`、调用窗口的 `showModel` / `clearModel`，并输出长度帧 `ready` 或固定 `error_*` ACK。

- [ ] **Step 1: 写 Python controller 与 daemon 接线失败测试**

将 `tests/test_overlay.py` 的 Xlib fake 替换为内存 `FakeProcess`（提供 `stdin: io.BytesIO`、
`stdout: io.BytesIO`、`poll()`、`terminate()`、`wait()`）。新增以下测试：

```python
def test_dtk_controller_starts_lazily_and_writes_a_bounded_show_frame() -> None:
    spawned: list[FakeProcess] = []
    controller = DtkOverlayController(
        executable=Path("/native/fun-voice-overlay"),
        popen=lambda _argv: spawned.append(FakeProcess()) or spawned[-1],
    )
    controller.show(OverlayModel(phase=DaemonState.RECORDING, level=42))
    assert len(spawned) == 1
    assert decode_one_frame(spawned[0].stdin.getvalue()) == {
        "command": "show", "phase": "recording", "stable_text": "",
        "provisional_text": "", "level": 42,
    }

def test_dtk_controller_clears_without_retaining_or_logging_transient_text() -> None:
    process = FakeProcess()
    controller = DtkOverlayController(executable=Path("overlay"), popen=lambda _argv: process)
    controller.show(OverlayModel(phase=DaemonState.RECORDING, stable_text="私密文本"))
    controller.clear()
    frames = decode_frames(process.stdin.getvalue())
    assert frames[-1] == {"command": "clear"}
    assert not hasattr(controller, "_last_model")

def test_dtk_controller_pipe_failure_is_best_effort_and_next_show_respawns() -> None:
    failed, recovered = BrokenPipeProcess(), FakeProcess()
    processes = iter((failed, recovered))
    controller = DtkOverlayController(executable=Path("overlay"), popen=lambda _argv: next(processes))
    controller.show(OverlayModel(phase=DaemonState.RECORDING))
    controller.show(OverlayModel(phase=DaemonState.FINALIZING))
    assert recovered.stdin.getvalue()
```

在 `tests/test_daemon.py` 保留 `FakeOverlay` 合约，并把生产入口测试断言从
`X11TransientOverlay` 更新为 `DtkOverlayController`，同时继续断言错误、提交和 shutdown 都会
调用 `clear()` / `close()`。

- [ ] **Step 2: 运行 Python RED 测试**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_overlay.py tests/test_daemon.py -q`

Expected: FAIL，原因是 `DtkOverlayController` 与 frame codec 尚未存在。

- [ ] **Step 3: 实现 stdin 路由与安全的 Python 子进程控制**

在 `main.cpp` 使用 `QSocketNotifier(STDIN_FILENO, QSocketNotifier::Read)`；把读取字节交给
`FrameDecoder`，按命令执行并立即写一个不含转写的 ACK。`show` 停止空闲计时并显示；`clear`
执行 `window.clearModel()` 后 `idleExitTimer.start(5000)`；`shutdown` 清空后 `QCoreApplication::quit()`。
stdin EOF 也必须执行清空并退出。标准错误保持为空，禁止 `qDebug`、`qWarning` 和文本日志。

在 `native/dtk-overlay/CMakeLists.txt` 于 Task 2 的窗口目标之后追加生产二进制：

```cmake
add_executable(fun-voice-overlay src/main.cpp src/overlay_window.cpp src/overlay_window.h)
target_link_libraries(fun-voice-overlay PRIVATE fun-voice-overlay-protocol Qt6::Gui Qt6::Widgets Dtk6::Gui Dtk6::Widget)
```

在 `overlay.py` 保留 `OverlayModel`、`OverlayFrame`、`OverlayController` 和 `NullOverlay`，删除
`X11TransientOverlay` 的显示实现。以以下方式编码 `show`，确保 Python 不保存额外模型快照：

```python
def _show_payload(model: OverlayModel) -> bytes:
    payload: dict[str, object] = {
        "command": "show",
        "phase": model.phase.value,
        "stable_text": model.stable_text,
        "provisional_text": model.provisional_text,
    }
    if model.level is not None:
        payload["level"] = max(0, min(100, model.level))
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > OVERLAY_MAX_FRAME_BYTES:
        raise ValueError("overlay frame exceeds bound")
    return len(encoded).to_bytes(4, "big") + encoded
```

`DtkOverlayController` 必须用 `subprocess.Popen([str(executable)], stdin=PIPE, stdout=PIPE,
stderr=DEVNULL, close_fds=True)`，在守护线程中只消费并验证固定 ACK，不记录 ACK 或异常详情。
`show` / `clear` 在 `BrokenPipeError`、`OSError`、超限 `ValueError` 时终止并丢弃该 process 引用；
所有这些异常都在 controller 内吞掉。`close` 写 `shutdown`，关闭 stdin，最多等 0.2 秒，仍未退出
才 `terminate()`，且无论结果均删除引用。下一次 `show` 若 `poll()` 非 `None` 必须拉起新进程。

在 `daemon.py` 将生产构造替换为：

```python
overlay: OverlayController = DtkOverlayController(
    executable=default_overlay_executable()
)
```

除 import 和这个构造点外，不得改变 `VoiceDaemon._show_overlay`、`_clear_overlay`、状态机或
最终上屏路径。

- [ ] **Step 4: 运行 Python GREEN 与静态检查**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_overlay.py tests/test_daemon.py -q && .venv/bin/ruff check src/fun_voice/overlay.py src/fun_voice/daemon.py tests/test_overlay.py tests/test_daemon.py && .venv/bin/mypy src/fun_voice/overlay.py src/fun_voice/daemon.py`

Expected: 全部 PASS，且 mypy 输出 `Success: no issues found`。

- [ ] **Step 5: 提交控制器接入**

```bash
git add native/dtk-overlay/src/main.cpp native/dtk-overlay/CMakeLists.txt \
  src/fun_voice/overlay.py src/fun_voice/daemon.py tests/test_overlay.py tests/test_daemon.py
git commit -m "feat: launch dtk overlay on demand"
```

## Task 4: 接入用户安装、卸载和运维文档

**Files:**

- Modify: `scripts/install-user.sh`
- Modify: `scripts/uninstall-user.sh`
- Modify: `tests/test_install_scripts.py`
- Modify: `README.md`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`

**Interfaces:**

- 构建产物必须位于 `build/dtk-overlay/fun-voice-overlay`。
- 已安装二进制必须位于 `~/.local/lib/fun-voice-ryan/fun-voice-overlay`，模式为 `0755`。
- Python `default_overlay_executable()` 的路径与安装路径必须字节一致。

- [ ] **Step 1: 写部署与文档失败测试**

在 `tests/test_install_scripts.py` 新增静态契约：

```python
def test_installer_validates_and_installs_the_private_dtk_overlay_binary() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    assert 'OVERLAY_BIN="${ROOT}/build/dtk-overlay/fun-voice-overlay"' in install
    assert 'OVERLAY_INSTALL_DIR="${HOME}/.local/lib/fun-voice-ryan"' in install
    assert 'install_file "${OVERLAY_BIN}" "${OVERLAY_INSTALL_DIR}/fun-voice-overlay" 755' in install
    assert "enable --now fun-voice-overlay" not in install

def test_uninstaller_removes_only_the_owned_overlay_binary() -> None:
    uninstall = (ROOT / "scripts/uninstall-user.sh").read_text(encoding="utf-8")
    assert 'remove_file "${OVERLAY_INSTALL_DIR}/fun-voice-overlay"' in uninstall
```

并断言 README 列出 `libdtk6gui-dev`、`libdtk6widget-dev`，操作手册说明缺少 native binary 时语音
可继续但无悬浮窗，验收清单包含深色、浅色、中文、透明和焦点保持检查。

- [ ] **Step 2: 运行 RED 测试**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_install_scripts.py -q`

Expected: FAIL，原因是 DTK overlay 的构建、安装和文档契约尚不存在。

- [ ] **Step 3: 实现用户范围部署与文档**

在 `install-user.sh` 定义并在写入任何用户文件前验证：

```bash
OVERLAY_BIN="${ROOT}/build/dtk-overlay/fun-voice-overlay"
OVERLAY_INSTALL_DIR="${HOME}/.local/lib/fun-voice-ryan"
[[ -f "${OVERLAY_BIN}" ]] \
    || die "source" "DTK overlay binary missing: ${OVERLAY_BIN} (build it first)"
```

在 Fcitx addon 安装后、autostart 前执行：

```bash
install_file "${OVERLAY_BIN}" "${OVERLAY_INSTALL_DIR}/fun-voice-overlay" 755
log "installed private DTK overlay binary into ${OVERLAY_INSTALL_DIR}"
```

在 `uninstall-user.sh` 新增同一 `OVERLAY_INSTALL_DIR` 常量，并仅调用：

```bash
remove_file "${OVERLAY_INSTALL_DIR}/fun-voice-overlay"
```

不要创建 systemd unit、autostart 文件或 socket 来常驻运行 overlay。README 的构建命令改为先构建
fcitx addon，再运行 `cmake -S native/dtk-overlay -B build/dtk-overlay && cmake --build build/dtk-overlay`。
将 DTK6 开发包加入开发前提。运维手册记录 `clear` 后 5 秒退出、二进制丢失时的无 UI 降级和重新
构建/执行 `scripts/install-user.sh` 的恢复步骤。人工验收清单必须逐项验证底部居中、主题切换、
中文/英文/命令混排、透明圆角或无合成器实色降级、以及不会夺取输入焦点。

- [ ] **Step 4: 运行部署 GREEN 与 shell 静态检查**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_install_scripts.py -q && bash -n scripts/install-user.sh scripts/uninstall-user.sh`

Expected: 全部 PASS，`bash -n` 无输出。

- [ ] **Step 5: 提交部署与文档**

```bash
git add scripts/install-user.sh scripts/uninstall-user.sh tests/test_install_scripts.py \
  README.md docs/operations.md docs/acceptance-checklist.md
git commit -m "docs: deploy dtk overlay with desktop assistant"
```

## Task 5: 端到端验证与 DDE 人工验收

**Files:**

- Modify: `docs/acceptance-checklist.md`

**Interfaces:**

- 产物 `build/dtk-overlay/fun-voice-overlay` 可由当前用户启动；daemon 按需拉起它并在 `clear` 后退出。
- 任何验证都不得输出或保存真实临时转写文本。

- [ ] **Step 1: 为人工检查添加不可泄露的验收条目**

在验收清单加入不包含真实语音文本的步骤：切换 DDE 深浅主题，按住/松开 `Super+C`，检查卡片位置、
圆角、半透明/模糊或实色降级、中文固定状态标签、焦点不变和清空后不再可见；再以
`ps -C fun-voice-overlay` 确认 5 秒后没有残留进程。

- [ ] **Step 2: 构建并运行所有自动验证**

Run: `cmake -S native/fcitx5-fun-voice -B build/fcitx && cmake --build build/fcitx && ctest --test-dir build/fcitx --output-on-failure && cmake -S native/dtk-overlay -B build/dtk-overlay && cmake --build build/dtk-overlay && ctest --test-dir build/dtk-overlay --output-on-failure && PYTHONPATH=src .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src/fun_voice && bash -n scripts/install-user.sh scripts/uninstall-user.sh`

Expected: 两套 CTest、全部 Python 测试、Ruff、mypy 和 shell 语法检查全部通过。

- [ ] **Step 3: 安装最新二进制并重启 daemon**

Run: `scripts/install-user.sh && systemctl --user restart fun-voice-daemon.service && systemctl --user is-active fun-voice-daemon.service`

Expected: 输出 `active`；此时 `ps -C fun-voice-overlay` 没有输出，证明登录/重启未常驻 Qt 进程。

- [ ] **Step 4: 执行人工 DDE 验收**

Run: `fun-voice-selftest --format json`

Expected: 在按住并松开一次 `Super+C` 后，`x11_hotkey` 为 `pass`；再按验收清单观察视觉、主题和
焦点行为。若 DTK overlay 无法启动，识别和最终上屏仍必须成功，且日志不含临时文本。

- [ ] **Step 5: 提交最终验收清单更新**

```bash
git add docs/acceptance-checklist.md
git commit -m "test: document dtk overlay acceptance"
```

## Plan Self-Review

- **Spec coverage:** Task 1 覆盖私有有界协议；Task 2 覆盖 DTK、中文、圆角、透明、主题、底部居中与无焦点；Task 3 覆盖按需生命周期、隐私清空和 daemon 接线；Task 4 覆盖可部署性；Task 5 覆盖全量回归与真实 DDE 验收。
- **Placeholder scan:** 没有占位项；每个任务包含具体文件、接口、测试、命令和提交范围。
- **Type consistency:** Python 一律使用 `OverlayModel` / `DtkOverlayController` / `OverlayController`；native 一律使用 `OverlayCommand` / `FrameDecoder` / `OverlayWindow`；安装产物路径与 `default_overlay_executable()` 均为 `~/.local/lib/fun-voice-ryan/fun-voice-overlay`。
