---
title: Fun Voice Ryan DTK 原生瞬态悬浮窗
status: approved
date: 2026-09-02
scope: DDE X11 底部居中状态窗、中文渲染、圆角透明主题与按需生命周期
supersedes:
  - src/fun_voice/overlay.py 的 Xlib Core Font 直接渲染实现
---

# Fun Voice Ryan DTK 原生瞬态悬浮窗

## 1. 目标

将当前纯 Xlib 白色矩形状态窗替换为原生 Qt 6 / DTK 6 悬浮窗。新窗口固定显示在当前
屏幕底部居中，正确显示普通话、英文、代码和计算机术语，具有 DDE 原生的圆角、透明、轻微
模糊和深浅主题适配效果。

本次改动只改临时显示层。语音模型、热键、录音、最终文本校正、剪贴板和 Fcitx 提交协议
均不改变。

## 2. 已确认约束

| 领域 | 决策 |
| --- | --- |
| 位置 | 当前活动屏幕底部居中；不再跟随鼠标。 |
| 技术 | 单独的 C++ Qt 6 / DTK 6 原生进程；不在 Python daemon 中嵌入 Qt。 |
| 主题 | 自动跟随 DDE 的深色/浅色主题与字体变化。 |
| 输入 | 不获取键盘焦点、鼠标穿透、不改变目标应用的输入上下文。 |
| 生命周期 | 收到首个 `show` 时按需启动；`clear` 后短暂保活并自动退出；daemon 退出时立即结束。 |
| 隐私 | 临时文本只存在 daemon 与 overlay 子进程内存中；不写日志、磁盘、剪贴板或目标输入框。 |
| 兼容性 | DDE X11 为目标；无合成器/无模糊能力时降级为不透明圆角卡片。 |

## 3. 根因与边界

当前 `X11TransientOverlay` 通过 Xlib Core Font 的 `draw_text` 直接绘制 Python UTF-8 字符串。
Core Font 不是 Unicode 文本布局系统，不能可靠选择中文字体或处理中英文混排，因此会产生
乱码或缺字。窗口同时使用父窗口深度和纯白背景像素，既没有 alpha 通道，也没有圆角裁剪或
合成器效果。

DTK overlay 负责展示一个不可交互的 `OverlayModel` 快照；它不解释、不校正、不记录文本。
daemon 仍然是会话状态机和清理时序的唯一权威。任何 native overlay 故障必须降级为无 UI，
绝不能阻塞录音、ASR、最终文本提交或模型卸载。

## 4. 架构

```text
Python Control Daemon
  └─ DtkOverlayController
       ├─ lazy spawn: fun-voice-overlay
       ├─ private inherited stdin/stdout pipe
       ├─ bounded length-prefixed UTF-8 frames
       └─ watchdog / clear / shutdown

fun-voice-overlay (C++ / Qt 6 / DTK 6)
  ├─ QApplication event loop
  ├─ OverlayWindow : QWidget
  │   ├─ DFloatingWidget / DBlurEffectWidget
  │   ├─ Qt text layout and system CJK fonts
  │   └─ DGuiApplicationHelper theme listener
  └─ protocol reader + idle-exit timer
```

### 4.1 进程与协议

`DtkOverlayController` 懒启动 `fun-voice-overlay`，以其专有的 stdin/stdout 作为进程私有
双向通道，不建立可被其他进程连接的监听 socket。每帧使用 4 字节大端长度前缀与 UTF-8 JSON
payload，单帧上限 64 KiB；超限、无效 UTF-8、未知命令或子进程退出均只作为固定枚举错误
处理，绝不记录 payload。

协议仅包含：

- `show`：`phase`、`stable_text`、`provisional_text`、`level`；更新内存态并显示窗口。
- `clear`：立即隐藏、清空所有文本和状态引用，并启动短暂空闲退出计时。
- `shutdown`：立即清空、隐藏和退出。
- `ready` / `error`：子进程回传不含文本的固定状态码。

daemon 向 overlay 写入失败时销毁该子进程引用，本次会话退化为无悬浮窗；下一次 `show` 可重新
拉起一次。退出和重启都必须丢弃旧会话帧，不能将旧文本显示到新会话。

### 4.2 DTK 窗口

`OverlayWindow` 使用无边框工具提示类窗口旗标，并额外设置：始终置顶、禁止激活、透明输入
区域和无焦点策略。窗口出现、隐藏和销毁都不调用输入法、剪贴板或 XTEST。

窗口的容器使用 `DFloatingWidget` 的圆角与背景能力；可用时启用其 `DBlurEffectWidget` 背景，
并以 `DWindowManagerHelper` 的合成/模糊能力作为开关。卡片使用 16 px 圆角、半透明主题色、
轻阴影和 12--16 px 内边距。无合成器或模糊失败时保留圆角、阴影和主题前景色，改为实色背景，
确保文本对比度。

`DGuiApplicationHelper` 的主题和字体变化信号驱动重绘：浅色模式为半透明浅色背景与深色正文，
深色模式为半透明深色背景与浅色正文。文本由 Qt 的 Unicode 排版和系统字体回退渲染；禁止
再调用 Xlib `draw_text`。

### 4.3 布局与状态

窗口在 `QGuiApplication::screenAt(cursorPos)` 所在屏幕的可用工作区底部居中，距底边 36 px；
没有光标信息时选主屏。最小宽度为 320 px，最大宽度为 `min(720 px, 可用宽度 - 48 px)`，高度
由内容决定且不超过工作区高度的三分之一。超过显示范围的文本按 Unicode 字边界省略，绝不
换行到屏幕外。

内容从上到下为：状态行（图标和固定中文状态标签）、可选音量指示、稳定转写、推测转写。
稳定转写使用主题正文色；推测转写使用较低对比度的辅助色；内容为空时只显示状态和音量。文本
仅在 `show` 至 `clear` 区间保留，`clear`、取消、失败、会话 supersede 和进程退出都会清空。

## 5. 故障处理与资源预算

- 缺少 overlay 二进制、DTK 运行库、显示服务或子进程启动超时：`NullOverlay` 语义生效，语音
  主链路继续运行。
- 合成器、透明或模糊不可用：使用主题适配的不透明圆角卡片；不回退至白色矩形。
- 进程崩溃、协议损坏、写入超时：丢弃所有待显示帧，不重试本次帧；下次会话按需重启一次。
- 登录空闲时不启动 Qt/DTK 进程；显示完成后的空闲退出时间固定为 5 秒。该进程不加载任何
  ASR/Qwen/CAM++ 模型，也不访问 XPU。

## 6. 测试与验收

1. Python 单元测试验证 lazy spawn、帧上限、隐私清空、崩溃降级、下一会话重启，以及旧帧不能
   跨会话显示。
2. C++ 单元测试验证协议拒绝规则、`clear` 后无文本引用、尺寸限制、底部居中计算、深浅主题
   颜色与无合成器降级。
3. X11 集成测试验证窗口不激活、不请求键盘焦点、输入区域为空、`clear` 后隐藏；中文、英文、
   代码混排使用 Qt `QString` 往返验证。
4. 人工 DDE 验收：在深色和浅色主题下按 `Super+C`，检查窗口底部居中、中文可读、透明圆角、
   模糊可用或优雅降级；录音与目标应用焦点保持不变。
5. 回归：完整 Python 测试、native CMake 测试、Ruff、mypy 与安装脚本测试均通过；无模型在登录
   或 overlay 显示时被额外加载。

## 7. 非目标

- 不在本次加入悬浮窗点击操作、编辑、历史记录、结果预览或视觉设置面板。
- 不改变 Wayland 支持范围，不通过 DDE 全局快捷键接口替换现有 X11 热键机制。
- 不保存、上报或复用临时转写文本。
