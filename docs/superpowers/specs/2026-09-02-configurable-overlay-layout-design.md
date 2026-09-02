---
title: Fun Voice Ryan 可配置 DTK 悬浮窗布局
status: approved
date: 2026-09-02
scope: 悬浮窗垂直位置、固定宽度和字体缩放的用户级 TOML 配置
supersedes:
  - 2026-09-02-dtk-transient-overlay-design.md 的固定底部位置与固定字号/宽度布局约束
---

# Fun Voice Ryan 可配置 DTK 悬浮窗布局

## 1. 目标

将 DTK 原生瞬态悬浮窗从固定的底部位置和隐含尺寸，调整为可由用户级配置文件控制的中下部
显示样式。默认窗口中心放在活动屏幕工作区高度的 70% 处，窗口比现有版本更宽，中文状态和
识别文本更大、更易读。

本次只调整悬浮窗的几何与字体。语音模型、录音、ASR、校正、说话人信息、结构化接口、热键、
剪贴板与 Fcitx 提交链路均不改变。

## 2. 已确认配置契约

配置仍只从 `${XDG_CONFIG_HOME:-~/.config}/fun-voice-ryan/config.toml` 读取。新增可选
`[overlay]` 表；旧配置文件不含该表时使用以下默认值：

```toml
[overlay]
# 悬浮窗中心在当前屏幕工作区中的垂直比例。
vertical_center_ratio = 0.70

# 悬浮窗固定逻辑宽度（像素）；必要时会为小屏幕自动收缩。
width_px = 680

# 状态、音量、稳定文本和推测文本使用的统一字号倍率。
font_scale = 1.0
```

| 字段 | 默认值 | 合法范围 | 语义 |
| --- | ---: | --- | --- |
| `vertical_center_ratio` | `0.70` | `0.50`--`0.85` | 窗口几何中心相对当前屏幕**可用工作区**的垂直位置。 |
| `width_px` | `680` | `420`--`1000` | 逻辑固定宽度；实际宽度不会超过工作区宽度减去 48 px。 |
| `font_scale` | `1.0` | `0.80`--`1.80` | 基准字号的缩放倍率。 |

缺失字段逐项回退默认值；未知字段保持当前配置系统的忽略语义。已知字段存在但类型错误、非有限
浮点数或超出范围时，daemon 在启动前报出字段名明确的 `ConfigError`，不会悄悄采用一个意外的
布局。

## 3. 布局与视觉规格

### 3.1 屏幕和位置

沿用现有的屏幕选择策略：在收到第一帧 `show` 时，选择鼠标所在的 `QScreen`；取不到时使用主
屏。不会在显示过程中跟随鼠标，也不会重新定位到其他屏幕。

记可用工作区为 `available`，窗口高度为 `height`，则目标中心 Y 为：

```text
available.top + round(available.height * vertical_center_ratio)
```

随后将窗口矩形夹紧到工作区四周至少 24 px 的安全边距内。这样在带顶栏、底部 Dock 或小分辨率
屏幕上，70% 仍是中下部位置且不会被裁切。水平位置始终工作区居中。

### 3.2 尺寸和文本

宽度为 `width_px`，在窄屏上按工作区宽度收缩；内容只决定高度。高度继续受工作区三分之一上限
约束，过长文本按 Qt Unicode 字边界右侧省略，不换行到屏幕外。

系统字体族和 CJK 回退继续交给 Qt/DTK。基准逻辑字号为：状态 18 pt（DemiBold）、稳定和推测
文本 15 pt、音量提示 13 pt；每项乘以 `font_scale`。`font_scale = 1.0` 已是比原有窗口更大的
默认字号，不依赖系统默认点数的差异。

圆角、透明/模糊主题适配、非激活、鼠标穿透和无焦点属性保持不变。

## 4. 架构和数据流

`Config` 新增不可变的 `OverlayConfig` 值对象。Python 是唯一的 TOML 解析和校验位置；native
DTK 程序不读取配置文件，也不会接触用户目录。

```text
config.toml [overlay]
        |
        v
Python load_config() --typed/validated--> Config.overlay
        |
        v
DtkOverlayController lazy spawn
        |
        |  --vertical-center-ratio / --width-px / --font-scale
        v
fun-voice-overlay (Qt/DTK)
        |
        v
OverlayWindow geometry and fonts
```

daemon 将经过验证的三个纯数字参数作为启动参数传给私有 overlay 子进程。私有 stdin/stdout 的
长度帧 JSON 协议保持不变，仍然只传递内存中的瞬态状态和转写文本。布局参数绝不进入 `show`
帧、日志、剪贴板或持久化存储。

C++ 侧以 `QCommandLineParser` 解析参数，并再次进行防御性范围校验；直接手工运行 native
二进制时，缺省值与 Python 默认值一致。参数无效时进程立即以非零状态退出；daemon 将本次
悬浮窗降级为无 UI，主语音链路不受影响。

配置只在 daemon 启动时读取。用户修改配置后需执行：

```bash
systemctl --user restart fun-voice-daemon.service
```

重启会关闭旧 overlay 子进程，因此下一次显示必然使用新布局。此次不加入文件监听、热更新或
设置面板。

## 5. 故障处理与兼容性

- 无 `[overlay]` 表或旧版本配置：完全兼容，使用本文默认布局。
- 小于最小宽度的屏幕：运行时以可用工作区宽度减 48 px 为准，绝不越界。
- 顶栏/Dock 改变可用工作区：每次显示重新读取 `availableGeometry()`，位置随工作区正确计算。
- native 参数或进程异常：沿用 `NullOverlay` 降级语义；录音、识别、校正和提交继续执行。
- 任何异常、clear、取消或退出：照旧清空所有瞬态文本引用；本次不增加新的文本保留路径。

## 6. 测试与验收

1. Python 配置单测：默认值、合法覆写、缺失字段、类型错误、边界值和越界值。
2. Python overlay 控制器单测：验证懒启动参数精确包含三项已校验的数值，且不把文本放入 argv。
3. native CTest：验证命令行默认值与边界拒绝、70% 几何中心、24 px 夹紧、固定宽度的小屏收缩，
   以及三类标签的缩放字号。
4. 回归：现有协议、clear 隐私清理、透明/无焦点属性、按需启动和 idle 退出测试继续通过。
5. 人工 DDE 验收：在常用屏幕按 `Super+C`，窗口处在屏幕中下部，宽度和中文字号明显大于旧版；
   修改 `[overlay]` 后重启用户服务，三项配置分别生效且焦点仍留在目标应用。

## 7. 非目标

- 不支持运行中的 TOML 热重载，也不添加图形化设置入口。
- 不开放透明度、圆角、颜色、动画、屏幕选择或文本换行策略配置。
- 不改变 overlay 私有协议，不在任何参数、日志或磁盘中保存识别文本。
