# 修复验证设计：配置、POC 与 DDE 证据

## 目标

在不改变「Deepin DDE X11 + Super+C 按住说话 + Fun-ASR-Nano XPU + 原样上屏」主链路的前提下，修复验证发现的剩余缺口，并让自检不再把模拟测试误报成真实桌面验收。

## 已确认保留的行为

- C 键松开由 daemon 的独立轮询线程检测；不依赖 DDE 是否发出松键 action。
- Worker 的响应上限为 4 MiB；Fcitx 多帧只在末帧原子提交。
- CPU FSMN-VAD 限制单段不超过 30 秒，Nano 在 XPU 上逐段直接拼接原始模型文本。
- 音频与转写文本不持久化，日志、报告和诊断数据均不得含两者。

## 修复方案

### 1. 单一 TOML 配置成为真实来源

`config.toml` 的 Fcitx 超时使用规范的 `commit_timeout_ms = 500` 毫秒整数；daemon 在唯一边界处换算为秒。Worker 启动时读取同一个 TOML，并将 `inference.device`、`dtype`、`gpu_memory_utilization`、`enforce_eager` 传入 `load_nano_runtime`。`device` 只接受 `xpu:0`，以防通过配置或 CLI 走 CPU 回退。

示例配置和运维文档只展示真正生效的键；固定不允许修改的热键、模型、Fcitx 主通道、录音上限和历史策略明确写为固定约束，而非伪配置。

### 2. POC 必须与 Worker 的分段契约完全一致

POC 的分段解码复用 Worker 同等条件：VAD 区域按起点排序、固定重叠切片、一次批量推理，且 Nano 返回结果数必须精确等于切片数，每项均必须是带字符串 `text` 的字典。否则 POC 返回该检查失败，不能以部分文本放行。

脚本以 `umask 077` 创建报告和临时样本，写入每个公开样本的来源、语言和时长；退出时仍删除样本。报告只含来源元数据、时长、段数、文本长度和设备指标。

### 3. 自检区分模拟桥接与真实按住事件

daemon 仅在某次 `start_if_idle` 读取到 `C` 为按下时，在进程内设置一个布尔诊断标记。该标记通过同 UID daemon socket 的 `diagnostics` 操作读取，重启即丢失，不包含时间、音频、文本、焦点或 token。自检保留 bridge 映射的 fake 测试，但 `bridge_hold_timing` 只有在标记为真时才通过。

这证明一次 start 请求在 C 按下时到达 daemon；浏览器、终端、IDE 上屏和「不误输入 c」仍由现有真实 DDE 验收清单人工确认，不能伪造为自动化通过。

### 4. 运行态卫生与文档一致性

删除经过路径和属主确认的历史 `investigate-samples` 临时 WAV。修复测试导入格式，更新运维文档、POC 文档与验收清单，使其描述当前配置、POC 证据和人工验收边界，不记录每次随机推理的精确文本长度。

## 备选方案与取舍

1. **推荐：上述方案。** 配置真正生效，POC 不能部分通过，DDE 验证拥有可重启即丢失的真实按住证据。
2. **仅更新文档和人工清单。** 改动较少，但错误配置仍静默无效，自检仍会把 fake 测试显示为 pass。
3. **移除全部可调推理配置。** 能避免未接线配置，却背离已确认的单 TOML 设计，也不解决毫秒单位与 Worker 读取问题。

## 验收

- 配置测试证明 `commit_timeout_ms=500` 传给 Fcitx 时为 `0.5` 秒，且 Worker 使用 TOML 的 XPU 参数并拒绝 CPU。
- POC 测试拒绝少结果、过多结果和非字符串结果；真实 POC 的 60 秒检查至少有两段，报告来源非空。
- daemon diagnostics 在 fake C 按下 start 后为真；无真实按住事件时 selftest 的 `bridge_hold_timing` 为 fail。
- pytest、ruff、mypy、原生 Fcitx CTest、真实 XPU POC 和 selftest 均以最新输出验证；最后由人工完成 DDE 验收清单。
