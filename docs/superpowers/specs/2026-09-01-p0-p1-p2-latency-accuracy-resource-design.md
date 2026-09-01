---
title: Fun Voice Ryan P0/P1/P2 准确率、实时性与 XPU 资源平衡设计
status: approved
date: 2026-09-01
scope: 本地指标、录音期 Nano 预加载，以及 Nano/Qwen 的互斥 XPU 生命周期
supersedes: []
---

# Fun Voice Ryan P0/P1/P2 准确率、实时性与 XPU 资源平衡设计

## 已确认决策

| 项目 | 决策 |
| --- | --- |
| 优先级 | 准确率优先；不能以 CPU、其他修正模型或静默降级换取速度。 |
| ASR | `Fun-ASR-Nano-2512` 仍为主模型；`SenseVoiceSmall` 仅处理 Nano `model_load` / `oom`。 |
| 修正模型 | 仅 `Qwen/Qwen3.5-0.8B`、BF16、`xpu:0`、非思考模式。 |
| 登录态 | 不加载任何神经网络；轻量 X11 daemon 可常驻。 |
| P1 | 一次有效按住 `Super+C` 后，在录音期间异步预加载 Nano 和 VAD。 |
| P2 | 不依赖不可用的跨进程显存读数；以进程级 XPU 租约保证 Nano 与 Qwen 不重叠。 |
| P0 隐私 | 运行指标仅保存在内存；基准报告只含聚合数值，绝不记录转写、参考文本、音频或路径。 |

## 现状与目标

一次约五秒的真实录音中，daemon 从 `transcribing` 到 `committing` 为约 31.4 秒；Nano
自身音频编码和生成日志分别约 1.22 秒和 0.65 秒。冷启动和逐次加载 Qwen 是主要等待来源。
当前 `LazyTranscriber` 只在 `transcribe` 请求中加载模型，因此仅提前启动 worker 不能缩短
加载时间；必须增加显式预加载操作。

目标是把 Nano 冷加载与用户说话时间重叠，保留已验证的 Nano 识别路径，同时确保 Nano
空闲期和 Qwen 单请求加载不会在 XPU 上重叠。真实准确率和延迟调优必须先有可复现、无敏感
内容的度量数据。

## 范围与非目标

本设计实现 P0/P1/P2，不实现流式局部上屏、连续录音识别、说话人识别、实时显存驱动查询或
第二个修正模型。模型输出的结构化时间段保持原样；本阶段不对 VAD 段边界做自动去重或补词。

## P0：内存指标和本地基准

### 运行时指标

daemon 持有至多 128 条内存 `SessionMetric`。每条只包含：

- 单调会话序号；
- `capture_duration_ms`、`preload_ms`、`asr_ms`、`correction_ms`、
  `commit_ms`、`end_to_end_ms`；
- `asr_profile`（`nano` / `sensevoice`）、`nano_preload`（未请求、进行中、就绪、失败）、
  `correction`（禁用、尝试/接受、跳过、超时或失败）和最终错误类别；
- `nano_was_stopped_for_qwen` 布尔值。

没有文本、音频、文件路径、窗口名、Fcitx token、模型输出或例外详情。daemon 私有 socket
新增 `{"op":"metrics"}`，仅同 uid 调用方可获取聚合计数及 P50/P95；不会落盘，也不会
写入 journal。

### 离线基准工具

新增本地 CLI，接受用户自有的 JSONL manifest。每行指定分类、音频文件和参考文本；该
manifest 不在仓库中，CLI 不会回显任何输入文本。工具在内存中计算并输出：中文 CER、术语
和代码 token 精确率、标点 F1、修正接受/拒绝计数、Nano/Qwen 的冷/热 P50/P95 与失败计数。
输出文件仅为分类聚合指标，可选且以 `0600` 写入。没有 manifest 时 CLI 拒绝运行，不伪造
准确率结论。

## P1：录音期 Nano 预加载

```text
有效按住 Super+C
  -> 建立焦点 token，成功启动 PipeWire 录音
  -> state=RECORDING
  -> 独立线程请求 Nano worker 的 preload
  -> worker 的 LazyTranscriber 构造 Nano + VAD
松键
  -> 正常 transcribe；若 preload 尚未完成，复用其同一加载操作并等待
```

worker 新增私有 `preload` 协议操作。它只调用 `LazyTranscriber` 的加载路径并返回
`model_ready`，不接收音频、不生成文本、不启动 SenseVoice。worker 仍是单线程服务，因此
`preload` 与随后的 `transcribe` 线性执行，不能并发操作同一 vLLM 引擎。

daemon 仅在录音真正成功且状态已进入 `RECORDING` 后发起后台预加载。短按取消、采集失败、
热键冲突和启动前焦点失败均不预加载。预加载线程不阻塞 X11 事件循环、录音启动或松键处理；
它的错误只写入无敏感指标，实际转写仍走现有 Nano/SenseVoice 错误分类与回退规则。

## P2：保守 XPU 租约与 Qwen 修正

当前机器没有可靠的跨进程 XPU 空闲显存接口，因此不根据猜测的“可用显存”决定并发。改用
daemon 所有的 `XpuLeaseCoordinator`：

1. Nano 转写完成后，若 Qwen 修正已启用，coordinator 请求停止 Nano template service；
2. 它必须确认 Nano 单元不再 active，才允许启动一次 Qwen 子进程；
3. Qwen 无论成功、超时、OOM、输出校验失败均退出；daemon 提交修正文本或原始文本；
4. 下一次有效按住再由 P1 预加载 Nano；SenseVoice 不参与预加载，也不作为修正回退。

停止 Nano 的确认有严格上界；停止失败或超时则不启动 Qwen，并立即使用 Nano 原始文本。这样
宁可少一次修正，也不会出现两大模型同时占用 XPU 或因为资源争用破坏原始识别结果。

修正候选继续经过长度和相似度校验，并新增受保护 token 检验：原文中的 URL、绝对/相对路径、
反引号代码、版本号、`snake_case`、`CamelCase`、命令选项和已配置术语必须在候选中按原顺序
逐字保留。违例即采用原始 Nano 文本。该规则不修改原始 ASR 文本。

## 有效配置

删除对 Transformers Qwen 不生效的 `correction.gpu_memory_utilization` 与
`correction.max_model_len` 概念。修正配置只保留真实生效的固定上限：

```toml
[enhanced.correction]
model = "Qwen/Qwen3.5-0.8B"
device = "xpu:0"
dtype = "bf16"
max_source_characters = 512
max_new_tokens = 512
timeout_seconds = 30
enable_thinking = false
```

30 秒是准确率优先但有界的同步修正预算；超时一定提交原始文本。Nano 的
`gpu_memory_utilization=0.15`、`max_model_len=1536` 和 `idle_unload_seconds=120` 不变。

## 验收条件

- daemon 登录后、未按热键时没有 Nano、SenseVoice 或 Qwen 模型进程；
- 有效按住期间只发生一次 Nano preload；松键后的 transcribe 不会再重复模型构造；
- 一次转写期内不启动 SenseVoice preload；只有 Nano `model_load` / `oom` 才触发它；
- Qwen 启动前 Nano service 已 inactive；停止未确认时 Qwen 不启动且原文可提交；
- 受保护 token 被 Qwen 修改时原文可提交；
- `metrics` 回复与基准报告不包含文本、音频、路径或窗口身份；
- 全部单元测试、端到端 fake 测试、Ruff、mypy 通过；真实基准报告提供冷/热 P50/P95 与各项
  准确率指标后，才调整 token 上限、超时或 XPU Graph 设置。
