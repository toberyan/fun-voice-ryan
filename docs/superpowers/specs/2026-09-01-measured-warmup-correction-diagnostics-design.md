---
title: Fun Voice Ryan 可测预热与 Qwen 校对诊断设计
status: approved
date: 2026-09-01
scope: 私有细粒度时延指标、Nano 合成预热、Qwen 校对拒绝原因
supersedes: []
---

# Fun Voice Ryan 可测预热与 Qwen 校对诊断设计

## 背景和目标

一次真实 3.13 秒录音的内存指标显示，松键到提交为 30.22 秒：Nano 预加载为 21.16 秒，
ASR 阶段为 20.12 秒，Qwen 校对为 8.58 秒，提交仅 5 毫秒。worker 日志中的实际 Nano 音频
编码和生成分别约为 1.20 秒和 0.49 秒。因此本阶段不降低 Nano 精度、不会换模型，也不会
让两个模型同时驻留；目标是精确量化其余等待，并把首次 Triton JIT 置于录音期。

## 已确认约束

- 所有模型仍只能使用 `xpu:0`；Nano 为主模型，SenseVoiceSmall 仅为 `model_load` / `oom`
  回退；Qwen 固定为 `Qwen/Qwen3.5-0.8B`。
- 不记录、不回显、不落盘音频、文本、路径、焦点、Fcitx token、模型错误详情或原始例外。
- metrics socket 仅返回 128 条内存记录的聚合计数、P50/P95 和固定枚举直方图。
- Qwen 在 producing ASR service 已确认 inactive/failed 后才启动；任一诊断或预热失败都不
  阻断原始 ASR 上屏。
- 禁止启动时加载神经模型，也不新增 CPU/CUDA fallback 或第二个校对模型。

## 方案比较

1. **仅调高 KV Cache 或缩短 token 限制。** 不能解决 21 秒模型加载，并会损害长句的准确率；
   不采用。
2. **登录时预热或并发保留 Nano/Qwen。** 可以降低首句延迟，但违背已确认的低内存和 XPU
   互斥要求；不采用。
3. **录音期合成预热 + 细粒度测量（采用）。** 在 Nano 权重/VAD/vLLM 已加载之后，使用仅在
   内存中的固定合成 PCM 触发一小段 engine generate。它不经过 VAD、不读取用户音频、不会
   产生或提交文本。预热与录音并行，失败只产生固定枚举。结合分阶段指标，可决定后续是否
   值得提供可选的轻量 worker 常驻模式。

## 数据流

```text
有效录音开始
  -> daemon 记录 preload 总时长
  -> Nano worker: runtime load -> 合成 PCM warmup -> 返回固定时间值
松键
  -> daemon: ASR wall time - worker time = 排队/transport
  -> Nano: audio load -> VAD -> engine generate，返回仅耗时
  -> daemon: stop producing ASR，记录 lease 停止时长
  -> Qwen child: model load -> generate -> validate，返回仅耗时或固定拒绝原因
  -> 仅聚合 metrics；最终仍提交修正文本或原始文本
```

## 私有指标合同

新增的 timing 字段均为非负整数毫秒：

- `preload_worker_ms`、`preload_runtime_load_ms`、`preload_warmup_ms`；前者是 worker 内
  preload 总时长，daemon `preload_ms` 减去它可定位 systemd/socket 等外部等待。
- `asr_worker_ms`、`asr_queue_transport_ms`、`asr_audio_load_ms`、`asr_vad_ms`、
  `asr_generate_ms`；queue/transport 是 daemon ASR wall time 与 worker elapsed 的差值。
- `asr_release_ms`、`correction_model_load_ms`、`correction_generate_ms`、
  `correction_validate_ms`；Qwen 总时长仍保留在 `correction_ms`。

所有字段只通过现有 `{"op":"metrics"}` 的汇总端点暴露；缺失阶段不填值。不会返回逐条
会话记录或任何输入输出负载。

## Nano 合成预热

`NanoRuntime.warmup()` 构造固定的一秒 16 kHz float32 零 PCM，直接调用现有受锁的 ASR
generate 路径。预热只在 `LazyTranscriber.preload()` 成功 materialize Nano runtime 后执行一次。
不调用 VAD，不持久化样本，不把返回的模型文本保存、记录、返回或注入。预热失败不会卸载
已可用 runtime；worker 返回 `warmup_ms` 为空且 daemon 增加 `nano_warmup=failed` 聚合。

## Qwen 拒绝原因

保留外部稳定错误码 `correction.invalid_output`，同时附带仅供本地聚合的固定原因枚举：
`envelope_missing`、`envelope_malformed`、`output_empty`、`output_too_long`、`similarity`、
`protected_token`、`input_too_large`、`model_load`、`oom`、`device`、`protocol`、`no_output`、
`generation`、`timeout`、`unavailable`、`internal`。Qwen child 仅回传该枚举和阶段耗时；
daemon 以 `correction_rejection` 直方图聚合。文本和受保护 token 本身永不进入该结果。

## 验收

- 对 fake runtime，预热只调用一次 generate、未调用 VAD、不会修改真实转写；首次真实
  transcribe 可复用同一 runtime。
- Nano response/daemon metrics 能提供上述 stage timing，报告 `repr` 不含音频路径或正文。
- Qwen 的 envelope、相似度和技术词拒绝原因分别聚合，但对桌面行为仍统一回退原始文本。
- 基线单测、ruff、mypy 和现有 XPU-only/lease 测试全部通过；真实下一次短录音可从 metrics
  判断预热是否覆盖 JIT，以及 Qwen 失败属于哪一个固定类别。
