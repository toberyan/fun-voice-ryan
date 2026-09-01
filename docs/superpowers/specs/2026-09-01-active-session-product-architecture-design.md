---
title: Fun Voice Ryan 活跃会话端侧语音助手产品架构
status: review
date: 2026-09-01
scope: 热会话 Nano、悬浮临时转写、条件 Qwen、结构化结果、说话人身份与质量闭环
supersedes: []
---

# Fun Voice Ryan 活跃会话端侧语音助手产品架构

## 1. 产品定位

Fun Voice Ryan 是 Deepin DDE X11 上的端侧语音输入助手。用户按住 `Super+C` 说普通话，
可夹杂英文、代码、命令和计算机术语；松键后将经确认的最终文本安全提交到录音开始时的
输入焦点。

本架构的核心目标不是让所有模型常驻，而是在一次成功使用后建立一个可控时长的 **Nano
活跃会话**：连续输入以热态 Nano 获得低延迟和高准确率；空闲、锁屏或资源压力出现时释放
模型。录音过程中以不写入目标应用的悬浮窗显示临时状态和临时转写。

## 2. 已确认约束

| 领域 | 决策 |
| --- | --- |
| 桌面平台 | Deepin DDE X11；全局热键固定为 `Super+C`。 |
| 主识别 | `FunAudioLLM/Fun-ASR-Nano-2512`，准确率优先。 |
| 设备 | 仅 `xpu:0`；禁止 CPU 或 CUDA 模型回退。 |
| 备用 ASR | `iic/SenseVoiceSmall` 只处理 Nano `model_load`/`oom`。 |
| 校正模型 | 仅 `Qwen/Qwen3.5-0.8B`，BF16、非思考、单次按需进程。 |
| 启动内存 | 登录只启动轻量 daemon；Nano、SenseVoice、Qwen、CAM++ 均不得随登录加载。 |
| 活跃会话 | 第一次成功识别后 Nano 默认保持 8 分钟；省内存策略缩短为 2 分钟。 |
| 隐私 | 不持久化音频、原文、临时文本、焦点、Fcitx token 或模型异常原文。 |
| 输出 | 仅最终文本可写剪贴板/Fcitx；结构化结果只经 owner-only 本地接口提供。 |
| 身份 | 仅加密保存经质量验证的声纹模板，不保存注册音频或转写。 |

## 3. 目标与非目标

### 目标

- 热态普通短句从松键到最终上屏 P95 不高于 3 秒，且不为无风险短句加载 Qwen。
- 冷态首次输入始终显示明确的“正在准备本地模型”状态；不伪造即时结果。
- 临时文本永不进入目标应用；最终结果保持现有焦点校验与 Fcitx 优先提交链路。
- 为每次完成会话提供内存态的文本、时间戳、说话人聚类、身份匹配和模型溯源结果。
- 所有 XPU 模型任务可抢占或取消，新的语音输入永远优先于后台富化。

### 非目标

- 不支持 Wayland、远程网络推理、多用户共享 daemon 或跨主机同步。
- 不将 SenseVoice 用作普通路径的低质量实时字幕，也不让 Nano 与 Qwen 为预热而并发驻留。
- 不用未校验的 Qwen 候选替换原文，不用低置信度自动切换 ASR 模型。
- 不把音频、文本或身份模板用于云端训练、遥测或远程诊断。

## 4. 组件边界

```text
Control Daemon
├─ X11 Hotkey / Focus Guard / Fcitx Committer
├─ Overlay Controller
├─ Active Session Controller
├─ XPU Scheduler
├─ Result Store API
└─ Identity Vault API

Capture Pipeline
├─ PipeWire Recorder
├─ in-memory Ring Buffer
└─ VAD Endpointer

Model Executors (all XPU-only)
├─ Nano Worker: provisional, stable and final ASR
├─ Qwen Corrector: conditional final-text transaction
└─ Enrichment Worker: timestamps, CAM++ diarization and identity matching
```

`Control Daemon` 不导入模型运行时，不保留模型权重，也不保存会话内容。每个执行器只通过
受限本地 socket 接受带 session capability 的请求；socket 目录为 `0700`，socket 为 `0600`，
且用 `SO_PEERCRED` 限制到当前 UID。

## 5. 会话状态机

```text
IDLE
  -> PREPARING       # 首次按键，启动/加载/预热 Nano
  -> RECORDING       # 模型可用，录音与 VAD endpointing
  -> FINALIZING      # 松键，处理尾段、合并最终 Nano 文本
  -> CORRECTING?     # Risk Gate 命中时才执行，且先释放 Nano
  -> COMMITTING      # 再次校验焦点，写剪贴板与 Fcitx
  -> REHYDRATING?    # Qwen 退出后，活跃会话内异步重新加载 Nano
  -> ENRICHING?      # 后台、低优先级结构化富化
  -> ACTIVE_IDLE     # Nano 继续热驻留，等待下一次按键
  -> IDLE            # 空闲超时、锁屏、资源压力或显式省内存
```

任何异常均可从当前状态回到 `ACTIVE_IDLE` 或 `IDLE`；只有录音成功且最终文本非空时才允许
进入 `COMMITTING`。`ENRICHING` 不是阻塞态：新一次热键会取消它并优先进入 `RECORDING`。
若本次运行过 Qwen，则 `REHYDRATING` 在 Qwen 子进程退出后异步恢复 Nano；它绝不阻塞当前
提交。重载未完成时的新热键仍可录音并显示准备状态，录音结束后使用完成的 Nano 处理。

## 6. 悬浮窗和临时转写

悬浮窗不请求键盘焦点，不修改目标应用光标位置，也不写入剪贴板。它有以下状态：

| 会话状态 | 悬浮内容 | 文本性质 |
| --- | --- | --- |
| `PREPARING` | 准备 Nano、加载或预热进度 | 无文本 |
| `RECORDING` | 音量、时长、VAD 状态 | 无文本或临时文本 |
| 稳定段完成 | 深色稳定片段 | 已封闭，后续不回滚 |
| 连续尾段 | 浅色推测片段 | 可被较新的结果替换 |
| `FINALIZING` / `CORRECTING` | 最终整理或精修状态 | 不写入目标应用 |
| 完成或失败 | 简短固定提示 | 会话内容立即清空 |

### 6.1 临时片段策略

1. VAD 检测到 400–800 ms 停顿后封闭稳定段，将其加入 Nano 队列。
2. 连续语音每 1.5 秒建立一个带固定重叠的推测尾段窗口；结果只用于浅色显示。
3. 松键后只对未封闭尾段做最终解码，再按音频时间顺序合并稳定段和尾段。
4. 每个 Nano runtime 同时最多一个 `generate`；队列优先级严格为：
   `final tail > stable segment > provisional tail`。
5. 当落后时丢弃过期推测任务，不丢弃稳定段或最终尾段。

Nano 的增量短窗输出必须先通过本机 XPU POC，验证不损害最终文本 CER、不会与最终任务死锁，
且不会因重叠片段产生重复上屏。POC 不通过时，悬浮窗仍显示录音、VAD 和已封闭稳定段状态，
但禁用连续尾段推测；不得以第二个 ASR 模型替代。

## 7. 模型生命周期和 XPU 调度

### 7.1 Nano 活跃会话

- 冷态时在热键确认后立刻启动 Nano worker，而不是等待松键。
- runtime 加载完成后，对固定一秒 16 kHz 零 PCM 做一次合成 generate 预热；不经过 VAD，
  不读取或保存用户音频，丢弃输出。
- 每一次完成的 Nano 最终识别刷新 8 分钟活跃截止时间。
- 锁屏、登出、显式“省内存”、持续内存压力、模型连续失败或活跃截止到期均停止 worker。
- `省内存` 策略使用 2 分钟；`持续输入` 策略仅接电时允许 30 分钟，但锁屏与资源压力仍立即卸载。

### 7.2 单一 XPU Scheduler

调度器是唯一可启动/停止模型执行器的组件，且所有状态变更带 session id 和 generation 编号，
防止过期后台结果作用于下一次录音。

```text
1. Nano final tail
2. Nano stable segment
3. Nano provisional tail
4. Qwen correction transaction
5. Enrichment / CAM++ / identity matching
```

Qwen 前必须停止并确认产生该结果的 Nano 或 SenseVoice worker 为 `inactive`/`failed`。Qwen
完成或失败后立即退出；若活跃会话截止时间尚未到期且无资源压力，scheduler 才异步进入
`REHYDRATING` 重新加载 Nano。后台富化只能在 scheduler 空闲时运行，且可被新的录音取消。

## 8. Qwen 风险门控与最终提交

Risk Gate 是无模型、无持久化、仅本次会话内运行的确定性规则。任一条件命中才允许进入
`CORRECTING`：

- 原文缺少句末标点且具有连续口述特征；
- 命中本地同音/术语候选词典，例如 `get commit`、`py test`；
- 句子包含命令、路径、版本号、英文与中文混排，且出现疑似空格或词边界错误；
- 用户在悬浮窗显式选择“精修本段”。

Qwen 候选必须满足受保护技术词、长度、相似度和输出包裹协议校验。任何失败、超时、OOM、
设备错误或租约拒绝均提交 Nano 原文。Risk Gate 的命中类型和 Qwen 拒绝原因只作为固定枚举
写入内存 metrics，不记录真实文本或词典命中词。

`Finalizer` 对最终文本执行以下固定顺序：写剪贴板备份、重新采集焦点、比较录音起始焦点、
优先 Fcitx token 提交、必要时 XTEST Ctrl+V。焦点不一致时永不注入。

## 9. 结构化结果 API

完成提交后，daemon 在仅内存的 `Result Store` 建立一条 `VoiceResult`。最多保存 8 条，固定
TTL 为 600 秒；到期、daemon 退出和用户执行清空操作时删除。查询 API 只允许同 UID，且不对
desktop commit socket 或日志复用。

```text
VoiceResult
├─ result_id, created_at, expires_at, state
├─ transcript
│  ├─ raw_text, final_text
│  ├─ correction: skipped | accepted | rejected + fixed reason
│  └─ engine, model_revision
├─ segments[]
│  ├─ start_ms, end_ms, raw_text, final_text
│  ├─ token_timestamps[]
│  ├─ speaker_cluster
│  ├─ speaker_profile_id?          # 仅可信匹配时存在
│  └─ provenance
└─ timing and fixed error codes
```

桌面输入路径只能读取 `final_text`，不能读取或修改其它字段。初始结果允许为 `final_ready`；
时间戳/说话人富化完成后以同一 `result_id` 更新为 `enriched`。若被新录音抢占，则标为
`enrichment_cancelled`，并保留已知字段。

## 10. 时间戳、说话人与可保存身份

### 10.1 富化任务

Nano 负责产生原始文本和可对齐的时间信息；富化执行器在不影响已上屏文本的前提下完成
token 时间戳、CAM++ 说话人聚类和段级归属。默认仅在 scheduler 空闲时执行。会议模式允许
用户接受更长的结果富化时间，但仍不允许后台任务阻塞新录音。

为支持富化，capture artifact 使用引用计数的内存句柄：`Finalizer` 完成提交后只在富化任务
仍存活时保持该句柄；任务完成、取消、超时或 daemon 退出即关闭。长录音继续使用匿名 memfd
切片，不创建稳定路径或可恢复文件。

Nano + CAM++ 同会话热驻留是可选优化，必须先通过本机 XPU 内存、准确率和连续输入 POC。
未通过时，默认采用提交后的、可取消的串行富化，不能为便利而让两个大模型并发常驻。

### 10.2 身份注册与匹配

```text
注册 3–5 段本人语音
  -> XPU CAM++ embedding
  -> 时长、信噪比、离群值质量门控
  -> 聚合为 centroid
  -> 使用 Secret Service 管理密钥进行加密持久化
```

身份资料仅保存随机 `profile_id`、用户显示名、加密 embedding、模型版本、注册时间和阈值版本；
绝不保存原始音频、转写或可重建音频的特征。匹配要求最佳候选超过接受阈值，且与第二候选的
差距超过间隔阈值；否则只返回匿名 `speaker_cluster`。产品必须提供重新注册、禁用、删除单个
资料和删除全部身份资料的功能。

## 11. 资源策略和可靠性

| 策略 | Nano 空闲窗口 | 临时尾段 | 自动富化 | 适用情况 |
| --- | --- | --- | --- | --- |
| 省内存 | 2 分钟 | 关闭 | 按需 | 电池、资源紧张 |
| 平衡（默认） | 8 分钟 | 开启 | 空闲时 | 日常桌面输入 |
| 持续输入 | 30 分钟 | 开启 | 空闲时 | 接电的持续录入 |

资源压力处理顺序为：取消推测尾段、取消富化、保留正在 finalizing 的文本、完成或安全放弃
当前提交、停止 Nano。不得为了继续后台任务牺牲焦点校验或让系统进入 OOM。

错误通知只使用固定类别：`capture`、`worker.model_load`、`worker.oom`、`worker.timeout`、
`correction.*`、`identity.unavailable`。通知、日志和 metrics 不得含音频路径、文本、窗口身份、
Fcitx token 或原始例外。

## 12. 可观测性、质量门与验收

### 12.1 内存聚合指标

现有 metrics 保留最近 128 个会话的 P50/P95 与固定枚举，并扩展以下产品判断所需字段：

- 冷加载、预热、稳定段/推测段/最终段 ASR 耗时；
- UI 状态到首次稳定文本、松键到最终提交、Qwen 租约/加载/生成/校验耗时；
- 热会话命中率、富化完成/取消率、身份匹配接受/匿名率；
- 无文本的错误码和资源策略切换计数。

### 12.2 私有评测基准

维护用户本机可删除的 20–50 条私有代表性样本，覆盖普通话、中英混说、代码术语、噪声、
长口述和多人音频。报告只输出聚合 CER、标点 F1、技术词保留率、身份误接受率、热/冷 P50/P95、
内存峰值与失败数。

升级 FunASR、vLLM、PyTorch XPU、Qwen、CAM++ 或任一模型 revision 前，必须跑全量 POC 和
该基准；任一准确率、安全、内存或 XPU-only 门不达标则拒绝升级。

### 12.3 验收标准

- 登录后 daemon 存活而任何模型进程均未加载。
- 热态无 Risk Gate 的短句 P95 松键到上屏不高于 3 秒。
- 临时文本从不写剪贴板或目标输入框，完成/取消后从 overlay 清除。
- 热键到新会话可取消过期推测/富化，不出现旧文本污染新会话。
- Qwen、CAM++、身份匹配失败不影响 Nano 原文提交。
- 结果 API 仅同 UID 可访问，8 条/600 秒上限与删除操作均可验证。
- 身份模糊时返回匿名 cluster，绝不猜测已注册身份。
- 全部模型参数及实际执行设备均验证为 `xpu:0`，没有 CPU/CUDA 回退。

## 13. 实施分期

1. **交互和热会话**：Active Session Controller、Overlay、Nano 8 分钟生命周期、XPU scheduler、
   稳定段与尾段任务取消。
2. **准确率事务**：Risk Gate、条件 Qwen、最终提交契约、热/冷性能基准。
3. **结构化结果**：时间戳、owner-only Result Store、TTL/容量、富化任务和抢占。
4. **身份和会议模式**：CAM++ POC、加密 Identity Vault、注册/删除流程、阈值评测。
5. **策略与发布门**：三档资源策略、锁屏/压力处理、版本升级质量门和产品验收。

每一阶段必须独立可测试、可回退，且不得在未完成上一阶段的安全/性能验收前启动下一阶段。
