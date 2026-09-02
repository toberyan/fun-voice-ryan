---
title: Fun Voice Ryan GPU、XPU 与 CPU 首启运行时选择
status: approved
date: 2026-09-02
scope: CUDA GPU → Intel XPU → CPU 的设备选择、隔离运行时、模型初始化和服务启动契约
supersedes:
  - Intel XPU 是安装与运行唯一硬门的约束
  - install-user.sh 只能接受 xpu POC 报告的部署前提
---

# Fun Voice Ryan GPU、XPU 与 CPU 首启运行时选择

## 1. 目标

将当前只能在 Intel XPU 上安装和运行的桌面语音助手，改为由首次初始化自动选择最佳已验证
推理路径：**CUDA GPU → Intel XPU → CPU**。选择必须以实际运行探测为准，而非只看 PCI 设备或
驱动是否存在。

CUDA 和 XPU 选择 Fun-ASR-Nano 为主识别模型，SenseVoiceSmall 为 ASR 故障备用；纯 CPU 选择
SenseVoiceSmall 为唯一 ASR 模型。纯 CPU 明确不启用、不下载 Qwen 3.5 0.8B 修正、说话人分离
或身份库。

本设计保留 DDE X11 的热键、录音、临时 DTK 悬浮窗、Fcitx/剪贴板提交、内存分片与按需模型
生命周期。它不增加联网推理，也不改变文本与音频的隐私边界。

## 2. 已确认产品决策

| 场景 | 选择 | ASR | 增强能力 |
| --- | --- | --- | --- |
| 可完成 CUDA 端到端探测 | `cuda` / `cuda:0` | Fun-ASR-Nano 主、SenseVoiceSmall 备用 | 启用 Qwen 修正、说话人、身份库。 |
| CUDA 不可用而 Intel XPU 探测成功 | `xpu` / `xpu:0` | Fun-ASR-Nano 主、SenseVoiceSmall 备用 | 启用 Qwen 修正、说话人、身份库。 |
| 两种加速器均不可用或验证失败 | `cpu` / `cpu` | **SenseVoiceSmall 唯一主模型** | **强制关闭且不下载** Qwen、说话人、身份库。 |

本设计中的“GPU”是 CUDA 兼容 GPU。AMD ROCm、Apple MPS、DirectML 和其他非 CUDA 设备不在本
次后端集合中，将进入 XPU 或 CPU 的后续候选路径，而不会被误标为 CUDA 可用。

用户可通过 `--backend cuda|xpu|cpu` 显式诊断一个后端；显式模式失败时直接报错，不跨后端。
默认 `--backend auto` 才执行上述降级序列。驱动或硬件改变后使用 `--force-reselect` 重做选择；
正常语音会话中的推理失败不会偷偷热切换设备。

## 3. 为什么使用隔离运行时

PyTorch 的 CUDA、XPU 和 CPU 分发包、底层驱动要求及可用设备 API 不相同。官方 API 将 CUDA
与 Intel XPU 分别暴露为 `torch.cuda.is_available()` 和 `torch.xpu.is_available()`；XPU 官方
安装也使用单独的 XPU wheel 索引。因此不能把当前 XPU 专用 `.venv` 当成同时可运行 CUDA 和
CPU 的环境。

采用用户私有、按需创建的隔离运行时：

```text
~/.local/share/fun-voice-ryan/
├── models/                         # 所有后端共用的 ModelScope 快照
├── runtimes/
│   ├── cuda/                       # 仅在 CUDA 候选被尝试时创建
│   ├── xpu/                        # 仅在 XPU 候选被尝试时创建
│   └── cpu/                        # 仅在最终 CPU 候选被尝试时创建
└── runtime/selection.json          # 当前唯一已验证选择，0600
```

每个运行时各自安装固定版本的 Python 依赖、FunASR 和对应 PyTorch wheel；不会覆盖开发目录的
`.venv`，也不会因一次 CUDA 初始化失败破坏已经可用的 XPU/CPU 环境。第一次初始化只创建尝试
链中实际需要的环境，而不是预装三份大型依赖。

## 4. 首次初始化流程

新增 `scripts/initialize-first-run.sh`，它是新机器的推荐唯一入口：

```bash
scripts/initialize-first-run.sh                 # 自动 CUDA → XPU → CPU
scripts/initialize-first-run.sh --backend cpu   # 明确诊断 CPU 路径
scripts/initialize-first-run.sh --force-reselect
```

脚本只编排，不处理转写文本；实际机器检测、环境创建、ModelScope 下载、端到端探测和原子清单写入
由可单测的 Python bootstrap 模块完成。流程为：

1. 验证 DDE X11、`XDG_RUNTIME_DIR`、磁盘可写、Python/uv、PipeWire、Fcitx 和原生 DTK/Fcitx
   构建依赖。缺少桌面依赖时明确失败，不将“CPU 推理可用”误报为“桌面助手可用”。
2. 对 `auto` 构造候选顺序 `cuda`、`xpu`、`cpu`。CUDA/XPU 的硬件提示只用于避免无意义下载；
   候选的通过条件始终是其独立运行时内的实际探测。
3. 为候选创建或复用 `runtimes/<backend>`；使用对应的带哈希锁文件安装：
   `requirements-cuda.lock`、`requirements-xpu.lock`、`requirements-cpu.lock`，并安装同一已固定
   FunASR 源版本。
4. 下载该候选最少的 ModelScope 快照并切换 hub 到离线模式：
   - CUDA/XPU：Nano、FSMN-VAD、SenseVoiceSmall、Qwen 3.5 0.8B、CAM++。
   - CPU：SenseVoiceSmall、FSMN-VAD；不得调用 Qwen/CAM++ 的下载代码。
5. 在该独立解释器中执行后端探测：导入 torch，运行目标设备张量创建/归约；验证可用 API、dtype，
   再用公开短音频完成一次目标 ASR 模型的端到端短推理。探测结果只保存后端、模型、耗时分桶和
   固定错误类别，不保存音频路径或识别文本。
6. 若 CUDA 或 XPU 任一环节失败，清理本次临时下载/构建状态并继续下一个候选；已完整的共用模型
   快照可保留复用。CPU 失败则首启失败。
7. 仅在某候选全部通过后原子替换 `selection.json`，构建原生 Fcitx/DTK 组件，调用安装流程并
   重启轻量 daemon。若所有候选失败或用户中断，保留旧的有效清单和服务，不写半成品选择。

CUDA 设备探测使用 `torch.cuda.is_available()`，Intel XPU 使用 `torch.xpu.is_available()`，并附加
实际张量运算，以防“API 返回可用但驱动/算子在首次推理失败”。Fun-ASR-Nano 的官方用法支持
`device="cuda:0"`；CPU 策略仍选择较轻的 SenseVoiceSmall，不以 Nano 的 CPU 兼容性作为桌面
首启可用性的前提。

## 5. 运行时选择清单

`selection.json` 是应用拥有的部署状态，不是用户随意编辑的 TOML 配置。父目录权限为 `0700`，
文件为 `0600`；通过同目录临时文件 + `os.replace()` 发布。其 schema 为：

```json
{
  "schema_version": 1,
  "backend": "cuda",
  "python": "/home/user/.local/share/fun-voice-ryan/runtimes/cuda/bin/python",
  "device": "cuda:0",
  "dtype": "bf16",
  "primary_asr_profile": "nano",
  "fallback_asr_profile": "sensevoice",
  "enhanced_enabled": true,
  "speaker_enabled": true,
  "model_revisions": {
    "nano": "master",
    "sensevoice": "master",
    "vad": "master",
    "qwen": "master",
    "campplus": "master"
  },
  "probe": {"status": "pass", "selected_at": 0}
}
```

CPU 清单固定为 `device="cpu"`、`dtype="float32"`、`primary_asr_profile="sensevoice"`、
`fallback_asr_profile=null`、`enhanced_enabled=false`、`speaker_enabled=false`，且模型修订中不
得出现 `qwen` 或 `campplus`。CUDA 的 dtype 优先 BF16；不支持时选择经过探测验证的 FP16。XPU
优先 BF16，不能安全执行则继续 CPU 候选而不是降低到未验证的 XPU 精度。

daemon、worker、corrector、selftest 与安装器都通过同一个只读 loader 获得 `RuntimeSelection`。
缺失、权限不安全、schema 不兼容、解释器不存在、解释器路径不在用户运行时根目录或模型集与
策略矛盾时，一律拒绝启动并提示重新执行首启脚本。

## 6. 应用与服务架构

```text
initialize-first-run.sh
  └─ bootstrap candidate probe (cuda → xpu → cpu)
       ├─ isolated runtime + required models
       ├─ atomic RuntimeSelection
       ├─ native component build
       └─ install-user.sh
            └─ ~/.local/bin/fun-voice-* launcher shims
                 └─ RuntimeSelection.python -m fun_voice.<entrypoint>
                      ├─ daemon / worker use selected device & ASR policy
                      └─ corrector refused before spawn when CPU policy applies
```

现有 `~/.local/bin/fun-voice-*` 不再是复制自仓库 `.venv` 的 console script，而是无文本、无模型
加载的 launcher shim。它先验证 `RuntimeSelection`，再以清单内解释器执行对应 `fun_voice` 模块。
systemd unit 保持调用这些固定用户路径，因此 daemon 和按需 worker 必然使用同一后端。

配置层拆分“用户偏好”和“部署事实”：`config.toml` 继续保存音频、输入法、悬浮窗和允许的功能
偏好；`RuntimeSelection` 决定实际 device、dtype、主/备 ASR profile 与增强功能上限。用户不能用
`config.toml` 将 CPU 清单强行改为 Qwen、CAM++、`cuda:0` 或 `xpu:0`。

worker 和 scheduler 去除“XPU 是唯一设备”的命名和硬约束，改为以 `RuntimeSelection` 驱动：

- CUDA/XPU：先运行 Nano，按现有条件才切 SenseVoice；Qwen 调度仍必须在 ASR worker 已确认停止后
  执行，但 lease 语义改为 accelerator-neutral。
- CPU：只注册 SenseVoice worker/socket；scheduler 不可调度 Nano，也不可提交 correction task。
  所有增强 API 仍返回结构化的 `disabled_by_runtime_policy`，不会 spawn corrector 子进程。
- corrector 与说话人模块接收 selection 的 device/dtype；不再硬编码 `xpu:0`。它们的 launcher 和
  请求入口都先验证 `enhanced_enabled/speaker_enabled`。

自检与诊断新增 `runtime_selection`，显示固定枚举 `cuda`、`xpu` 或 `cpu`、主 ASR profile 与增强
开关；原 `xpu_hard_gate` 改为“所选 runtime 已验证”检查，CPU 下不因缺少 XPU 而失败。

## 7. 失败处理、资源与隐私

- 没有 CUDA/XPU、驱动不可用、对应 wheel 安装失败、模型下载失败、设备张量失败或 ASR 冒烟失败：
  以固定枚举记录候选失败原因并继续 `auto` 的下一候选；日志中不含音频路径或文本。
- CPU 只有 SenseVoice/VAD，首次下载量、内存占用和按需加载显著小于加速器完整路径；Qwen、CAM++
  均不在 CPU 模型目录内。
- `--force-reselect` 在新清单成功发布前保留旧清单；初始化失败时不停止已能工作的服务。成功发布后
  才重启 daemon，并保持 worker、Qwen 在登录时不加载。
- 首启不发送任何音频、文本、设备清单或模型结果到远程服务；ModelScope 仅在模型缺失时下载公开
  模型文件。探测用公开样本、临时文件与报告均在任务结束时删除。
- 用户显式要求某后端时，失败即退出并保留旧选择，避免意外的性能/隐私预期改变。

## 8. 测试与验收

1. 单测以 fake runtime probe 覆盖候选优先序、CUDA 失败转 XPU、CUDA/XPU 均失败转 CPU、显式后端
   不跨候选，以及任何成功清单的类型/模型集不变量。
2. 清单测试覆盖 `0600`/`0700`、原子替换、损坏/越权/越界解释器拒绝、旧清单在新初始化失败时不变。
3. 配置和 daemon 测试覆盖 CPU 强制关闭增强、只注册 SenseVoice、禁止 corrector spawn；CUDA/XPU
   覆盖 device/dtype 注入、Nano 主路径与 SenseVoice 备用路径。
4. 安装/launcher 测试覆盖无有效清单拒绝、shim 使用清单解释器、systemd worker 与 daemon 使用相同
   选择、安装器不再引用 XPU POC 作为唯一硬门。
5. 脚本测试覆盖 `--backend`、`--force-reselect`、dry-run 候选序和 CPU 模型清单；网络下载与真实
   设备推理由人工首启验收完成，不放入常规 CI。
6. 真实验收分别在 CUDA、Intel XPU 与纯 CPU 机器上执行 `initialize-first-run.sh`；确认选择、模型
   集、首次语音输入、CPU 无 Qwen/CAM++ 进程、DDE 热键/悬浮窗/上屏和注销后无模型常驻。

## 9. 非目标

- 不支持 AMD ROCm、Apple MPS、Wayland、云端 ASR 或运行中热切换后端。
- 不允许 CPU Qwen、CPU 说话人识别或 CPU 身份库；这些能力不能通过用户 TOML 绕过。
- 不预装全部三个重型运行时，不在登录时预加载 ASR/Qwen，不保存转写历史。
- 不修改 Fun-ASR-Nano、SenseVoiceSmall 或 Qwen 的模型权重与原始输出契约。

## 10. 参考

- [PyTorch CUDA availability](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html)
- [PyTorch Intel XPU API](https://docs.pytorch.org/docs/stable/xpu.html)
- [PyTorch Intel GPU installation](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Fun-ASR-Nano ModelScope model card](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)
