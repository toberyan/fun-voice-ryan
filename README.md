# Fun Voice Ryan

Fun Voice Ryan 是运行在 Deepin DDE X11 上的本地语音输入助手：按住 `Super+C`
说普通话（可夹杂英文、代码与计算机术语），松开后把结果提交到录音开始时的焦点窗口，
并将最终文本保留在剪贴板。推理运行时由首次初始化根据硬件验证结果选择，不在登录时加载模型。

加速器路径以准确率优先：Fun-ASR-Nano-2512 为主识别，SenseVoiceSmall 为加载/OOM
备用，识别 worker 完全停止后才按需运行 Qwen3.5-0.8B 修正。
纯 CPU 路径为低内存兜底，只使用 SenseVoiceSmall 与 VAD，提交原始识别文本；不下载、
不启动 Nano、Qwen 或 CAM++。

SenseVoice 的 `<|...|>` 语言、情绪、事件和 ITN 控制元数据会在识别边界移除；桌面提交、
剪贴板和结构化段落文本只保留识别文字。清理不会将元数据替换为表情，也不会改写普通话、
英文或代码内容。

当前版本未实现 CAM++ 加载、说话人分离/身份或结构化结果接口；加速器清单中的
`campplus` revision 及其预下载快照只是后续能力预留，不代表已有可调用的产品能力。

## 开发与桌面前提

- Python 3.12、`uv`、CMake 与 pkg-config。
- Deepin DDE X11（首版不支持 Wayland）、PipeWire、fcitx5。
- Fcitx5Core 开发库、Qt 6、`libdtk6gui-dev` 与 `libdtk6widget-dev`，用于构建
  Fcitx 插件和 `native/dtk-overlay` 原生 DDE 悬浮窗。
- CUDA、Intel XPU 均为可选；没有可用加速器时可选择 CPU 路径。

仓库 `.venv` 只用于开发和测试，不是生产模型运行时：

```bash
uv sync
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src/fun_voice
```

## 首次初始化

在真实 DDE X11 会话中执行：

```bash
scripts/initialize-first-run.sh
scripts/initialize-first-run.sh --backend cpu
scripts/initialize-first-run.sh --force-reselect
fun-voice-selftest --format json
```

通常只需第一条。`auto` 固定按 **CUDA → Intel XPU → CPU** 尝试；每个候选必须完成
独立环境安装、张量探测、模型下载和无正文 ASR smoke test 后才能被选中。显式
`--backend cuda|xpu|cpu` 只验证该后端，失败时不回退，也不会覆盖旧的有效选择。
`--force-reselect` 用于驱动、硬件或运行时发生变化后的重新选择。

初始化会先验证或构建 Fcitx 插件与 DTK overlay；缺少开发依赖时以固定
`native_prerequisite` 类别退出，尚不会创建候选运行时或下载模型。各后端的模型下载位于
独立候选缓存，失败候选会被丢弃，只有成功选择允许的模型快照会提升到正式缓存。

初始化会一次性下载所选后端允许的模型集：

- CUDA / Intel XPU：Nano、SenseVoiceSmall、VAD、Qwen3.5-0.8B，以及仅作后续能力预留的 CAM++ 快照；
- CPU：SenseVoiceSmall、VAD，严格为 **SenseVoice-only**，无 Qwen、无说话人能力。

CUDA 优先使用通过探测的 BF16，不支持时允许探测并选择 FP16；Intel XPU 只接受 BF16；
CPU 固定 FP32。TOML 中旧的 `device` / `dtype` 输入不会覆盖这项有效运行时策略；
用户偏好 `enhanced.enabled = false` 只能进一步关闭加速器上的修正能力。

## 运行时与安全边界

生产依赖不安装到仓库 `.venv`，而是隔离在：

```text
${XDG_DATA_HOME:-$HOME/.local/share}/fun-voice-ryan/
├── runtimes/{cuda|xpu|cpu}-<generation>/
├── runtime/selection.json
└── models/
```

`runtime/` 目录固定为 `0700`，`selection.json` 固定为 `0600`。启动器每次加载并校验
该清单，然后只用其中已验证的 Python 执行六个固定模块；清单、解释器或依赖不安全时
失败关闭。安装的启动器仍引用当前仓库的只读代码入口，因此移动仓库后应在新路径重新初始化。

- 不持久化录音或转写文本。短音频只驻留内存；超阈值音频仅在
  `$XDG_RUNTIME_DIR` 的用户专属目录以 `0700`/`0600` 暂存，任务结束即删除。
- 日志、通知和探测结果不含音频、转写正文或模型路径，只记录长度、状态、固定错误类别和请求 id。
- 加速器修正失败、超时或租约失败时提交原始 ASR 文本；URL、路径、代码、选项、版本和
  受保护术语被改动时拒绝修正结果。

## 使用与诊断

登录后 daemon 常驻但不加载模型；Nano、SenseVoiceSmall 和 Qwen 均按需启动并在空闲时
卸载。按住 `Super+C` 录音，松开后识别、上屏并更新剪贴板。DTK 原生悬浮窗会在清空后
空闲 5 秒退出，守护进程会回收该子进程，不会积累后台的 `fun-voice-overlay` 无响应条目。

```bash
fun-voice-selftest --format json
systemctl --user status fun-voice-daemon.service --no-pager
journalctl --user -u fun-voice-daemon.service -f
journalctl --user -u fun-voice-worker@nano.service -f
journalctl --user -u fun-voice-worker@sensevoice.service -f
fun-voice-benchmark --manifest /path/to/private-manifest.jsonl
```

完整运维见 `docs/operations.md`，真实机器验收见 `docs/acceptance-checklist.md`；
`docs/xpu-poc.md` 的九项 POC 现在是显式 Intel XPU 诊断，不会阻断 CUDA 或 CPU 初始化。

## 卸载

```bash
scripts/uninstall-user.sh
```

卸载只移除应用拥有的启动器、systemd user unit、桌面入口、Fcitx 插件、DTK 二进制和
临时 socket/分片；模型快照、隔离运行时、`selection.json` 与用户配置始终保留。
