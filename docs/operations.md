# 运维手册

Fun Voice Ryan 只在 Deepin DDE X11 用户会话中运行。首次初始化负责验证桌面前提、
选择硬件后端、创建隔离 Python 环境、下载该后端模型、发布安全清单并安装用户服务；
无需 `sudo`，也不触碰系统目录。

## 1. 首次初始化与重新选择

在图形会话中执行：

```bash
scripts/initialize-first-run.sh
scripts/initialize-first-run.sh --backend cpu
scripts/initialize-first-run.sh --force-reselect
fun-voice-selftest --format json
```

默认 `auto` 顺序固定为 **CUDA → Intel XPU → CPU**。候选只有在隔离环境安装、设备张量
探测、模型下载和固定无正文 ASR smoke test 全部通过后才会被选择。显式
`--backend cuda|xpu|cpu` 只尝试一个候选，失败时不回退，旧的有效 `selection.json` 保持
不变。驱动、GPU 或依赖版本变化后使用 `--force-reselect`；并发初始化由私有文件锁串行化。

初始化前会检查 `DISPLAY`、X11/DDE 会话、权限安全的 `XDG_RUNTIME_DIR`、PipeWire、
fcitx5、`uv`、CMake、pkg-config 和数据目录可写性。`--dry-run` 只显示候选顺序，不做这些
桌面检查，也不写文件。

选中后端的有效策略是：

| 后端 | 精度 | ASR | 修正/说话人 |
| --- | --- | --- | --- |
| CUDA | 优先 BF16，探测不支持时允许 FP16 | Nano primary，SenseVoiceSmall fallback | Qwen3.5-0.8B / CAM++ 按需 |
| Intel XPU | 仅 BF16 | Nano primary，SenseVoiceSmall fallback | Qwen3.5-0.8B / CAM++ 按需 |
| CPU | FP32 | **SenseVoice-only**，无 fallback | 禁用，不下载 Qwen/CAM++ |

`docs/xpu-poc.md` 的九项 POC 仅为显式 Intel XPU 诊断：该命令自身失败关闭，但不能阻断
CUDA 或 CPU 初始化。

## 2. 隔离运行时与模型缓存

生产模型依赖不会安装进仓库 `.venv`。仓库 `.venv` 只供开发测试；生产布局为：

```text
${XDG_DATA_HOME:-$HOME/.local/share}/fun-voice-ryan/
├── runtimes/
│   └── {cuda|xpu|cpu}-<32位随机generation>/
├── runtime/
│   └── selection.json
└── models/
```

`runtime/` 与各运行时目录固定为 `0700`，`selection.json` 固定为 `0600`。清单记录后端、
解释器、device/dtype、ASR profile、能力集合和精确模型 revision；启动器每次都重新验证
清单、路径所有权、权限、解释器与策略，任一异常即失败关闭。六个安装启动器只映射到六个
固定 Python module，不解析任意模块名，也不使用 shell 拼接用户命令。

首次成功探测会一次性下载该后端允许的模型集：CUDA/XPU 下载 Nano、SenseVoiceSmall、
VAD、Qwen3.5-0.8B、CAM++；CPU 只下载 SenseVoiceSmall 与 VAD。重新安装不会隐式删除或
重复下载已有快照。

## 3. 配置来源与覆盖规则

配置文件为 `${XDG_CONFIG_HOME:-~/.config}/fun-voice-ryan/config.toml`；不存在时使用安全
默认值，可从 `scripts/config.example.toml` 复制。热键固定为 `<Super>C`，录音上限、
内存分片阈值与不保留历史是不可配置的安全约束。

TOML 中历史 `inference.device`、`inference.dtype`、`correction.device` 与
`correction.dtype` 是兼容输入，**被有效 runtime policy 忽略**，不能覆盖
`selection.json`。`inference.allow_sensevoice_fallback` 也不能让 CPU 启用 Nano。
只有 `enhanced.enabled = false` 可以进一步关闭 CUDA/XPU 的 Qwen 修正；设为 `true`
不能在 CPU selection 下启用修正或说话人能力。

仍可配置的 Qwen 偏好包括 `correction.max_source_characters`、`max_new_tokens`、
`timeout_seconds`、`protected_terms` 和 `enable_thinking`，纯 CPU 下均不生效。overlay
支持 `vertical_center_ratio`（0.50--0.85）、`width_px`（420--1000）与
`font_scale`（0.80--1.80）；越界会明确拒绝 daemon 启动。

## 4. 服务与按需模型生命周期

安装后 daemon 由 DDE autostart 在导入 `DISPLAY`、`XAUTHORITY` 和 D-Bus 环境后启动。
worker template 没有 `[Install]`，不能成为开机模型服务：

```bash
systemctl --user status fun-voice-daemon.service --no-pager
systemctl --user status fun-voice-worker@nano.service --no-pager
systemctl --user status fun-voice-worker@sensevoice.service --no-pager
journalctl --user -u fun-voice-daemon.service -f
```

有效录音期间 daemon 会异步**预加载**所选 ASR profile，松键后通过对应私有 socket
识别。加速器修正必须先**停止 Nano**（或本次实际使用的 SenseVoice worker），确认服务
已经 `inactive`/`failed` 且 transport 不可达，再启动一次 Qwen 子进程。停止确认失败、
Qwen 超时/OOM/校验拒绝时直接提交原始转写，绝不让 ASR 与 Qwen 同时抢占设备。

CPU selection 的调度器只允许 SenseVoice profile：不会探测或启动 Nano socket，不创建
Qwen child，也不会加载 CAM++。所有模型按需加载并在空闲窗口后卸载；登录本身不占用数 GB
模型内存。

## 5. 自检与故障排查

```bash
fun-voice-selftest --format json
```

自检按当前 `selection.json` 检查后端、有效 policy、桌面链路与允许的 worker。CPU 机器只
探测 SenseVoice；CUDA/XPU 机器探测对应加速器路径。`x11_hotkey` 要在本次 daemon 启动后
真实按住/松开一次 `Super+C` 才为 pass，只记录 `registered` 与 `press_seen` 布尔值。

- **清单或依赖失败**：重新执行首次初始化；不要手改 `selection.json` 或把仓库 `.venv`
  软链到生产 runtime。
- **Super+C 冲突**：查看 daemon journal；X11 grab 冲突会以退出码 2 失败且不循环重启。
- **Fcitx 上屏失败**：确认用户 addon 与 `$XDG_RUNTIME_DIR/fun-voice-ryan-fcitx.sock`；
  Fcitx 失败时按策略写 clipboard，再允许 XTEST Ctrl+V fallback。
- **模型启动失败**：查看固定错误类别和聚合时延，不要在日志中加入音频路径或识别正文。

显式后端初始化失败不会覆盖当前清单。例如不支持 CUDA 的机器执行
`scripts/initialize-first-run.sh --backend cuda --force-reselect` 必须非零退出，daemon 继续
使用此前的安全选择。

## 6. DTK 悬浮窗

悬浮窗位于 `~/.local/lib/fun-voice-ryan/fun-voice-overlay`，按需启动、5 秒无状态后退出，
不会获取焦点或写剪贴板。二进制缺失或 DTK 运行库异常时，识别和上屏仍继续，但进入
**无悬浮窗**模式。

重新构建后用首次初始化重新核验并部署：

```bash
cmake -S native/dtk-overlay -B build/dtk-overlay
cmake --build build/dtk-overlay
scripts/initialize-first-run.sh --force-reselect
```

## 7. 内存与时延诊断

daemon 的 owner-only control socket 只返回最近 128 次会话的聚合计数、P50/P95 和固定
枚举，不返回音频、文本、路径、窗口信息或模型异常原文：

- `preload_runtime_load_ms` / `preload_warmup_ms`：按需 runtime 构造与固定静音预热；
- `asr_audio_load_ms` / `asr_vad_ms` / `asr_generate_ms`：读取、VAD、推理；
- `asr_queue_transport_ms` / `asr_release_ms`：外部排队与释放 ASR worker；
- `correction_model_load_ms` / `correction_generate_ms` / `correction_validate_ms`：
  加速器 Qwen 单次修正阶段，CPU 不产生这些指标。

不要为了单次冷启动数字恢复登录常驻模型。若录音交互慢，先区分是模型冷加载、ASR、
worker 交换、Qwen 还是输入法提交，再调整现有空闲策略。

## 8. 本地准确率与时延基准

基准只在显式执行时加载模型。私有 JSONL 清单每行包含类别、16 kHz 单声道音频绝对路径、
参考文本和可选术语；不得提交到仓库：

```bash
fun-voice-benchmark --manifest /path/to/private-manifest.jsonl \
  --output /path/to/private-benchmark-report.json
```

输出只含类别级 CER、术语准确率、标点 P/R/F1 与冷/热 P50/P95 聚合值，可选报告固定
`0600`。加速器可分别比较 Nano 原始结果与串行 Qwen 修正；CPU 只测 SenseVoice 原始结果。

## 9. 隐私与长录音

- 短音频只在内存处理；超过阈值后才在 `$XDG_RUNTIME_DIR` 用户专属 tmpfs 以
  `0700`/`0600` 切分，结束后删除。
- 10 分钟后按固定 60 秒分片识别并按时间顺序拼接；25 分钟提醒一次，30 分钟强制停止。
- 日志、通知、probe 与 selftest 均不包含语音、转写正文或模型绝对路径。
- 最终上屏文本（修正成功时为修正文本，否则为原始 ASR）留在剪贴板，结构化结果只经接口提供。

## 10. 卸载

```bash
scripts/uninstall-user.sh
```

卸载会停止服务并移除六个启动器、systemd unit、autostart、Fcitx addon、DTK 二进制、
runtime socket 与 capture 分片。模型快照、`runtimes/`、`runtime/selection.json` 和用户配置
始终保留，没有删除模型的隐式选项。

## 11. Wayland 非支持声明

首版不支持 Wayland。X11 焦点查询、`Super+C` 原子 grab、C 键物理状态与 XTEST fallback
在 Wayland 下不可用；请使用 Deepin DDE X11 会话。
