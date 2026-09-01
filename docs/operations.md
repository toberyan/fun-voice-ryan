# 运维手册

Fun Voice Ryan 的安装、运行与故障排查。安装脚本为 `scripts/install-user.sh`，
卸载脚本为 `scripts/uninstall-user.sh`，两者都只操作用户级路径（`~/.local`、
`~/.config`、`$XDG_RUNTIME_DIR`），全程不 `sudo`、不触碰系统目录。

## 1. 安装前硬件核验（XPU / 驱动）

本项目推理依赖 Intel Arc XPU，安装前必须先通过全部九项硬门检查：

1. 确认显卡为 Intel Arc（或带 Arc 核显的 Arrow Lake / Meteor Lake 等），且
   `ls /dev/dri` 能看到 render 节点。
2. 确认已安装 Intel GPU compute runtime 与 Level Zero：
   ```bash
   clinfo | grep -i "level zero\|device name"   # 或 sycl-ls
   ```
3. 运行 POC 脚本，产出 `ready=true` 的报告（安装脚本会读取它作为硬门）：
   ```bash
   scripts/run-nano-xpu-poc.sh
   ```
   报告位于 `$XDG_RUNTIME_DIR/fun-voice-ryan/poc-report.json`。`ready` 不为
   `true` 时 `install-user.sh` 会直接拒绝安装；**绝不允许静默退回 CPU**。

## 2. 首次模型下载

模型（Fun-ASR-Nano-2512 与 FSMN-VAD）由 POC 脚本在首次运行时下载到：

```
~/.local/share/fun-voice-ryan/models
```

之后 worker 直接复用该缓存，不会重复下载。卸载时该目录默认保留（见第 8 节）。

## 3. 配置来源

配置文件为 `${XDG_CONFIG_HOME:-~/.config}/fun-voice-ryan/config.toml`；不存在时使用
安全默认值。可从 `scripts/config.example.toml` 复制后修改，daemon 与 worker 都在启动时
读取它。当前生效的键包括 PipeWire `audio.source`、Fcitx
`input_method.commit_timeout_ms`（毫秒）及 `input_method.allow_x11_paste_fallback`、Nano
的 `inference.*`，以及 `enhanced.enabled` 和 Qwen 的
`correction.max_source_characters`、`max_new_tokens`、`timeout_seconds`、`protected_terms`。
Qwen 的模型、设备和精度固定，不能改为其他模型或 CPU。

`inference.device` 只能是 `xpu:0`，任何 CPU/CUDA 设置都会拒绝启动，绝不静默回退。
热键固定为 `<Super>C`，模型固定为 `FunAudioLLM/Fun-ASR-Nano-2512`，Fcitx 主通道固定为
fcitx5；录音上限、内存阈值和不保留历史均为不可配置的安全约束。卸载默认保留用户配置。

> 另请注意：安装后的 console script（shebang）与 autostart `Exec` 都指向
> **仓库与 `.venv` 的绝对路径**（当前 `~/workspace/fun-voice-ryan`）。**移动或
> 删除仓库目录会使语音输入静默失效**；仓库路径变更时请先
> `scripts/uninstall-user.sh`，再在新位置重新安装。

## 4. 隐私说明

- **不持久化录音或转写文本**：短音频只驻留内存；超过阈值才在
  `$XDG_RUNTIME_DIR`（用户专属 tmpfs）下以 `0700`/`0600` 权限暂存，任务结束即删除。
- **日志与通知不含音频内容或转写正文**，只记录长度、状态、错误类别和请求 id。
- **原始转写始终可用**：仅 Qwen3.5-0.8B 可做一次本地校对；URL、路径、反引号代码、
  命令选项、版本、`snake_case`、`CamelCase` 和配置技术词必须保留。校对超时、失败或
  校验不通过时提交原始转写。

## 5. 日志脱敏

应用日志只写「长度、状态、错误类别、请求 id」，不写音频路径或转写正文。
排查时如果发现某条日志疑似包含正文，请优先当作 bug 报告（隐私红线）。

## 6. 服务诊断（journalctl --user）

安装后有 daemon 与按需 worker template：

```bash
systemctl --user status fun-voice-worker@nano.service
systemctl --user status fun-voice-daemon
journalctl --user -u fun-voice-worker -f
journalctl --user -u fun-voice-daemon -f
```

自检入口：

```bash
fun-voice-selftest --format json
```

`x11_hotkey` 会在 daemon 成功独占抓取 `Super+C`、且本次启动已经观测到一次真实按下后
才返回 `pass`。这是预期的验收门：先在 X11 会话中按住并松开一次 `Super+C`，再运行
自检；它只读取 daemon 内存中的 `registered`、`press_seen` 两个布尔值，不保存按键时间、
音频或转写文本。该项通过后仍须完成 `docs/acceptance-checklist.md` 的目标应用人工验收。

首次有效录音会在录音期间预加载 Nano；若此前没有模型驻留，worker 状态会短暂为
`activating`。daemon 依赖
图形会话环境（DISPLAY/XAUTHORITY），由登录时的 autostart 入口导入（见第 8 节）。

### 6.1 内存时延诊断

daemon 的 owner-only control socket 支持 `{"op":"metrics"}`。它只返回当前进程最近
128 次会话的聚合计数、P50/P95 与固定枚举直方图；不会返回会话明细，也不会包含音频、文本、
路径、窗口信息或模型异常原文。重启 daemon 会清空这些内存指标。

- `preload_runtime_load_ms` 是 worker 内 Nano/VAD/runtime 构造耗时；
  `preload_warmup_ms` 是构造完成后对固定一秒静音 PCM 的一次生成预热，二者均在录音期发生。
  `nano_warmup=failed` 只表示这次预热不可用，真实 ASR 仍会继续使用已加载的 Nano。
- `asr_queue_transport_ms` 是 daemon 端 ASR 总耗时扣除 worker 执行耗时后的外部等待；
  `asr_audio_load_ms`、`asr_vad_ms`、`asr_generate_ms` 分别定位音频读取、VAD 和 Nano 生成。
  `asr_release_ms` 是启动 Qwen 前确认对应 ASR worker 已停止的耗时。
- `correction_model_load_ms`、`correction_generate_ms`、`correction_validate_ms` 分别表示
  一次 Qwen3.5-0.8B 子进程的加载、生成和确定性校验耗时。`correction_rejection` 仅为固定原因，
  例如 `envelope_missing`、`similarity`、`protected_token`、`oom` 或 `timeout`；它不携带候选
  文本或被保护的技术词。任何此类拒绝仍提交原始 ASR 文本。

使用这些字段先判断瓶颈是否在 runtime 加载、首次预热、ASR 推理、worker 交换，还是 Qwen
加载；不要为了追求单次时延在登录时常驻 Nano/Qwen，或让两个模型同时占用 XPU。

## 7. 本地准确率与时延基准

仅在你明确执行时运行基准。清单为本机自有的 JSONL 文件，每行包含安全类别名
（如 `mixed`）、16 kHz 单声道 WAV/PCM 的绝对路径、参考文本，以及可选的技术词
`terms`。它不应加入仓库。示例命令：

```bash
fun-voice-benchmark --manifest /path/to/private-manifest.jsonl \
  --output /path/to/private-benchmark-report.json
```

命令会依次测量首个请求的冷启动和后续请求的热态时延，计算字级 CER、技术词精确率
与标点 P/R/F1。清单、音频、参考文本和识别结果只保留在该进程内用于评分；终端输出和
可选报告都只包含类别级计数、P50/P95 聚合值。显式指定的报告权限固定为 `0600`。

基准先采集不含 Qwen 的 Nano ASR 基线；日常输入链路则在 PipeWire 录音成功后异步**预加载**
Nano。松键后 worker 对预加载与转写串行执行。若启用校对，daemon 必须先**停止 Nano**（若
本次使用备用模型则停止 SenseVoice）并确认对应 user service 已是 `inactive` 或 `failed`，
才启动一次 Qwen3.5-0.8B 子进程。任何停止确认失败都会跳过 Qwen、直接提交原始转写；Qwen
退出后，下一次有效录音才会重新预加载 Nano。此顺序不依赖不可靠的跨进程显存读数。

## 8. X11 热键 / Fcitx 故障处理

- **Fcitx addon 未加载**：确认 fcitx5 正在运行，且
  `~/.local/lib/fcitx5/fcitx5-fun-voice.so` 与
  `~/.local/share/fcitx5/addon/fcitx5-fun-voice.conf` 已就位；重启 fcitx5 后
  观察 `$XDG_RUNTIME_DIR/fun-voice-ryan-fcitx.sock` 是否出现。
- **Super+C 失效或 daemon 启动失败**：查看
  `journalctl --user -u fun-voice-daemon.service -b --no-pager`。若出现
  `X11 hotkey unavailable`，说明另一个 X11 客户端已抢占该组合；停止或改配冲突客户端后
  执行 `systemctl --user restart fun-voice-daemon.service`。退出码 `2` 是确定的冲突失败，
  systemd 不会对它循环重启。
- **自检的 `x11_hotkey` 未通过**：先确认 `registered=true`；若 `press_seen=false`，在
  任意输入框按住并松开一次 `Super+C` 后重新执行自检。不要将它改成切换式录音。
- **上屏失败回退**：Fcitx 提交失败时会回退到剪贴板（需要 `xclip`/`xsel`），
  再失败会尝试 XTEST（Ctrl+V，需要 python-xlib 且 X 可连接）。
- **daemon 反复失败**：`journalctl --user -u fun-voice-daemon` 查看退出原因；
  常见为缺少 `DISPLAY`（需重启会话让 autostart 入口导入环境）。

## 9. 如何保持 Super+C 可用

- 不要让其他 X11 全局热键工具抢占 `Super+C`。
- daemon 在启动时会原子抓取含 Caps Lock、Num Lock、Scroll Lock 变体的 `Super+C`；任何
  一个变体冲突都会使整个服务以退出码 `2` 失败，而不会退回到轮询、切换录音或 raw input。
- 重启 daemon 后应重新执行一次按住/松开，再以 `fun-voice-selftest --format json` 确认
  `x11_hotkey` 为 `pass`。

## 10. 卸载

```bash
scripts/uninstall-user.sh            # 保留模型缓存与用户配置
scripts/uninstall-user.sh --purge    # 二次确认后连模型缓存与配置一并删除
```

卸载会停止并 disable 两个 systemd 服务、移除 unit/desktop/addon 文件、
`~/.local/bin` 下的 6 个 console script，以及
`$XDG_RUNTIME_DIR/fun-voice-ryan/` 下的 daemon/worker socket、fcitx socket 与
capture 分片。

## 11. Wayland 非支持声明

首版**不支持 Wayland**。本项目依赖 X11 焦点查询（`_NET_ACTIVE_WINDOW`）、
C 键物理状态查询与 XTEST 注入，这些能力在 Wayland 会话下不可用。请在
Deepin DDE **X11** 会话下使用；Wayland 会话下安装脚本可执行，但 daemon 的 X11
热键、焦点校验与 Fcitx 上屏链路无法正常工作。
