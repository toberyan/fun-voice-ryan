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
读取它。当前生效的键为 PipeWire `audio.source`、Fcitx
`input_method.commit_timeout_ms`（毫秒）及 `input_method.allow_x11_paste_fallback`，以及
`inference.device`、`dtype`、`gpu_memory_utilization`、`enforce_eager`。

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
- **模型输出原样保留**，不做任何词典、正则或 LLM 改写。

## 5. 日志脱敏

应用日志只写「长度、状态、错误类别、请求 id」，不写音频路径或转写正文。
排查时如果发现某条日志疑似包含正文，请优先当作 bug 报告（隐私红线）。

## 6. 服务诊断（journalctl --user）

安装后有两个 systemd user 服务：

```bash
systemctl --user status fun-voice-worker
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

worker 首次启动需要几分钟加载模型（状态会先处于 `activating`）；daemon 依赖
图形会话环境（DISPLAY/XAUTHORITY），由登录时的 autostart 入口导入（见第 8 节）。

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
`~/.local/bin` 下的 4 个 console script，以及
`$XDG_RUNTIME_DIR/fun-voice-ryan/` 下的 daemon/worker socket、fcitx socket 与
capture 分片。

## 11. Wayland 非支持声明

首版**不支持 Wayland**。本项目依赖 X11 焦点查询（`_NET_ACTIVE_WINDOW`）、
C 键物理状态查询与 XTEST 注入，这些能力在 Wayland 会话下不可用。请在
Deepin DDE **X11** 会话下使用；Wayland 会话下安装脚本可执行，但 daemon 的 X11
热键、焦点校验与 Fcitx 上屏链路无法正常工作。
