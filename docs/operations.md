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

## 3. 配置来源（source）

当前版本没有独立配置文件，运行参数来自代码内安全默认值
（`fun_voice.config.Config`）：热键 `<Super>C`、音频源 `default`（PipeWire）、
输入法 `fcitx5`、提交超时 500ms、模型 `FunAudioLLM/Fun-ASR-Nano-2512`、
设备 `xpu:0`、dtype `bf16`、`gpu_memory_utilization=0.35`。未来若引入配置文件，
将落在 `~/.config/fun-voice-ryan/`，卸载默认保留。

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

worker 首次启动需要几分钟加载模型（状态会先处于 `activating`）；daemon 依赖
图形会话环境（DISPLAY/XAUTHORITY），由登录时的 autostart 入口导入（见第 7 节）。

## 7. DDE / Fcitx 故障处理

- **Fcitx addon 未加载**：确认 fcitx5 正在运行，且
  `~/.local/lib/fcitx5/fcitx5-fun-voice.so` 与
  `~/.local/share/fcitx5/addon/fcitx5-fun-voice.conf` 已就位；重启 fcitx5 后
  观察 `$XDG_RUNTIME_DIR/fun-voice-ryan-fcitx.sock` 是否出现。
- **Super+C 失效**：确认 DDE 快捷键已注册（`~/.config/fun-voice-ryan/dde-shortcut-id`
  存在），且没有被其他程序占用（`fun-voice-selftest` 的 `super_c_conflict` 项）。
- **上屏失败回退**：Fcitx 提交失败时会回退到剪贴板（需要 `xclip`/`xsel`），
  再失败会尝试 XTEST（Ctrl+V，需要 python-xlib 且 X 可连接）。
- **daemon 反复失败**：`journalctl --user -u fun-voice-daemon` 查看退出原因；
  常见为缺少 `DISPLAY`（需重启会话让 autostart 入口导入环境）。

## 8. 如何保持 Super+C 可用

- 不要把 `Super+C` 分配给其他 DDE 快捷键或第三方工具。
- 卸载前先运行 `scripts/uninstall-user.sh`，它会注销快捷键并删除 id 文件。
- 若手动在 DDE「快捷键」里删除了本助手，残留的
  `~/.config/fun-voice-ryan/dde-shortcut-id` 会让安装脚本误以为已注册而**跳过**注册，
  导致 `Super+C` 未注册；删除该文件后重跑 `scripts/install-user.sh` 即可复位。

## 9. 卸载

```bash
scripts/uninstall-user.sh            # 保留模型缓存与用户配置
scripts/uninstall-user.sh --purge    # 二次确认后连模型缓存与配置一并删除
```

卸载会停止并 disable 两个 systemd 服务、注销 Super+C、移除 unit/desktop/addon
文件、`~/.local/bin` 下的 5 个 console script，以及
`$XDG_RUNTIME_DIR/fun-voice-ryan/` 下的 daemon/worker socket、fcitx socket 与
capture 分片。

## 10. Wayland 非支持声明

首版**不支持 Wayland**。本项目依赖 X11 焦点查询（`_NET_ACTIVE_WINDOW`）、
C 键物理状态查询与 XTEST 注入，这些能力在 Wayland 会话下不可用。请在
Deepin DDE **X11** 会话下使用；Wayland 会话下安装脚本可执行，但 daemon /
bridge / Fcitx 上屏链路无法正常工作。
