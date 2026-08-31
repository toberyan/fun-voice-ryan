# Fun Voice Ryan

Fun Voice Ryan 是一个运行在 Deepin DDE X11 上的本地语音输入助手：按住 `Super+C` 说普通话（可夹杂英文、代码与计算机术语），松开后由本机 Fun-ASR-Nano 转写，并把**未经改写的模型原始文本**提交到录音开始时的焦点窗口（首选 Fcitx 上屏，剪贴板作备份）。

> [!WARNING]
> **未通过 Intel XPU POC 之前，禁止安装或启动桌面服务。**
>
> 本项目推理依赖 Intel Arc XPU。安装前必须先跑通 `fun-voice-preflight` 的全部硬门检查：`torch.xpu` 可用、vLLM 能在 XPU 加载 Qwen 解码、Fun-ASR-Nano 的 encoder / adaptor / prompt embeddings 都运行在 `xpu:0`、10 秒与 60 秒中英混合样本通过，且日志证明没有静默 CPU 解码回退。任一检查失败即停止部署，**不得**静默退回 CPU 或切换其他后端。

## 开发前提

- Python 3.12 与 `uv`。
- 桌面环境为 Deepin DDE X11（首版不支持 Wayland）。
- 已安装 fcitx5 及 Fcitx5Core 开发库。
- 默认输入走 PipeWire，能提供 48 kHz / 4 通道音频。
- Intel Arc 显卡，并装好 Intel GPU compute runtime 与 Level Zero。

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

## 隐私边界

- **不持久化录音或转写文本。** 短音频只驻留内存；超过阈值才在 `$XDG_RUNTIME_DIR`（用户专属 tmpfs）下以 `0700`/`0600` 权限暂存，任务结束即删除。
- **日志与通知不含音频内容或转写正文**，只记录长度、状态、错误类别和请求 id。
- **模型输出原样保留**，不做任何词典、正则或 LLM 改写。

## 安装

安装脚本只操作用户级路径，不 `sudo`、不触碰系统目录；安装前会读取
`$XDG_RUNTIME_DIR/fun-voice-ryan/poc-report.json` 作为硬门，`ready` 不为 `true`
即拒绝安装（详见上方 WARNING）。

```bash
# 1) 先构建 fcitx addon 与 Python 环境（一次性）
cmake -S native/fcitx5-fun-voice -B build/fcitx && cmake --build build/fcitx
uv sync

# 2) 安装到用户会话（systemd user 服务 + autostart + DDE Super+C 快捷键）
scripts/install-user.sh
```

安装内容：5 个 console script 拷贝到 `~/.local/bin/`；两个 systemd user unit
（worker、daemon）写入 `~/.config/systemd/user/`；fcitx addon 的 `.so` 与
`.conf` 写入 `~/.local/lib/fcitx5/` 与 `~/.local/share/fcitx5/addon/`；登录自启
入口写入 `~/.config/autostart/`。脚本幂等，可重复执行。

> [!IMPORTANT]
> 安装的 console script（shebang）与 autostart `Exec` 都指向**仓库与 `.venv`
> 的绝对路径**（当前为 `~/workspace/fun-voice-ryan`）。**移动或删除仓库目录后
> 语音输入会静默失效**；如仓库路径变更，请先 `scripts/uninstall-user.sh` 再在新
> 位置重新安装。

## 使用

1. 重新登录 DDE 会话（或手动 `systemctl --user start fun-voice-worker fun-voice-daemon`）。
2. 在任意输入框**按住** `Super+C` 说话，松开后文本上屏到录音开始时的焦点窗口。
3. 自检与诊断：

```bash
fun-voice-selftest --format json
journalctl --user -u fun-voice-worker -f
journalctl --user -u fun-voice-daemon -f
```

详见 `docs/operations.md`；真实环境人工验收见 `docs/acceptance-checklist.md`。

## 卸载

```bash
scripts/uninstall-user.sh            # 保留模型缓存与用户配置
scripts/uninstall-user.sh --purge    # 二次确认后连模型缓存与配置一并删除
```
