# 真实 Deepin X11 会话验收清单

> 本清单由**人工**在真实 Deepin DDE **X11** 会话中执行。先完成第 0 节首次初始化，再确认
> daemon 已运行。Nano/SenseVoice worker 与 Qwen 都是按需进程，登录后未启动是预期行为。
> `x11_hotkey` 在本次 daemon 启动后首次真实按住 `Super+C` 前会是 `fail`；完成第 1 节的
> 第一次按住/松开后重新运行 `fun-voice-selftest --format json`，应全部 `pass`。

## 0. 选择一个互斥的硬件验收路径

以下三节按机器实际硬件**只执行一节**。不要在一台不支持目标后端的机器上依次运行三种
显式初始化；显式模式失败不会回退，也不得改变已有 `selection.json`。

### CUDA 机器

- [ ] 执行 `scripts/initialize-first-run.sh --backend cuda`，确认私有 `selection.json`
      的 backend/device 是 `cuda`/`cuda:0`，dtype 是探测通过的 BF16，硬件不支持时才是 FP16。
- [ ] 确认 Nano 是 primary、SenseVoiceSmall 是 fallback；有效录音前没有模型进程常驻。
- [ ] 启用修正时确认 `fun-voice-worker@nano.service`（或实际使用的 SenseVoice worker）
      已停止，随后才出现 Qwen 进程。当前版本不验收 CAM++、说话人或结构化结果 API。
- [ ] 按住/松开 `Super+C` 完成一次中英混合输入，确认最终文本上屏并留在 clipboard。

### Intel XPU 机器

- [ ] 执行 `scripts/initialize-first-run.sh --backend xpu`，确认 `selection.json` 是
      `xpu`/`xpu:0`/BF16，不接受 FP16 或 CPU 静默回退。
- [ ] 确认 Nano primary、SenseVoiceSmall fallback，且 Qwen 只在 ASR worker 已停止后运行；
      当前版本不应启动 CAM++，也不验收说话人或结构化结果 API。
- [ ] `docs/xpu-poc.md` 的九项 POC 可作为显式 Intel XPU 诊断运行，但它不是 CUDA/CPU
      初始化的前置条件。
- [ ] 按住/松开 `Super+C` 完成输入，确认最终文本上屏并留在 clipboard。

### 纯 CPU 机器

- [ ] 执行 `scripts/initialize-first-run.sh --backend cpu`，确认 `selection.json` 是
      `cpu`/`cpu`/FP32，ASR 策略为 **SenseVoice-only** 且没有 fallback。
- [ ] 确认不存在 `fun-voice-worker@nano` socket、服务进程或模型加载；只允许
      SenseVoiceSmall worker 按需启动。
- [ ] 检查 `${XDG_DATA_HOME:-$HOME/.local/share}/fun-voice-ryan/models/models`，确认本次
      初始化没有下载 Qwen3.5-0.8B 或 CAM++ 快照，也没有 Qwen/CAM++ 进程。
- [ ] 按住/松开 `Super+C`，确认原始 SenseVoiceSmall 文本经 Fcitx 上屏并写入 clipboard；
      悬浮窗、焦点保护、长录音切分等非模型桌面行为与加速器路径一致。

完成所选初始化路径后执行：

```bash
systemctl --user status fun-voice-daemon.service --no-pager
```

确认 daemon 为 `active` 后再继续以下交互验收。

## 1. X11 Super+C 独占、按住录音、松开识别

- [ ] 执行一次 `Super+C` 按住/松开后运行 `fun-voice-selftest --format json`，确认
      `x11_hotkey` 为 `pass`，且 detail 仅含 `registered=true`、`press_seen=true`。
- [ ] 按住 `Super+C` 说一句普通话（如「今天天气不错」），松开后文本在**录音开始时**的焦点窗口上屏。
- [ ] 松开后完成识别与上屏，无重复上屏、无乱码。第一次使用时 Nano 会在录音期预加载；
      若录音很短或模型仍在加载，耗时可能高于热态。
- [ ] 先松开 Super、保持 C 按下，确认录音不中断；最后松开 C 后才识别。
- [ ] 长按触发自动重复时只产生一次录音；目标应用不收到字母 `c`。
- [ ] 让其他 X11 客户端临时抢占 `Super+C` 后启动 daemon，确认服务退出码为 `2` 且不循环重启；
      移除冲突后重启服务可重新抓取热键。

## 2. DTK 悬浮窗视觉与焦点

- [ ] 在浅色主题按住 `Super+C`，确认悬浮窗位于当前屏幕中下部、水平居中，中文状态标签清晰可读，
      且窗口具有圆角、半透明和轻微模糊效果。
- [ ] 切换 DDE 深色主题后再次按住 `Super+C`，确认窗口自动使用深色卡片和浅色文字；如果系统
      无合成器或模糊能力，确认仍显示可读的实色圆角卡片。
- [ ] 说含中文、英文与命令的短句，确认临时显示无乱码；点击或继续在原输入框键入，确认悬浮窗
      不夺取焦点、不拦截鼠标、不改变输入位置。
- [ ] 完成或取消后等待 5 秒，执行 `pgrep -af '[f]un-voice-overlay'`，确认没有残留进程。

### 悬浮窗布局配置

- [ ] 将 `vertical_center_ratio` 设为 `0.70`，重启 daemon 后按住 `Super+C`，确认卡片位于当前
      屏幕中下部、水平居中且未压到 Dock。
- [ ] 将 `width_px` 改为 `420` 与 `1000` 分别重启测试，确认卡片宽度变化；在窄屏上确认它自动
      收缩且仍在工作区内。
- [ ] 将 `font_scale` 改为 `0.80` 与 `1.80` 分别重启测试，确认四类文字同比缩放且中文、英文、
      代码混排无乱码。
- [ ] 将任一字段设为越界值，重启后确认 daemon 明确拒绝启动；恢复合法值后确认语音与目标窗口
      焦点正常。

## 3. 中英夹杂 / 代码原样输入

- [ ] 按住 `Super+C` 说「定义变量 count 等于 42，然后 print 出来」，松开后确认英文与
      代码没有被翻译或重排；允许 Qwen 修正明显术语、标点与空格错误。
- [ ] 说一段含符号的口语（如「路径是 usr 斜杠 local 斜杠 bin」），确认输出为
      「路径是 usr/local/bin」等符合预期的原样文本，没有被翻译成中文或改写。

## 4. 剪贴板留存（识别结果）

- [ ] 完成一次语音输入并成功上屏。
- [ ] 在别处粘贴（Ctrl+V），确认剪贴板内容为本次**最终上屏文本**（修正成功时为修正后
      文本，其他情况为原始识别结果；成功注入后不恢复旧剪贴板）。
- [ ] （可选）在 Fcitx 不可用的场景触发剪贴板回退，确认语音文本进入剪贴板且可粘贴。

## 5. 切窗 / 异常 / 空音频不误输入

- [ ] **切窗**：按住 `Super+C` 开始录音后，切换到另一个窗口再松开，确认文本**不会**误输入到新窗口（或明确拒绝上屏）。
- [ ] **空音频**：按住 `Super+C` 不说话直接松开，确认无任何文本上屏、无报错弹窗。
- [ ] **异常中断**：录音中途强制结束 daemon（`systemctl --user stop fun-voice-daemon`），确认不会把半截/空内容输入到窗口，且服务重启后恢复正常。

## 6. 长录音切分与自动停止

> 用 `fun-voice-daemon` 日志确认阈值行为（日志只含时长/状态，不含正文）。

- [ ] **>10 分钟切分**：模拟超长录音，确认超过阈值后按 60 秒切分为多个分片，最终仍能给出完整转写。
- [ ] **25 分钟提醒**：录音接近 25 分钟时收到一次通知提醒。
- [ ] **30 分钟停止**：达到 30 分钟硬上限后自动停止，不再继续录音。

## 7. 加速器无静默 CPU 回退

- [ ] 在 CUDA 或 XPU 验收路径中观察 `journalctl --user -u fun-voice-worker@nano`，确认
      推理设备与 `selection.json` 一致，无「CPU fallback」等字样。
- [ ] 启用 Qwen 后确认其在 Nano/SenseVoice worker 停止后才运行；停止确认失败时应直接得到
      原始转写，而不应同时驻留两个模型。
- [ ] `fun-voice-selftest --format json` 的 runtime selection 与 worker health 均为 `pass`。

## 8. 重启后无残留

- [ ] 注销并重新登录 DDE 会话。
- [ ] 确认 `fun-voice-daemon` 随登录运行、Nano/SenseVoice/Qwen 均未因登录常驻加载。
- [ ] 确认 `Super+C` 仍然可用（X11 grab 未因重启失效）。
- [ ] 确认没有残留的 `~/.local/bin/fun-voice-*` 之外的临时文件或 socket 遗留（`$XDG_RUNTIME_DIR/fun-voice-ryan/` 下仅应有运行时 socket，无 capture 分片残留）。
