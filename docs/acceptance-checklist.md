# 真实 Deepin X11 会话验收清单

> 本清单由**人工**在真实 Deepin DDE **X11** 会话中执行。执行前先确认
> `systemctl --user status fun-voice-daemon` 为 `active`。Nano/SenseVoice worker 与 Qwen
> 都是按需进程，登录后未启动是预期行为。
> `x11_hotkey` 在本次 daemon 启动后首次真实按住 `Super+C` 前会是 `fail`；完成第 1 节的
> 第一次按住/松开后重新运行 `fun-voice-selftest --format json`，应全部 `pass`。

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

- [ ] 在浅色主题按住 `Super+C`，确认悬浮窗位于当前屏幕底部居中，中文状态标签清晰可读，
      且窗口具有圆角、半透明和轻微模糊效果。
- [ ] 切换 DDE 深色主题后再次按住 `Super+C`，确认窗口自动使用深色卡片和浅色文字；如果系统
      无合成器或模糊能力，确认仍显示可读的实色圆角卡片。
- [ ] 说含中文、英文与命令的短句，确认临时显示无乱码；点击或继续在原输入框键入，确认悬浮窗
      不夺取焦点、不拦截鼠标、不改变输入位置。
- [ ] 完成或取消后等待 5 秒，执行 `pgrep -af '[f]un-voice-overlay'`，确认没有残留进程。

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

## 7. XPU 无 CPU 回退

- [ ] 录音输入期间观察 `journalctl --user -u fun-voice-worker@nano`，确认推理设备为
      `xpu:0`，无「CPU fallback」「decoder device type is cpu」等字样。
- [ ] 启用 Qwen 后确认其在 Nano/SenseVoice worker 停止后才运行；停止确认失败时应直接得到
      原始转写，而不应同时驻留两个模型。
- [ ] `fun-voice-selftest --format json` 中 `xpu_hard_gate` 与 `worker_health` 均为 `pass`。

## 8. 重启后无残留

- [ ] 注销并重新登录 DDE 会话。
- [ ] 确认 `fun-voice-daemon` 随登录运行、Nano/SenseVoice/Qwen 均未因登录常驻加载。
- [ ] 确认 `Super+C` 仍然可用（X11 grab 未因重启失效）。
- [ ] 确认没有残留的 `~/.local/bin/fun-voice-*` 之外的临时文件或 socket 遗留（`$XDG_RUNTIME_DIR/fun-voice-ryan/` 下仅应有运行时 socket，无 capture 分片残留）。
