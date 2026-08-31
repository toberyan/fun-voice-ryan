# 真实 DDE X11 会话验收清单

> 本清单由**人工**在真实 Deepin DDE **X11** 会话中执行。执行前先确认：
> `fun-voice-selftest --format json` 全部 `pass`，且
> `systemctl --user status fun-voice-worker fun-voice-daemon` 均为 `active`。

## 1. Super+C 无冲突，按住录音、松开识别

- [ ] 在 DDE「控制中心 → 键盘 → 快捷键」中确认 `Super+C` 只属于「Fun Voice Ryan — 按住说话」，无红色冲突提示。
- [ ] 按住 `Super+C` 说一句普通话（如「今天天气不错」），松开后文本在**录音开始时**的焦点窗口上屏。
- [ ] 松开后约 1~2 秒内完成识别与上屏，无重复上屏、无乱码。

## 2. 中英夹杂 / 代码原样输入

- [ ] 按住 `Super+C` 说「定义变量 count 等于 42，然后 print 出来」，松开后得到
      「定义变量 count 等于 42，然后 print 出来」（英文/代码按口语原样保留，**未经改写**）。
- [ ] 说一段含符号的口语（如「路径是 usr 斜杠 local 斜杠 bin」），确认输出为
      「路径是 usr/local/bin」等符合预期的原样文本，没有被翻译成中文或改写。

## 3. 剪贴板留存

- [ ] 先复制一段文本到剪贴板（Ctrl+C）。
- [ ] 完成一次语音输入并成功上屏。
- [ ] 再次粘贴（Ctrl+V），确认剪贴板里仍然是刚才 Ctrl+C 的原文，未被语音文本污染或清空。
- [ ] （可选）在 Fcitx 不可用的场景触发剪贴板回退，确认语音文本进入剪贴板且可粘贴。

## 4. 切窗 / 异常 / 空音频不误输入

- [ ] **切窗**：按住 `Super+C` 开始录音后，切换到另一个窗口再松开，确认文本**不会**误输入到新窗口（或明确拒绝上屏）。
- [ ] **空音频**：按住 `Super+C` 不说话直接松开，确认无任何文本上屏、无报错弹窗。
- [ ] **异常中断**：录音中途强制结束 daemon（`systemctl --user stop fun-voice-daemon`），确认不会把半截/空内容输入到窗口，且服务重启后恢复正常。

## 5. 长录音切分与自动停止

> 用 `fun-voice-daemon` 日志确认阈值行为（日志只含时长/状态，不含正文）。

- [ ] **>10 分钟切分**：模拟超长录音，确认超过阈值后按 60 秒切分为多个分片，最终仍能给出完整转写。
- [ ] **25 分钟提醒**：录音接近 25 分钟时收到一次通知提醒。
- [ ] **30 分钟停止**：达到 30 分钟硬上限后自动停止，不再继续录音。

## 6. XPU 无 CPU 回退

- [ ] 录音输入期间观察 `journalctl --user -u fun-voice-worker`，确认推理设备为 `xpu:0`，无「CPU fallback」「decoder device type is cpu」等字样。
- [ ] `fun-voice-selftest --format json` 中 `xpu_hard_gate` 与 `worker_health` 均为 `pass`。

## 7. 重启后无残留

- [ ] 注销并重新登录 DDE 会话。
- [ ] 确认 `fun-voice-worker` / `fun-voice-daemon` 随登录自动运行（autostart 生效）。
- [ ] 确认 `Super+C` 仍然可用（快捷键未因重启失效或重复注册）。
- [ ] 确认没有残留的 `~/.local/bin/fun-voice-*` 之外的临时文件或 socket 遗留（`$XDG_RUNTIME_DIR/fun-voice-ryan/` 下仅应有运行时 socket，无 capture 分片残留）。
