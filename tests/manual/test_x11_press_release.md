# 手动验收：X11 `Super+C` 按住说话

本清单验证 daemon 直接以 X11 `XGrabKey` 独占监听 `Super+C`。不需要、也不得注册
DDE 快捷键或运行 bridge。记录过程不得保存音频、转写正文，亦不得读取
`/dev/input/event*`。

## 前置条件

- Deepin DDE 的 X11 会话（`echo "$XDG_SESSION_TYPE"` 输出 `x11`）。
- Intel XPU POC 的 9 项硬门均已通过，且已安装 Fcitx5。
- 执行 `scripts/install-user.sh`，随后确认服务运行：

  ```bash
  systemctl --user status fun-voice-daemon.service --no-pager
  ```

- 重启 daemon 后先在任意输入框按住并松开一次 `Super+C`，再执行：

  ```bash
  fun-voice-selftest --format json
  ```

  其中 `x11_hotkey` 必须为 `pass`，detail 只应包含 `registered` 与
  `press_seen` 两个布尔值。

## 验收清单

在浏览器、终端和 IDE/文本编辑器各做一轮。

| # | 操作 | 预期 | 结果 |
| --- | --- | --- | --- |
| 1 | 先按 Super，再按 C 并说一段普通话混合英文、代码或术语 | 只开始一次录音 | ☐ |
| 2 | 一直按住 C（触发键盘自动重复） | 不产生第二个录音 session | ☐ |
| 3 | 先松开 Super，继续按住 C | 仍保持录音 | ☐ |
| 4 | 最后松开 C | 停止录音，识别后原样上屏 | ☐ |
| 5 | 录音中切换到另一窗口 | 不向新窗口注入，只写剪贴板并通知 | ☐ |
| 6 | 松开后观察目标应用 | 不误输入字母 `c` | ☐ |
| 7 | 连续三次按住/松开 | 无重复文本、无串台、无 daemon 泄漏 | ☐ |

## 冲突与故障行为

- 使用另一 X11 客户端抢占 `Super+C` 后启动 daemon：应退出码为 `2`；
  `systemctl --user status` 显示失败，且不会不断重启。
- 移除冲突后重启 daemon：应重新取得 grab，`x11_hotkey.registered` 为真。
- 检查日志只含状态和错误类别，不含录音路径、音频内容或转写正文：

  ```bash
  journalctl --user -u fun-voice-daemon.service --since '10 minutes ago'
  ```

## 记录

- 日期 / 会话类型：
- X11 热键注册与首次按下：☐ 通过
- 各目标应用结果：见上表
- 冲突退出码 2：☐ 通过
- 备注：
