# 手动验收：DDE Super+C 按住说话时序

> 本文件是**手动验收记录模板**。当前环境尚未注册快捷键（注册脚本仅供 Task 8 安装阶段使用），
> 因此下面的条目在安装并运行 `scripts/register-dde-shortcut.sh` 后逐条填写。
> 记录过程严禁读取 `/dev/input`，也不得把转写正文写进日志或本文件。

## 前置条件

- 运行在 Deepin DDE X11 会话（`echo $XDG_SESSION_TYPE` 应为 `x11`）。
- 已通过 Task 2 XPU POC 硬门。
- 已安装 Fcitx5（`input_method = fcitx5`）。
- `python-xlib`、`xclip`（或 `xsel`）可用。

## DDE 冲突检查（只读，已执行）

命令：

```bash
dbus-send --session --dest=org.deepin.dde.Keybinding1 --print-reply \
  /org/deepin/dde/Keybinding1 \
  org.deepin.dde.Keybinding1.LookupConflictShortcut string:"<Super>C"
```

实际输出（2026-08-31，Super+C 无冲突；空 struct 表示未占用）：

```text
method return time=1788159659.256536 sender=:1.142 -> destination=:1.1544 serial=132 reply_serial=2
   struct {
      string ""
      string ""
      string ""
      array [
      ]
      string ""
      string ""
      boolean false
      boolean false
   }
```

busctl 等价只读调用：

```bash
busctl --user call org.deepin.dde.Keybinding1 /org/deepin/dde/Keybinding1 \
  org.deepin.dde.Keybinding1 LookupConflictShortcut s '<Super>C>'
# (sssasssbb) "" "" "" 0 "" "" false false
```

> 对照组（已被 UOS AI Talk 占用的快捷键，证明「不得假设 Ctrl+Super+Space 可用」）：
>
> ```bash
> busctl --user call org.deepin.dde.Keybinding1 /org/deepin/dde/Keybinding1 \
>   org.deepin.dde.Keybinding1 LookupConflictShortcut s '<Control><Super>space'
> # (sssasssbb) "org.deepin.dde.keybinding.shortcut.app.uos-ai-talk" "UOS AI Talk" "UOS AI" 1 "<Control><Super>space" ... false true
> ```

## 验收清单

目标应用各做一轮（浏览器、终端、IDE/文本编辑器）。

| # | 操作 | 预期 | 结果 |
| --- | --- | --- | --- |
| 1 | 按住 Super+C 说一句话 | 触发 bridge，`start_if_idle`，开始录音 | ☐ |
| 2 | 按住期间长按不动 | C 键持续 down；不因自动重复产生重复 session | ☐ |
| 3 | 松开 C | bridge 发 `stop`，进入识别并上屏 | ☐ |
| 4 | 录音中切换到另一窗口 | 不注入目标窗口，只写剪贴板并通知 | ☐ |
| 5 | 目标内 Fcitx 切换中/英文后按住说话 | 文本原样上屏到录音开始时的焦点窗口 | ☐ |
| 6 | 松开后观察目标应用 | 不误输入字母 `c` | ☐ |
| 7 | 连续 3 次按住/松开 | 无重复文本、无串台、无进程泄漏 | ☐ |

### POC 门：DDE 是否在按住阶段触发

- [ ] 按住 Super+C **不松开**，确认 bridge 在按键仍按下时被调用（可观察 daemon 日志的
      `start_if_idle` 时间点，或临时打印时间戳）。
- [ ] 若 DDE 只在**松开后**才触发 action，则本方案「按住说话」不成立：**记录 POC fail**，
      **不得**暗改为「按一下开始、再按一下停止」的切换模式；按流程请求人工决定是否研究
      纯 X11 全局抓键（XGrabKey）回退方案。

### 隐私边界确认

- [ ] 代码与日志均未读取 `/dev/input/event*`（`grep -r "/dev/input" src scripts` 为空）。
- [ ] 日志/通知只含长度、状态、错误类别与请求 id，不含转写正文。

## 记录

- 日期 / 环境：
- DDE 按住触发：☐ 按住阶段 ☐ 松开后（POC fail）
- 各条目结果：见上表
- 备注：
