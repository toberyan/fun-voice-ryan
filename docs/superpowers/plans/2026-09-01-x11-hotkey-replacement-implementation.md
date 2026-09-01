# X11 全局热键替换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 DDE/bridge 快捷键机制，以 daemon 内置 X11 `XGrabKey` 监听器可靠实现 `Super+C` 按住录音、松开识别。

**Architecture:** `X11HotkeyListener` 独立持有一个 X11 Display，在 root window 原子注册 `Super+C` 及锁定修饰键组合，并把有序按下/松开事件直接交给 `VoiceDaemon`。daemon 保留当前 capture → worker → Fcitx 管道与焦点保护，但删除 DDE 诊断、bridge IPC 依赖和 C 键轮询释放。安装升级仅一次性、安全地注销本项目留下的旧 DDE shortcut；新运行时完全不依赖 DDE。

**Tech Stack:** Python 3.12、python-xlib、X11/XGrabKey、systemd --user、Bash、pytest、ruff、mypy、CMake/CTest。

## Global Constraints

- 只支持 Deepin DDE **X11**；不得加入 Wayland、`/dev/input`、evdev、root 权限或切换录音。
- `Super+C` 必须由本助手独占；抓取失败即以专用退出码失败，不得回退 DDE 或轮询。
- 热键日志和 diagnostics 仅保存布尔状态；不得记录转写、音频、按键时间、焦点或 Fcitx token。
- 不改 Fun-ASR-Nano、XPU、VAD、PipeWire、Fcitx、剪贴板和焦点校验的功能语义。
- 现有工作树包含用户未提交修改；每个提交只能暂存本任务实际修改的文件。

---

### Task 1: 建立可原子清理的 X11 `Super+C` 监听器

**Files:**
- Modify: `src/fun_voice/desktop.py:1-330,497-667`
- Modify: `tests/test_desktop.py:1-390`

**Interfaces:**
- Produces `X11HotkeyUnavailable(X11Error)`。
- Produces `X11HotkeyListener(on_press: Callable[[], None], on_release: Callable[[], None], *, make_display: Callable[[], XDisplay] = default_make_display, select_ready: Callable[[int, float], bool] = ...)`。
- `start() -> None` registers all combinations then starts one daemon event thread; `close() -> None` is idempotent.
- `handle_event(event: object) -> None` is package-visible for deterministic unit tests.

- [ ] **Step 1: 写出失败的监听器测试**

  删除 DDE/bridge 测试、相关 fake runner 与 D-Bus 常量。扩展 `FakeDisplay` / `_FakeRootWindow`，记录 `grab_key(key, modifiers, owner_events, pointer_mode, keyboard_mode, onerror)`、`ungrab_key(key, modifiers)`、`sync()`、`get_modifier_mapping()`、`fileno()` 和事件；添加下面的行为测试：

  ```python
  def test_hotkey_grabs_super_c_with_every_lock_combination() -> None:
      display = FakeHotkeyDisplay(super_index=6, num_index=4, scroll_index=7)
      listener = X11HotkeyListener(lambda: None, lambda: None, make_display=lambda: display)

      listener.start()

      assert {modifiers for _key, modifiers in display.grabs} == {
          1 << 6, (1 << 6) | X_LOCK_MASK, (1 << 6) | (1 << 4),
          (1 << 6) | (1 << 7), (1 << 6) | X_LOCK_MASK | (1 << 4),
          (1 << 6) | X_LOCK_MASK | (1 << 7), (1 << 6) | (1 << 4) | (1 << 7),
          (1 << 6) | X_LOCK_MASK | (1 << 4) | (1 << 7),
      }
      listener.close()

  def test_hotkey_rolls_back_every_successful_grab_on_bad_access() -> None:
      display = FakeHotkeyDisplay(fail_modifier=(1 << 6) | X_LOCK_MASK)
      listener = X11HotkeyListener(lambda: None, lambda: None, make_display=lambda: display)

      with pytest.raises(X11HotkeyUnavailable, match="already grabbed"):
          listener.start()

      assert display.ungrabs == display.grabs_before_failure
      assert display.closed is True

  def test_hotkey_press_repeat_and_release_call_each_callback_once() -> None:
      calls: list[str] = []
      display = FakeHotkeyDisplay()
      listener = X11HotkeyListener(
          lambda: calls.append("start"), lambda: calls.append("stop"),
          make_display=lambda: display,
      )
      listener.start()

      listener.handle_event(FakeKeyEvent(X_KEY_PRESS, display.c_keycode, X_SUPER_MASK))
      listener.handle_event(FakeKeyEvent(X_KEY_PRESS, display.c_keycode, X_SUPER_MASK))
      listener.handle_event(FakeKeyEvent(X_KEY_RELEASE, display.c_keycode, 0))
      listener.handle_event(FakeKeyEvent(X_KEY_RELEASE, display.c_keycode, 0))

      assert calls == ["start", "stop"]
      listener.close()
  ```

  另加：Super 未在 modifier mapping 中、C keycode 为 `0`、无 Super 的 press、`close()` 两次、事件线程退出前释放每个成功 grab 的测试。

- [ ] **Step 2: 运行测试确认 RED**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_desktop.py -q`

  Expected: FAIL，原因是 `X11HotkeyListener`、`X11HotkeyUnavailable` 和 fake X11 grab 接口尚不存在；不得是导入路径或测试拼写错误。

- [ ] **Step 3: 最小实现监听器并删除 DDE/bridge Python API**

  将 `desktop.py` 的模块说明更新为 X11 focus/hotkey、clipboard、XTEST。删除 `argparse`、DDE runner/client、busctl parser、shortcut state、`HotkeyBridge` 和 CLI `main()`；保留 `X11FocusGuard`、`ClipboardMirror`、`XTestInjector`。

  增加稳定常量及协议：

  ```python
  XK_C = 0x63
  XK_NUM_LOCK = 0xFF7F
  XK_SCROLL_LOCK = 0xFF14
  XK_SUPER_L = 0xFFEB
  XK_SUPER_R = 0xFFEC
  X_LOCK_MASK = 1 << 1
  HOTKEY_EVENT_WAIT_SECONDS = 0.1

  class X11HotkeyUnavailable(X11Error):
      """The X server cannot exclusively grab Super+C."""
  ```

  扩展 `XDisplay` / `XWindow` Protocol，声明 `get_modifier_mapping()`、`fileno()`、`pending_events()`、`next_event()`，以及 root 的 `grab_key()` / `ungrab_key()`。监听器使用 `display.keysym_to_keycode(XK_C)`；在 `get_modifier_mapping()` 的八组 keycode 中找同时包含 `XK_SUPER_L` 或 `XK_SUPER_R` keycode 的 modifier index，再以 `1 << index` 得到 Super mask。再找到 Num Lock、Scroll Lock 所在 modifier mask；与 `X_LOCK_MASK` 生成去重的 8 个忽略组合。

  `start()` 必须按组合调用 root `grab_key(keycode, modifiers, False, X.GrabModeAsync, X.GrabModeAsync, onerror=...)`，再 `display.sync()` 收集 `BadAccess`。没有 Super、没有 C、任一 X 错误或任何 grab error 都调用 `_release_grabs()`、`display.close()` 并抛 `X11HotkeyUnavailable`；不得留下部分注册。

  ```python
  def handle_event(self, event: object) -> None:
      if getattr(event, "detail", None) != self._keycode:
          return
      event_type = getattr(event, "type", None)
      if event_type == X_KEY_PRESS and self._press_matches_super(event):
          if not self._held:
              self._held = True
              self._on_press()
      elif event_type == X_KEY_RELEASE and self._held:
          self._held = False
          self._on_release()
  ```

  `_press_matches_super()` 必须忽略 Caps/Num/Scroll mask 后精确匹配 Super；release 不再检查当前 Super 状态，因此可处理“先松 Super，再松 C”。事件线程在 `select.select([display.fileno()], [], [], 0.1)` 就绪后读取 `next_event()` 并排空 `pending_events()`；`close()` 先设 stop event、join，再 ungrab 已成功的组合、sync、close。

- [ ] **Step 4: 运行局部测试确认 GREEN，并静态检查**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_desktop.py -q && .venv/bin/ruff check src/fun_voice/desktop.py tests/test_desktop.py && .venv/bin/mypy src/fun_voice/desktop.py`

  Expected: PASS；DDE、busctl、`HotkeyBridge` 的 Python 导入不再出现，X11 focus/XTEST 原有测试继续通过。

- [ ] **Step 5: 提交这一独立适配器变更**

  ```bash
  git add src/fun_voice/desktop.py tests/test_desktop.py
  git commit -m "feat: add X11 Super+C hotkey listener"
  ```

### Task 2: 将热键事件接入 daemon，并移除 C 释放轮询

**Files:**
- Modify: `src/fun_voice/daemon.py:1-110,372-610,827-1019`
- Modify: `tests/test_daemon.py:280-665`
- Modify: `tests/test_end_to_end_fakes.py:1-815`

**Interfaces:**
- `VoiceDaemon.handle_hotkey_press() -> str` sets only `hotkey_press_seen` then calls `start_if_idle()`.
- `VoiceDaemon.handle_hotkey_release() -> None` starts one daemon thread that calls `stop()`; it does not block the X11 event thread.
- `VoiceDaemon.mark_hotkey_registered() -> None` enables the diagnostic registration flag only after `X11HotkeyListener.start()` succeeds.
- `serve(..., hotkey_listener: HotkeyLifecycle | None = None) -> int` starts the listener before binding `daemon.sock`, and always calls `close()`.

- [ ] **Step 1: 写出失败的 daemon 集成测试**

  为 test harness 加入 `FakeHotkeyListener`，其 `start()` 可同步调用传入的 press/release callbacks、可抛 `X11HotkeyUnavailable`、并记录 `closed`。添加：

  ```python
  def test_hotkey_press_starts_recording_and_keeps_private_boolean() -> None:
      h = Harness(guard=FakeGuard(c_down=True))
      assert h.daemon.diagnostics() == {
          "hotkey_registered": False, "hotkey_press_seen": False,
      }

      assert h.daemon.handle_hotkey_press() == "started"

      assert h.daemon.diagnostics() == {
          "hotkey_registered": False, "hotkey_press_seen": True,
      }

  def test_hotkey_release_stops_without_blocking_listener_thread() -> None:
      h = Harness(guard=FakeGuard(c_down=True), worker=BlockingWorker())
      assert h.daemon.handle_hotkey_press() == "started"

      h.daemon.handle_hotkey_release()

      assert h.worker.started.wait(timeout=1.0)
      assert h.daemon.state is DaemonState.TRANSCRIBING

  def test_serve_closes_listener_when_the_server_exits(tmp_path: Path) -> None:
      listener = FakeHotkeyListener()
      # Trigger the existing signal/shutdown seam, then assert listener.closed.
      assert run_server_to_shutdown(tmp_path, listener) == 0
      assert listener.started is True
      assert listener.closed is True
  ```

  再加 listener `start()` 抛 `X11HotkeyUnavailable` 时 `serve()` 返回 `HOTKEY_UNAVAILABLE_EXIT=2`、不创建 socket、并执行 daemon cleanup 的测试。删除 `poll_c_release`、`_release_poller`、`C_RELEASE_*` 的全部测试；保留 500 ms 的开始阶段 C 确认作为事件-录音之间的防御层。

- [ ] **Step 2: 运行测试确认 RED**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_daemon.py tests/test_end_to_end_fakes.py -q`

  Expected: FAIL，因为热键 handler、listener 生命周期和新 diagnostics key 尚未实现。

- [ ] **Step 3: 最小接线实现**

  在 `daemon.py` 导入 `X11HotkeyListener`、`X11HotkeyUnavailable` 并定义小 Protocol：

  ```python
  class HotkeyLifecycle(Protocol):
      def start(self) -> None: ...
      def close(self) -> None: ...
  ```

  删除 C release 常量、`self._c_up_since`、`poll_c_release()`、`DaemonServer._release_poller`、`_ensure_release_poller()` 和对它们的调用。将 diagnostics 换成：

  ```python
  def diagnostics(self) -> dict[str, bool]:
      return {
          "hotkey_registered": self._hotkey_registered,
          "hotkey_press_seen": self._hotkey_press_seen,
      }

  def handle_hotkey_press(self) -> str:
      self._hotkey_press_seen = True
      return self.start_if_idle()

  def handle_hotkey_release(self) -> None:
      threading.Thread(target=self.stop, daemon=True, name="hotkey-stop").start()
  ```

  `serve()` 在 socket unlink 后、`DaemonServer(...)` 前启动可选 listener；只有成功后调用 `daemon.mark_hotkey_registered()`。失败时记录异常类别、调用 `daemon.shutdown()`、返回 `2`。`finally` 中先 `listener.close()`，再关闭 server、daemon、socket。`main()` 创建 `X11HotkeyListener(daemon.handle_hotkey_press, daemon.handle_hotkey_release)` 并传给 `serve()`。

- [ ] **Step 4: 运行 daemon 全量测试和静态检查**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_daemon.py tests/test_end_to_end_fakes.py -q && .venv/bin/ruff check src/fun_voice/daemon.py tests/test_daemon.py tests/test_end_to_end_fakes.py && .venv/bin/mypy src/fun_voice/daemon.py`

  Expected: PASS；一次 KeyPress 仅启动一次、KeyRelease 可在转写期间异步执行、抓取失败不绑定 socket，旧 release poller 完全移除。

- [ ] **Step 5: 提交 daemon 接线变更**

  ```bash
  git add -p src/fun_voice/daemon.py tests/test_daemon.py tests/test_end_to_end_fakes.py
  git commit -m "feat: drive voice daemon from X11 hotkey events"
  ```

  在提交前必须检查暂存 diff；这些文件当前已有未提交内容，只暂存本任务 hunk。

### Task 3: 以 X11 热键状态替换自检 DDE 项

**Files:**
- Modify: `src/fun_voice/selftest.py:1-452`
- Modify: `tests/test_selftest.py:1-364`

**Interfaces:**
- `probe_hotkey_state(socket_path: Path | None = None) -> dict[str, bool] | None` reads same-UID daemon diagnostics with a 1 s bound.
- `check_x11_hotkey(probe: Callable[[], dict[str, bool] | None]) -> SelfTestResult` is the single hotkey check.
- `run_selftest()` has no DDE factory or DDE checks.

- [ ] **Step 1: 写出失败的自检测试**

  删除 `_FakeDdeClient`、DDE imports 和三项 DDE/bridge tests，改写 report fixture。添加：

  ```python
  def test_x11_hotkey_passes_after_real_press() -> None:
      result = check_x11_hotkey(
          lambda: {"hotkey_registered": True, "hotkey_press_seen": True}
      )
      assert result.status == STATUS_PASS
      assert result.detail == {"registered": True, "press_seen": True}

  def test_x11_hotkey_fails_before_any_press_without_sensitive_data() -> None:
      result = check_x11_hotkey(
          lambda: {"hotkey_registered": True, "hotkey_press_seen": False}
      )
      assert result.status == STATUS_FAIL
      assert result.detail == {"registered": True, "press_seen": False}

  def test_x11_hotkey_fails_when_daemon_cannot_be_reached() -> None:
      result = check_x11_hotkey(lambda: None)
      assert result.status == STATUS_FAIL
      assert result.detail["reason"] == "daemon diagnostics unavailable"
  ```

  另加 socket probe 遇到非布尔/多余 payload 时返回 `None`，以及 `run_selftest()` checks 中只包含一次 `x11_hotkey`、不存在 `dde_service` / `super_c_conflict` / `bridge_hold_timing` 的测试。

- [ ] **Step 2: 运行测试确认 RED**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_selftest.py -q`

  Expected: FAIL，原因是新 probe/check 未定义或旧 `run_selftest` 仍请求 DDE client。

- [ ] **Step 3: 最小实现新自检**

  删除 `DdeKeybindingClient`、`DdeKeybindingError` 的导入，以及 `check_dde_service`、`check_super_c_conflict`、`check_bridge_timing`、`probe_held_trigger_seen`。复用现有 Unix socket 请求逻辑发送 `{"op":"diagnostics"}`，只接受：

  ```python
  {"status": "ok", "hotkey_registered": bool, "hotkey_press_seen": bool}
  ```

  `check_x11_hotkey` 的 detail 只输出 `registered`、`press_seen` 和固定 `reason`；不把 socket reply 原样回显。将其作为 `run_selftest()` 的第一项；其余 PipeWire、Fcitx、clipboard、XTEST、worker、XPU 检查顺序保持不变。

- [ ] **Step 4: 验证自检模块**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_selftest.py -q && .venv/bin/ruff check src/fun_voice/selftest.py tests/test_selftest.py && .venv/bin/mypy src/fun_voice/selftest.py`

  Expected: PASS；测试和生产自检没有 DDE class、D-Bus 或 bridge 用语。

- [ ] **Step 5: 提交自检变更**

  ```bash
  git add -p src/fun_voice/selftest.py tests/test_selftest.py
  git commit -m "feat: report X11 hotkey readiness in selftest"
  ```

  在提交前必须检查暂存 diff；只暂存本任务 hunk。

### Task 4: 删除 DDE 资产，并实现一次性安全升级清理

**Files:**
- Delete: `src/fun_voice/bridge.py`
- Delete: `scripts/register-dde-shortcut.sh`
- Delete: `scripts/unregister-dde-shortcut.sh`
- Delete: `scripts/start-session-bridge.sh`
- Create: `scripts/import-session-environment.sh`
- Modify: `pyproject.toml:9-14`
- Modify: `scripts/install-user.sh:1-160`
- Modify: `scripts/uninstall-user.sh:1-150`
- Modify: `systemd/fun-voice-daemon.service:1-20`
- Modify: `systemd/fun-voice-session.desktop:1-8`
- Create: `tests/test_install_scripts.py`
- Delete: `tests/manual/test_dde_press_release.md`
- Create: `tests/manual/test_x11_press_release.md`

**Interfaces:**
- Installer has no DDE registration path. A legacy `dde-shortcut-id` is retired only if `GetShortcutCommand` proves its action ends in `fun-voice-bridge`.
- `import-session-environment.sh` imports `DISPLAY`, `XAUTHORITY`, `DBUS_SESSION_BUS_ADDRESS`, validates `DISPLAY`, and restarts the daemon; it never sends start/stop.
- `fun-voice-daemon.service` uses `RestartPreventExitStatus=2`.

- [ ] **Step 1: 写出失败的安装/遗留清理静态测试**

  新建 `tests/test_install_scripts.py`：

  ```python
  ROOT = Path(__file__).resolve().parents[1]

  def test_new_install_has_no_dde_registration_or_bridge_console_script() -> None:
      install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
      project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
      assert "AddCustomShortcut" not in install
      assert "register-dde-shortcut.sh" not in install
      assert "fun-voice-bridge =" not in project

  def test_installer_retires_only_verified_legacy_bridge_shortcut() -> None:
      install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
      assert "GetShortcutCommand" in install
      assert "fun-voice-bridge" in install
      assert "DeleteCustomShortcut" in install
      assert install.index("GetShortcutCommand") < install.index("DeleteCustomShortcut")

  def test_dde_and_bridge_source_assets_are_removed() -> None:
      assert not (ROOT / "src/fun_voice/bridge.py").exists()
      assert not (ROOT / "scripts/register-dde-shortcut.sh").exists()
      assert not (ROOT / "scripts/unregister-dde-shortcut.sh").exists()
      assert not (ROOT / "scripts/start-session-bridge.sh").exists()
  ```

  增加检查：新 session script 不含 `fun-voice-bridge`，desktop entry 引用新脚本，uninstaller 的 console list 没有 bridge，且 daemon service 有 `RestartPreventExitStatus=2`。

- [ ] **Step 2: 运行测试确认 RED**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_install_scripts.py -q`

  Expected: FAIL，因为 DDE/bridge 文件、入口和安装逻辑仍存在。

- [ ] **Step 3: 移除资产并实现迁移**

  删除上述 DDE/bridge 文件，移除 `pyproject.toml` 的 `fun-voice-bridge` entry。`install-user.sh` 的 `CONSOLE_SCRIPTS`、源验证、日志、步骤注释不再含 bridge 或 DDE 注册。

  在首次重启服务前加入 `retire_legacy_dde_shortcut()`：若 `${XDG_CONFIG_HOME:-${HOME}/.config}/fun-voice-ryan/dde-shortcut-id` 不存在，直接返回且不调用 `busctl`。若存在，读取单行 id，调用：

  ```bash
  busctl --user call org.deepin.dde.Keybinding1 /org/deepin/dde/Keybinding1 \
      org.deepin.dde.Keybinding1 GetShortcutCommand s "${shortcut_id}"
  ```

  仅当回复可识别且 command 以旧 `fun-voice-bridge` wrapper 为目标时，才调用 `DeleteCustomShortcut` 并删除 id 文件；任何 D-Bus 错误、空/未知 command、非预期 id 均 `die "legacy-dde" ...`，不继续安装也不删除未知快捷键。随后始终删除旧 `${BIN_DIR}/fun-voice-bridge` regular file/symlink。这样新安装无 DDE 依赖，旧安装可安全去除遗留全局 grab。

  新 `import-session-environment.sh` 只执行 `systemctl --user import-environment DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS`、检查 `DISPLAY`、`systemctl --user restart fun-voice-daemon.service`；重命名 desktop entry 文案并指向新脚本。将安装步骤改为部署后 restart 服务，保证已运行的旧 daemon 被替换。删除 uninstaller 的 DDE shortcut 逻辑和 bridge console script；补入 `RestartPreventExitStatus=2` 到 daemon unit。

  将手工文档替换为 `test_x11_press_release.md`：包括无 DDE 前提、先按 C 后松 C、先松 Super 后松 C、自动重复、目标应用不收到 C、X11 冲突时 daemon exit 2、日志不含敏感数据。

- [ ] **Step 4: 验证资产清理和 Bash 语法**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_install_scripts.py -q && bash -n scripts/install-user.sh scripts/uninstall-user.sh scripts/import-session-environment.sh && rg -n -i 'DdeKeybinding|AddCustomShortcut|register-dde|unregister-dde|fun-voice-bridge|HotkeyBridge' src scripts systemd pyproject.toml --glob '!scripts/install-user.sh'`

  Expected: pytest 与 `bash -n` PASS；最后的 `rg` 没有输出。`install-user.sh` 中仅可保留注释明确标注为一次性 legacy cleanup 的 `GetShortcutCommand` / `DeleteCustomShortcut`。

- [ ] **Step 5: 提交部署与迁移变更**

  ```bash
  git add -u -- src/fun_voice/bridge.py scripts/register-dde-shortcut.sh \
    scripts/unregister-dde-shortcut.sh scripts/start-session-bridge.sh \
    tests/manual/test_dde_press_release.md
  git add pyproject.toml scripts/import-session-environment.sh systemd/fun-voice-daemon.service \
    systemd/fun-voice-session.desktop \
    tests/test_install_scripts.py tests/manual/test_x11_press_release.md
  git add -p scripts/install-user.sh scripts/uninstall-user.sh
  git commit -m "refactor: replace DDE shortcut deployment with X11 hotkey"
  ```

  在提交前必须检查暂存 diff；`scripts/install-user.sh` 已有未提交修改，只暂存本任务 hunk。

### Task 5: 更新用户文档、全量回归并完成真实 X11 部署验证

**Files:**
- Modify: `README.md:1-80`
- Modify: `docs/operations.md:1-140`
- Modify: `docs/acceptance-checklist.md:1-80`
- Modify: `docs/superpowers/specs/2026-08-31-fun-asr-nano-intel-xpu-voice-assistant-design.md:40-105,230-250`

**Interfaces:**
- 所有面向用户的文档把快捷键所有权描述为 X11 daemon grab，而非 DDE 注册。
- 历史原始设计添加“已由 2026-09-01 X11 设计替代”的明确链接，不重写历史决策。

- [ ] **Step 1: 写出失败的文档一致性测试**

  将 `tests/test_install_scripts.py` 扩展：

  ```python
  def test_current_user_docs_describe_x11_not_dde_bridge() -> None:
      for relative in ("README.md", "docs/operations.md", "docs/acceptance-checklist.md"):
          text = (ROOT / relative).read_text(encoding="utf-8")
          assert "DDE 快捷键" not in text
          assert "bridge_hold_timing" not in text
          assert "X11" in text
  ```

  历史规格的测试只断言存在到新规格的 `2026-09-01-x11-hotkey-replacement-design.md` 链接，避免要求历史文本改写。

- [ ] **Step 2: 运行测试确认 RED**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_install_scripts.py -q`

  Expected: FAIL，因为 README、operations、acceptance 仍说明 DDE/bridge。

- [ ] **Step 3: 更新文档与验收命令**

  README 安装说明改为“systemd user 服务 + session 环境导入 + X11 `Super+C` grab”。Operations 说明 daemon 必须成功抓取、冲突时查看 `journalctl --user -u fun-voice-daemon` 并停止冲突客户端后 `systemctl --user restart fun-voice-daemon`；不再提 DDE id 或控制中心注册。Acceptance checklist 将首项改为执行一次 `Super+C` 后 `fun-voice-selftest --format json` 的 `x11_hotkey` 为 pass，并加入先松 Super、冲突 exit 2 与无字母 C 漏入验证。

  原始设计的顶部加入：

  ```markdown
  > **已替代（2026-09-01）：** DDE/bridge 的实际 POC 不成立；快捷键设计见
  > [X11 全局热键替换设计](2026-09-01-x11-hotkey-replacement-design.md)。
  ```

- [ ] **Step 4: 执行完整自动质量门**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest -q
  .venv/bin/ruff check src tests
  .venv/bin/mypy src
  cmake --build build/fcitx
  ctest --test-dir build/fcitx --output-on-failure
  git diff --check
  ```

  Expected: 全部 PASS；pytest 不再收集 DDE/bridge 测试，ruff/mypy 无新增错误，原生 Fcitx CTest 通过。

- [ ] **Step 5: 在当前用户会话部署并做真实验证**

  按顺序执行（这会注销已验证归属的旧 DDE shortcut）：

  ```bash
  uv sync --inexact
  scripts/install-user.sh
  systemctl --user status fun-voice-daemon.service --no-pager
  fun-voice-selftest --format json
  ```

  随后完成 `tests/manual/test_x11_press_release.md`：第一次真实长按/松开后 `x11_hotkey` 为 pass；说一段普通话混合英文/代码；验证先松 Super；切窗不误注入。保存的证据只能是 service 状态、布尔 diagnostics 和 POC gate，不保存音频或转写正文。

- [ ] **Step 6: 提交文档和测试收尾**

  ```bash
  git add README.md docs/superpowers/specs/2026-08-31-fun-asr-nano-intel-xpu-voice-assistant-design.md \
    tests/test_install_scripts.py
  git add -p docs/operations.md docs/acceptance-checklist.md
  git commit -m "docs: document X11 hotkey operation"
  ```

  在提交前必须检查暂存 diff；operations、acceptance 与测试文件可能已有未提交修改，只暂存本任务 hunk。

## Plan Self-Review

- **Spec coverage:** Task 1 covers exclusive, atomic X11 grab and press/release semantics; Task 2 covers daemon lifecycle and no polling; Task 3 covers privacy-preserving diagnostics; Task 4 removes DDE/bridge and safely retires the currently registered DDE shortcut; Task 5 covers all user docs, tests and live X11 acceptance.
- **Placeholder scan:** 未发现占位内容；每个实现步骤均列出接口、文件和命令。
- **Type consistency:** `X11HotkeyListener` is the concrete `HotkeyLifecycle`; it calls `VoiceDaemon.handle_hotkey_press/release`; `serve` owns listener start/close; selftest reads only the boolean diagnostics those handlers maintain.
