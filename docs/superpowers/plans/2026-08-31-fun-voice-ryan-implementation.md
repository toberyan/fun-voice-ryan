# Fun Voice Ryan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 在 Deepin DDE X11 上交付一个本地按住说话助手：按住 Super+C 录音，松开后以 Fun-ASR-Nano 识别普通话混合英文、代码和计算机术语；推理优先使用 Intel Arc XPU；成功后向当前 Fcitx 输入上下文提交未经改写的原始模型文本，并保留剪贴板副本。

**Architecture:** DDE Keybinding1 将 Super+C 交给轻量 bridge；bridge 只把 start_if_idle 发给常驻 VoiceDaemon。VoiceDaemon 在按下、松开间以 pw-record 采集音频，持有开始时的 X11 焦点标识和 Fcitx focus token。它通过私有 Unix socket 调用常驻 Worker，Worker 先用 CPU FSMN-VAD 切段、再在 XPU 上运行 Fun-ASR-Nano。daemon 只在焦点仍匹配时请求 Fcitx addon 原子提交；剪贴板镜像总是独立尝试。Fcitx 不可用时才允许 X11 Ctrl+V 回退，且同样要求焦点未变化。

**Tech Stack:** Python 3.12、uv、FunASR 官方主线固定提交 8cd758c0ced576516b05a749194e6a94cdd38f99、ModelScope、PyTorch XPU、vLLM XPU、PipeWire pw-record、Xlib/XTEST、Fcitx5 C++ addon、DDE D-Bus、systemd --user、pytest。

**Precondition / hard gate:** Intel XPU POC 是桌面功能的阻断门。必须同时证明 torch.xpu 可用、vLLM 能在 XPU 加载 Qwen decoder、Nano encoder/adaptor 和 prompt embeddings 在 xpu:0、10 秒与 60 秒中英混合样本可返回文本，且没有静默 CPU decoder 回退。任何一步失败都停止桌面集成，不自动切换模型或后端；把证据记录到自检报告，等待用户选择下一条路线。

**Global constraints:**
- 模型输出必须原样保留：不得用 LLM、词典、正则、拼写或标点处理改写文本。
- 不持久化录音、转写或文本日志。短音频在内存；超过 10 分钟才在 XDG_RUNTIME_DIR 的私有文件切分，终态、进程重启、启动清理时删除。
- 录音上限 30 分钟，25 分钟通知，异常或识别失败不注入部分结果。
- Fcitx addon socket、worker socket、临时文件均为本用户私有，目录及文件权限为 0700/0600。
- 不读取 /dev/input；DDE bridge 结合 X11 C 键状态实现按住语义。
- 当前范围仅 DDE X11；不支持 Wayland、实时字幕、说话人分离、命令执行、历史记录或密码输入框识别。
- 任何向日志或通知输出的文本只能是长度、状态、错误类别和请求标识，不能包含音频内容或转写正文。

## Target file map

- pyproject.toml: Python 项目元数据、开发依赖与脚本入口。
- requirements-xpu.lock: 从已验证环境导出的精确 XPU 依赖快照，随源码提交。
- .gitignore: 忽略 .venv、models、运行态音频、用户配置；不忽略 requirements-xpu.lock。
- src/fun_voice/config.py: 有类型的配置和运行时路径解析。
- src/fun_voice/contracts.py: bridge、daemon、worker 和 Fcitx 的协议、状态、异常与不可变数据类型。
- src/fun_voice/preflight.py: XPU 环境核验、模型加载 POC、结构化证据输出。
- src/fun_voice/desktop.py: DDE Keybinding1、X11 键状态/焦点、剪贴板和 XTEST 注入适配器。
- src/fun_voice/capture.py: PipeWire 录制、停止语义、内存和分段临时存储。
- src/fun_voice/worker.py: VAD、Nano 推理、worker Unix socket 服务和客户端。
- src/fun_voice/daemon.py: 状态机、焦点保护、输出路由、通知。
- src/fun_voice/bridge.py: DDE action 入口与 daemon 请求客户端。
- src/fun_voice/selftest.py: 端到端前置检查及可机器读取报告。
- native/fcitx5-fun-voice/: Fcitx5 addon、CMake 与安装清单。
- scripts/: 环境、模型、DDE、安装、卸载和 POC 帮助脚本。
- systemd/: user service unit 和 DDE session autostart desktop entry。
- tests/: 单元、协议、适配器替身、集成与手动验收说明。
- docs/operations.md: 安装、运行、隐私、故障排查和卸载手册。

## Task 1: 建立项目契约、配置和测试骨架

**Files:**
- Create: pyproject.toml
- Create: .gitignore
- Create: src/fun_voice/__init__.py
- Create: src/fun_voice/config.py
- Create: src/fun_voice/contracts.py
- Create: tests/conftest.py
- Create: tests/test_config.py
- Create: tests/test_contracts.py
- Create: README.md

- [ ] 先写失败测试：配置默认运行目录应为 XDG_RUNTIME_DIR/fun-voice-ryan；若 XDG_RUNTIME_DIR 不存在或不属当前用户，配置应拒绝启动；临时文件目录和两个 socket 的权限策略应可测试。
- [ ] 在 contracts.py 定义明确类型：DaemonState 为 IDLE、RECORDING、TRANSCRIBING、COMMITTING、ERROR；StartRequest、StopRequest、FocusSnapshot、CaptureArtifact、Segment、Transcription、CommitResult、WorkerHealth 和结构化 ErrorCode。
- [ ] 定义稳定协议：bridge/daemon/worker 消息为单行 UTF-8 JSON、最大 64 KiB；Fcitx 使用下列 header 加 UTF-8 正文的独立帧协议、最大 64 KiB：
  ~~~text
  bridge -> daemon: {"op":"start_if_idle"} 或 {"op":"stop"}
  daemon -> worker: {"id":"uuid","op":"transcribe","audio":"path-or-memfd","sample_rate":16000}
  daemon -> fcitx: COMMIT <focus-token> <sequence> <total>\n<utf8-text>
  fcitx -> daemon: OK | REJECT stale-focus | REJECT no-input-context | ERROR <code>
  ~~~
  Fcitx 的文本行在协议层单独限制为 64 KiB；超过时必须由 daemon 以 Unicode 边界切成最多 8 KiB 的有序块。
- [ ] 添加 Python 脚本入口 fun-voice-daemon、fun-voice-worker、fun-voice-bridge、fun-voice-preflight、fun-voice-selftest。
- [ ] 写 README 的开发前提、隐私边界和未通过 XPU POC 不可安装桌面服务的醒目说明。
- [ ] 运行测试，确认先失败：
  ~~~bash
  uv run pytest tests/test_config.py tests/test_contracts.py -q
  ~~~
- [ ] 实现最小代码并再次运行：
  ~~~bash
  uv run pytest tests/test_config.py tests/test_contracts.py -q
  uv run ruff check src tests
  uv run mypy src
  ~~~
- [ ] 提交：git add pyproject.toml .gitignore README.md src tests && git commit -m "feat: add voice assistant contracts and config"

## Task 2: 锁定 XPU 环境并实现 Fun-ASR-Nano 阻断 POC

**Files:**
- Create: scripts/create-xpu-env.sh
- Create: scripts/run-nano-xpu-poc.sh
- Create: src/fun_voice/preflight.py
- Create: tests/test_preflight.py
- Create: docs/xpu-poc.md
- Create: requirements-xpu.lock

- [ ] 先写纯单元测试，使用假 torch、假 vLLM 与假 Nano loader，断言 preflight 只有在下列所有 checks 为 pass 时返回 ready：xpu_visible、vllm_xpu_decoder、nano_encoder_xpu、nano_adaptor_xpu、prompt_embeddings_xpu、decode_10s、decode_60s、no_cpu_decoder_fallback、oom_survives。
- [ ] create-xpu-env.sh 应创建项目 venv，并使用 vLLM 官方 XPU nightly 索引安装 vllm 与 torch XPU；FunASR 必须固定到提交 8cd758c0ced576516b05a749194e6a94cdd38f99。脚本打印安装版本和设备信息，不打印任何用户音频路径。
- [ ] 安装完成后执行 uv pip freeze --exclude-editable | LC_ALL=C sort，审核内容后以 requirements-xpu.lock 提交；禁止将这个锁文件加入 .gitignore。
- [ ] preflight 实现必须显式检查：
  ~~~python
  assert torch.xpu.is_available()
  device = "xpu:0"
  assert next(nano_encoder.parameters()).device.type == "xpu"
  assert next(nano_adaptor.parameters()).device.type == "xpu"
  ~~~
  并记录 vLLM engine 的设备配置和一次显存分配前后统计。若 decoder device 或引擎日志表明 CPU 回退，结果必须 fail。
- [ ] 用 FunASR 官方 Nano inference_vllm.py 的当前调用约定实现加载：audio encoder、audio adaptor 与 prompt embedding 均显式移动到 xpu:0；vLLM 使用 tensor_parallel_size=1、dtype=bf16、gpu_memory_utilization=0.35、max_model_len=4096、enforce_eager=True 作为初始值。
- [ ] run-nano-xpu-poc.sh 接收两个调用方提供的无敏感样本路径：--short 和 --long。它只在 XDG_RUNTIME_DIR 中生成报告和临时转换文件，trap 删除临时目录。短样本约 10 秒、长样本约 60 秒，均需含普通话、英文和计算机术语。
- [ ] POC 额外对一个超出预留显存的受控请求执行 OOM 测试；捕获异常后再运行一次短样本，证明 worker 进程仍能服务。不得为通过测试改为 CPU。
- [ ] 文档写明成功证据格式、失败分类、Intel 驱动/Level Zero 核验命令，以及失败后仅允许人工决定是否研究 llama.cpp/Vulkan 路线。
- [ ] 运行：
  ~~~bash
  uv run pytest tests/test_preflight.py -q
  scripts/create-xpu-env.sh
  scripts/run-nano-xpu-poc.sh --short /absolute/short.wav --long /absolute/long.wav
  ~~~
  预期：脚本退出 0 且报告所有 hard gate 为 pass；任一 fail 时后续任务不得部署服务。
- [ ] 提交：git add scripts src tests docs requirements-xpu.lock pyproject.toml && git commit -m "feat: add Nano XPU preflight gate"

## Task 3: 实现 DDE、X11 焦点与剪贴板桌面适配器

**Files:**
- Create: src/fun_voice/desktop.py
- Create: scripts/register-dde-shortcut.sh
- Create: scripts/unregister-dde-shortcut.sh
- Create: tests/test_desktop.py
- Create: tests/manual/test_dde_press_release.md

- [ ] 先写替身驱动测试：DDE LookupConflictShortcut 返回空时注册 Super+C；返回冲突时不修改任何 DDE 配置并清晰报出拥有者；注册得到的 custom shortcut id 必须保存到本地 config。
- [ ] 将 DDE 调用集中在 Keybinding1 客户端，使用 LookupConflictShortcut、AddCustomShortcut、DeleteCustomShortcut。action 只能执行 bridge 命令，不能直接运行模型。
- [ ] 注册前再次确认 Super+C 当前未冲突；不得假设 Ctrl+Super+Space 可用，因为系统已被 UOS AI Talk 占用。
- [ ] bridge 收到 DDE action 后读取 X11 键盘状态：C 为 down 则向 daemon 发送 start_if_idle；C 为 up 则发送 stop。连续重复 action 必须幂等。若 DDE 不能在按住期间触发，记录 POC fail，不得暗改为切换模式；按流程请求人工决定是否研究纯 X11 grab。
- [ ] FocusSnapshot 包含 X11 active window、窗口进程、输入焦点窗口和单调时间；提交前做完全相等比较。窗口切换、焦点丢失或 X server 异常都只保留剪贴板，不注入。
- [ ] 实现 UTF-8 剪贴板写入，成功转写后始终尝试写入；剪贴板失败不会撤销已成功 Fcitx 提交。实现 Ctrl+V XTEST 仅作 Fcitx 失败后的可选回退，并在注入前再次检查 snapshot。
- [ ] 运行：
  ~~~bash
  uv run pytest tests/test_desktop.py -q
  dbus-send --session --dest=org.deepin.dde.Keybinding1 --print-reply /org/deepin/dde/Keybinding1 org.deepin.dde.Keybinding1.LookupConflictShortcut string:"<Super>C"
  ~~~
- [ ] 手动验收记录：按住、自动重复、松开、切窗、Fcitx 切换中英文、目标为浏览器和终端；确认不读取 /dev/input。
- [ ] 提交：git add src tests scripts docs && git commit -m "feat: add DDE and X11 desktop adapters"

## Task 4: 实现 PipeWire 采集与受限临时分段存储

**Files:**
- Create: src/fun_voice/capture.py
- Create: tests/test_capture.py
- Create: tests/test_capture_integration.py
- Create: scripts/check-audio.sh

- [ ] 先写使用 fake subprocess 和 fake clock 的测试：开始调用 pw-record 为 16 kHz、单声道、s16le；stop 发送 SIGINT；退出码 0、130 或负 SIGINT 在存在最小有效音频时均按正常停止处理。
- [ ] 将前 10 分钟的 PCM 放在受限内存对象或匿名临时 fd。超过 10 分钟时才建立运行时临时目录，目录 0700、每个分片 0600、仅存有界大小的 16 kHz 单声道 PCM。
- [ ] 将 30 分钟限制编码为不可配置的安全上界；25 分钟只发 DDE 通知，30 分钟停止采集并进识别。无字节、少于最小时长、格式异常和子进程错误转为 CaptureError，且不得触发注入。
- [ ] 支持 source 名称显式配置，默认 PipeWire default source；rnnoise effect source 仅在用户配置时使用。
- [ ] 处理进程退出、SIGTERM、worker 失败、启动遗留目录：均调用同一 cleanup，删除本次精确创建的 runtime 目录。不得对 XDG_RUNTIME_DIR 做广泛删除。
- [ ] integration 测试仅在 pw-record 可用且 CI_AUDIO=1 时运行，录制 1 秒并验证采样格式；其他环境跳过。
- [ ] 运行：
  ~~~bash
  uv run pytest tests/test_capture.py tests/test_capture_integration.py -q
  scripts/check-audio.sh
  ~~~
- [ ] 提交：git add src tests scripts && git commit -m "feat: add PipeWire capture lifecycle"

## Task 5: 实现带焦点令牌的 Fcitx5 addon 与客户端

**Files:**
- Create: native/fcitx5-fun-voice/CMakeLists.txt
- Create: native/fcitx5-fun-voice/src/addon.cpp
- Create: native/fcitx5-fun-voice/src/addon.h
- Create: native/fcitx5-fun-voice/fcitx5-fun-voice.conf
- Create: native/fcitx5-fun-voice/README.md
- Create: src/fun_voice/fcitx.py
- Create: tests/test_fcitx_client.py
- Create: native/fcitx5-fun-voice/tests/protocol_test.cpp

- [ ] 先写 C++ 协议测试与 Python 客户端测试：PING 返回 PONG；未知命令拒绝；过期 focus token、无 InputContext、错序分块和超过 64 KiB 的消息都不得提交文本。
- [ ] addon 在启动时创建 $XDG_RUNTIME_DIR/fun-voice-ryan-fcitx.sock，删除既有同名 socket 前必须 lstat 校验其 owner 是当前用户且类型为 socket；bind 后 chmod 0600。
- [ ] daemon 的 START_FOCUS 请求从 addon 取得不可预测令牌，addon 将其关联到当前 InputContext UUID。COMMIT 的 token 必须对应当前仍活跃的同一 InputContext；不匹配返回 REJECT stale-focus。
- [ ] Python 客户端把转写分割为最多 8 KiB 的 Unicode 边界分块，携带顺序号与总数。收到任一 reject 时停止后续分块并返回未提交；不得重试到另一个输入框。
- [ ] addon 以 Fcitx InputContext.commitString 提交，严禁模拟键盘输入。只保留活跃令牌的短期内存映射，在 focus change、context destroy 和 daemon disconnect 时清理。
- [ ] 编译并运行：
  ~~~bash
  cmake -S native/fcitx5-fun-voice -B build/fcitx
  cmake --build build/fcitx
  ctest --test-dir build/fcitx --output-on-failure
  uv run pytest tests/test_fcitx_client.py -q
  ~~~
- [ ] 手动验收 Fcitx 能接收 PING，连续中英文本原样提交，没有写入插件日志。
- [ ] 提交：git add native src tests && git commit -m "feat: add focus-safe Fcitx commit addon"

## Task 6: 实现常驻 Nano worker、VAD 与私有 socket

**Files:**
- Create: src/fun_voice/worker.py
- Create: src/fun_voice/nano_runtime.py
- Create: tests/test_worker.py
- Create: tests/test_worker_protocol.py
- Create: systemd/fun-voice-worker.service

- [ ] 先写测试，以 fake VAD 和 fake Nano runtime 验证 VAD 空结果返回 EmptySpeech、多个 segment 保留严格时间顺序、最终文本按模型返回顺序直接拼接，绝不插入或删除字符。
- [ ] worker 只绑定 $XDG_RUNTIME_DIR/fun-voice-ryan/worker.sock，目录 0700、socket 0600，且仅接受同 uid 客户端。请求里不得传送裸文本，只传内存句柄或受控 runtime 文件标识。
- [ ] CPU FSMN-VAD 的实测 FunASR 返回形状在 Task 2 POC 中断言并写入适配层；适配层产出规范 Segment(start_ms, end_ms)。在 VAD 片段边界加入固定小重叠，拼接顺序严格按原音频时间。
- [ ] NanoRuntime 启动时一次性加载模型并保温到 logout；每个 segment 都复用运行时。调用期间强制 audio encoder、adaptor 和 prompt embeddings 在 xpu:0，若检测到非 XPU 设备即拒绝请求。
- [ ] 对单个请求设置可配置的推理超时和显存异常捕获。OOM、vLLM 异常、模型无输出、格式错误返回明确 error；worker 清理请求资源并保持监听，允许下一次短请求。
- [ ] worker 健康端点返回版本、模型状态、device 和最后一次错误类别；不得含录音路径和文字内容。
- [ ] 运行：
  ~~~bash
  uv run pytest tests/test_worker.py tests/test_worker_protocol.py -q
  systemctl --user daemon-reload
  systemctl --user start fun-voice-worker.service
  uv run fun-voice-preflight --require-live-worker
  ~~~
- [ ] 提交：git add src tests systemd && git commit -m "feat: add warm Nano XPU worker"

## Task 7: 实现 VoiceDaemon 状态机、输出路由与错误策略

**Files:**
- Create: src/fun_voice/daemon.py
- Create: src/fun_voice/bridge.py
- Create: tests/test_daemon.py
- Create: tests/test_end_to_end_fakes.py
- Create: systemd/fun-voice-daemon.service

- [ ] 先写状态机参数化测试：IDLE 只接受 start；RECORDING 重复 start 无副作用；stop 只对 RECORDING 生效；错误、取消和完成始终回到 IDLE；任一路径最终清理 capture artifact。
- [ ] 在 start_if_idle 原子记录 FocusSnapshot 和从 Fcitx 请求 focus token，然后开始采集。若未获得 token 仍可录音，但后续只允许剪贴板和严格受焦点保护的 XTEST 回退。
- [ ] stop 后按 capture -> worker -> output 线性运行。同一时间只能有一个 session；新 start 在 TRANSCRIBING 或 COMMITTING 时只返回 busy，不可排队。
- [ ] worker 返回文本后立刻独立写剪贴板。随后重新比较 X11 FocusSnapshot；若变更，发送 DDE 通知 stale focus 且禁止 Fcitx/XTEST 注入。
- [ ] 焦点未变时优先 Fcitx commit。Fcitx 无响应、socket 失效或返回错误时，只有 X11 snapshot 仍一致且 XTEST 可用才尝试 Ctrl+V。Fcitx 显式 stale-focus 拒绝不能回退 XTEST。
- [ ] 除成功外，任何 CaptureError、EmptySpeech、WorkerError、XPU error、focus change 或 injection error 都显示 DDE 通知；不显示和不记录转写正文，不注入部分文本。
- [ ] 端到端 fake 测试最少覆盖：正常 Fcitx 提交；窗口切换只写剪贴板；Fcitx 不可用的 XTEST 回退；Fcitx stale reject 不回退；剪贴板写失败不撤销 Fcitx 成功；worker OOM 后下一次可成功；长文本分块中途 reject。
- [ ] 运行：
  ~~~bash
  uv run pytest tests/test_daemon.py tests/test_end_to_end_fakes.py -q
  uv run ruff check src tests
  uv run mypy src
  ~~~
- [ ] 提交：git add src tests systemd && git commit -m "feat: add guarded voice daemon"

## Task 8: 打包、安装、自检与真实系统验收

**Files:**
- Create: scripts/install-user.sh
- Create: scripts/uninstall-user.sh
- Create: scripts/start-session-bridge.sh
- Create: src/fun_voice/selftest.py
- Create: systemd/fun-voice-session.desktop
- Create: docs/operations.md
- Create: docs/acceptance-checklist.md
- Modify: README.md

- [ ] 先写 selftest 单元测试：每项结果有 name、status、detail；失败项使进程非零；输出 JSON 不带敏感数据。测试 DDE service、Super+C 冲突、bridge 按住时序 POC、PipeWire、Fcitx PING、clipboard、XTEST 回退资格、worker health 和 XPU hard gate。
- [ ] install-user.sh 在运行 Task 2 POC 报告为全部 pass 后才允许继续。它安装 Python package、Fcitx addon、systemd user units 和 desktop autostart，daemon-reload、enable --now worker/daemon，并调用 DDE 注册脚本。每个写入目标必须明确、可逆。
- [ ] uninstall-user.sh 先 disable --now 服务、注销记录的 custom shortcut id、移除本项目明确安装的 addon/unit/desktop 文件和 runtime socket；保留模型缓存及用户配置，除非用户显式要求删除。
- [ ] session bridge desktop entry 只在 DDE session 登录后拉起 bridge，bridge 对 daemon 不可用时退出非零但不重启风暴；服务使用 Restart=on-failure、合理 StartLimit 和 XDG_RUNTIME_DIR 依赖。
- [ ] operations.md 包含安装前硬件核验、首次模型下载、配置 source、隐私说明、日志脱敏、服务诊断、DDE/Fcitx 故障处理、如何保持 Super+C 可用、卸载和 Wayland 非支持声明。
- [ ] 按 docs/acceptance-checklist.md 在真实 DDE X11 会话逐条验证：
  1. Super+C 无冲突并能按住录音、松开识别；
  2. 普通话夹英文、代码、计算机术语可原样输入；
  3. 成功文本同时留在剪贴板；
  4. 切窗、Fcitx focus 变化、异常和空音频均不误输入；
  5. 超过 10 分钟切分、25 分钟提醒、30 分钟停止；
  6. XPU 报告没有 CPU decoder fallback；
  7. 重启服务和注销登录后没有遗留录音或文本。
- [ ] 运行完整质量门：
  ~~~bash
  uv run pytest -q
  uv run ruff check src tests
  uv run mypy src
  git diff --check
  scripts/install-user.sh
  uv run fun-voice-selftest --format json
  ~~~
- [ ] 提交：git add scripts src systemd docs README.md tests && git commit -m "feat: package local voice input assistant"

## Final review and release gate

- [ ] 逐项对照设计文档，确认没有引入持久化音频/文本、LLM 后处理、/dev/input、自动模型后端切换或 Wayland 承诺。
- [ ] 审阅 requirements-xpu.lock、POC JSON 和 worker health：Nano 关键模块与 decoder 均有 XPU 证据。
- [ ] 运行 git diff --check、完整 pytest、ruff 与 mypy；在干净工作树确认结果。
- [ ] 执行 git status --short；预期为空。若不为空，只审阅本计划引入的文件，勿覆盖用户既有改动。
- [ ] 建立最终 tag 或发布包前，人工完成真实桌面验收清单并保存不含文本/音频的通过摘要。
