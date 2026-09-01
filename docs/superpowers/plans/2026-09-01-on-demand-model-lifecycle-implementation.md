# On-Demand Model Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove boot-time model residency while retaining Nano accuracy, an XPU-only SenseVoiceSmall fallback, and on-demand Qwen3.5 correction.

**Architecture:** Keep the X11 daemon login-resident but make model workers non-enabled template units. Each worker binds first and lazily builds exactly one XPU runtime on its first transcription; a monitor exits the process after 120 seconds of no active work. The daemon starts Nano after capture and only switches to a separately started SenseVoiceSmall worker on Nano model-load or OOM failure.

**Tech Stack:** Python 3.12, systemd user template units, Unix sockets, FunASR, vLLM Intel XPU, pytest, ruff, mypy.

## Global Constraints

- All neural inference uses `xpu:0`; no CPU or CUDA fallback.
- Nano defaults are BF16, `gpu_memory_utilization=0.15`, `max_model_len=1536`.
- SenseVoiceSmall is only a `worker.model_load`/`worker.oom` fallback.
- Qwen is exactly `Qwen/Qwen3.5-0.8B`, text-only and on-demand.
- Models, raw text and audio never appear in logs or boot-time processes.
- The current worktree is shared and dirty; stage only files changed by each task.

---

## File structure

| File | Responsibility |
| --- | --- |
| `src/fun_voice/config.py` | Low-KV and lifecycle config validation |
| `src/fun_voice/worker.py` | Lazy runtime construction, profile selection, idle exit |
| `src/fun_voice/nano_runtime.py` | Nano local loading and SenseVoice runtime adapter |
| `src/fun_voice/daemon.py` | Condition-based template service launch and restricted fallback |
| `systemd/fun-voice-worker@.service` | Non-enabled per-profile model worker |
| `scripts/install-user.sh` | Install template, disable legacy warm worker, enable daemon only |
| `scripts/config.example.toml` | Document safe runtime defaults |
| `tests/test_config.py` | Config contract regressions |
| `tests/test_worker_protocol.py` | Lazy model / idle-lifecycle protocol regressions |
| `tests/test_end_to_end_fakes.py` | Daemon startup and fallback regressions |
| `tests/test_install_scripts.py` | No warm-worker-enable deployment contract |

### Task 1: Lock low-KV lifecycle configuration

**Files:** Modify `src/fun_voice/config.py`, `scripts/config.example.toml`, `tests/test_config.py`.

- [ ] Write tests asserting defaults `0.15`, `1536`, `120`, and rejection of out-of-range KV and idle values.
- [ ] Run `PYTHONPATH=src .venv/bin/pytest tests/test_config.py -q`; verify the new assertions fail.
- [ ] Add typed config fields, TOML parsing and explicit validation.
- [ ] Run the focused test command again; verify it passes.

### Task 2: Add lazy profile-owned worker runtime and idle exit

**Files:** Modify `src/fun_voice/worker.py`, `src/fun_voice/nano_runtime.py`, `tests/test_worker_protocol.py`.

- [ ] Write tests proving server bind/health do not call a model factory, first transcription creates it once, and a 120-second idle monitor shuts down only with no active request.
- [ ] Run the focused worker tests; verify each new test fails for the missing lazy lifecycle API.
- [ ] Implement `LazyTranscriber`, profile validation (`nano`/`sensevoice`), model-load error mapping, and an external idle monitor that invokes server shutdown after the last completed request.
- [ ] Implement a local-snapshot SenseVoiceSmall XPU runtime adapter that exposes the existing `Transcriber` protocol and refuses non-XPU modules.
- [ ] Run the worker and runtime tests; verify they pass.

### Task 3: Use profile template services and restricted fallback

**Files:** Modify `src/fun_voice/daemon.py`, create `systemd/fun-voice-worker@.service`, modify `scripts/install-user.sh`, `tests/test_end_to_end_fakes.py`, `tests/test_install_scripts.py`.

- [ ] Write tests that require socket readiness polling after `systemctl --user start fun-voice-worker@nano.service`, prohibit start-at-boot, and invoke SenseVoice only for model-load/OOM errors after stopping Nano.
- [ ] Run focused daemon/install tests; verify failure.
- [ ] Implement template service start/stop and condition polling; construct a Nano client plus fallback client in daemon startup.
- [ ] Install the template and explicitly disable/stop legacy `fun-voice-worker.service`; enable/restart daemon only.
- [ ] Run focused daemon/install tests; verify pass.

### Task 4: Make Qwen correction process on demand

**Files:** Modify `src/fun_voice/enhanced_poc.py`, future correction-service files from `2026-09-01-structured-identity-correction-implementation.md`, related tests.

- [ ] Extend the existing corrector service task so request dispatch starts Qwen only for a correction and process termination releases it after the response.
- [ ] Preserve `Qwen/Qwen3.5-0.8B`, text-only overrides and validation fallback to raw text.
- [ ] Add fake-process tests for no daemon-start preload, one request/one model lifetime, and raw fallback on any Qwen failure.
- [ ] Run those tests plus the enhanced POC after completing the relevant prior-plan tasks.

### Task 5: Verify deployment and real XPU behavior

**Files:** Modify `docs/operations.md`, `README.md`, `docs/xpu-poc.md` as necessary.

- [ ] Run `scripts/run-nano-xpu-poc.sh` with `--gpu-memory-utilization 0.15 --max-model-len 1536` support and record only device/memory evidence.
- [ ] Run a Nano real recording, wait 120 seconds, and prove `systemctl --user is-active fun-voice-worker@nano.service` is inactive.
- [ ] Run a controlled fallback test and Qwen correction test; confirm no CPU backend and no model service is enabled.
- [ ] Run `PYTHONPATH=src .venv/bin/pytest -q`, `ruff check src tests`, `mypy src`, `git diff --check`.

## Plan review

Task 1 bounds the source of the old KV allocation. Task 2 ensures model allocation occurs only after a request and can be released. Task 3 isolates fallback model allocation from Nano and prevents boot activation. Task 4 applies the same lifecycle invariant to the selected Qwen model. Task 5 covers the actual XPU and privacy-sensitive acceptance evidence.
