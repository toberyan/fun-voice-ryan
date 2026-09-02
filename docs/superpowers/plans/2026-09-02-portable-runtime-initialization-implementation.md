# Portable Runtime Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a first-run initializer select and verify CUDA GPU, then Intel XPU, then CPU; run every service through the selected isolated runtime; use Nano plus enhancements on accelerators and SenseVoiceSmall without enhancements on CPU.

**Architecture:** A stdlib-only `RuntimeSelection` manifest is the sole authority for deployment facts: interpreter, device, dtype, ASR profiles, installed model revisions, and enhancement capability. The bootstrap process creates only the candidate virtual environments it needs, proves tensor execution and a local end-to-end ASR run in each candidate, then atomically publishes a valid selection. Existing configuration remains user preference only; daemon, worker, corrector, scheduler, self-test, installer, and launcher consume the immutable runtime policy derived from that selection.

**Tech Stack:** Python 3.11+ stdlib, pytest, `uv`, hash-locked PyTorch/FunASR runtimes, ModelScope snapshots, native FunASR/PyTorch, Bash, user-level systemd, Deepin DDE X11, CMake/CTest for existing native components.

## Global Constraints

- Backend priority in `auto` mode is exactly **CUDA GPU → Intel XPU → CPU**; only CUDA is considered a GPU backend in this release. ROCm, MPS, DirectML, and unknown accelerators must continue to the next candidate.
- `--backend cuda|xpu|cpu` is an explicit one-backend diagnostic: failure exits non-zero and must not silently select another backend. Only `--backend auto` falls through candidates.
- CUDA/XPU use `FunAudioLLM/Fun-ASR-Nano-2512` as primary ASR and `iic/SenseVoiceSmall` as fallback. CPU uses `iic/SenseVoiceSmall` as its only ASR profile.
- CPU must set `device="cpu"`, `dtype="float32"`, `fallback_asr_profile=null`, `enhanced_enabled=false`, and `speaker_enabled=false`; it must not download, load, invoke, or expose Qwen 3.5 0.8B, CAM++, diarization, or identity capabilities.
- CUDA/XPU download Nano, SenseVoiceSmall, FSMN-VAD, `Qwen/Qwen3.5-0.8B`, and `iic/speech_campplus_sv_zh-cn_16k-common`; CPU downloads only SenseVoiceSmall and FSMN-VAD.
- Store shared snapshots and per-backend virtual environments only below `${XDG_DATA_HOME:-$HOME/.local/share}/fun-voice-ryan`; do not overwrite the repository `.venv`.
- Store `${data_root}/runtime/selection.json` in a `0700` parent directory with file mode `0600`, publish with temp-file plus `os.replace()`, and reject an unsafe, malformed, incompatible, or out-of-root selection at every use site.
- A successful `--force-reselect` replaces the selection only after complete environment, model, tensor, dtype, and ASR checks pass. A failed or interrupted run leaves the previous usable selection and daemon unchanged.
- Probing uses public temporary audio only, deletes it before exit, and records fixed backend/model/error categories and durations only—never audio paths, audio bytes, recognition text, focus data, or user data.
- Login must remain model-free: daemon is lightweight, ASR starts only after speech capture, Qwen remains one-request/on-demand, and model workers never become enabled systemd login services.
- Keep the existing DDE **X11** `Super+C` hold/release, PipeWire capture, DTK overlay, Fcitx-first commit, clipboard fallback, in-memory capture, and privacy contracts unchanged.
- Do not add CPU Qwen, CPU speaker/identity, cloud inference, runtime hot device switching, Wayland support, or a settings UI.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/fun_voice/runtime_selection.py` | Stdlib-only manifest schema, backend policy, safe path/permission validation, atomic persistence, and selection loader. |
| `tests/test_runtime_selection.py` | Manifest schema, CPU/accelerator invariants, safe permissions/path checks, and atomic replacement tests. |
| `src/fun_voice/config.py` | Keep TOML as preference; derive immutable effective device/dtype/profile/enhancement settings from `RuntimeSelection`. |
| `src/fun_voice/nano_runtime.py` | Load Nano/SenseVoice/VAD against the selected device and expected device type rather than an XPU constant. |
| `src/fun_voice/worker.py` | Admit only the selected ASR profile and use the effective runtime policy for model construction. |
| `src/fun_voice/scheduler.py`, `src/fun_voice/xpu_lease.py` | Use device-neutral model scheduling/lease terminology while retaining serial ASR-before-Qwen release semantics for accelerator selections. |
| `src/fun_voice/daemon.py`, `src/fun_voice/corrector.py`, `src/fun_voice/selftest.py` | Wire selection into process startup; suppress corrector/spawn and Nano fallback on CPU; report selected-runtime health instead of an XPU-only POC gate. |
| `tests/test_config.py`, `tests/test_worker.py`, `tests/test_worker_protocol.py`, `tests/test_scheduler.py`, `tests/test_xpu_lease.py`, `tests/test_daemon.py`, `tests/test_corrector.py`, `tests/test_selftest.py` | Prove policy injection, CPU denial, accelerator behavior, profile restrictions, and text-free diagnostics with fakes. |
| `src/fun_voice/backend_probe.py` | Runs inside a candidate runtime: downloads the policy model set, proves Torch/device/dtype tensor execution, and performs one offline local ASR smoke inference on a temporary public WAV. |
| `src/fun_voice/bootstrap.py` | Stdlib-only candidate ordering, subprocess orchestration, temporary-candidate cleanup, force-reselection safety, native/install handoff, and fixed-category diagnostic output. |
| `src/fun_voice/runtime_launcher.py`, `scripts/run-selected-runtime.sh` | Validate `RuntimeSelection` in a lightweight host Python, map public command names to fixed modules, then `execve` the selected interpreter. |
| `scripts/initialize-first-run.sh`, `scripts/create-runtime-env.sh` | Public first-run CLI and reusable isolated-environment builder. |
| `requirements-cuda.in/.lock`, `requirements-xpu.in/.lock`, `requirements-cpu.in/.lock`, `scripts/compile-runtime-locks.sh` | Backend-specific, hash-locked PyTorch distributions with one shared, pinned FunASR dependency graph. |
| `scripts/create-xpu-env.sh` | Retained explicit XPU developer/POC entry point, refactored to reuse the generic environment builder without becoming deployment authority. |
| `scripts/install-user.sh`, `scripts/uninstall-user.sh`, `systemd/fun-voice-*.service` | Install source-root-aware launcher shims and native desktop artifacts only after a validated selection; retain on-demand worker services. |
| `tests/test_bootstrap.py`, `tests/test_backend_probe.py`, `tests/test_runtime_launcher.py`, `tests/test_install_scripts.py` | Unit/static contracts for candidate fallback, model lists, no CPU enhancement download, launcher execution, and install hard gates. |
| `scripts/config.example.toml`, `README.md`, `docs/operations.md`, `docs/acceptance-checklist.md`, `docs/xpu-poc.md` | Publish selection behavior, migration rules, CPU limitations, real-device checks, and the new status of the XPU POC. |

### Task 1: Define the safe runtime-selection contract

**Files:**

- Create: `src/fun_voice/runtime_selection.py`
- Create: `tests/test_runtime_selection.py`
- Modify: `src/fun_voice/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**

- Produces `Backend = Literal["cuda", "xpu", "cpu"]`, `AsrProfile = Literal["nano", "sensevoice"]`, `RuntimeSelectionError`, immutable `RuntimeSelection`, and `RuntimePolicy`.
- Produces `data_root(env: Mapping[str, str] | None = None) -> Path`, `selection_path(root: Path | None = None) -> Path`, `load_runtime_selection(root: Path | None = None) -> RuntimeSelection`, and `write_runtime_selection(selection: RuntimeSelection, root: Path | None = None) -> Path`.
- Produces `effective_runtime_config(user: Config, selection: RuntimeSelection) -> EffectiveRuntimeConfig`; daemon/worker/corrector must use this instead of trusting TOML device/dtype fields.
- Consumes only stdlib modules in `runtime_selection.py`; it must remain importable by host `python3` before any backend runtime exists.

- [ ] **Step 1: Write manifest and policy tests before defining the module**

  Create `tests/test_runtime_selection.py`. Use a `tmp_path / "data"` root and an executable `tmp_path / "data/runtimes/cpu/bin/python"` fixture so every selected interpreter stays inside the owned runtime root. The central test data and assertions must be exactly equivalent to the following:

  ```python
  from fun_voice.runtime_selection import (
      RuntimeSelection,
      RuntimeSelectionError,
      load_runtime_selection,
      write_runtime_selection,
  )

  def _selection(root: Path, backend: str = "cpu") -> RuntimeSelection:
      python = root / "runtimes" / backend / "bin" / "python"
      python.parent.mkdir(parents=True)
      python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
      python.chmod(0o700)
      if backend == "cpu":
          return RuntimeSelection(
              schema_version=1, backend="cpu", python=python, device="cpu",
              dtype="float32", primary_asr_profile="sensevoice",
              fallback_asr_profile=None, enhanced_enabled=False,
              speaker_enabled=False, model_revisions={"sensevoice": "master", "vad": "master"},
              probe_status="pass", selected_at=1,
          )
      return RuntimeSelection(
          schema_version=1, backend=backend, python=python, device=f"{backend}:0",
          dtype="bf16", primary_asr_profile="nano",
          fallback_asr_profile="sensevoice", enhanced_enabled=True,
          speaker_enabled=True,
          model_revisions={"nano": "master", "sensevoice": "master", "vad": "master",
                           "qwen": "master", "campplus": "master"},
          probe_status="pass", selected_at=1,
      )

  def test_cpu_manifest_round_trip_forbids_accelerator_models(tmp_path: Path) -> None:
      root = tmp_path / "data"
      expected = _selection(root)
      path = write_runtime_selection(expected, root)
      assert path.stat().st_mode & 0o777 == 0o600
      assert path.parent.stat().st_mode & 0o777 == 0o700
      assert load_runtime_selection(root) == expected

  def test_cpu_rejects_qwen_and_speaker_enablement(tmp_path: Path) -> None:
      root = tmp_path / "data"
      invalid = dataclasses.replace(_selection(root), enhanced_enabled=True)
      with pytest.raises(RuntimeSelectionError, match="CPU runtime"):
          write_runtime_selection(invalid, root)

  @pytest.mark.parametrize("mutate", [
      lambda selection, root: dataclasses.replace(selection, python=Path("/usr/bin/python3")),
      lambda selection, root: dataclasses.replace(selection, dtype="bf16"),
      lambda selection, root: dataclasses.replace(selection, fallback_asr_profile="nano"),
  ])
  def test_selection_rejects_unsafe_interpreter_or_cpu_policy(
      tmp_path: Path, mutate: Callable[[RuntimeSelection, Path], RuntimeSelection]
  ) -> None:
      root = tmp_path / "data"
      with pytest.raises(RuntimeSelectionError):
          write_runtime_selection(mutate(_selection(root), root), root)
  ```

  Add a separate test that writes one valid CPU selection, then calls `write_runtime_selection` with an invalid CUDA selection. Assert the original JSON bytes and `load_runtime_selection(root)` are unchanged. Add parameterized malformed JSON, schema version `2`, mode `0644`, parent mode `0755`, non-owner file, missing interpreter, and a symlink escaping `root / "runtimes"` cases; each must make `load_runtime_selection` raise `RuntimeSelectionError` without echoing its content.

  In `tests/test_config.py`, replace XPU-only device-rejection expectations with effective-policy tests:

  ```python
  def test_effective_runtime_config_cpu_overrides_toml_devices(tmp_path: Path) -> None:
      path = tmp_path / "config.toml"
      path.write_text(
          "[inference]\ndevice = 'xpu:0'\n[enhanced]\nenabled = true\n"
          "[correction]\ndevice = 'xpu:0'\ndtype = 'bf16'\n",
          encoding="utf-8",
      )
      effective = effective_runtime_config(load_config(path), _cpu_selection(tmp_path))
      assert effective.inference.device == "cpu"
      assert effective.inference.dtype == "float32"
      assert effective.primary_asr_profile == "sensevoice"
      assert effective.fallback_asr_profile is None
      assert effective.enhanced.enabled is False
      assert effective.speaker_enabled is False
  ```

- [ ] **Step 2: Run the focused tests and confirm the missing contract fails**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_runtime_selection.py tests/test_config.py -q
  ```

  Expected: collection fails with `ModuleNotFoundError: No module named 'fun_voice.runtime_selection'` and `ImportError` for `effective_runtime_config`.

- [ ] **Step 3: Implement the stdlib-only JSON schema, path validation, and atomic publication**

  In `src/fun_voice/runtime_selection.py`, use the following public shape. `model_revisions` must be copied to a read-only mapping in `__post_init__`; `to_dict()` must serialize `Path` as a string and the writer must use `json.dumps(selection.to_dict(), ensure_ascii=False, sort_keys=True)`.

  ```python
  Backend = Literal["cuda", "xpu", "cpu"]
  AsrProfile = Literal["nano", "sensevoice"]
  SELECTION_SCHEMA_VERSION = 1
  DIRECTORY_MODE = 0o700
  FILE_MODE = 0o600

  @dataclass(frozen=True, slots=True)
  class RuntimePolicy:
      backend: Backend
      device: str
      dtype: str
      primary_asr_profile: AsrProfile
      fallback_asr_profile: AsrProfile | None
      enhanced_enabled: bool
      speaker_enabled: bool

      @property
      def allowed_profiles(self) -> tuple[AsrProfile] | tuple[AsrProfile, AsrProfile]:
          return ((self.primary_asr_profile,) if self.fallback_asr_profile is None
                  else (self.primary_asr_profile, self.fallback_asr_profile))

  @dataclass(frozen=True, slots=True)
  class RuntimeSelection:
      schema_version: int
      backend: Backend
      python: Path
      device: str
      dtype: str
      primary_asr_profile: AsrProfile
      fallback_asr_profile: AsrProfile | None
      enhanced_enabled: bool
      speaker_enabled: bool
      model_revisions: Mapping[str, str]
      probe_status: Literal["pass"]
      selected_at: int

      def policy(self) -> RuntimePolicy:
          return RuntimePolicy(self.backend, self.device, self.dtype,
                               self.primary_asr_profile, self.fallback_asr_profile,
                               self.enhanced_enabled, self.speaker_enabled)

      def to_dict(self) -> dict[str, object]:
          return {
              "schema_version": self.schema_version, "backend": self.backend,
              "python": str(self.python), "device": self.device, "dtype": self.dtype,
              "primary_asr_profile": self.primary_asr_profile,
              "fallback_asr_profile": self.fallback_asr_profile,
              "enhanced_enabled": self.enhanced_enabled,
              "speaker_enabled": self.speaker_enabled,
              "model_revisions": dict(self.model_revisions),
              "probe": {"status": self.probe_status, "selected_at": self.selected_at},
          }
      @classmethod
      def from_dict(cls, raw: Mapping[str, object]) -> "RuntimeSelection":
          probe = raw.get("probe")
          if not isinstance(probe, Mapping):
              raise RuntimeSelectionError("invalid selection schema")
          return cls(
              schema_version=cast(int, raw.get("schema_version")),
              backend=cast(Backend, raw.get("backend")),
              python=Path(cast(str, raw.get("python"))),
              device=cast(str, raw.get("device")), dtype=cast(str, raw.get("dtype")),
              primary_asr_profile=cast(AsrProfile, raw.get("primary_asr_profile")),
              fallback_asr_profile=cast(AsrProfile | None, raw.get("fallback_asr_profile")),
              enhanced_enabled=cast(bool, raw.get("enhanced_enabled")),
              speaker_enabled=cast(bool, raw.get("speaker_enabled")),
              model_revisions=cast(Mapping[str, str], raw.get("model_revisions")),
              probe_status=cast(Literal["pass"], probe.get("status")),
              selected_at=cast(int, probe.get("selected_at")),
          )
  ```

  Implement `data_root()` with `XDG_DATA_HOME` when set, otherwise `Path.home() / ".local/share"`, and append `fun-voice-ryan`. `selection_path(root)` is `root / "runtime" / "selection.json"`.

  Before reading or writing, create/check `root / "runtime"` as the effective uid's owned `0700` directory; reject any group/other permission bit, wrong uid, symlink, or non-directory. Validate `selection.python.resolve()` is executable and is strictly below `root.resolve() / "runtimes"`; use `Path.is_relative_to()` and never accept the root itself. Write with `tempfile.NamedTemporaryFile(dir=parent, prefix=".selection.", delete=False)`, `json.dump(selection.to_dict(), ensure_ascii=False, sort_keys=True)`, `flush`, `os.fsync`, `os.chmod(temp, 0o600)`, and `os.replace(temp, path)`. On every exception after temporary file creation, unlink that exact temp path only.

  Enforce these complete policy invariants in one private `validate_selection(selection, root)` called by both write/load:

  ```python
  ACCELERATOR_MODELS = frozenset({"nano", "sensevoice", "vad", "qwen", "campplus"})
  CPU_MODELS = frozenset({"sensevoice", "vad"})

  # cpu: device cpu, float32, SenseVoice-only, enhancements/speaker false,
  # exact CPU_MODELS keys.
  # cuda: device cuda:0, dtype bf16 or fp16, Nano primary/SenseVoice fallback,
  # enhancement/speaker true, exact ACCELERATOR_MODELS keys.
  # xpu: device xpu:0, dtype bf16, Nano primary/SenseVoice fallback,
  # enhancement/speaker true, exact ACCELERATOR_MODELS keys.
  # all revisions are non-empty ASCII-safe strings, probe_status is pass,
  # selected_at is a positive non-bool integer, schema_version is 1.
  ```

  A CUDA selection is allowed to use `fp16` only after the candidate probe writes it. An XPU selection cannot self-downgrade to fp16; failed BF16 falls through to CPU.

- [ ] **Step 4: Bind user preferences to the immutable deployment policy**

  In `src/fun_voice/config.py`, retain the current TOML parsing and bounded values such as source, overlay, recording policy, Qwen token limits, and user preference `enhanced.enabled`. Add this immutable aggregate without importing `torch`, FunASR, or ModelScope:

  ```python
  @dataclass(frozen=True)
  class EffectiveRuntimeConfig:
      selection: RuntimeSelection
      inference: InferenceConfig
      active_session: ActiveSessionConfig
      enhanced: EnhancedInferenceConfig
      primary_asr_profile: AsrProfile
      fallback_asr_profile: AsrProfile | None
      speaker_enabled: bool

  def effective_runtime_config(
      user: Config, selection: RuntimeSelection
  ) -> EffectiveRuntimeConfig:
      policy = selection.policy()
      inference = replace(
          user.inference, device=policy.device, dtype=policy.dtype,
          allow_sensevoice_fallback=policy.fallback_asr_profile == "sensevoice",
      )
      active = replace(user.active_session, device=policy.device)
      enhanced = replace(
          user.enhanced,
          enabled=user.enhanced.enabled and policy.enhanced_enabled,
          correction_device=policy.device,
          correction_dtype=policy.dtype,
          identity_enabled=user.enhanced.identity_enabled and policy.speaker_enabled,
          identity_device=policy.device,
      )
      return EffectiveRuntimeConfig(
          selection, inference, active, enhanced, policy.primary_asr_profile,
          policy.fallback_asr_profile, enhanced.identity_enabled,
      )
  ```

  Replace `validate_inference_config`, `validate_active_session_config`, and `validate_enhanced_inference_config` XPU literals with validation against the passed `RuntimePolicy`. Preserve all current limits. `load_config()` must parse legacy `inference.device`, `inference.dtype`, `active_session.device`, `correction.device`, `correction.dtype`, and `speaker_identity.device` only for backward-compatible syntax, but these values must be discarded before construction of `EffectiveRuntimeConfig`; they must never select hardware. Remove those keys from the example file in Task 5 and document this migration.

- [ ] **Step 5: Run focused quality checks and commit the manifest boundary**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_runtime_selection.py tests/test_config.py -q
  .venv/bin/ruff check src/fun_voice/runtime_selection.py src/fun_voice/config.py tests/test_runtime_selection.py tests/test_config.py
  .venv/bin/mypy src/fun_voice/runtime_selection.py src/fun_voice/config.py
  ```

  Expected: all tests pass; lint and type checking emit no diagnostics.

  Commit:

  ```bash
  git add src/fun_voice/runtime_selection.py src/fun_voice/config.py tests/test_runtime_selection.py tests/test_config.py
  git commit -m "feat: add validated runtime selection"
  ```

### Task 2: Make model loading and workers backend-neutral

**Files:**

- Modify: `src/fun_voice/nano_runtime.py`
- Modify: `src/fun_voice/worker.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_worker_protocol.py`
- Modify: `tests/test_config.py`

**Interfaces:**

- Consumes `RuntimeSelection` / `EffectiveRuntimeConfig` from Task 1.
- Produces `device_type(device: str) -> Literal["cuda", "xpu", "cpu"]`, selected-device `NanoRuntime`/`SenseVoiceRuntime` loaders, and worker `main()` profile admission based on `selection.policy().allowed_profiles`.
- Preserves the existing same-UID socket protocol, in-memory audio descriptor protocol, VAD ranges, verbatim segment join, model-on-demand lifecycle, error codes, and no-text logging.

- [ ] **Step 1: Write failing selected-device loader and CPU profile tests**

  In `tests/test_worker.py`, replace assertions which rely on module-level `DEVICE`/`EXPECTED_DEVICE_TYPE` with a fake selection fixture. Add:

  ```python
  def test_engine_device_check_uses_selected_cuda_type() -> None:
      engine = FakeEngine(texts=["ok"])
      engine.audio_encoder = FakeModule("cuda")
      engine.audio_adaptor = FakeModule("cuda")
      engine.embed_tokens = FakeModule("cuda")
      check_engine_devices(engine, expected="cuda")

  def test_engine_device_check_rejects_cpu_when_cuda_selected() -> None:
      engine = FakeEngine(texts=["ok"])
      engine.audio_encoder = FakeModule("cpu")
      engine.audio_adaptor = FakeModule("cuda")
      engine.embed_tokens = FakeModule("cuda")
      with pytest.raises(DeviceMismatchError, match="expected 'cuda'"):
          check_engine_devices(engine, expected="cuda")
  ```

  Add CPU SenseVoice tests which assert the loader receives `device="cpu"`, `expected_device_type="cpu"`, and `dtype="float32"`; add accelerator tests which assert Nano gets `cuda:0/bf16` and `xpu:0/bf16` from their selection, not TOML. Keep the existing no-fallback-on-normal-result behavior.

  In `tests/test_worker_protocol.py`, add a `main()` seam test with monkeypatched `load_runtime_selection`, `config.load_config`, and `serve`:

  ```python
  def test_worker_cpu_rejects_nano_before_creating_a_socket(monkeypatch) -> None:
      monkeypatch.setattr(worker_module, "load_runtime_selection", lambda: _cpu_selection())
      assert worker_module.main(["--profile", "nano"]) == 2
      assert _served_workers == []

  def test_worker_cpu_starts_only_sensevoice(monkeypatch) -> None:
      monkeypatch.setattr(worker_module, "load_runtime_selection", lambda: _cpu_selection())
      assert worker_module.main(["--profile", "sensevoice"]) == 0
      assert _served_workers[0].health()["device"] == "cpu"
  ```

- [ ] **Step 2: Run selected worker tests and confirm the old XPU guards fail**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_worker.py tests/test_worker_protocol.py tests/test_config.py -q
  ```

  Expected: the new tests fail because `nano_runtime.py` defines `DEVICE = "xpu:0"` and worker startup has no runtime-selection load or profile policy check.

- [ ] **Step 3: Parameterize Nano, SenseVoice, and VAD loaders by selection**

  In `src/fun_voice/nano_runtime.py`, delete `DEVICE` and `EXPECTED_DEVICE_TYPE`. Keep the `NanoRuntimeError` hierarchy and stable wire codes. Add a strict device helper and pass its result to every device assertion:

  ```python
  SelectedDeviceType = Literal["cuda", "xpu", "cpu"]

  def device_type(device: str) -> SelectedDeviceType:
      value = device.split(":", 1)[0]
      if value not in {"cuda", "xpu", "cpu"}:
          raise DeviceMismatchError("unsupported selected device")
      return cast(SelectedDeviceType, value)

  def check_engine_devices(engine: Any, *, expected: SelectedDeviceType) -> None:
      for name in ("audio_encoder", "audio_adaptor", "embed_tokens"):
          actual = _module_device_type(getattr(engine, name, None))
          if actual != expected:
              raise DeviceMismatchError(f"{name} is on {actual!r}, expected {expected!r}")
  ```

  Change `NanoRuntime.__init__`, `SenseVoiceRuntime.__init__`, `_load_vad`, `load_native_nano_engine`, `load_nano_runtime`, `load_sensevoice_runtime`, `health()`, and the private module assertions to accept `selection: RuntimeSelection` (or local `device`, `dtype`, `expected_device_type` extracted once from it). Every check must compare with `device_type(selection.device)`, never hard-code XPU. `load_nano_runtime` must reject a selection whose allowed profiles omit `nano`; `load_sensevoice_runtime` must reject a selection whose allowed profiles omit `sensevoice`. CPU therefore never instantiates Nano even if a caller bypasses daemon policy.

  Keep model directories under the shared cache exactly as today. Do not create a CPU-special ModelScope cache, CPU fallback inside a Nano call, or CPU Qwen import.

- [ ] **Step 4: Enforce profile policy at worker process startup**

  In `src/fun_voice/worker.py`, update CLI wording to `On-demand selected-runtime ASR worker`. Load `RuntimeSelection` before `resolve_runtime_dir()` and reject a requested profile that is not in `selection.policy().allowed_profiles` with exit code `2`, a fixed log category `unsupported_profile`, and no socket creation. Build `effective = config.effective_runtime_config(config.load_config(), selection)` once.

  Replace the old free device override authority with compatibility-only flags: `--device` and `--dtype` may be parsed to produce a deterministic `2`/`runtime_policy_override` error when they differ from `effective.inference`; they must never alter it. Leave `--timeout-ms` and lifecycle bounds as bounded operational inputs. Instantiate loaders exactly as follows:

  ```python
  if args.profile == "nano":
      return load_nano_runtime(
          selection=selection,
          inference=effective.inference,
          default_timeout=args.timeout_ms / 1000.0,
      )
  return load_sensevoice_runtime(
      selection=selection,
      inference=effective.inference,
      default_timeout=args.timeout_ms / 1000.0,
  )
  ```

  Keep the lazy `LazyTranscriber` construction, but populate the health device from `effective.inference.device`. This keeps CPU SenseVoice idle at startup and makes systemd attempts to launch `worker@nano` on a CPU machine harmless.

- [ ] **Step 5: Run worker checks and commit device-neutral ASR**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_worker.py tests/test_worker_protocol.py tests/test_config.py -q
  .venv/bin/ruff check src/fun_voice/nano_runtime.py src/fun_voice/worker.py tests/test_worker.py tests/test_worker_protocol.py
  .venv/bin/mypy src/fun_voice/nano_runtime.py src/fun_voice/worker.py
  ```

  Expected: all tests pass with fake CPU/CUDA/XPU selections and no third-party model import during collection.

  Commit:

  ```bash
  git add src/fun_voice/nano_runtime.py src/fun_voice/worker.py tests/test_worker.py tests/test_worker_protocol.py tests/test_config.py
  git commit -m "feat: select ASR runtime by verified backend"
  ```

### Task 3: Apply CPU capability denial across daemon, correction, scheduling, and self-test

**Files:**

- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/corrector.py`
- Modify: `src/fun_voice/scheduler.py`
- Modify: `src/fun_voice/xpu_lease.py`
- Modify: `src/fun_voice/selftest.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_corrector.py`
- Modify: `tests/test_scheduler.py`
- Modify: `tests/test_xpu_lease.py`
- Modify: `tests/test_selftest.py`

**Interfaces:**

- Consumes `RuntimeSelection` and `EffectiveRuntimeConfig` from Task 1 and selected-profile worker admission from Task 2.
- Produces `ModelScheduler` and `ModelLeaseCoordinator` names; their serial ordering remains ASR release confirmed before accelerator Qwen work.
- Produces `check_runtime_selection(loader: Callable[[], RuntimeSelection] = load_runtime_selection) -> SelfTestResult` and `run_selftest(selection_loader: Callable[[], RuntimeSelection] = load_runtime_selection) -> SelfTestReport`.
- Preserves the daemon's raw-text commit fallback: a disabled, failed, rejected, or timed-out correction commits raw ASR text and does not leak it to logs/notifications.

- [ ] **Step 1: Write failing CPU policy tests at every spawn boundary**

  Add the following tests, retaining current XPU behavioral tests by replacing their scheduler class import with `ModelScheduler`:

  ```python
  def test_daemon_cpu_registers_only_sensevoice_and_never_constructs_qwen(monkeypatch) -> None:
      selection = _cpu_selection()
      created_profiles: list[str] = []
      monkeypatch.setattr(daemon_module, "SocketWorkerClient", _recording_worker_factory(created_profiles))
      monkeypatch.setattr(daemon_module, "OnDemandQwenCorrector", _must_not_construct)
      daemon = daemon_module.build_voice_daemon(_fakes(), selection=selection)
      assert created_profiles == ["sensevoice"]
      assert daemon._fallback_worker is None
      assert daemon._corrector is None

  def test_cpu_asr_commits_raw_text_without_correction_spawn() -> None:
      daemon = _daemon_with_cpu_policy(worker=FakeWorker(text="get commit"))
      _run_one_capture(daemon)
      assert _fcitx.commits == [("tok-123", "get commit")]
      assert _corrector_calls == []

  def test_corrector_refuses_cpu_selection_before_subprocess(monkeypatch) -> None:
      corrector = OnDemandQwenCorrector(inference=_cpu_effective().enhanced,
                                        selection=_cpu_selection())
      with pytest.raises(CorrectionError, match="disabled_by_runtime_policy"):
          corrector.correct("get commit")
      assert _runner_calls == []
  ```

  In `tests/test_selftest.py`, assert CPU self-test requests health only for `("sensevoice",)` and reports `runtime_selection` pass with `{"backend": "cpu", "primary_profile": "sensevoice", "enhanced": False}`. Assert a missing/unsafe manifest makes `runtime_selection` fail, not `xpu_hard_gate`.

- [ ] **Step 2: Run daemon/corrector/self-test tests and verify old assumptions fail**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_daemon.py tests/test_corrector.py tests/test_scheduler.py tests/test_xpu_lease.py tests/test_selftest.py -q
  ```

  Expected: the CPU daemon test fails because `daemon.main()` unconditionally creates Nano/fallback workers and `OnDemandQwenCorrector`; self-test still exposes `xpu_hard_gate`.

- [ ] **Step 3: Rename the accelerator ownership abstractions without weakening their ordering**

  In `src/fun_voice/scheduler.py`, rename `XpuScheduler` to `ModelScheduler`, update its thread name to `fun-voice-model-scheduler`, and keep its public `activate`, `submit`, `run_asr`, `run_correction`, `profile_state`, `wait_idle`, and `close` contracts. In `src/fun_voice/xpu_lease.py`, rename `XpuLeaseCoordinator` to `ModelLeaseCoordinator` and its docstrings to “selected accelerator”. Update all imports and annotations in daemon/tests in the same commit; do not leave an alias whose normal production code still calls an XPU-named type.

  `run_correction` must continue to stop every profile it has observed, call health after stopping, deny execution on uncertainty, and execute Qwen only after confirmed `INACTIVE`/`FAILED` states. CPU code must not call `run_correction`; this task does not create a CPU model lease.

- [ ] **Step 4: Make daemon and corrector selection-driven**

  In `src/fun_voice/daemon.py`, load the manifest before constructing any model client. Extract the former final `main()` object wiring into a testable factory:

  ```python
  def build_voice_daemon(
      dependencies: DaemonDependencies,
      *,
      selection: RuntimeSelection,
      user_config: config.Config | None = None,
  ) -> VoiceDaemon:
      user = config.load_config() if user_config is None else user_config
      effective = config.effective_runtime_config(user, selection)
      primary = SocketWorkerClient(
          profile=effective.primary_asr_profile,
          socket_path=dependencies.paths.worker_socket,
          start_service=dependencies.start_worker_service,
          stop_service=dependencies.stop_worker_service,
      )
      fallback = (
          SocketWorkerClient(
              profile=effective.fallback_asr_profile,
              socket_path=dependencies.paths.worker_socket,
              start_service=dependencies.start_worker_service,
              stop_service=dependencies.stop_worker_service,
          )
          if effective.fallback_asr_profile is not None else None
      )
      corrector = (
          OnDemandQwenCorrector(inference=effective.enhanced, selection=selection)
          if effective.enhanced.enabled else None
      )
      scheduler = ModelScheduler(
          start_profile=dependencies.start_worker_service,
          stop_profile=dependencies.stop_worker_service,
          health_profile=dependencies.health_worker_profile,
      )
      return VoiceDaemon(
          guard=dependencies.guard, recorder=dependencies.recorder,
          fcitx_factory=dependencies.fcitx_factory, clipboard=dependencies.clipboard,
          injector=dependencies.injector, notifier=dependencies.notifier,
          overlay=dependencies.overlay, worker=primary, fallback_worker=fallback,
          corrector=corrector, scheduler=scheduler,
          nano_preloader=primary.preload,
          capture_config=dependencies.capture_config,
      )
  ```

  Make `nano_preloader` generic to the primary profile; CPU must preload SenseVoice only after capture begins. Keep accelerator Nano primary plus SenseVoice fallback. `default_start_worker_service`, `default_stop_worker_service`, and `worker_service_name` must validate profile against `selection.policy().allowed_profiles`, so a CPU run cannot manually start Nano via daemon code.

  In `src/fun_voice/corrector.py`, replace `DEVICE = "xpu:0"` with `selection.device` and `selection.dtype`, receive `selection: RuntimeSelection` in `generate_enveloped_correction` and `OnDemandQwenCorrector`, and call `selection.policy()` before loading Transformers or constructing a subprocess request. If `enhanced_enabled` is false, raise `CorrectionError("disabled_by_runtime_policy")`; `VoiceDaemon` never reaches this branch in normal CPU operation. Preserve low dynamic KV behavior, max token/source bounds, protected-token validation, and child-process release.

- [ ] **Step 5: Replace the self-test XPU gate with the selected-runtime check**

  In `src/fun_voice/selftest.py`, remove `PreflightReport`, `CHECK_NAMES`, `REPORT_RELATIVE_PATH`, `load_preflight_report`, and `check_xpu_hard_gate` from the normal self-test path. Keep `fun-voice-preflight` and `docs/xpu-poc.md` as an explicit XPU diagnostic, but no installed service may require its report.

  Add:

  ```python
  def check_runtime_selection(
      loader: Callable[[], RuntimeSelection] = load_runtime_selection,
  ) -> SelfTestResult:
      try:
          selection = loader()
      except RuntimeSelectionError as exc:
          return SelfTestResult("runtime_selection", STATUS_FAIL,
                                {"reason": "invalid_or_missing"})
      return SelfTestResult(
          "runtime_selection", STATUS_PASS,
          {"backend": selection.backend,
           "primary_profile": selection.primary_asr_profile,
           "enhanced": selection.enhanced_enabled},
      )
  ```

  Make `check_worker_health` accept `profiles: tuple[AsrProfile] | tuple[AsrProfile, AsrProfile]` from `selection.policy().allowed_profiles`, then set `CHECK_NAMES_SELFTEST` to `("x11_hotkey", "pipewire", "fcitx_ping", "clipboard", "xtest_eligibility", "worker_health", "runtime_selection")`. The CLI has no `--report` option afterward. Details remain enum/boolean-only.

- [ ] **Step 6: Run component tests and commit CPU capability enforcement**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_daemon.py tests/test_corrector.py tests/test_scheduler.py tests/test_xpu_lease.py tests/test_selftest.py -q
  .venv/bin/ruff check src/fun_voice/daemon.py src/fun_voice/corrector.py src/fun_voice/scheduler.py src/fun_voice/xpu_lease.py src/fun_voice/selftest.py tests/test_daemon.py tests/test_corrector.py tests/test_scheduler.py tests/test_xpu_lease.py tests/test_selftest.py
  .venv/bin/mypy src/fun_voice/daemon.py src/fun_voice/corrector.py src/fun_voice/scheduler.py src/fun_voice/xpu_lease.py src/fun_voice/selftest.py
  ```

  Expected: all CPU tests prove no Nano/Qwen construction and accelerator tests still prove release-before-Qwen.

  Commit:

  ```bash
  git add src/fun_voice/daemon.py src/fun_voice/corrector.py src/fun_voice/scheduler.py src/fun_voice/xpu_lease.py src/fun_voice/selftest.py tests/test_daemon.py tests/test_corrector.py tests/test_scheduler.py tests/test_xpu_lease.py tests/test_selftest.py
  git commit -m "feat: enforce runtime capability policy"
  ```

### Task 4: Build, probe, and atomically select isolated backend runtimes

**Files:**

- Create: `src/fun_voice/backend_probe.py`
- Create: `src/fun_voice/bootstrap.py`
- Create: `scripts/initialize-first-run.sh`
- Create: `scripts/create-runtime-env.sh`
- Create: `scripts/compile-runtime-locks.sh`
- Create: `requirements-cuda.in`
- Create: `requirements-xpu.in`
- Create: `requirements-cpu.in`
- Create: `requirements-cuda.lock`
- Create: `requirements-cpu.lock`
- Modify: `requirements-xpu.lock`
- Modify: `scripts/create-xpu-env.sh`
- Create: `tests/test_backend_probe.py`
- Create: `tests/test_bootstrap.py`
- Modify: `tests/test_install_scripts.py`

**Interfaces:**

- `scripts/initialize-first-run.sh [--backend auto|cuda|xpu|cpu] [--force-reselect] [--dry-run]` executes `python3 -m fun_voice.bootstrap` with the repository `src` path.
- `fun_voice.bootstrap.candidate_backends(requested: str) -> tuple[Backend] | tuple[Backend, Backend] | tuple[Backend, Backend, Backend]`, `run_initialization(options: InitializationOptions, runner: CommandRunner) -> RuntimeSelection`, and `main(argv) -> int` are unit-testable without Torch.
- `fun_voice.backend_probe.run_probe(request: ProbeRequest) -> ProbeResult` runs inside the candidate interpreter and emits one compact JSON result to stdout; stdout contains no paths or text.
- `scripts/create-runtime-env.sh --backend BACKEND --runtime-dir PATH --models-root PATH` creates/syncs only the requested backend runtime and installs the verified FunASR source.

- [ ] **Step 1: Write candidate-order, preservation, and model-list tests**

  Create `tests/test_bootstrap.py` with a fake `CommandRunner` that records argv and returns prebuilt JSON `ProbeResult` values. Cover exactly these paths:

  ```python
  def test_auto_tries_cuda_xpu_cpu_in_priority_order() -> None:
      runner = FakeRunner({"cuda": fail("tensor"), "xpu": fail("asr"), "cpu": passed_cpu()})
      selected = run_initialization(_options("auto"), runner=runner)
      assert runner.probed == ["cuda", "xpu", "cpu"]
      assert selected.backend == "cpu"

  @pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu"])
  def test_explicit_backend_does_not_fall_through(backend: str) -> None:
      runner = FakeRunner({backend: fail("unavailable")})
      with pytest.raises(InitializationError, match="selected backend failed"):
          run_initialization(_options(backend), runner=runner)
      assert runner.probed == [backend]

  def test_failed_force_reselect_keeps_existing_manifest(tmp_path: Path) -> None:
      previous = _write_selection(tmp_path, "xpu")
      with pytest.raises(InitializationError):
          run_initialization(_options("auto", force=True, root=tmp_path), runner=FakeRunner.all_fail())
      assert load_runtime_selection(tmp_path) == previous
  ```

  Assert accelerator probe commands contain only `nano`, `sensevoice`, `vad`, `qwen`, and `campplus`; CPU contains exactly `sensevoice` and `vad`; no CPU command argv, environment, or model request includes `Qwen`, `qwen`, `campplus`, or `CAM++`. Test `--dry-run` returns candidate order and performs no venv/model/probe subprocess.

  Add a `DesktopPrerequisites` fake which supplies X11/DDE session values, an owned `XDG_RUNTIME_DIR`, executable paths, and writable data root. Assert each missing prerequisite (`DISPLAY`, DDE X11 session, `XDG_RUNTIME_DIR`, `uv`, `pw-cli`/PipeWire socket, `fcitx5-remote`, `cmake`, `pkg-config`, writable data root) causes `run_initialization()` to raise `InitializationError("desktop_prerequisite")` before the fake runner records an environment or probe command. The `--dry-run` path is the sole exception: it returns candidate order without desktop checks or subprocesses.

  Create `tests/test_backend_probe.py` with fake `torch`, snapshot downloader, WAV downloader, and `load_nano_runtime`/`load_sensevoice_runtime` seams. Assert CUDA follows BF16 then FP16 when BF16 tensor operation fails, XPU only permits BF16, CPU only permits float32 and SenseVoice. Assert each passing probe calls both a device tensor reduction and exactly one local `transcribe()` against the temporary public sample; result JSON omits sample path and transcript.

- [ ] **Step 2: Run bootstrap/probe tests and confirm the new modules are absent**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_bootstrap.py tests/test_backend_probe.py -q
  ```

  Expected: collection fails because `fun_voice.bootstrap` and `fun_voice.backend_probe` do not exist.

- [ ] **Step 3: Implement deterministic backend environment inputs and hash locks**

  Create one `.in` file per backend and one compiler script. The compiler must use Python 3.12, the existing verified FunASR commit `8cd758c0ced576516b05a749194e6a94cdd38f99`, and archive SHA-256 `f8b2c9b9954c463b5c0e433bd1f2706b5c6c28f16f755f55ec66365960c06da0`. The `.in` files identify the PyTorch index and backend distribution; all non-Torch transitive dependencies must be resolved from the sanitized pinned FunASR source and the project runtime dependencies.

  `requirements-xpu.in` must be added as the source of the existing XPU lock rather than treating an opaque lock as manually edited input. Pin the three backend inputs exactly as follows, with `modelscope==1.39.1` and `transformers==5.16.1` in every one:

  ```text
  # requirements-cuda.in; --extra-index-url https://download.pytorch.org/whl/cu130
  torch==2.13.0+cu130
  torchaudio==2.11.0+cu130

  # requirements-xpu.in; --extra-index-url https://download.pytorch.org/whl/xpu
  torch==2.13.0+xpu
  torchaudio==2.11.0+xpu

  # requirements-cpu.in; --extra-index-url https://download.pytorch.org/whl/cpu
  torch==2.13.0+cpu
  torchaudio==2.11.0+cpu
  ```

  Use `https://pypi.tuna.tsinghua.edu.cn/simple` as the primary index, matching the current XPU install, and retain `--index-strategy unsafe-best-match` because local-version CUDA/XPU/CPU wheels must win over the primary index. The compiler must run one import/tensor smoke in a temporary Python 3.12 venv for each lock and commit the resolved versions plus hashes—not a range, marker, or unpinned package. The CUDA wheel/index pair exists for CPython 3.12 on the official PyTorch CUDA 13.0 index; keep the exact pin and index in source, rather than selecting versions dynamically at end-user first run.

  Implement `scripts/compile-runtime-locks.sh` to:

  1. download and SHA-verify the FunASR tarball into a temporary directory;
  2. unpack it, delete only its absolute symlinks as `create-xpu-env.sh` currently does;
  3. invoke `uv pip compile --generate-hashes --python-version 3.12 --no-emit-package funasr` three times with that sanitized local source plus `requirements-cuda.in`, `requirements-xpu.in`, and `requirements-cpu.in`, producing `requirements-cuda.lock`, `requirements-xpu.lock`, and `requirements-cpu.lock` respectively;
  4. reject a lock missing `--hash=sha256:`, a backend's required Torch package, `modelscope==1.39.1`, `transformers==5.16.1`, or containing `vllm`, `vllm-xpu-kernels`, `cuda-python`, or `flashinfer-python`;
  5. set all temporary directory permissions to owner-only and delete its exact `mktemp -d` directory in a `trap`.

  Do not hand-edit hash blocks. The committed generated locks are the exact deployment artifact; all installers use `uv pip sync --require-hashes` and never resolve at end-user initialization time.

- [ ] **Step 4: Implement the generic isolated environment builder**

  Create `scripts/create-runtime-env.sh` with strict Bash (`set -euo pipefail`, `umask 077`) and this command contract:

  ```bash
  scripts/create-runtime-env.sh --backend cuda --runtime-dir "$DATA/runtimes/cuda" --models-root "$DATA/models"
  ```

  Parse only the three named options and reject an unknown/missing value. Verify `runtime-dir` resolves under `${XDG_DATA_HOME:-$HOME/.local/share}/fun-voice-ryan/runtimes`, is not a symlink, and is not a repository `.venv`. Select `requirements-${backend}.lock` via a closed `case` statement. Use `uv venv "$runtime_dir" --python 3.12`, then `uv pip sync --python "$runtime_dir/bin/python" --require-hashes "$lock_file"`; do not use `uv sync`, editable installation, or a shared virtual environment.

  Reuse the verified tarball/download/unpack logic from `create-xpu-env.sh`, but stage it at `"${runtime_dir}/.funasr-src"`, delete only absolute symlinks, and run `uv pip install --python "$runtime_dir/bin/python" --no-deps "$stage"`. Finish by importing `torch`, `funasr`, `modelscope`, `transformers`, and `Xlib`; the import check reports only package/version/error class. It must not call ModelScope snapshot download or allocate a model.

  Change `scripts/create-xpu-env.sh` into an explicit developer/POC compatibility wrapper which calls this builder with `--backend xpu --runtime-dir "${FUN_VOICE_VENV_DIR:-${ROOT_DIR}/.venv}"` only after the builder receives a deliberate `--allow-project-venv` internal option. The generic initializer never passes that option. Retain its printed XPU diagnostic information and remove its role as a desktop installation hard gate.

- [ ] **Step 5: Implement in-runtime model download and actual tensor/ASR probing**

  In `src/fun_voice/backend_probe.py`, import Torch/FunASR/ModelScope lazily inside probe functions. Define fixed model constants and a closed model mapping:

  ```python
  MODEL_IDS = {
      "nano": "FunAudioLLM/Fun-ASR-Nano-2512",
      "sensevoice": "iic/SenseVoiceSmall",
      "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
      "qwen": "Qwen/Qwen3.5-0.8B",
      "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
  }
  PUBLIC_SAMPLE_URL = (
      "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/"
      "test_audio/asr_example_zh.wav"
  )
  ```

  `ProbeRequest` must include only backend, runtime root, models root, and revision. `ProbeResult` contains `backend`, `status` (`pass`/`fail`), `error_category` (one of `environment`, `import`, `availability`, `tensor`, `dtype`, `model_download`, `asr`, `internal`), optional successful `dtype`, model-key set, and non-negative `tensor_ms`/`asr_ms`; it contains no path, URL, name, text, stderr, or exception message.

  For CUDA, require `torch.cuda.is_available()` then try `torch.ones(32, device="cuda:0", dtype=torch.bfloat16).sum().item()`; if that fails, try the same reduction with `torch.float16`. Use the first successful dtype. For XPU, require `torch.xpu.is_available()` and successfully run the BF16 reduction on `xpu:0`; any failure returns an XPU candidate failure. CPU runs only float32 reduction on `cpu`. Before model load, invoke `snapshot_download(model_id, revision="master")` for each model id in the precise policy set, verify each expected snapshot metadata file, then set `MODELSCOPE_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

  Download `PUBLIC_SAMPLE_URL` into an owner-only `NamedTemporaryFile(suffix=".wav", delete=False)`. For CUDA/XPU call `load_nano_runtime(selection=probe_selection, inference=probe_inference, default_timeout=120.0)`; for CPU call `load_sensevoice_runtime(selection=probe_selection, inference=probe_inference, default_timeout=120.0)`; call `runtime.transcribe(temp.name, sample_rate=16000, timeout=120.0)` exactly once, discard the returned text/segments, close the runtime, and unlink the exact temp file in `finally`. `probe_inference` is `InferenceConfig(device=probe_selection.device, dtype=probe_selection.dtype)` and its remaining values are the existing bounded defaults. The success result writes model revisions from the closed policy mapping, not arbitrary downloader output.

- [ ] **Step 6: Implement bootstrap ordering, safe publish, and public CLI**

  In `src/fun_voice/bootstrap.py`, keep imports to stdlib plus `fun_voice.runtime_selection`. Define:

  ```python
  @dataclass(frozen=True, slots=True)
  class InitializationOptions:
      backend: Literal["auto", "cuda", "xpu", "cpu"] = "auto"
      force_reselect: bool = False
      dry_run: bool = False
      data_root: Path | None = None
      project_root: Path | None = None

  def candidate_backends(
      requested: str,
  ) -> tuple[Backend] | tuple[Backend, Backend] | tuple[Backend, Backend, Backend]:
      if requested == "auto":
          return ("cuda", "xpu", "cpu")
      if requested in {"cuda", "xpu", "cpu"}:
          return (cast(Backend, requested),)
      raise InitializationError("invalid backend")
  ```

  `run_initialization` first loads an existing manifest only to determine whether a successful new selection must replace it; it must not delete, stop services, or modify that manifest. On normal invocation with an existing valid selection and no `--force-reselect`, print a fixed `already_selected` diagnostic and return it without changing environments. With `--dry-run`, print JSON containing only ordered candidate names and exit `0` before any subprocess.

  Before examining an existing selection or creating a candidate, call `validate_desktop_prerequisites(options, environment=os.environ, which=shutil.which)`. It must reject a missing `DISPLAY`; `XDG_SESSION_TYPE` other than `x11` when that variable is present; an `XDG_CURRENT_DESKTOP` value that does not contain `DDE` case-insensitively when that variable is present; an unset, missing, non-owned, or group/world-accessible `XDG_RUNTIME_DIR`; a missing `uv`, `cmake`, or `pkg-config`; no `pw-cli` executable and no owned `${XDG_RUNTIME_DIR}/pipewire-0` socket; missing `fcitx5-remote`; or a data-root parent that cannot create and remove one exact owner-only temporary file. Return only fixed prerequisite category names, never the rejected path or variable value. The subsequent `scripts/install-user.sh` remains responsible for checking the project-built Fcitx and DTK artifacts before it writes desktop files.

  For each candidate call `scripts/create-runtime-env.sh` through an argv list, then call the new interpreter as `python -m fun_voice.backend_probe --backend <candidate> --models-root <root/models> --json`. Pass `PYTHONPATH=<project_root>/src`, `MODELSCOPE_CACHE=<root/models>`, and no user audio/text environment values. Parse only the closed `ProbeResult` JSON schema. In `auto`, record a fixed category and proceed on any failed candidate. In explicit mode, raise `InitializationError("selected backend failed")` after its one failure.

  On success, construct the exact `RuntimeSelection`, call `write_runtime_selection`, invoke `scripts/install-user.sh --runtime-selection "${data_root}/runtime/selection.json"` only after the write returns, and finally run `systemctl --user daemon-reload` plus `systemctl --user restart fun-voice-daemon.service`. If install/native build/restart fails, restore the prior selection bytes with the same mode only when they existed and were valid, then report `install`; do not start a model worker. A fresh initialization without an old selection leaves no manifest on failure. Do not remove a previously created `runtimes/cuda`, `runtimes/xpu`, or `runtimes/cpu` directory during an unsuccessful re-selection; each is reusable and cannot invalidate the old manifest.

  `scripts/initialize-first-run.sh` resolves its repository root, sets only `PYTHONPATH="${ROOT}/src"`, and executes `python3 -m fun_voice.bootstrap "$@"`. It must have no model IDs, device rules, `curl`, package installer, or direct file deletion logic.

- [ ] **Step 7: Run initializer tests, shell checks, and commit the bootstrap**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_runtime_selection.py tests/test_bootstrap.py tests/test_backend_probe.py tests/test_install_scripts.py -q
  .venv/bin/ruff check src/fun_voice/backend_probe.py src/fun_voice/bootstrap.py tests/test_bootstrap.py tests/test_backend_probe.py
  .venv/bin/mypy src/fun_voice/backend_probe.py src/fun_voice/bootstrap.py
  bash -n scripts/initialize-first-run.sh scripts/create-runtime-env.sh scripts/compile-runtime-locks.sh scripts/create-xpu-env.sh
  ```

  Expected: all fake tests pass, the generated locks contain hashes, and every shell script parses without executing installation/download work.

  Commit:

  ```bash
  git add src/fun_voice/backend_probe.py src/fun_voice/bootstrap.py scripts/initialize-first-run.sh scripts/create-runtime-env.sh scripts/compile-runtime-locks.sh scripts/create-xpu-env.sh requirements-cuda.in requirements-xpu.in requirements-cpu.in requirements-cuda.lock requirements-xpu.lock requirements-cpu.lock tests/test_backend_probe.py tests/test_bootstrap.py tests/test_install_scripts.py
  git commit -m "feat: initialize verified portable runtimes"
  ```

### Task 5: Install selection-aware launchers and document/verify real operation

**Files:**

- Create: `src/fun_voice/runtime_launcher.py`
- Create: `scripts/run-selected-runtime.sh`
- Create: `tests/test_runtime_launcher.py`
- Modify: `scripts/install-user.sh`
- Modify: `scripts/uninstall-user.sh`
- Modify: `tests/test_install_scripts.py`
- Modify: `scripts/config.example.toml`
- Modify: `README.md`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`
- Modify: `docs/xpu-poc.md`

**Interfaces:**

- Consumes the safe `load_runtime_selection` contract and source root supplied by the installer.
- Produces installed `$HOME/.local/bin/fun-voice-{daemon,worker,preflight,selftest,corrector,benchmark}` shims which dispatch only to fixed module names using `RuntimeSelection.python`.
- Preserves systemd unit `ExecStart` paths and on-demand worker template semantics, so no unit needs a model-specific interpreter path.

- [ ] **Step 1: Write failing launcher and installer contract tests**

  Create `tests/test_runtime_launcher.py` with monkeypatched `os.execvpe`:

  ```python
  def test_launcher_execs_the_manifest_python_with_fixed_daemon_module(monkeypatch) -> None:
      selection = _selection(tmp_path, "cpu")
      monkeypatch.setattr(runtime_launcher, "load_runtime_selection", lambda: selection)
      monkeypatch.setattr(os, "execvpe", _capture_exec)
      assert runtime_launcher.main(["fun-voice-daemon", "--log-level", "DEBUG"]) == 0
      assert _exec.path == str(selection.python)
      assert _exec.argv == [str(selection.python), "-m", "fun_voice.daemon", "--log-level", "DEBUG"]
      assert _exec.env["PYTHONPATH"].split(":")[0].endswith("/src")

  def test_launcher_rejects_unknown_binary_without_exec() -> None:
      assert runtime_launcher.main(["fun-voice-arbitrary"]) == 2
      assert _exec.calls == []

  def test_launcher_rejects_unsafe_selection_without_echoing_path(monkeypatch) -> None:
      monkeypatch.setattr(runtime_launcher, "load_runtime_selection", _raise_unsafe)
      assert runtime_launcher.main(["fun-voice-worker"]) == 2
      assert _exec.calls == []
  ```

  Update `tests/test_install_scripts.py` to assert the installer no longer refers to `POC_REPORT`, `poc-report.json`, `Nano POC backend`, `uv sync --inexact`, or `${ROOT}/.venv/bin/fun-voice-`. Instead assert it requires `--runtime-selection`, runs `fun_voice.runtime_selection` load/import validation through the manifest Python, and writes each shim through one `install_launcher` function. Keep all existing assertions for removal of DDE registration, private DTK installation, worker non-autostart, and the native artifact paths.

- [ ] **Step 2: Run launcher/install tests and confirm the old copied-console-script contract fails**

  Run:

  ```bash
  PYTHONPATH=src .venv/bin/pytest tests/test_runtime_launcher.py tests/test_install_scripts.py -q
  ```

  Expected: collection fails for the absent launcher module and installer assertions fail because it still copies repository `.venv` console scripts and requires the XPU POC report.

- [ ] **Step 3: Implement fixed-command launcher dispatch**

  In `src/fun_voice/runtime_launcher.py`, use a closed dictionary and no shell interpolation:

  ```python
  ENTRYPOINTS = {
      "fun-voice-daemon": "fun_voice.daemon",
      "fun-voice-worker": "fun_voice.worker",
      "fun-voice-preflight": "fun_voice.preflight",
      "fun-voice-selftest": "fun_voice.selftest",
      "fun-voice-corrector": "fun_voice.corrector",
      "fun-voice-benchmark": "fun_voice.benchmark",
  }

  def main(argv: Sequence[str] | None = None) -> int:
      values = list(sys.argv[1:] if argv is None else argv)
      if not values or values[0] not in ENTRYPOINTS:
          return 2
      try:
          selection = load_runtime_selection()
      except RuntimeSelectionError:
          return 2
      source_root = Path(__file__).resolve().parents[2]
      env = os.environ.copy()
      env["PYTHONPATH"] = str(source_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
      os.execvpe(str(selection.python), [str(selection.python), "-m", ENTRYPOINTS[values[0]], *values[1:]], env)
      return 127
  ```

  `scripts/run-selected-runtime.sh` must be a minimal strict Bash adapter: resolve the repository root, require one command-name argument, and execute `python3 -m fun_voice.runtime_launcher "$@"` with `PYTHONPATH="$ROOT/src"`. It must not parse `selection.json` itself, substitute arbitrary Python, or import models.

- [ ] **Step 4: Replace installation hard gate and copied wrappers**

  In `scripts/install-user.sh`, parse exactly one optional `--runtime-selection PATH` argument; default only to the canonical data-root selection path. Before any user-scoped write, invoke the host launcher loader to validate the manifest and invoke `selection.python -c 'import torch, funasr, modelscope, transformers, Xlib'`. Its failure must stop installation before writing a file; output includes fixed category `runtime_selection_invalid` or `runtime_import_failed`, never a transcript or model path.

  Replace the existing copy of `${ROOT}/.venv/bin/<console script>` with:

  ```bash
  install_launcher() {
      local name="$1"
      local target="${BIN_DIR}/${name}"
      install -d -m 700 "${BIN_DIR}"
      umask 077
      printf '%s\n' '#!/usr/bin/env bash' \
          "exec \"${ROOT}/scripts/run-selected-runtime.sh\" \"${name}\" \"\$@\"" \
          > "${target}.tmp"
      chmod 700 "${target}.tmp"
      mv -f "${target}.tmp" "${target}"
  }
  ```

  Call this for the six closed command names. Retain native Fcitx/DTK validation and installation, autostart/environment import, legacy bridge retirement, daemon unit reload, and disabled worker services. The installer still leaves daemon disabled until the graphical session importer; only `bootstrap.py` restarts a successfully installed daemon after a selection has been published. `uninstall-user.sh` removes the same six owned launchers and native artifacts only; it must not delete model snapshots, runtimes, or `selection.json`.

- [ ] **Step 5: Update operator material and real-machine acceptance paths**

  In `README.md` and `docs/operations.md`, replace `.venv`/XPU-POC-first installation instructions with:

  ```bash
  scripts/initialize-first-run.sh
  scripts/initialize-first-run.sh --backend cpu
  scripts/initialize-first-run.sh --force-reselect
  fun-voice-selftest --format json
  ```

  Document `auto` order, explicit-mode no-fallback behavior, runtime root layout, no-repo-`.venv` policy, selection file modes, one-shot model download sets, CUDA BF16/FP16 behavior, XPU BF16-only behavior, and CPU SenseVoice-only/no-Qwen/no-speaker limitations. State clearly that TOML device/dtype keys are legacy inputs ignored by the effective runtime policy; only user preference `enhanced.enabled` may further disable accelerator correction.

  Update `scripts/config.example.toml` to remove device/dtype examples and replace CPU-inaccurate Nano fallback wording with a note that ASR profile is selected at first initialization. Preserve Qwen max tokens/source/timeout/protected term knobs as accelerator preferences, and state they have no effect under CPU selection.

  Update `docs/acceptance-checklist.md` with three mutually exclusive real-machine sections:

  1. CUDA: `selection.json` is `cuda`, Nano primary/SenseVoice fallback, corrector starts only after ASR worker stops, and `Super+C` still commits text.
  2. Intel XPU: same behavior with `xpu`, BF16, and the existing explicit XPU POC as optional diagnostics.
  3. CPU: selection says CPU/SenseVoice-only, no `fun-voice-worker@nano` socket/process, no Qwen/CAM++ process/download, raw SenseVoice text commits through Fcitx/clipboard, and all non-model desktop behavior works.

  In `docs/xpu-poc.md`, change “desktop deployment hard gate” to “explicit Intel XPU diagnostic”; it remains fail-closed for that command but cannot block CUDA or CPU initialization.

- [ ] **Step 6: Run repository verification, native checks, and manual real-device commands**

  Run the automated suite:

  ```bash
  PYTHONPATH=src .venv/bin/pytest -q
  .venv/bin/ruff check src tests
  .venv/bin/mypy src/fun_voice
  bash -n scripts/initialize-first-run.sh scripts/create-runtime-env.sh scripts/compile-runtime-locks.sh scripts/run-selected-runtime.sh scripts/install-user.sh scripts/uninstall-user.sh scripts/create-xpu-env.sh
  cmake -S native/fcitx5-fun-voice -B build/fcitx
  cmake --build build/fcitx
  ctest --test-dir build/fcitx --output-on-failure
  cmake -S native/dtk-overlay -B build/dtk-overlay
  cmake --build build/dtk-overlay
  ctest --test-dir build/dtk-overlay --output-on-failure
  ```

  Expected: all Python/native tests pass, lint/type checks are clean, and shell syntax checks pass.

  Perform one controlled real initialization per available machine, never placing user speech in the probe:

  ```bash
  scripts/initialize-first-run.sh --backend cuda
  scripts/initialize-first-run.sh --backend xpu
  scripts/initialize-first-run.sh --backend cpu
  fun-voice-selftest --format json
  systemctl --user status fun-voice-daemon.service --no-pager
  ```

  On each machine, run only the command matching its supported backend; unsupported explicit commands must fail without altering the existing manifest. In a CPU-only test, inspect `${XDG_DATA_HOME:-$HOME/.local/share}/fun-voice-ryan/models/models` and confirm it lacks `Qwen--Qwen3.5-0.8B` and `iic--speech_campplus_sv_zh-cn_16k-common` after initialization. Then validate `Super+C` with a harmless test utterance in a real DDE X11 session and follow the matching checklist section.

- [ ] **Step 7: Commit launch/deployment/documentation changes**

  ```bash
  git add src/fun_voice/runtime_launcher.py scripts/run-selected-runtime.sh scripts/install-user.sh scripts/uninstall-user.sh tests/test_runtime_launcher.py tests/test_install_scripts.py scripts/config.example.toml README.md docs/operations.md docs/acceptance-checklist.md docs/xpu-poc.md
  git commit -m "feat: deploy selected portable runtime"
  ```

## Final Verification Matrix

| Scenario | Automated proof | Real-machine proof |
| --- | --- | --- |
| CUDA succeeds | Bootstrap selects only `cuda`; CUDA probe has tensor + Nano ASR smoke; manifest permits Nano/SenseVoice/Qwen/CAM++. | `--backend cuda` produces `cuda:0`, BF16 or verified FP16; no models load at login; Qwen runs only after ASR release. |
| CUDA fails, XPU succeeds | Fake runner sees `cuda`, then `xpu`; selected manifest is XPU BF16. | `--backend auto` settles on `xpu:0`; XPU explicit POC remains usable as diagnostic. |
| CUDA/XPU fail, CPU succeeds | Fake runner sees `cuda`, `xpu`, `cpu`; CPU manifest exact-model invariant passes. | CPU init downloads SenseVoice/VAD only; SenseVoice worker starts on speech and commits raw ASR result. |
| Explicit backend fails | One candidate only; old manifest bytes unchanged. | Unsupported `--backend <name>` exits non-zero and daemon remains on prior valid selection. |
| Unsafe manifest | Unit tests reject mode, owner, schema, escaping symlink, and interpreter. | Installer/launcher/self-test fail closed with fixed category and no model starts. |
| CPU enhancement denial | Daemon/corrector tests prove no Qwen object/child; worker rejects Nano. | No Qwen/CAM++ snapshot or process; self-test reports `enhanced: false`. |
| Desktop regression | Existing full Python/native suite stays green. | DDE X11 `Super+C`, overlay, Fcitx and clipboard behavior are unchanged for the selected runtime. |

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-02-portable-runtime-initialization-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
