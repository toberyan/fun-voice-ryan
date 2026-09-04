"""Selected-runtime ASR adapters for Nano and SenseVoiceSmall.

The runtime loads the Nano engine and the FSMN-VAD model exactly once and keeps
them warm for the worker's lifetime. Each request reuses both models:

1. audio is loaded to a float32 16 kHz mono array (WAV or raw s16le),
2. the selected-runtime FSMN-VAD segments speech, returning ``[start_ms, end_ms]``
   regions,
3. a fixed small overlap is added to each region boundary when slicing audio,
4. the Nano engine transcribes the slices in original-audio-time order, and
5. the per-segment texts are concatenated verbatim (no character is inserted or
   removed) into the final text.

Privacy: this module never logs audio paths or transcription text; only counts,
lengths, and durations are logged.

This module does not import ``torch`` / ``funasr`` at import time so
the orchestration logic stays unit-testable with fakes.
"""

from __future__ import annotations

import io
import logging
import os
import re
import stat
import threading
import time
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from fun_voice.config import InferenceConfig
from fun_voice.contracts import (
    AsrStageTiming,
    ErrorCode,
    Segment,
    Transcription,
    WorkerHealth,
)
from fun_voice.runtime_selection import RuntimeSelection

logger = logging.getLogger(__name__)

VERSION = "0.1.0"
VAD_OVERLAP_MS = 250  # fixed small overlap applied to VAD region boundaries
VAD_MAX_SINGLE_SEGMENT_TIME_MS = 30000  # FSMN-VAD hard cap per speech segment (30 s)

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_NEW_TOKENS = 512
WARMUP_SAMPLE_COUNT = 16_000
MAX_LIVE_WINDOW_MS = 60_000
MAX_LIVE_PCM_BYTES = 16_000 * 2 * MAX_LIVE_WINDOW_MS // 1000
SENSEVOICE_CONTROL_TOKEN = re.compile(r"<\|[^|>\r\n]*\|>")

# Model cache layout mirrors scripts/run-nano-xpu-poc.sh:
#   ${XDG_DATA_HOME:-~/.local/share}/fun-voice-ryan/models/
#   models/<owner>--<name>/snapshots/master
NANO_MODEL_RELPATH = (
    "models",
    "FunAudioLLM--Fun-ASR-Nano-2512",
    "snapshots",
    "master",
)
SENSEVOICE_MODEL_RELPATH = (
    "models",
    "iic--SenseVoiceSmall",
    "snapshots",
    "master",
)
VAD_MODEL_RELPATH = (
    "models",
    "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "snapshots",
    "master",
)

class NanoRuntimeError(RuntimeError):
    """Base class for worker-runtime errors; subclasses carry a stable code."""

    error_code: ErrorCode = ErrorCode("worker", "internal")


class EmptySpeechError(NanoRuntimeError):
    """VAD detected no speech."""

    error_code: ErrorCode = ErrorCode("worker", "empty_speech")


class DeviceMismatchError(NanoRuntimeError):
    """Audio encoder/adaptor/prompt embeddings are not on the expected device."""

    error_code: ErrorCode = ErrorCode("worker", "device")


class InferenceTimeoutError(NanoRuntimeError):
    """Inference exceeded the per-request timeout."""

    error_code: ErrorCode = ErrorCode("worker", "timeout")


class OomError(NanoRuntimeError):
    """Device memory exhausted during inference."""

    error_code: ErrorCode = ErrorCode("worker", "oom")


class VllmError(NanoRuntimeError):
    """An ASR engine raised an inference error.

    ``worker.vllm`` is retained as the stable wire error code for already
    installed clients.  The Nano implementation is native FunASR/PyTorch, not
    vLLM; new callers must treat this as a backend-neutral inference failure.
    """

    error_code: ErrorCode = ErrorCode("worker", "vllm")


class ModelOutputError(NanoRuntimeError):
    """The model returned no output or a malformed result."""

    error_code: ErrorCode = ErrorCode("worker", "no_output")


class ModelLoadError(NanoRuntimeError):
    """A local selected-runtime model could not be constructed for a request."""

    error_code: ErrorCode = ErrorCode("worker", "model_load")


class AudioFormatError(NanoRuntimeError):
    """The audio input could not be loaded at the expected format."""

    error_code: ErrorCode = ErrorCode("worker", "format")


class LiveAudioProtocolError(NanoRuntimeError):
    """A received live descriptor violates the bounded local socket contract."""

    error_code: ErrorCode = ErrorCode("worker", "protocol")


SelectedDeviceType = Literal["cuda", "xpu", "cpu"]


# --- Protocols (the seams the fakes implement) ------------------------------


class AsrEngine(Protocol):
    def generate(
        self, inputs: list[np.ndarray], max_new_tokens: int
    ) -> list[dict[str, Any]]: ...


class VadSegmenter(Protocol):
    def detect(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[int, int]]: ...


class NativeNanoEngine:
    """Adapt the local FunASR/PyTorch Nano model to the runtime engine seam.

    The installed vLLM 0.28.0 XPU stack stalls after its first prompt-embedding
    request on the supported Arc platform.  This adapter keeps the same local
    Nano checkpoint and XPU-only execution while delegating decoding to the
    stable native FunASR/PyTorch path.
    """

    def __init__(self, model: Any) -> None:
        native = getattr(model, "model", None)
        llm = getattr(native, "llm", None)
        llm_model = getattr(llm, "model", None)
        get_embeddings = getattr(llm_model, "get_input_embeddings", None)
        if native is None or not callable(get_embeddings):
            raise ModelLoadError("native Nano model has no inspectable decoder")
        self._model = model
        self.backend = "native_funasr_pytorch"
        self.audio_encoder = getattr(native, "audio_encoder", None)
        self.audio_adaptor = getattr(native, "audio_adaptor", None)
        self.embed_tokens = get_embeddings()
        if any(
            module is None
            for module in (self.audio_encoder, self.audio_adaptor, self.embed_tokens)
        ):
            raise ModelLoadError("native Nano model has incomplete audio components")
        self.decoder_device_type = _module_device_type(llm) or _module_device_type(
            llm_model
        )

    def generate(
        self, inputs: list[np.ndarray], max_new_tokens: int
    ) -> list[dict[str, Any]]:
        """Decode in-memory float32 slices without a vLLM process or cache."""
        if not inputs:
            return []
        import torch

        samples = [
            torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
            for value in inputs
        ]
        results = self._model.generate(
            input=samples,
            cache={},
            batch_size_s=1,
            max_length=max_new_tokens,
            llm_kwargs={"do_sample": False},
        )
        if not isinstance(results, list):
            raise ModelOutputError("native Nano returned a malformed result list")
        return results


# --- Device helpers ---------------------------------------------------------


def _module_device_type(module: Any) -> str | None:
    """Best-effort device type (``"xpu"``, ``"cpu"`` …) of a torch module."""
    value = getattr(module, "device", None)
    if value is None:
        try:
            value = next(module.parameters()).device
        except (StopIteration, AttributeError):
            return None
    if isinstance(value, str):
        return value.split(":")[0].strip().lower() or None
    device_type = getattr(value, "type", None)
    return device_type if isinstance(device_type, str) else None


def _normalized_dtype(value: object) -> str | None:
    name = str(value).lower().removeprefix("torch.")
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "float32"}
    return aliases.get(name)


def _module_dtype(module: Any) -> str | None:
    direct = _normalized_dtype(getattr(module, "dtype", None))
    if direct is not None:
        return direct
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    observed: set[str] = set()
    try:
        values = parameters()
    except Exception:
        return None
    for parameter in values:
        floating = getattr(parameter, "is_floating_point", None)
        if callable(floating) and not floating():
            continue
        normalized = _normalized_dtype(getattr(parameter, "dtype", None))
        if normalized is not None:
            observed.add(normalized)
    return next(iter(observed)) if len(observed) == 1 else None


def device_type(device: str) -> SelectedDeviceType:
    """Return the supported backend type for one selected device string."""
    value = device.split(":", 1)[0]
    if value not in {"cuda", "xpu", "cpu"}:
        raise DeviceMismatchError("unsupported selected device")
    return cast(SelectedDeviceType, value)


def check_engine_devices(
    engine: Any,
    *,
    expected: SelectedDeviceType,
    expected_dtype: str | None = None,
) -> None:
    """Raise if encoder, adaptor, or embeddings miss the selected policy."""
    for name in ("audio_encoder", "audio_adaptor", "embed_tokens"):
        module = getattr(engine, name, None)
        if module is None:
            raise DeviceMismatchError(f"engine is missing {name}")
        actual = _module_device_type(module)
        if actual != expected:
            raise DeviceMismatchError(f"{name} is on {actual!r}, expected {expected!r}")
        if expected_dtype is not None:
            actual_dtype = _module_dtype(module)
            if actual_dtype != expected_dtype:
                raise DeviceMismatchError(
                    f"{name} dtype is {actual_dtype!r}, expected {expected_dtype!r}"
                )


def _funasr_precision_kwargs(dtype: str) -> dict[str, object]:
    if dtype not in {"float32", "bf16", "fp16"}:
        raise DeviceMismatchError("unsupported selected dtype")
    return {
        "dtype": dtype,
        "bf16": dtype == "bf16",
        "fp16": dtype == "fp16",
    }


def _is_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "out of memory" in message


def _remove_sensevoice_control_tokens(text: str) -> str:
    """Remove SenseVoice's ``<|...|>`` protocol metadata, preserving text."""
    return SENSEVOICE_CONTROL_TOKEN.sub("", text)


def _elapsed_ms(started: float) -> int:
    """Return a non-negative monotonic duration rounded to milliseconds."""
    return max(0, round((time.perf_counter() - started) * 1000))


# --- Audio loading ----------------------------------------------------------


def _load_audio_samples(path: str, sample_rate: int) -> np.ndarray:
    """Load 16 kHz mono audio as a float32 array in [-1, 1].

    Accepts WAV (RIFF) and raw s16le PCM. The path itself is never logged.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise AudioFormatError("unable to read audio input") from exc
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with wave.open(io.BytesIO(data), "rb") as wav:
            rate = wav.getframerate()
            if rate != sample_rate:
                raise AudioFormatError(f"WAV sample rate {rate} != {sample_rate}")
            if wav.getnchannels() != 1:
                raise AudioFormatError(
                    f"WAV has {wav.getnchannels()} channels, expected mono"
                )
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    else:
        pcm = np.frombuffer(data, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _load_pcm_fd(fd: int) -> np.ndarray:
    """Read raw s16le PCM from one received descriptor without a pathname."""
    try:
        descriptor = os.fstat(fd)
    except OSError as exc:
        raise LiveAudioProtocolError("cannot inspect live audio descriptor") from exc
    if not stat.S_ISREG(descriptor.st_mode):
        raise LiveAudioProtocolError("live audio descriptor is not a regular file")
    if descriptor.st_size > MAX_LIVE_PCM_BYTES:
        raise LiveAudioProtocolError("live audio descriptor exceeds the window limit")
    try:
        with os.fdopen(os.dup(fd), "rb", closefd=True) as audio:
            audio.seek(0)
            data = audio.read(MAX_LIVE_PCM_BYTES + 1)
    except OSError as exc:
        raise AudioFormatError("unable to read live audio descriptor") from exc
    if len(data) > MAX_LIVE_PCM_BYTES:
        raise LiveAudioProtocolError("live audio descriptor exceeds the window limit")
    if not data or len(data) % 2:
        raise AudioFormatError("live audio descriptor is not s16le PCM")
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


# --- VAD segmenter ----------------------------------------------------------


class FsmnVadSegmenter:
    """Selected-runtime FSMN-VAD adapter.

    FunASR ``fsmn-vad`` returns ``[{"key": str, "value": [[start_ms, end_ms], ...]}]``
    where ``value`` is a list of ``[start_ms, end_ms]`` integer pairs in
    milliseconds ordered by speech onset (verified against
    ``iic/speech_fsmn_vad_zh-cn-16k-common-pytorch``: a 3 s probe with two tone
    bursts yielded ``[[220, 2980]]``). Empty speech yields ``value == []``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def detect(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[int, int]]:
        result = self._model.generate(
            input=samples,
            cache={},
            is_final=True,
            max_single_segment_time=VAD_MAX_SINGLE_SEGMENT_TIME_MS,
        )
        if not result:
            raise ModelOutputError("VAD returned no result")
        first = result[0]
        if not isinstance(first, dict) or "value" not in first:
            raise ModelOutputError("malformed VAD result")
        value = first["value"]
        if not isinstance(value, list):
            raise ModelOutputError("malformed VAD value")
        regions: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ModelOutputError("malformed VAD segment")
            try:
                regions.append((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                raise ModelOutputError("malformed VAD segment") from None
        return regions


# --- Slicing ----------------------------------------------------------------


def _slice_windows(
    regions: list[tuple[int, int]], total_samples: int, sample_rate: int
) -> list[tuple[int, int]]:
    """Map VAD regions to sample-index windows with a fixed small overlap.

    The overlap is applied only to the audio fed to the ASR; the reported
    ``Segment`` times keep the raw VAD boundaries.
    """
    overlap = int(VAD_OVERLAP_MS * sample_rate / 1000)
    windows: list[tuple[int, int]] = []
    for start_ms, end_ms in regions:
        start = max(0, int(start_ms * sample_rate / 1000) - overlap)
        end = min(total_samples, int(end_ms * sample_rate / 1000) + overlap)
        windows.append((start, end))
    return windows


# --- Runtime ----------------------------------------------------------------


class NanoRuntime:
    """Owns the warm Nano engine + VAD and orchestrates per-request transcription."""

    def __init__(
        self,
        engine: AsrEngine,
        vad: VadSegmenter,
        *,
        selection: RuntimeSelection,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.engine = engine
        self.vad = vad
        self.selection = selection
        self.device = selection.device
        self.dtype = selection.dtype
        self.expected_device_type = device_type(selection.device)
        self.default_timeout = default_timeout
        self.last_error: ErrorCode | None = None
        self._generate_lock = threading.Lock()
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    def health(self) -> WorkerHealth:
        xpu_ready = self._xpu_ready()
        model_ready = self._model_ready()
        return WorkerHealth(
            version=VERSION,
            xpu_ready=xpu_ready,
            model_ready=model_ready,
            device=self.device,
            last_error=self.last_error,
            lifecycle=(
                "inactive"
                if self._closed
                else "failed"
                if self.last_error is not None or not (xpu_ready and model_ready)
                else "ready"
            ),
        )

    def device_evidence(self) -> tuple[str, str]:
        """Return separately verified Nano/VAD selected-device evidence."""
        check_engine_devices(
            self.engine,
            expected=self.expected_device_type,
            expected_dtype=self.dtype,
        )
        vad_model = getattr(self.vad, "_model", None)
        if vad_model is None:
            raise DeviceMismatchError("FSMN-VAD exposes no inspectable model")
        _assert_funasr_model_device(
            vad_model,
            name="FSMN-VAD",
            expected=self.expected_device_type,
            expected_dtype=self.dtype,
        )
        return self.device, self.device

    def close(self) -> None:
        """Mark the runtime closed; the worker process owns native model release."""
        self._closed = True

    def _xpu_ready(self) -> bool:
        try:
            import torch

            if self.expected_device_type == "cpu":
                return True
            backend = getattr(torch, self.expected_device_type, None)
            available = getattr(backend, "is_available", None)
            return bool(available()) if callable(available) else False
        except Exception:
            return False

    def _model_ready(self) -> bool:
        if self._closed:
            return False
        try:
            self.device_evidence()
        except DeviceMismatchError:
            return False
        return True

    # -- transcription ------------------------------------------------------

    def transcribe(
        self,
        audio: str,
        *,
        sample_rate: int = 16000,
        timeout: float | None = None,
    ) -> Transcription:
        """Transcribe an audio path after checking the selected backend."""
        try:
            check_engine_devices(
                self.engine,
                expected=self.expected_device_type,
                expected_dtype=self.dtype,
            )
            load_started = time.perf_counter()
            samples = _load_audio_samples(audio, sample_rate)
        except NanoRuntimeError as exc:
            self.last_error = exc.error_code
            raise
        return self.transcribe_samples(
            samples,
            sample_rate=sample_rate,
            timeout=timeout,
            audio_load_ms=_elapsed_ms(load_started),
        )

    def transcribe_samples(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int = 16000,
        timeout: float | None = None,
        audio_load_ms: int | None = None,
    ) -> Transcription:
        """VAD -> slice (with overlap) -> ASR -> verbatim concatenation."""
        try:
            return self._transcribe_impl(
                samples,
                sample_rate=sample_rate,
                timeout=timeout,
                audio_load_ms=audio_load_ms,
            )
        except NanoRuntimeError as exc:
            self.last_error = exc.error_code
            raise

    def detect_vad_fd(
        self, fd: int, *, sample_rate: int = 16000
    ) -> tuple[tuple[int, int], ...]:
        """Run the already-loaded VAD over a received anonymous PCM window."""
        self._require_selected_device_for_live_audio()
        try:
            regions = self.vad.detect(_load_pcm_fd(fd), sample_rate)
            return tuple(sorted(regions, key=lambda region: region[0]))
        except NanoRuntimeError as exc:
            self.last_error = exc.error_code
            raise

    def transcribe_window_fd(
        self,
        fd: int,
        *,
        sample_rate: int = 16000,
        source_start_ms: int,
        source_end_ms: int,
        timeout: float | None = None,
    ) -> Transcription:
        """Decode one live window and express its segments in source time."""
        if source_start_ms < 0 or source_end_ms <= source_start_ms:
            raise AudioFormatError("invalid live source range")
        self._require_selected_device_for_live_audio()
        try:
            result = self.transcribe_samples(
                _load_pcm_fd(fd), sample_rate=sample_rate, timeout=timeout
            )
        except NanoRuntimeError as exc:
            self.last_error = exc.error_code
            raise
        return Transcription(
            text=result.text,
            segments=tuple(
                Segment(
                    start_ms=source_start_ms + segment.start_ms,
                    end_ms=source_start_ms + segment.end_ms,
                    text=segment.text,
                )
                for segment in result.segments
            ),
            request_id=result.request_id,
            engine=result.engine,
            timing=result.timing,
            worker_elapsed_ms=result.worker_elapsed_ms,
        )

    def _require_selected_device_for_live_audio(self) -> None:
        """Reject every live descriptor unless Nano is permitted by its selection."""
        if "nano" not in self.selection.policy().allowed_profiles:
            raise DeviceMismatchError("live ASR profile is not allowed")

    def _transcribe_impl(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        timeout: float | None,
        audio_load_ms: int | None,
    ) -> Transcription:
        vad_started = time.perf_counter()
        regions = self.vad.detect(samples, sample_rate)
        vad_ms = _elapsed_ms(vad_started)
        if not regions:
            raise EmptySpeechError("VAD detected no speech")

        # Strict original-audio-time order, whatever the VAD returned.
        regions = sorted(regions, key=lambda region: region[0])
        windows = _slice_windows(regions, len(samples), sample_rate)
        slices = [samples[start:end] for start, end in windows]
        generate_started = time.perf_counter()
        texts = self._run_asr(slices, timeout)
        generate_ms = _elapsed_ms(generate_started)

        text = "".join(texts)  # direct concatenation: never insert/delete
        segments = tuple(
            Segment(start_ms=int(start_ms), end_ms=int(end_ms), text=text_i)
            for (start_ms, end_ms), text_i in zip(regions, texts, strict=True)
        )
        logger.debug("transcribed %d segments -> %d chars", len(segments), len(text))
        return Transcription(
            text=text,
            segments=segments,
            timing=AsrStageTiming(
                audio_load_ms=audio_load_ms,
                vad_ms=vad_ms,
                generate_ms=generate_ms,
            ),
        )

    def warmup(self) -> int:
        """Compile the Nano generate path with a fixed in-memory PCM buffer.

        The synthetic zero PCM intentionally bypasses VAD, never touches a user
        audio handle, and discards all model output. ``_run_asr`` preserves the
        existing engine serialization guarantee against real transcription.
        """
        if self._closed:
            raise ModelLoadError("Nano runtime is closed")
        started = time.perf_counter()
        self._run_asr(
            [np.zeros(WARMUP_SAMPLE_COUNT, dtype=np.float32)], self.default_timeout
        )
        return _elapsed_ms(started)

    # -- ASR with timeout + error taxonomy -----------------------------------

    def _run_asr(self, slices: list[np.ndarray], timeout: float | None) -> list[str]:
        """Run one batch through the engine, mapping failures to typed errors.

        A native ``generate`` cannot be safely interrupted mid-decode, so a
        timed-out call keeps running on a daemon thread. ``_generate_lock``
        serializes engine access, which both
        prevents concurrent ``generate`` calls from corrupting the engine and
        guarantees the lock is released once the straggler finishes — a
        subsequent (short) request then proceeds normally.
        """
        effective = self.default_timeout if timeout is None else timeout
        if effective <= 0:
            effective = self.default_timeout

        holder: dict[str, Any] = {}

        def _generate() -> None:
            # Serialize generate: the engine is not re-entrant, and a
            # timed-out call may still be finishing on a background thread.
            with self._generate_lock:
                try:
                    holder["result"] = self.engine.generate(
                        slices, max_new_tokens=MAX_NEW_TOKENS
                    )
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    holder["error"] = exc

        thread = threading.Thread(
            target=_generate, name="nano-asr-generate", daemon=True
        )
        thread.start()
        thread.join(timeout=effective)
        if thread.is_alive():
            raise InferenceTimeoutError(f"inference exceeded {effective:.1f}s")

        error = holder.get("error")
        if error is not None:
            if _is_oom_error(error):
                raise OomError("out of memory") from error
            raise VllmError(type(error).__name__) from error

        results = holder.get("result")
        if not results:
            raise ModelOutputError("model returned no output")
        if len(results) != len(slices):
            raise ModelOutputError(
                f"model returned {len(results)} results for {len(slices)} segments"
            )
        texts: list[str] = []
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                raise ModelOutputError("malformed model result")
            texts.append(result["text"])
        return texts


# --- Model loading ----------------------------------------------------------


def models_root(env: Mapping[str, str] | None = None) -> Path:
    """Return the local model-cache root (``<data-home>/fun-voice-ryan/models``)."""
    if env is None:
        env = os.environ
    override = env.get("FUN_VOICE_MODELS_ROOT")
    if override:
        return Path(override)
    data_home = env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "fun-voice-ryan" / "models"


def nano_model_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the Fun-ASR-Nano snapshot directory (contains ``model.pt``)."""
    return models_root(env).joinpath(*NANO_MODEL_RELPATH)


def sensevoice_model_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the locally installed SenseVoiceSmall snapshot directory."""
    return models_root(env).joinpath(*SENSEVOICE_MODEL_RELPATH)


def vad_model_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the locally installed FSMN-VAD snapshot directory."""
    return models_root(env).joinpath(*VAD_MODEL_RELPATH)


def load_nano_runtime(
    *,
    selection: RuntimeSelection,
    inference: InferenceConfig,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> NanoRuntime:
    """Load native Fun-ASR-Nano and FSMN-VAD for one selected accelerator.

    vLLM-specific sizing options remain accepted for configuration/API
    compatibility, but the native backend owns neither a KV cache nor a vLLM
    engine process.
    """
    if "nano" not in selection.policy().allowed_profiles:
        raise ModelLoadError("nano is not allowed by selected runtime")
    selected_device = selection.device
    expected_device_type = device_type(selected_device)
    if selection.dtype not in {"bf16", "fp16"}:
        raise ModelLoadError("native Nano requires an accelerator dtype")
    if (
        inference.gpu_memory_utilization != 0.15
        or not inference.enforce_eager
        or inference.max_model_len != 1536
    ):
        logger.warning(
            "native Nano ignores deprecated vLLM-only KV/cache settings; "
            "it allocates no persistent vLLM KV cache"
        )
    # Reuse the already-downloaded FSMN-VAD instead of re-downloading it.
    os.environ.setdefault("MODELSCOPE_CACHE", str(models_root()))

    engine = load_native_nano_engine(selected_device, selection.dtype)
    check_engine_devices(
        engine,
        expected=expected_device_type,
        expected_dtype=selection.dtype,
    )
    vad = _load_vad(selected_device, selection.dtype)
    return NanoRuntime(
        engine=engine,
        vad=vad,
        selection=selection,
        default_timeout=default_timeout,
    )


def load_native_nano_engine(
    device: str, dtype: str = "bf16", *, model_dir: str | Path | None = None
) -> NativeNanoEngine:
    """Load a local native Nano checkpoint for a selected supported device.

    ``model_dir`` is only a local snapshot path.  Supplying it lets preflight
    load exactly the same native backend as the worker without downloading or
    using the unstable vLLM prompt-embedding route.
    """
    expected_device_type = device_type(device)
    from funasr import AutoModel

    model = AutoModel(
        model=str(nano_model_dir() if model_dir is None else model_dir),
        trust_remote_code=True,
        device=device,
        **_funasr_precision_kwargs(dtype),
        disable_update=True,
    )
    _assert_funasr_model_device(
        model,
        name="Fun-ASR-Nano",
        expected=expected_device_type,
        expected_dtype=dtype,
    )
    return NativeNanoEngine(model)


def _load_vad(device: str, dtype: str = "bf16") -> FsmnVadSegmenter:
    expected_device_type = device_type(device)
    from funasr import AutoModel

    model = AutoModel(
        model=str(vad_model_dir()),
        device=device,
        **_funasr_precision_kwargs(dtype),
        disable_update=True,
        max_single_segment_time=VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    )
    _assert_funasr_model_device(
        model,
        name="FSMN-VAD",
        expected=expected_device_type,
        expected_dtype=dtype,
    )
    return FsmnVadSegmenter(model)


def _assert_funasr_model_device(
    model: Any,
    *,
    name: str,
    expected: SelectedDeviceType,
    expected_dtype: str | None = None,
) -> None:
    """Reject a FunASR model unless it is entirely on the selected backend."""
    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(model, "network", None),
    ):
        if candidate is None:
            continue
        parameters = getattr(candidate, "parameters", None)
        if not callable(parameters):
            continue
        devices = {str(parameter.device.type) for parameter in parameters()}
        if devices:
            if devices != {expected}:
                raise DeviceMismatchError(
                    f"{name} is not entirely on {expected!r}"
                )
            if expected_dtype is not None:
                actual_dtype = _module_dtype(candidate)
                if actual_dtype != expected_dtype:
                    raise DeviceMismatchError(
                        f"{name} dtype is {actual_dtype!r}, expected {expected_dtype!r}"
                    )
            return
    raise DeviceMismatchError(f"{name} exposes no inspectable parameters")


def _assert_sensevoice_components(
    model: Any,
    *,
    expected: SelectedDeviceType,
    expected_dtype: str,
) -> None:
    _assert_funasr_model_device(
        model,
        name="SenseVoiceSmall",
        expected=expected,
        expected_dtype=expected_dtype,
    )
    vad_model = getattr(model, "vad_model", None)
    if vad_model is not None:
        _assert_funasr_model_device(
            vad_model,
            name="FSMN-VAD",
            expected=expected,
            expected_dtype=expected_dtype,
        )


class SenseVoiceRuntime:
    """Local-snapshot SenseVoiceSmall runtime on the selected backend."""

    def __init__(
        self,
        model: Any,
        *,
        selection: RuntimeSelection,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self.selection = selection
        self.device = selection.device
        self.dtype = selection.dtype
        self.expected_device_type = device_type(selection.device)
        self.default_timeout = default_timeout
        self.last_error: ErrorCode | None = None
        self._closed = False

    def health(self) -> WorkerHealth:
        ready = not self._closed
        if ready:
            try:
                _assert_sensevoice_components(
                    self._model,
                    expected=self.expected_device_type,
                    expected_dtype=self.dtype,
                )
            except DeviceMismatchError:
                ready = False
        return WorkerHealth(
            version=VERSION,
            xpu_ready=ready,
            model_ready=ready,
            device=self.device,
            last_error=self.last_error,
            lifecycle=(
                "inactive"
                if self._closed
                else "failed"
                if self.last_error is not None or not ready
                else "ready"
            ),
        )

    def close(self) -> None:
        self._closed = True

    def transcribe(
        self, audio: str, *, sample_rate: int = 16000, timeout: float | None = None
    ) -> Transcription:
        del timeout
        if self._closed:
            raise ModelLoadError("SenseVoiceSmall runtime is closed")
        try:
            _assert_sensevoice_components(
                self._model,
                expected=self.expected_device_type,
                expected_dtype=self.dtype,
            )
            # PipeWire capture is raw s16le PCM in an anonymous memory-backed
            # file.  FunASR treats a string input as a container path and
            # delegates decoding to ffmpeg, which cannot infer raw PCM's
            # format.  Passing 16 kHz float samples is an official AutoModel
            # input form and keeps the complete capture-to-ASR path in memory.
            samples = _load_audio_samples(audio, sample_rate)
            results = self._model.generate(input=samples)
        except NanoRuntimeError as exc:
            self.last_error = exc.error_code
            raise
        except BaseException as exc:  # noqa: BLE001 - stable worker taxonomy
            if _is_oom_error(exc):
                self.last_error = OomError.error_code
                raise OomError("out of memory") from exc
            self.last_error = VllmError.error_code
            raise VllmError(type(exc).__name__) from exc
        if not results or not isinstance(results[0], dict):
            self.last_error = ModelOutputError.error_code
            raise ModelOutputError("SenseVoiceSmall returned no result")
        first = results[0]
        sentence_info = first.get("sentence_info")
        segments: tuple[Segment, ...] = ()
        if isinstance(sentence_info, list):
            parsed: list[Segment] = []
            for item in sentence_info:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", item.get("sentence", ""))
                if not isinstance(text, str):
                    continue
                text = _remove_sensevoice_control_tokens(text)
                parsed.append(
                    Segment(
                        start_ms=int(item.get("start", 0)),
                        end_ms=int(item.get("end", 0)),
                        text=text,
                    )
                )
            segments = tuple(parsed)
        text = first.get("text")
        if not isinstance(text, str):
            text = "".join(segment.text for segment in segments)
        else:
            text = _remove_sensevoice_control_tokens(text)
        if not text:
            self.last_error = EmptySpeechError.error_code
            raise EmptySpeechError("SenseVoiceSmall returned empty text")
        return Transcription(text=text, segments=segments, engine="sensevoice")


def load_sensevoice_runtime(
    *,
    selection: RuntimeSelection,
    inference: InferenceConfig,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SenseVoiceRuntime:
    """Load SenseVoiceSmall and FSMN-VAD from local selected-runtime snapshots."""
    if "sensevoice" not in selection.policy().allowed_profiles:
        raise ModelLoadError("sensevoice is not allowed by selected runtime")
    selected_device = selection.device
    expected_device_type = device_type(selected_device)
    if inference.dtype != selection.dtype or inference.device != selection.device:
        raise ModelLoadError("SenseVoice configuration differs from selected runtime")
    os.environ.setdefault("MODELSCOPE_CACHE", str(models_root()))
    from funasr import AutoModel

    model = AutoModel(
        model=str(sensevoice_model_dir()),
        vad_model=str(vad_model_dir()),
        vad_kwargs=_funasr_precision_kwargs(selection.dtype),
        device=selected_device,
        **_funasr_precision_kwargs(selection.dtype),
        disable_update=True,
        max_single_segment_time=VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    )
    _assert_sensevoice_components(
        model,
        expected=expected_device_type,
        expected_dtype=selection.dtype,
    )
    return SenseVoiceRuntime(
        model, selection=selection, default_timeout=default_timeout
    )
