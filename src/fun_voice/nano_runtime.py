"""XPU ASR runtime adapters for Nano and SenseVoiceSmall.

The runtime loads the Nano engine and the FSMN-VAD model exactly once and keeps
them warm for the worker's lifetime. Each request reuses both models:

1. audio is loaded to a float32 16 kHz mono array (WAV or raw s16le),
2. the CPU FSMN-VAD segments speech, returning ``[start_ms, end_ms]`` regions,
3. a fixed small overlap is added to each region boundary when slicing audio,
4. the Nano engine transcribes the slices in original-audio-time order, and
5. the per-segment texts are concatenated verbatim (no character is inserted or
   removed) into the final text.

Privacy: this module never logs audio paths or transcription text; only counts,
lengths, and durations are logged.

This module does not import ``torch`` / ``vllm`` / ``funasr`` at import time so
the orchestration logic stays unit-testable with fakes.
"""

from __future__ import annotations

import io
import logging
import os
import stat
import threading
import time
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fun_voice.contracts import (
    AsrStageTiming,
    ErrorCode,
    Segment,
    Transcription,
    WorkerHealth,
)

logger = logging.getLogger(__name__)

VERSION = "0.1.0"
DEVICE = "xpu:0"
EXPECTED_DEVICE_TYPE = "xpu"
VAD_OVERLAP_MS = 250  # fixed small overlap applied to VAD region boundaries
VAD_MAX_SINGLE_SEGMENT_TIME_MS = 30000  # FSMN-VAD hard cap per speech segment (30 s)

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_NEW_TOKENS = 512
WARMUP_SAMPLE_COUNT = 16_000
MAX_LIVE_WINDOW_MS = 60_000
MAX_LIVE_PCM_BYTES = 16_000 * 2 * MAX_LIVE_WINDOW_MS // 1000

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
    """The vLLM engine raised an error during inference."""

    error_code: ErrorCode = ErrorCode("worker", "vllm")


class ModelOutputError(NanoRuntimeError):
    """The model returned no output or a malformed result."""

    error_code: ErrorCode = ErrorCode("worker", "no_output")


class ModelLoadError(NanoRuntimeError):
    """A local XPU model could not be constructed for a request."""

    error_code: ErrorCode = ErrorCode("worker", "model_load")


class AudioFormatError(NanoRuntimeError):
    """The audio input could not be loaded at the expected format."""

    error_code: ErrorCode = ErrorCode("worker", "format")


class LiveAudioProtocolError(NanoRuntimeError):
    """A received live descriptor violates the bounded local socket contract."""

    error_code: ErrorCode = ErrorCode("worker", "protocol")


# --- Protocols (the seams the fakes implement) ------------------------------


class AsrEngine(Protocol):
    def generate(
        self, inputs: list[np.ndarray], max_new_tokens: int
    ) -> list[dict[str, Any]]: ...


class VadSegmenter(Protocol):
    def detect(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[int, int]]: ...


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


def check_engine_devices(engine: Any, *, expected: str = EXPECTED_DEVICE_TYPE) -> None:
    """Raise :class:`DeviceMismatchError` if encoder/adaptor/embeds aren't on XPU."""
    for name in ("audio_encoder", "audio_adaptor", "embed_tokens"):
        module = getattr(engine, name, None)
        if module is None:
            raise DeviceMismatchError(f"engine is missing {name}")
        actual = _module_device_type(module)
        if actual != expected:
            raise DeviceMismatchError(f"{name} is on {actual!r}, expected {expected}")


def _is_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "out of memory" in message


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
    """CPU FSMN-VAD adapter.

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
        device: str = DEVICE,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.engine = engine
        self.vad = vad
        self.device = device
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
        """Return separately verified Nano/VAD XPU evidence for the POC gate."""
        if self.device != DEVICE:
            raise DeviceMismatchError("Nano runtime requires xpu:0")
        check_engine_devices(self.engine)
        vad_model = getattr(self.vad, "_model", None)
        if vad_model is None:
            raise DeviceMismatchError("FSMN-VAD exposes no inspectable model")
        _assert_funasr_model_xpu(vad_model, name="FSMN-VAD")
        return DEVICE, DEVICE

    def close(self) -> None:
        """Mark the runtime closed (offline vLLM has no explicit close API)."""
        self._closed = True

    def _xpu_ready(self) -> bool:
        try:
            import torch

            return bool(torch.xpu.is_available())
        except Exception:
            return False

    def _model_ready(self) -> bool:
        if self._closed:
            return False
        engine = self.engine
        for name in ("audio_encoder", "audio_adaptor", "embed_tokens"):
            module = getattr(engine, name, None)
            if module is None or _module_device_type(module) != EXPECTED_DEVICE_TYPE:
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
        """Transcribe an audio path, refusing non-XPU devices before any work."""
        try:
            check_engine_devices(self.engine)
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
        self._require_xpu_zero_for_live_audio()
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
        self._require_xpu_zero_for_live_audio()
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

    def _require_xpu_zero_for_live_audio(self) -> None:
        """Reject every live descriptor before it is read on a wrong device."""
        if self.device != DEVICE:
            raise DeviceMismatchError("live ASR requires xpu:0")

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

        Timeout is a design trade-off: vLLM's offline ``generate`` cannot be
        safely interrupted mid-decode, so a timed-out call keeps running on a
        daemon thread. ``_generate_lock`` serializes engine access, which both
        prevents concurrent ``generate`` calls from corrupting the engine and
        guarantees the lock is released once the straggler finishes — a
        subsequent (short) request then proceeds normally.
        """
        effective = self.default_timeout if timeout is None else timeout
        if effective <= 0:
            effective = self.default_timeout

        holder: dict[str, Any] = {}

        def _generate() -> None:
            # Serialize vLLM generate: the engine is not re-entrant, and a
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
    device: str = DEVICE,
    dtype: str = "bf16",
    gpu_memory_utilization: float = 0.15,
    enforce_eager: bool = True,
    max_model_len: int = 1536,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> NanoRuntime:
    """Load the Fun-ASR-Nano engine and FSMN-VAD once, on XPU."""
    if device != DEVICE:
        raise DeviceMismatchError("Nano runtime requires xpu:0")
    # Reuse the already-downloaded FSMN-VAD instead of re-downloading it.
    os.environ.setdefault("MODELSCOPE_CACHE", str(models_root()))

    from fun_voice.preflight import load_nano_engine

    engine = load_nano_engine(
        nano_model_dir(),
        device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
    )
    check_engine_devices(engine)
    vad = _load_vad(device)
    return NanoRuntime(
        engine=engine, vad=vad, device=device, default_timeout=default_timeout
    )


def _load_vad(device: str) -> FsmnVadSegmenter:
    from funasr import AutoModel

    model = AutoModel(
        model=str(vad_model_dir()),
        device=device,
        disable_update=True,
        max_single_segment_time=VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    )
    _assert_funasr_model_xpu(model, name="FSMN-VAD")
    return FsmnVadSegmenter(model)


def _assert_funasr_model_xpu(model: Any, *, name: str) -> None:
    """Reject a FunASR model unless an inspectable module is entirely on XPU."""
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
            if devices != {EXPECTED_DEVICE_TYPE}:
                raise DeviceMismatchError(f"{name} is not entirely on XPU")
            return
    raise DeviceMismatchError(f"{name} exposes no inspectable parameters")


class SenseVoiceRuntime:
    """Local-snapshot SenseVoiceSmall fallback, entirely on the Intel XPU."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = DEVICE,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self.device = device
        self.default_timeout = default_timeout
        self.last_error: ErrorCode | None = None
        self._closed = False

    def health(self) -> WorkerHealth:
        ready = not self._closed
        if ready:
            try:
                _assert_funasr_model_xpu(self._model, name="SenseVoiceSmall")
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
        del sample_rate, timeout
        if self._closed:
            raise ModelLoadError("SenseVoiceSmall runtime is closed")
        try:
            _assert_funasr_model_xpu(self._model, name="SenseVoiceSmall")
            results = self._model.generate(input=audio)
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
        if not text:
            self.last_error = EmptySpeechError.error_code
            raise EmptySpeechError("SenseVoiceSmall returned empty text")
        return Transcription(text=text, segments=segments, engine="sensevoice")


def load_sensevoice_runtime(
    *, device: str = DEVICE, default_timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> SenseVoiceRuntime:
    """Load SenseVoiceSmall and FSMN-VAD from local snapshots on XPU only."""
    os.environ.setdefault("MODELSCOPE_CACHE", str(models_root()))
    from funasr import AutoModel

    model = AutoModel(
        model=str(sensevoice_model_dir()),
        vad_model=str(vad_model_dir()),
        device=device,
        disable_update=True,
        max_single_segment_time=VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    )
    _assert_funasr_model_xpu(model, name="SenseVoiceSmall")
    return SenseVoiceRuntime(model, device=device, default_timeout=default_timeout)
