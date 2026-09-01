"""Warm Fun-ASR-Nano runtime: CPU FSMN-VAD + XPU vLLM ASR.

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
import threading
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fun_voice.contracts import ErrorCode, Segment, Transcription, WorkerHealth

logger = logging.getLogger(__name__)

VERSION = "0.1.0"
DEVICE = "xpu:0"
EXPECTED_DEVICE_TYPE = "xpu"
VAD_OVERLAP_MS = 250  # fixed small overlap applied to VAD region boundaries
VAD_MAX_SINGLE_SEGMENT_TIME_MS = 30000  # FSMN-VAD hard cap per speech segment (30 s)

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_NEW_TOKENS = 512

# Model cache layout mirrors scripts/run-nano-xpu-poc.sh:
#   ${XDG_DATA_HOME:-~/.local/share}/fun-voice-ryan/models/
#   models/<owner>--<name>/snapshots/master
NANO_MODEL_RELPATH = (
    "models",
    "FunAudioLLM--Fun-ASR-Nano-2512",
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


class AudioFormatError(NanoRuntimeError):
    """The audio input could not be loaded at the expected format."""

    error_code: ErrorCode = ErrorCode("worker", "format")


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
        return WorkerHealth(
            version=VERSION,
            xpu_ready=self._xpu_ready(),
            model_ready=self._model_ready(),
            device=self.device,
            last_error=self.last_error,
        )

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
            samples = _load_audio_samples(audio, sample_rate)
        except NanoRuntimeError as exc:
            self.last_error = exc.error_code
            raise
        return self.transcribe_samples(
            samples, sample_rate=sample_rate, timeout=timeout
        )

    def transcribe_samples(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int = 16000,
        timeout: float | None = None,
    ) -> Transcription:
        """VAD -> slice (with overlap) -> ASR -> verbatim concatenation."""
        try:
            return self._transcribe_impl(
                samples, sample_rate=sample_rate, timeout=timeout
            )
        except NanoRuntimeError as exc:
            self.last_error = exc.error_code
            raise

    def _transcribe_impl(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        timeout: float | None,
    ) -> Transcription:
        regions = self.vad.detect(samples, sample_rate)
        if not regions:
            raise EmptySpeechError("VAD detected no speech")

        # Strict original-audio-time order, whatever the VAD returned.
        regions = sorted(regions, key=lambda region: region[0])
        windows = _slice_windows(regions, len(samples), sample_rate)
        slices = [samples[start:end] for start, end in windows]
        texts = self._run_asr(slices, timeout)

        text = "".join(texts)  # direct concatenation: never insert/delete
        segments = tuple(
            Segment(start_ms=int(start_ms), end_ms=int(end_ms), text=text_i)
            for (start_ms, end_ms), text_i in zip(regions, texts, strict=True)
        )
        logger.debug("transcribed %d segments -> %d chars", len(segments), len(text))
        return Transcription(text=text, segments=segments)

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


def load_nano_runtime(
    *,
    device: str = DEVICE,
    dtype: str = "bf16",
    gpu_memory_utilization: float = 0.35,
    enforce_eager: bool = True,
    max_model_len: int = 4096,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> NanoRuntime:
    """Load the Fun-ASR-Nano engine and FSMN-VAD once, on XPU."""
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
    vad = _load_vad()
    return NanoRuntime(
        engine=engine, vad=vad, device=device, default_timeout=default_timeout
    )


def _load_vad() -> FsmnVadSegmenter:
    from funasr import AutoModel

    model = AutoModel(
        model="fsmn-vad",
        device="cpu",
        disable_update=True,
        max_single_segment_time=VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    )
    return FsmnVadSegmenter(model)
