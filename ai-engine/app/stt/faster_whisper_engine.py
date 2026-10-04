"""faster-whisper STT backend.

faster-whisper drives a CTranslate2-converted Whisper model. It is chosen over
``openai-whisper`` because it runs the same weights without PyTorch, quantises
to int8 on CPU (a 4-8x speed-up at negligible WER cost for short utterances),
and produces a CTranslate2 model directory that Phase 5 can either serve
directly or export to ONNX -- the same artifact family Triton expects.

The model is loaded lazily and behind a lock: ``GET /health`` must answer
without downloading or allocating ~150 MB of weights.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

from app.audio_io import ensure_rate
from app.stt.base import STTEngine, STTUnavailableError, TranscriptionResult, TranscriptSegment

logger = logging.getLogger(__name__)


class FasterWhisperSTT(STTEngine):
    """Whisper transcription via CTranslate2."""

    name = "faster-whisper"

    def __init__(
        self,
        model: str = "base.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        sample_rate: int = 16000,
        download_root: str | None = None,
        hf_cache_dir: str | None = None,
    ):
        self._model_name = model
        # CTranslate2 has no Metal/MPS backend; anything but cuda runs on CPU.
        self._device = "cuda" if device == "cuda" else "cpu"
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._sample_rate = sample_rate
        self._download_root = download_root
        self._hf_cache_dir = hf_cache_dir
        self._model = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:  # pragma: no cover - import guard
                raise STTUnavailableError(
                    "faster-whisper is not installed; `pip install faster-whisper`"
                ) from exc
            if self._hf_cache_dir:
                os.environ.setdefault("HF_HOME", self._hf_cache_dir)
            logger.info(
                "loading faster-whisper model=%s device=%s compute_type=%s",
                self._model_name, self._device, self._compute_type,
            )
            try:
                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=self._download_root,
                )
            except Exception as exc:
                raise STTUnavailableError(
                    f"failed to load faster-whisper model {self._model_name!r}: {exc}"
                ) from exc
            return self._model

    def load(self) -> None:
        self._load()

    def close(self) -> None:
        self._model = None

    # -- inference ----------------------------------------------------------
    def transcribe(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
    ) -> TranscriptionResult:
        model = self._load()

        source = audio if isinstance(audio, str) else "<array>"
        if isinstance(audio, str):
            audio_input: str | np.ndarray = audio
            duration = 0.0
        else:
            samples = np.asarray(audio, dtype=np.float32)
            rate = sample_rate or self._sample_rate
            if rate != self._sample_rate:
                samples = ensure_rate(samples, rate, self._sample_rate)
            audio_input = np.ascontiguousarray(samples, dtype=np.float32)
            duration = audio_input.shape[0] / float(self._sample_rate)

        segments_iter, info = model.transcribe(
            audio_input,
            beam_size=self._beam_size,
            language=None if not self._model_name.endswith(".en") else "en",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segments = tuple(
            TranscriptSegment(
                text=seg.text.strip(),
                start_s=float(seg.start),
                end_s=float(seg.end),
            )
            for seg in segments_iter
        )
        text = " ".join(s.text for s in segments if s.text).strip()
        if not duration:
            duration = float(getattr(info, "duration", 0.0) or 0.0)

        logger.debug("stt %s -> %r (%.2fs audio)", source, text[:80], duration)
        return TranscriptionResult(
            text=text,
            language=str(getattr(info, "language", "") or ""),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration_s=duration,
            segments=segments,
            backend=self.name,
        )
