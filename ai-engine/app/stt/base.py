"""STT engine interface.

The rest of the service depends on :class:`STTEngine` and the two frozen
dataclasses below -- never on faster-whisper's own types. That keeps the
transcriber swappable (Phase 5 moves inference to Triton/ONNX) without touching
the pipeline or the serving layer.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


class STTUnavailableError(RuntimeError):
    """Raised when the STT backend cannot be loaded (missing package/weights)."""


@dataclass(frozen=True)
class TranscriptSegment:
    """One decoded span, with its position in the source audio."""

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TranscriptionResult:
    """A full transcription of one utterance."""

    text: str
    language: str = ""
    language_probability: float = 0.0
    duration_s: float = 0.0
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)
    backend: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class STTEngine(abc.ABC):
    """Turns audio into text. Implementations load weights lazily."""

    #: Short identifier surfaced in metrics labels and API responses.
    name: str = "stt"

    @property
    def is_loaded(self) -> bool:
        """True once weights are resident (never triggers a load)."""
        return False

    def load(self) -> None:
        """Eagerly load weights. Optional; :meth:`transcribe` loads on demand."""

    def close(self) -> None:
        """Release weights. Optional; safe to call more than once."""

    @abc.abstractmethod
    def transcribe(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
    ) -> TranscriptionResult:
        """Transcribe a mono ``float32`` array (or a path) to text.

        ``sample_rate`` is required when ``audio`` is an array and is assumed to
        be the service's configured rate otherwise.
        """
