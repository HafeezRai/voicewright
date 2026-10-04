"""TTS engine interface.

Synthesis comes in two shapes:

* :meth:`TTSEngine.synthesize` -- the whole utterance as one array (the batch
  path, and what tests assert on);
* :meth:`TTSEngine.stream` -- the same audio as an iterator of chunks, which is
  what barge-in needs: the caller can stop consuming (or cancel the token) and
  synthesis halts, rather than finishing an utterance nobody will hear.

The base class provides a working ``stream`` (synthesise-then-chunk) so a new
backend is usable before it grows true incremental synthesis; the shipped
backend overrides it to cancel *during* the model's forward pass.
"""

from __future__ import annotations

import abc
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from app.audio_io import duration_s, iter_chunks, to_wav_bytes


class TTSUnavailableError(RuntimeError):
    """Raised when the TTS backend cannot be loaded (missing package/weights)."""


class CancelToken:
    """A thread-safe one-way cancel flag.

    Set with :meth:`cancel`, tested with :attr:`cancelled`. One token covers one
    utterance; :meth:`reset` re-arms it for the next.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()


@dataclass(frozen=True)
class SynthesisResult:
    """Synthesised mono audio plus the rate it was produced at."""

    audio: np.ndarray
    sample_rate: int
    backend: str = ""
    #: True when synthesis was stopped early by a cancel.
    cancelled: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return duration_s(self.audio, self.sample_rate)

    def to_wav_bytes(self, *, subtype: str = "PCM_16") -> bytes:
        return to_wav_bytes(self.audio, self.sample_rate, subtype=subtype)


class TTSEngine(abc.ABC):
    """Turns text into audio. Implementations load weights lazily."""

    name: str = "tts"

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz (valid once loaded; a sane guess before)."""
        return 0

    @property
    def is_loaded(self) -> bool:
        return False

    def load(self) -> None:
        """Eagerly load weights. Optional; synthesis loads on demand."""

    def close(self) -> None:
        """Release weights. Optional; safe to call more than once."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        speaker_id: int | None = None,
    ) -> SynthesisResult:
        """Synthesise ``text`` in full and return the audio."""

    def stream(
        self,
        text: str,
        *,
        chunk_samples: int,
        speed: float | None = None,
        speaker_id: int | None = None,
        cancel: CancelToken | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield ``text``'s audio as chunks, honouring ``cancel`` between chunks.

        The default implementation synthesises fully, then chunks -- so a cancel
        still stops *playback* promptly, it just cannot stop the model's forward
        pass. Backends that can interrupt synthesis itself override this.
        """
        result = self.synthesize(text, speed=speed, speaker_id=speaker_id)
        for chunk in iter_chunks(result.audio, chunk_samples):
            if cancel is not None and cancel.cancelled:
                return
            yield chunk
