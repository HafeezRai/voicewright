"""Barge-in: let the user interrupt the agent mid-sentence.

While the agent is speaking, sustained user speech must cancel playback and
hand the floor back to the user. This module owns that decision; it does *not*
own playback. :class:`PlaybackSink` is the seam where Phase 3's TTS lands —
Phase 2 ships :class:`StubPlaybackSink` only, per the phase boundary.

Design notes
------------
* **Hysteresis.** Two time constants, both in milliseconds, govern the machine:
  ``min_speech_ms`` of *continuous* speech before an interrupt fires, and
  ``min_silence_ms`` of sustained silence before the user's turn ends and
  barge-in re-arms. A click, a lip smack or a cough is shorter than the former;
  a breath pause mid-sentence is shorter than the latter.
* **No self-trigger (assumption).** The agent's own voice reaching the mic is
  assumed to be removed upstream by WebRTC's acoustic echo canceller (AEC),
  which is on by default in the browser peer connection. That is an *assumption
  about the client*, not something this server can enforce — if AEC is disabled
  or leaky, ``min_speech_ms`` is the only remaining defence and should be
  raised. This is called out in the Phase 2 report as a known risk.
* **Testability.** The controller is pure: it consumes booleans from a VAD and
  a :class:`PlaybackSink`, and emits :class:`BargeInEvent` objects. No audio
  device, event loop or network is involved, so it is fully unit-testable.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field


class PlaybackSink(abc.ABC):
    """The agent's outbound audio, as seen by the barge-in controller.

    Phase 3 will implement this over the real TTS stream. Implementations must
    report, at any instant, whether audio is currently being played, and must
    make :meth:`stop` idempotent.
    """

    @property
    @abc.abstractmethod
    def is_speaking(self) -> bool:
        """True while the agent is actively playing audio."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Cancel playback immediately. Idempotent."""


@dataclass
class StubPlaybackSink(PlaybackSink):
    """In-memory playback sink used by tests and as a Phase 2 placeholder."""

    speaking: bool = False
    stop_calls: int = field(default=0, init=False)

    @property
    def is_speaking(self) -> bool:
        return self.speaking

    def start(self) -> None:
        """Begin 'playback' (test helper; real TTS drives this itself)."""
        self.speaking = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.speaking = False


class BargeInEventKind(str, enum.Enum):
    BARGE_IN = "barge_in"      # user interrupted the agent; playback cancelled
    SPEECH_END = "speech_end"  # user stopped talking; turn over, barge-in re-armed


@dataclass(frozen=True)
class BargeInEvent:
    kind: BargeInEventKind
    #: Duration in ms of the speech run (BARGE_IN) or silence run (SPEECH_END).
    duration_ms: int


class BargeInController:
    """Frame-by-frame barge-in state machine.

    Feed it one ``bool`` per VAD frame via :meth:`process`; it returns the list
    of events (usually empty) produced by that frame.
    """

    def __init__(
        self,
        sink: PlaybackSink,
        *,
        frame_duration_ms: int,
        min_speech_ms: int = 200,
        min_silence_ms: int = 400,
        enabled: bool = True,
    ):
        if frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive")
        self._sink = sink
        self._frame_ms = frame_duration_ms
        self._min_speech_frames = max(1, -(-min_speech_ms // frame_duration_ms)) if min_speech_ms else 1
        self._min_silence_frames = max(1, -(-min_silence_ms // frame_duration_ms)) if min_silence_ms else 1
        self._enabled = enabled
        self._speech_run = 0
        self._silence_run = 0
        self._interrupted = False

    @property
    def interrupted(self) -> bool:
        """True from the moment barge-in fires until the user's turn ends."""
        return self._interrupted

    @property
    def sink(self) -> PlaybackSink:
        return self._sink

    def reset(self) -> None:
        self._speech_run = 0
        self._silence_run = 0
        self._interrupted = False

    def process(self, is_speech: bool) -> list[BargeInEvent]:
        """Advance the machine by one frame and return any events produced."""
        if not self._enabled:
            return []

        events: list[BargeInEvent] = []

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
            # Fire once per user turn, and only while the agent still speaks.
            if (
                not self._interrupted
                and self._sink.is_speaking
                and self._speech_run >= self._min_speech_frames
            ):
                self._sink.stop()
                self._interrupted = True
                events.append(
                    BargeInEvent(BargeInEventKind.BARGE_IN, self._speech_run * self._frame_ms)
                )
        else:
            self._silence_run += 1
            self._speech_run = 0
            if self._interrupted and self._silence_run >= self._min_silence_frames:
                events.append(
                    BargeInEvent(BargeInEventKind.SPEECH_END, self._silence_run * self._frame_ms)
                )
                self._interrupted = False

        return events
