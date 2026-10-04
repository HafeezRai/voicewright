"""The TTS side of barge-in: a real :class:`PlaybackSink`.

``backend-core/app/pipeline/bargein.py`` defines ``PlaybackSink`` as the seam
where the agent's outbound audio is exposed to its barge-in state machine: an
``is_speaking`` property and an idempotent ``stop()``. Phase 2 shipped only a
stub. :class:`TTSPlaybackSink` is the real one.

It deliberately does **not** import backend-core. The two services are separate
processes with separate dependency trees, and a cross-package import would couple
them for no benefit; the contract is small and duck-typed, so this class
implements it structurally and ``tests/test_bargein_sink.py`` proves that
conformance against backend-core's actual ABC (loaded by path, so nothing in
backend-core has to change).

Cancellation crosses the thread boundary through a :class:`CancelToken`: the
barge-in controller calls :meth:`stop` from its own thread, and the TTS streaming
generator, running wherever playback happens, observes the token between -- and,
for sherpa-onnx, *during* -- chunks.
"""

from __future__ import annotations

import threading

from app.tts.base import CancelToken


class TTSPlaybackSink:
    """Playback control surface for one TTS utterance.

    Contract (matches ``backend-core.app.pipeline.bargein.PlaybackSink``):
    :attr:`is_speaking` reports whether audio is currently playing, and
    :meth:`stop` cancels playback and is idempotent.
    """

    def __init__(self) -> None:
        self._speaking = False
        self._token = CancelToken()
        self._lock = threading.Lock()
        #: Number of times :meth:`stop` has been called (observability/tests).
        self.stop_calls = 0

    # -- PlaybackSink -------------------------------------------------------
    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def stop(self) -> None:
        """Cancel playback immediately. Idempotent: safe to call repeatedly."""
        with self._lock:
            self.stop_calls += 1
            self._speaking = False
            self._token.cancel()

    # -- lifecycle helpers --------------------------------------------------
    @property
    def cancel_token(self) -> CancelToken:
        """Token the streaming generator polls; set by :meth:`stop`."""
        return self._token

    @property
    def cancelled(self) -> bool:
        return self._token.cancelled

    def start(self) -> None:
        """Mark playback started and re-arm the cancel token."""
        with self._lock:
            self._token.reset()
            self._speaking = True

    def finish(self) -> None:
        """Mark playback completed normally, *without* cancelling."""
        with self._lock:
            self._speaking = False

    def reset(self) -> None:
        """Return to the idle state without recording a stop."""
        with self._lock:
            self._speaking = False
            self._token.reset()
