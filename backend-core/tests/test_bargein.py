"""Barge-in state machine — driven entirely by synthetic VAD results.

No audio device, no event loop, no network: exactly the unit-testability the
phase boundary requires.
"""

from __future__ import annotations

from app.pipeline.bargein import (
    BargeInController,
    BargeInEventKind,
    StubPlaybackSink,
)

FRAME_MS = 32      # 512 samples @ 16 kHz
MIN_SPEECH_MS = 200   # -> 7 frames
MIN_SILENCE_MS = 400  # -> 13 frames


def _controller(**kwargs) -> tuple[BargeInController, StubPlaybackSink]:
    sink = StubPlaybackSink()
    controller = BargeInController(
        sink,
        frame_duration_ms=FRAME_MS,
        min_speech_ms=kwargs.pop("min_speech_ms", MIN_SPEECH_MS),
        min_silence_ms=kwargs.pop("min_silence_ms", MIN_SILENCE_MS),
        **kwargs,
    )
    return controller, sink


def test_no_barge_in_while_agent_is_silent() -> None:
    controller, sink = _controller()
    # Agent not speaking: user speech must never produce an interrupt.
    for _ in range(50):
        assert controller.process(True) == []
    assert sink.stop_calls == 0
    assert controller.interrupted is False


def test_barge_in_fires_only_after_min_speech() -> None:
    controller, sink = _controller()
    sink.start()

    events = [e for _ in range(6) for e in controller.process(True)]
    assert events == []  # 6 frames = 192 ms < 200 ms threshold
    assert sink.stop_calls == 0

    events = controller.process(True)  # 7th frame crosses the threshold
    assert [e.kind for e in events] == [BargeInEventKind.BARGE_IN]
    assert events[0].duration_ms == 7 * FRAME_MS
    assert sink.stop_calls == 1
    assert sink.is_speaking is False


def test_short_burst_does_not_trigger_barge_in() -> None:
    """A cough/click shorter than min_speech_ms must be ignored."""
    controller, sink = _controller()
    sink.start()
    for _ in range(3):  # 96 ms burst
        assert controller.process(True) == []
    for _ in range(20):  # then quiet again
        assert controller.process(False) == []
    assert sink.stop_calls == 0


def test_speech_run_must_be_continuous() -> None:
    """A silence gap resets the speech run, so two short bursts do not add up."""
    controller, sink = _controller()
    sink.start()
    for _ in range(5):
        controller.process(True)
    controller.process(False)  # gap resets the run
    events = [e for _ in range(5) for e in controller.process(True)]
    assert events == []
    assert sink.stop_calls == 0


def test_speech_end_requires_sustained_silence_then_rearms() -> None:
    controller, sink = _controller()
    sink.start()
    for _ in range(7):
        controller.process(True)
    assert controller.interrupted is True

    # One silent frame is not enough (hysteresis).
    assert controller.process(False) == []

    events: list = []
    for _ in range(12):
        events += controller.process(False)
    assert [e.kind for e in events] == [BargeInEventKind.SPEECH_END]
    assert controller.interrupted is False

    # Re-armed: a fresh sustained utterance triggers a second barge-in.
    sink.start()
    fired = [e for _ in range(7) for e in controller.process(True)]
    assert [e.kind for e in fired] == [BargeInEventKind.BARGE_IN]
    assert sink.stop_calls == 2


def test_barge_in_fires_once_per_turn() -> None:
    controller, sink = _controller()
    sink.start()
    fired = 0
    for _ in range(40):
        fired += sum(1 for e in controller.process(True) if e.kind is BargeInEventKind.BARGE_IN)
    assert fired == 1
    assert sink.stop_calls == 1


def test_disabled_controller_never_fires() -> None:
    controller, sink = _controller(enabled=False)
    sink.start()
    for _ in range(100):
        assert controller.process(True) == []
    assert sink.stop_calls == 0


def test_reset_clears_state() -> None:
    controller, sink = _controller()
    sink.start()
    for _ in range(4):
        controller.process(True)
    controller.reset()
    # After reset the run counter is zero, so 6 frames is again below threshold.
    assert [e for _ in range(6) for e in controller.process(True)] == []


def test_stop_is_idempotent() -> None:
    sink = StubPlaybackSink()
    sink.start()
    sink.stop()
    sink.stop()
    assert sink.stop_calls == 2
    assert sink.is_speaking is False
