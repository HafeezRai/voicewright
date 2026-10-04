"""Conformance of ai-engine's playback sink with backend-core's barge-in.

``backend-core/app/pipeline/bargein.py`` declares ``PlaybackSink`` as an ABC and
``BargeInController`` consumes it. ai-engine implements it structurally and does
not import backend-core (separate service, separate dependency tree).

So this test proves the conformance *for real*, two ways:

1. statically -- every abstract method backend-core declares is present on
   :class:`TTSPlaybackSink`, and it registers as a virtual subclass; and
2. dynamically -- a genuine ``BargeInController`` from backend-core, fed frames,
   actually interrupts a genuine sherpa-onnx TTS stream.

backend-core's module is loaded **by file path**, so nothing in backend-core has
to change and its own ``app`` package is not shadowed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.tts import TTSPlaybackSink
from app.tts.base import CancelToken

REPO_ROOT = Path(__file__).resolve().parents[2]
BARGEIN_PATH = REPO_ROOT / "backend-core" / "app" / "pipeline" / "bargein.py"


def _load_backend_core_bargein():
    """Import backend-core's bargein module without touching sys.path.

    A plain ``sys.path`` insert would shadow ai-engine's own ``app`` package
    (both services have one), so the module is loaded from its file under a
    private name instead.
    """
    if not BARGEIN_PATH.is_file():
        pytest.skip(f"backend-core bargein module not found at {BARGEIN_PATH}")
    name = "backend_core_bargein_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BARGEIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sink_implements_every_abstract_method() -> None:
    mod = _load_backend_core_bargein()
    required = set(mod.PlaybackSink.__abstractmethods__)
    assert required == {"is_speaking", "stop"}, "backend-core's contract changed"
    assert required <= set(dir(TTSPlaybackSink))


def test_sink_registers_as_a_virtual_subclass() -> None:
    mod = _load_backend_core_bargein()
    mod.PlaybackSink.register(TTSPlaybackSink)
    assert issubclass(TTSPlaybackSink, mod.PlaybackSink)


def test_bargein_controller_stops_the_sink() -> None:
    mod = _load_backend_core_bargein()
    sink = TTSPlaybackSink()
    controller = mod.BargeInController(sink, frame_duration_ms=32, min_speech_ms=192)

    sink.start()
    assert controller.sink is sink

    events = []
    for _ in range(5):                     # silence: nothing happens
        events += controller.process(False)
    assert events == []
    assert sink.is_speaking

    for _ in range(6):                     # sustained speech: barge-in fires
        events += controller.process(True)

    kinds = [e.kind for e in events]
    assert mod.BargeInEventKind.BARGE_IN in kinds
    assert controller.interrupted is True
    assert sink.is_speaking is False, "barge-in did not stop playback"
    assert sink.cancelled is True, "barge-in did not set the cancel token"


def test_bargein_rearms_after_the_user_stops() -> None:
    mod = _load_backend_core_bargein()
    sink = TTSPlaybackSink()
    controller = mod.BargeInController(sink, frame_duration_ms=32, min_speech_ms=192)

    sink.start()
    for _ in range(6):
        controller.process(True)
    assert controller.interrupted

    events = []
    for _ in range(13):                    # 416 ms of silence > 400 ms default
        events += controller.process(False)
    assert mod.BargeInEventKind.SPEECH_END in [e.kind for e in events]
    assert controller.interrupted is False


def test_bargein_interrupts_a_real_tts_stream(tts_engine) -> None:
    """The whole point, end to end: a real interrupt cuts a real synthesis.

    Playback is driven by a background thread that consumes the TTS stream, while
    a genuine backend-core ``BargeInController`` (fed speech frames on the main
    thread) calls ``stop()`` -- exactly the threading the two services have.
    """
    import threading

    mod = _load_backend_core_bargein()
    text = "I can help you cancel your account. " * 4

    full_len = len(tts_engine.synthesize(text).audio)

    sink = TTSPlaybackSink()
    controller = mod.BargeInController(sink, frame_duration_ms=32, min_speech_ms=192)
    sink.start()

    produced = 0
    lock = threading.Lock()

    def _play() -> None:
        nonlocal produced
        for chunk in tts_engine.stream(text, chunk_samples=3200, cancel=sink.cancel_token):
            if not sink.is_speaking:
                break
            with lock:
                produced += len(chunk)

    player = threading.Thread(target=_play, daemon=True)
    player.start()

    # The user starts talking while the agent speaks.
    for _ in range(8):
        controller.process(True)
    player.join(timeout=15.0)

    assert controller.interrupted is True
    assert sink.stop_calls >= 1
    assert produced < full_len, (
        f"barge-in did not shorten playback (played {produced} of {full_len} samples)"
    )


def test_sink_uses_a_shared_cancel_token_type() -> None:
    """The token the sink exposes is the type the TTS layer polls."""
    sink = TTSPlaybackSink()
    assert isinstance(sink.cancel_token, CancelToken)
