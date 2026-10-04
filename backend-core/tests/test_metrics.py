"""GET /metrics — the endpoint infra/prometheus/prometheus.yml scrapes."""

from __future__ import annotations

import asyncio

import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.audio.vad import VAD
from app.main import app
from app.metrics import VAD_FRAMES_PROCESSED
from app.ws.manager import VoiceSession

client = TestClient(app)


def test_metrics_endpoint_returns_200() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_exposes_expected_names() -> None:
    # Create at least one labelled VAD sample so the series is rendered.
    VAD_FRAMES_PROCESSED.labels(backend="energy", speech="true").inc()

    body = client.get("/metrics").text
    for name in (
        "cloudpso_active_webrtc_sessions",
        "cloudpso_webrtc_sessions_total",
        "cloudpso_bargein_events_total",
        "cloudpso_vad_frames_processed_total",
        "cloudpso_signaling_connections",
    ):
        assert name in body, f"missing metric {name} in /metrics output"


def test_metrics_active_session_gauge_starts_at_zero() -> None:
    body = client.get("/metrics").text
    assert "cloudpso_active_webrtc_sessions 0.0" in body


class _AlwaysSpeechVAD(VAD):
    """Test VAD that classifies every frame as speech."""

    backend_name = "test"
    frame_samples = 512

    def is_speech(self, frame: np.ndarray) -> bool:  # noqa: ARG002 - always speech
        return True


def test_bargein_counter_increments_once_per_real_interrupt() -> None:
    """A real BARGE_IN bumps cloudpso_bargein_events_total exactly once.

    Drives the session audio pipeline (the same path the gateway's
    ``on_audio_frame`` callback uses) while the sink is "speaking": the counter
    must move by one on the interrupt and not again on subsequent speech frames.
    """

    async def _run() -> None:
        session = VoiceSession(
            "bargein-metric-test",
            vad=_AlwaysSpeechVAD(),
            frame_samples=512,
            frame_duration_ms=32,
            min_speech_ms=200,
        )
        try:
            before = REGISTRY.get_sample_value("cloudpso_bargein_events_total")
            session.sink.start()  # the agent is speaking, so barge-in can fire
            frame = np.zeros(512, dtype=np.float32)
            for _ in range(10):  # 10 * 32 ms = 320 ms > 200 ms min_speech
                session._on_audio_frame(frame)
            after_interrupt = REGISTRY.get_sample_value("cloudpso_bargein_events_total")
            assert after_interrupt == before + 1, "barge-in was not counted exactly once"

            # Once interrupted, further speech in the same turn must not recount.
            for _ in range(5):
                session._on_audio_frame(frame)
            assert REGISTRY.get_sample_value("cloudpso_bargein_events_total") == after_interrupt
        finally:
            await session.close()

    asyncio.run(_run())
