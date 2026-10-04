"""GET /metrics — the endpoint infra/prometheus/prometheus.yml scrapes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.metrics import VAD_FRAMES_PROCESSED

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
