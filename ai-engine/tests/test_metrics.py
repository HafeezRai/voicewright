"""``GET /metrics`` exposes the ai-engine Prometheus series."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

EXPECTED = {
    # STT
    "cloudpso_ai_stt_requests_total",
    "cloudpso_ai_stt_latency_seconds",
    "cloudpso_ai_stt_audio_seconds",
    # NLU
    "cloudpso_ai_nlu_requests_total",
    "cloudpso_ai_nlu_latency_seconds",
    # TTS
    "cloudpso_ai_tts_requests_total",
    "cloudpso_ai_tts_latency_seconds",
    "cloudpso_ai_tts_audio_seconds",
    "cloudpso_ai_tts_cancelled_total",
    # pipeline
    "cloudpso_ai_pipeline_requests_total",
    "cloudpso_ai_pipeline_latency_seconds",
    # gRPC
    "cloudpso_ai_grpc_audio_frames_total",
    "cloudpso_ai_grpc_events_total",
}


def test_metrics_endpoint_returns_prometheus_text() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP" in response.text and "# TYPE" in response.text


def test_metrics_endpoint_advertises_the_expected_series() -> None:
    body = client.get("/metrics").text
    missing = {name for name in EXPECTED if name not in body}
    assert not missing, f"missing metrics: {sorted(missing)}"


def test_every_ai_engine_metric_is_namespaced() -> None:
    """Mirror backend-core's convention: this service owns ``cloudpso_ai_*``."""
    body = client.get("/metrics").text
    cloudpso_series = {
        line.split("{", 1)[0].split(" ", 1)[0]
        for line in body.splitlines()
        if line.startswith("cloudpso_")
    }
    assert cloudpso_series, "no cloudpso_ series exported at all"
    assert all(name.startswith("cloudpso_ai_") for name in cloudpso_series)


# ---------------------------------------------------------------------------
# The per-stage endpoints must be visible in Prometheus
#
# /stt, /nlu and /tts are thin wrappers, but they are still requests served by
# this process. If they called the engines directly they would be invisible in
# metrics while the identical call inside a turn was counted -- so this pins
# that they go through the pipeline.
# ---------------------------------------------------------------------------


def _total(body: str, series: str) -> float:
    """Sum every sample of a counter family in the exposition text."""
    total = 0.0
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith(series):
            continue
        total += float(line.rsplit(" ", 1)[1])
    return total


def _before_after(series: str, call) -> tuple[float, float]:
    before = _total(client.get("/metrics").text, series)
    call()
    after = _total(client.get("/metrics").text, series)
    return before, after


def test_nlu_endpoint_records_metrics() -> None:
    before, after = _before_after(
        "cloudpso_ai_nlu_requests_total",
        lambda: client.post("/nlu", json={"text": "I want to book an appointment"}),
    )
    assert after > before


def test_tts_endpoint_records_metrics() -> None:
    before, after = _before_after(
        "cloudpso_ai_tts_requests_total",
        lambda: client.post("/tts", json={"text": "hello there"}),
    )
    assert after > before


def test_stt_endpoint_records_metrics(speech_wav) -> None:
    body = speech_wav.read_bytes()
    before, after = _before_after(
        "cloudpso_ai_stt_requests_total",
        lambda: client.post("/stt", content=body),
    )
    assert after > before


def test_pipeline_endpoint_records_metrics(speech_wav) -> None:
    body = speech_wav.read_bytes()
    before, after = _before_after(
        "cloudpso_ai_pipeline_requests_total",
        lambda: client.post("/pipeline", content=body),
    )
    assert after > before
