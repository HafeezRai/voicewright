"""``GET /metrics`` exposes the ai-engine Prometheus series."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

EXPECTED = {
    # STT
    "voicewright_ai_stt_requests_total",
    "voicewright_ai_stt_latency_seconds",
    "voicewright_ai_stt_audio_seconds",
    # LLM
    "voicewright_ai_llm_requests_total",
    "voicewright_ai_llm_latency_seconds",
    "voicewright_ai_llm_first_token_seconds",
    "voicewright_ai_llm_tokens_total",
    "voicewright_ai_llm_cancelled_total",
    # TTS
    "voicewright_ai_tts_requests_total",
    "voicewright_ai_tts_latency_seconds",
    "voicewright_ai_tts_audio_seconds",
    "voicewright_ai_tts_cancelled_total",
    # pipeline
    "voicewright_ai_pipeline_requests_total",
    "voicewright_ai_pipeline_latency_seconds",
    # gRPC
    "voicewright_ai_grpc_audio_frames_total",
    "voicewright_ai_grpc_events_total",
}

#: The retired intent stage. A dashboard still querying these shows a flat zero
#: with no error at all, so their absence is asserted rather than assumed.
RETIRED = {
    "voicewright_ai_nlu_requests_total",
    "voicewright_ai_nlu_latency_seconds",
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


def test_the_nlu_series_are_gone() -> None:
    body = client.get("/metrics").text
    present = {name for name in RETIRED if name in body}
    assert not present, f"retired series still exported: {sorted(present)}"


def test_every_ai_engine_metric_is_namespaced() -> None:
    """Mirror backend-core's convention: this service owns ``voicewright_ai_*``."""
    body = client.get("/metrics").text
    voicewright_series = {
        line.split("{", 1)[0].split(" ", 1)[0]
        for line in body.splitlines()
        if line.startswith("voicewright_")
    }
    assert voicewright_series, "no voicewright_ series exported at all"
    assert all(name.startswith("voicewright_ai_") for name in voicewright_series)


def test_pipeline_requests_are_labelled_by_llm_provider() -> None:
    """The old label was ``nlu_backend``; a renamed label is a silent dashboard
    break, so the new one is pinned."""
    body = client.get("/metrics").text
    pipeline_lines = [
        line for line in body.splitlines() if line.startswith("voicewright_ai_pipeline_requests_total")
    ]
    assert pipeline_lines, "no pipeline request samples exported"
    assert any("llm_provider=" in line for line in pipeline_lines)
    assert not any("nlu_backend=" in line for line in pipeline_lines)


def test_first_token_histogram_has_coarse_buckets() -> None:
    """The shared buckets top out at 10s; a cold model load must not read +Inf."""
    body = client.get("/metrics").text
    bucket_lines = [
        line
        for line in body.splitlines()
        if line.startswith("voicewright_ai_llm_first_token_seconds_bucket")
    ]
    assert bucket_lines, "first-token histogram is not exported"
    assert any('le="60.0"' in line for line in bucket_lines), (
        "first-token buckets do not reach a cold-start timescale"
    )


# ---------------------------------------------------------------------------
# The per-stage endpoints must be visible in Prometheus
#
# /stt, /llm and /tts are thin wrappers, but they are still requests served by
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


def test_llm_endpoint_records_metrics() -> None:
    before, after = _before_after(
        "voicewright_ai_llm_requests_total",
        lambda: client.post("/llm", json={"text": "I want to book an appointment"}),
    )
    assert after > before


def test_llm_endpoint_records_token_usage() -> None:
    """The fake reports usage, so the counter must move."""
    before, after = _before_after(
        "voicewright_ai_llm_tokens_total",
        lambda: client.post("/llm", json={"text": "hello there"}),
    )
    assert after > before


def test_tts_endpoint_records_metrics() -> None:
    before, after = _before_after(
        "voicewright_ai_tts_requests_total",
        lambda: client.post("/tts", json={"text": "hello there"}),
    )
    assert after > before


def test_stt_endpoint_records_metrics(speech_wav) -> None:
    body = speech_wav.read_bytes()
    before, after = _before_after(
        "voicewright_ai_stt_requests_total",
        lambda: client.post("/stt", content=body),
    )
    assert after > before


def test_pipeline_endpoint_records_metrics(speech_wav) -> None:
    body = speech_wav.read_bytes()
    before, after = _before_after(
        "voicewright_ai_pipeline_requests_total",
        lambda: client.post("/pipeline", content=body),
    )
    assert after > before
