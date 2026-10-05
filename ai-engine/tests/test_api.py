"""The pipeline HTTP endpoints.

These drive the app's own process-wide pipeline (the one ``GET /metrics``
observes), so they also confirm the served pipeline is wired end to end. The
LLM in that pipeline is the deterministic fake installed by ``conftest`` -- the
real wire protocol has its own live test.
"""

from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient

from app import main as app_main
from app.main import app

client = TestClient(app)


def _wav_bytes(path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _system_messages(messages) -> list[str]:
    return [m.content for m in messages if m.role == "system"]


# ---------------------------------------------------------------------------
# per-stage endpoints
# ---------------------------------------------------------------------------


def test_tts_endpoint_returns_a_wav() -> None:
    response = client.post("/tts", json={"text": "I can help you with that."})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[:4] == b"RIFF"
    assert int(response.headers["X-Sample-Rate"]) == 16000
    assert float(response.headers["X-Duration-Seconds"]) > 0.4

    with wave.open(io.BytesIO(response.content), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getnframes() > 8000


def test_llm_endpoint_generates_a_reply() -> None:
    response = client.post("/llm", json={"text": "I want to book an appointment"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply_text"].strip()
    assert body["model"]
    assert body["provider"]
    assert body["finish_reason"]
    assert body["usage"]["total_tokens"] > 0
    assert body["latency_ms"] >= 0


def test_llm_endpoint_rejects_empty_text() -> None:
    assert client.post("/llm", json={"text": ""}).status_code == 422


def test_the_nlu_endpoint_is_gone() -> None:
    """The path returned intents; it now returns nothing, on purpose.

    Keeping ``/nlu`` while it answers with prose would be a lie for every future
    reader, so it was renamed rather than re-pointed.
    """
    assert client.post("/nlu", json={"text": "book an appointment"}).status_code == 404


def test_stt_endpoint_transcribes(speech_wav) -> None:
    response = client.post("/stt", content=_wav_bytes(speech_wav))
    assert response.status_code == 200
    body = response.json()
    assert body["text"].strip()
    assert body["language"] == "en"
    assert body["duration_s"] == 2.0


def test_stt_endpoint_rejects_an_empty_body() -> None:
    assert client.post("/stt", content=b"").status_code == 400


def test_stt_endpoint_rejects_non_wav_bytes() -> None:
    assert client.post("/stt", content=b"this is not a wav file").status_code == 400


# ---------------------------------------------------------------------------
# the whole turn
# ---------------------------------------------------------------------------


def test_pipeline_endpoint_runs_the_whole_turn(speech_wav) -> None:
    response = client.post("/pipeline", content=_wav_bytes(speech_wav))
    assert response.status_code == 200
    body = response.json()

    assert body["transcript"].strip()
    assert body["reply_text"].strip()
    assert body["model"] and body["provider"]
    assert body["audio"]["duration_s"] > 0.4
    assert set(body["timings_ms"]) == {"stt", "llm", "tts", "total"}
    assert all(v >= 0 for v in body["timings_ms"].values())
    # No intent anywhere in the response -- the concept is gone, not renamed.
    assert "intent" not in body


def test_pipeline_endpoint_can_include_audio(speech_wav) -> None:
    response = client.post("/pipeline?include_audio=true", content=_wav_bytes(speech_wav))
    assert response.status_code == 200
    import base64

    wav = base64.b64decode(response.json()["audio"]["wav_base64"])
    assert wav[:4] == b"RIFF"


def test_pipeline_endpoint_applies_a_system_prompt_override(speech_wav) -> None:
    """The per-request persona must reach the model, not just be accepted."""
    body = _wav_bytes(speech_wav)

    client.post("/pipeline?system_prompt=You%20are%20a%20pirate.", content=body)
    pirate = _system_messages(app_main.pipeline.llm.last_messages)

    client.post("/pipeline?system_prompt=You%20are%20a%20butler.", content=body)
    butler = _system_messages(app_main.pipeline.llm.last_messages)

    assert pirate == ["You are a pirate."]
    assert butler == ["You are a butler."]
    assert pirate != butler


def test_pipeline_endpoint_uses_the_configured_prompt_by_default(speech_wav) -> None:
    client.post("/pipeline", content=_wav_bytes(speech_wav))
    systems = _system_messages(app_main.pipeline.llm.last_messages)
    assert systems, "a configured system prompt must be sent even with no override"


def test_health_is_unchanged() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ai-engine"}
