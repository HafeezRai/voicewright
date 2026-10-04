"""The pipeline HTTP endpoints.

These drive the app's own process-wide pipeline (the one ``GET /metrics``
observes), so they also confirm the served pipeline is wired end to end.
"""

from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _wav_bytes(path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


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


def test_nlu_endpoint_classifies() -> None:
    response = client.post("/nlu", json={"text": "I want to book an appointment"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "book_appointment"
    assert body["backend"]
    assert body.get("scores")


def test_nlu_endpoint_rejects_empty_text() -> None:
    assert client.post("/nlu", json={"text": ""}).status_code == 422


def test_stt_endpoint_transcribes(speech_wav) -> None:
    response = client.post("/stt", content=_wav_bytes(speech_wav))
    assert response.status_code == 200
    body = response.json()
    assert body["text"].strip()
    assert body["language"] == "en"
    assert body["duration_s"] == 2.0


def test_stt_endpoint_rejects_an_empty_body() -> None:
    response = client.post("/stt", content=b"")
    assert response.status_code == 400


def test_stt_endpoint_rejects_non_wav_bytes() -> None:
    response = client.post("/stt", content=b"this is not a wav file")
    assert response.status_code == 400


def test_pipeline_endpoint_runs_the_whole_turn(speech_wav) -> None:
    response = client.post("/pipeline", content=_wav_bytes(speech_wav))
    assert response.status_code == 200
    body = response.json()

    assert body["transcript"].strip()
    assert body["intent"]["intent"]
    assert body["reply_text"].strip()
    assert body["audio"]["duration_s"] > 0.4
    assert set(body["timings_ms"]) == {"stt", "nlu", "tts", "total"}
    assert all(v >= 0 for v in body["timings_ms"].values())


def test_pipeline_endpoint_can_include_audio(speech_wav) -> None:
    response = client.post("/pipeline?include_audio=true", content=_wav_bytes(speech_wav))
    assert response.status_code == 200
    import base64

    wav = base64.b64decode(response.json()["audio"]["wav_base64"])
    assert wav[:4] == b"RIFF"
