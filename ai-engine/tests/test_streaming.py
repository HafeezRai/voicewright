"""The streaming (SSE) turn endpoint.

``POST /pipeline`` returns one JSON body once the whole turn exists. Voice is
the wrong shape for that: the caller waits in silence for the entire turn.
``POST /pipeline/stream`` forwards the pipeline's event stream instead, so audio
starts after the first sentence.

The pipeline under test is the app's process-wide one with the deterministic
``FakeLLM`` that ``conftest`` installs, so these tests need no model and no
network -- but they do exercise the real STT and TTS.
"""

from __future__ import annotations

import base64
import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _frames(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    parsed: list[tuple[str, dict]] = []
    for block in response.text.split("\n\n"):
        event = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event:
            parsed.append((event, json.loads(data) if data else {}))
    return parsed


def _post(speech_wav, **params):
    with open(speech_wav, "rb") as handle:
        return client.post(
            "/pipeline/stream",
            content=handle.read(),
            headers={"Content-Type": "audio/wav"},
            params=params,
        )


def test_stream_emits_transcript_deltas_audio_then_done(speech_wav) -> None:
    response = _post(speech_wav)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _frames(response)
    kinds = [event for event, _ in events]

    # Ordering is the contract: the caller cannot render a reply before the
    # transcript, and must not be told the turn finished before the audio.
    assert kinds[0] == "transcript"
    assert "reply_delta" in kinds
    assert "audio" in kinds
    assert kinds[-1] == "done"
    assert kinds.index("transcript") < kinds.index("reply_delta")
    assert kinds.index("reply_delta") < kinds.index("audio")
    assert kinds.index("audio") < kinds.index("done")


def test_deltas_concatenate_to_the_final_reply(speech_wav) -> None:
    events = _frames(_post(speech_wav))
    deltas = [data["text"] for event, data in events if event == "reply_delta"]
    final = [data for event, data in events if event == "done"][-1]

    # DONE.text is *defined* as the concatenation of every REPLY_DELTA. That is
    # what lets a client render the reply as it arrives and still end up with
    # exactly the text the server considers final.
    assert "".join(deltas) == final["reply_text"]
    assert final["reply_text"]


def test_audio_frames_decode_to_pcm16(speech_wav) -> None:
    events = _frames(_post(speech_wav))
    audio = [data for event, data in events if event == "audio"]
    assert audio, "expected at least one audio frame"

    total_bytes = 0
    for frame in audio:
        assert frame["sample_rate"] > 0
        raw = base64.b64decode(frame["pcm16"])
        assert len(raw) % 2 == 0, "PCM16 must be an even number of bytes"
        total_bytes += len(raw)
    # A spoken reply, not a stub -- roughly a second of 16-bit audio at least.
    assert total_bytes > 8000


def test_terminal_event_carries_the_model_and_grounding_count(speech_wav) -> None:
    events = _frames(_post(speech_wav))
    _, data = events[-1]
    assert data["model"], "the terminal frame names the model that answered"
    # No agent_id, so there is no knowledge base and nothing was retrieved.
    assert data["knowledge_chunks"] == 0


def test_unknown_agent_is_404_before_any_streaming(speech_wav) -> None:
    response = _post(speech_wav, agent_id="does-not-exist")
    # Must be a plain 404, not an `error` frame: a client has to be able to
    # tell "no such agent" from "the turn failed halfway through".
    assert response.status_code == 404
    assert "text/event-stream" not in response.headers.get("content-type", "")


def test_empty_body_is_rejected(speech_wav) -> None:
    response = client.post("/pipeline/stream", content=b"", headers={"Content-Type": "audio/wav"})
    assert response.status_code == 400
