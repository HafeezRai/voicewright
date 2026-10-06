"""The knowledge-base HTTP surface, and the pipeline's use of it.

Document *ingestion* needs the real MiniLM encoder, so the tests that store a
document skip with an actionable message when it is not cached -- exactly like
the STT/TTS engine tests. The tests that do not embed (upload validation, org
scoping, the unavailable-encoder fast paths) run everywhere.

The pipeline tests drive the served app's own pipeline with the deterministic
``FakeLLM`` that ``conftest`` installs, and inspect the messages that reached
it: that is how grounding is proven without Ollama. The real two-model proof
lives in ``test_knowledge_live.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.agents.store import DEFAULT_ORG_ID, AgentStore
from app.knowledge.embeddings import EmbeddingUnavailableError
from app.main import app
from app.stt.base import TranscriptionResult

client = TestClient(app)

ORG_B = "org_kb_api_b"
app_main.agent_store.create_org(ORG_B, "kb-api-tenant-b")

FETCH_HINT = "run `ai-engine/scripts/fetch_models.sh` to download model weights"

#: A distinctive fact no base model would know, used to spot grounding.
POLICY = "Saturday appointments are only for existing patients, never new ones."
#: A question the policy chunk actually answers. Grounding is *supposed* to be
#: silent when the utterance is unrelated (that is the threshold doing its job),
#: so a test of the injection seam has to ask a question that is on topic. The
#: ``speech_16k`` fixture transcribes to "percent global wealth" -- unrelated to
#: any policy -- so the audio->text step is scripted here; retrieval, the prompt
#: splice and the LLM call are all real.
QUESTION = "Are Saturday appointments available for new patients?"


class _ScriptedSTT:
    """An STT engine that returns a fixed transcript, ignoring the audio.

    Only the audio->text stage is replaced: it is orthogonal to grounding, and
    pinning it is what makes the assertion below about the *knowledge base*
    rather than about Whisper's opinion of a 2-second fixture.
    """

    name = "scripted"

    def __init__(self, text: str):
        self._text = text

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def transcribe(self, audio, sample_rate=None) -> TranscriptionResult:
        return TranscriptionResult(text=self._text, language="en", duration_s=2.0, backend="scripted")


@pytest.fixture(scope="module")
def embedder_ready():
    """Skip the calling test unless the MiniLM encoder can actually load."""
    try:
        app_main.embedder.load()
    except Exception as exc:  # pragma: no cover - offline dev box
        pytest.skip(f"MiniLM embedding model unavailable: {exc}; {FETCH_HINT}")


def _make_agent(name: str, *, org: str | None = None, system_prompt: str = "") -> dict:
    headers = {"X-Org-Id": org} if org else None
    response = client.post(
        "/agents",
        json={"name": name, "system_prompt": system_prompt},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def _add_text_document(agent_id: str, text: str, *, org: str | None = None) -> dict:
    headers = {"X-Org-Id": org} if org else None
    response = client.post(
        f"/agents/{agent_id}/documents", json={"title": "Policy", "text": text}, headers=headers
    )
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------


def test_json_document_is_chunked_and_listed(embedder_ready) -> None:
    agent = _make_agent("KB json")
    document = _add_text_document(agent["id"], POLICY)
    assert document["chunk_count"] >= 1
    assert document["char_count"] == len(POLICY.strip())
    assert document["source"] == "text"

    listed = client.get(f"/agents/{agent['id']}/documents").json()
    assert [row["id"] for row in listed] == [document["id"]]


def test_markdown_upload_is_accepted(embedder_ready) -> None:
    agent = _make_agent("KB upload")
    response = client.post(
        f"/agents/{agent['id']}/documents",
        files={"file": ("policy.md", b"# Hours\n\nWe open at nine.", "text/markdown")},
        data={"title": "Hours"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "upload"
    assert body["title"] == "Hours"
    assert body["chunk_count"] >= 1


def test_non_text_upload_is_rejected() -> None:
    agent = _make_agent("KB bad upload")
    response = client.post(
        f"/agents/{agent['id']}/documents",
        files={"file": ("diagram.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 422


def test_oversized_document_is_rejected(monkeypatch) -> None:
    agent = _make_agent("KB oversized")
    monkeypatch.setattr(app_main.settings, "knowledge_max_document_chars", 10)
    response = client.post(
        f"/agents/{agent['id']}/documents", json={"title": "Big", "text": "x" * 50}
    )
    assert response.status_code == 413


def test_empty_document_is_rejected() -> None:
    agent = _make_agent("KB empty")
    # A whitespace-only body fails the pydantic min_length on a blank string, or
    # reaches the ingest guard; either way it must not be stored.
    assert client.post(
        f"/agents/{agent['id']}/documents", json={"title": "Blank", "text": "   "}
    ).status_code == 422


def test_documents_require_an_existing_agent() -> None:
    assert client.get("/agents/nope/documents").status_code == 404
    assert client.post("/agents/nope/documents", json={"text": "x"}).status_code == 404


def test_delete_document_then_it_is_gone(embedder_ready) -> None:
    agent = _make_agent("KB delete")
    document = _add_text_document(agent["id"], POLICY)
    assert client.delete(f"/documents/{document['id']}").status_code == 204
    assert client.delete(f"/documents/{document['id']}").status_code == 404
    assert client.get(f"/agents/{agent['id']}/documents").json() == []


# ---------------------------------------------------------------------------
# tenant scoping of documents
# ---------------------------------------------------------------------------


def test_org_cannot_read_or_delete_another_orgs_document(embedder_ready) -> None:
    b_agent = _make_agent("KB B agent", org=ORG_B)
    b_document = _add_text_document(b_agent["id"], POLICY, org=ORG_B)

    # Org A (default) must not see the agent's documents, nor delete one.
    assert client.get(f"/agents/{b_agent['id']}/documents").status_code == 404
    assert client.delete(f"/documents/{b_document['id']}").status_code == 404
    # It survives for its real owner.
    assert client.get(
        f"/agents/{b_agent['id']}/documents", headers={"X-Org-Id": ORG_B}
    ).json()[0]["id"] == b_document["id"]


# ---------------------------------------------------------------------------
# the pipeline seam: grounding reaches the model's system message
# ---------------------------------------------------------------------------


def _system_messages() -> list[str]:
    return [m.content for m in app_main.pipeline.llm.last_messages if m.role == "system"]


def _wav(path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def test_pipeline_injects_agent_knowledge_and_prompt(
    speech_wav, embedder_ready, monkeypatch
) -> None:
    agent = _make_agent("KB grounded", system_prompt="You are a clinic receptionist.")
    _add_text_document(agent["id"], POLICY)
    monkeypatch.setattr(app_main.pipeline, "stt", _ScriptedSTT(QUESTION))

    response = client.post(f"/pipeline?agent_id={agent['id']}", content=_wav(speech_wav))
    assert response.status_code == 200, response.text
    assert response.json()["knowledge_chunks"] >= 1

    systems = _system_messages()
    assert any("clinic receptionist" in s for s in systems), systems
    assert any("existing patients" in s for s in systems), systems


def test_pipeline_leaves_an_off_topic_utterance_ungrounded(
    speech_wav, embedder_ready
) -> None:
    """The threshold is the point: an unrelated question injects no knowledge.

    Uses the *real* transcriber on the fixture ("percent global wealth"), which
    is on no policy the agent holds. The agent has documents, so retrieval runs;
    nothing clears the bar, so nothing is spliced -- a knowledge base that
    injected its best three chunks regardless would make this answer worse.
    """
    agent = _make_agent("KB off-topic", system_prompt="You are a clinic receptionist.")
    _add_text_document(agent["id"], POLICY)

    response = client.post(f"/pipeline?agent_id={agent['id']}", content=_wav(speech_wav))
    assert response.status_code == 200, response.text
    assert response.json()["knowledge_chunks"] == 0
    assert _system_messages() == ["You are a clinic receptionist."]


def test_pipeline_without_documents_injects_nothing(speech_wav) -> None:
    agent = _make_agent("KB bare", system_prompt="You are a clinic receptionist.")
    response = client.post(f"/pipeline?agent_id={agent['id']}", content=_wav(speech_wav))
    assert response.status_code == 200, response.text
    assert response.json()["knowledge_chunks"] == 0
    systems = _system_messages()
    assert systems == ["You are a clinic receptionist."], systems


def test_explicit_system_prompt_overrides_the_agents(speech_wav) -> None:
    agent = _make_agent("KB override", system_prompt="AGENT PROMPT")
    response = client.post(
        f"/pipeline?agent_id={agent['id']}&system_prompt=OVERRIDE", content=_wav(speech_wav)
    )
    assert response.status_code == 200, response.text
    assert _system_messages() == ["OVERRIDE"]


def test_unknown_agent_on_the_pipeline_is_404(speech_wav) -> None:
    response = client.post("/pipeline?agent_id=does_not_exist", content=_wav(speech_wav))
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# degradation: retrieval must never break a turn
# ---------------------------------------------------------------------------


def test_knowledge_for_skips_encoding_with_no_documents(monkeypatch) -> None:
    store = AgentStore(app_main.settings.agent_db_path)
    agent = store.create_agent(DEFAULT_ORG_ID, {"name": "no docs"})

    def _explode(_text):
        raise AssertionError("the encoder must not be touched when there are no chunks")

    monkeypatch.setattr(app_main.embedder, "encode_one", _explode)
    assert app_main._knowledge_for(DEFAULT_ORG_ID, agent["id"], "any question") == []


def test_knowledge_for_degrades_when_the_encoder_is_unavailable(monkeypatch) -> None:
    store = AgentStore(app_main.settings.agent_db_path)
    agent = store.create_agent(DEFAULT_ORG_ID, {"name": "has docs"})
    store.add_document(
        DEFAULT_ORG_ID,
        agent["id"],
        title="d",
        source="text",
        text="something",
        chunks=[("something", np.ones(2, dtype=np.float32).tobytes(), 2)],
    )

    def _unavailable(_text):
        raise EmbeddingUnavailableError("model missing")

    monkeypatch.setattr(app_main.embedder, "encode_one", _unavailable)
    # No exception, no grounding: the turn proceeds, just ungrounded.
    assert app_main._knowledge_for(DEFAULT_ORG_ID, agent["id"], "question") == []
