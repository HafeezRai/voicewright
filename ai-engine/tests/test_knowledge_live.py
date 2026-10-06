"""Live proof that the knowledge base changes what a real model answers.

This is the headline claim of step 2B, and the only test that proves it end to
end: a real MiniLM embedder, the real retriever, a real ``llama3:8b`` through a
real ``POST /pipeline``, and the question itself spoken by the service's own TTS
and transcribed by its own Whisper -- so the whole path is

    TTS -> WAV -> POST /pipeline -> STT -> retrieve -> LLM

with nothing faked. Two agents share one persona that knows nothing about the
clinic's Saturday policy; one has the policy as a document, one has none. If the
answers differ, and only the grounded one states the policy, the document -- not
the persona -- produced it.

It skips, cleanly and with an actionable message, when Ollama or the model is
absent, or when MiniLM is not cached, so a bare checkout stays green. A cold
``llama3:8b`` is a 4.66 GB load, hence the generous timeout.
"""

from __future__ import annotations

import re

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.config import Settings
from app.llm.factory import resolve_base_url
from app.llm.openai_compat import OpenAICompatLLM
from app.main import app

client = TestClient(app)

#: A cold `llama3:8b`, plus the TTS/STT models, on the first call.
_LIVE_TIMEOUT_S = 300.0

#: The question, asked through /pipeline as audio.
QUESTION = "Are Saturday appointments available for new patients?"
#: The fact. It appears only in the document -- the persona below never mentions
#: it -- so an answer that states it can only have come from retrieval.
POLICY = (
    "Saturday appointments are only for existing patients. "
    "We never book a new patient on a Saturday."
)
#: Deliberately silent on the policy, so a Saturday answer cannot be persona.
PERSONA = (
    "You are a clinic receptionist answering the phone. "
    "Keep every reply to one short sentence."
)


#: A denial cue, word-bounded so "know"/"not" inside other words do not count.
_NEGATION = re.compile(r"\b(no|not|never|only|unable|cannot|can't|don't|doesn't|do not)\b")


def _denies_new_patient_saturdays(answer: str) -> bool:
    """Whether the reply states the policy: Saturday is closed to new patients.

    Checks the *claim*, not the wording, because the model paraphrases. The
    ungrounded agent tends to hallucinate the opposite ("Yes, we do have
    availability on Saturdays for new patients"), which is exactly the failure
    the knowledge base exists to prevent.
    """
    lowered = answer.lower()
    mentions_patients = "new patient" in lowered or "existing" in lowered
    return mentions_patients and _NEGATION.search(lowered) is not None


def _ollama_has_model(settings: Settings) -> bool:
    root = settings.llm_base_url or "http://127.0.0.1:11434/v1"
    tags_url = root.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        response = httpx.get(tags_url, timeout=2.0)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    return settings.llm_model in {m.get("name") for m in response.json().get("models", [])}


@pytest.fixture(scope="module")
def live_pipeline(settings) -> None:
    """Point the served pipeline's LLM at a real, cold-tolerant Ollama model.

    ``conftest`` installs the deterministic ``FakeLLM`` on the served pipeline so
    the HTTP suite needs no Ollama. This fixture swaps in the real client for the
    duration and puts the fake back afterwards, so grounding is proven against
    real generation without disturbing any other test.
    """
    if not _ollama_has_model(settings):
        pytest.skip(
            f"model {settings.llm_model!r} is not served by Ollama at "
            f"{settings.llm_base_url or 'http://127.0.0.1:11434'}; "
            f"run `ollama serve` and `ollama pull {settings.llm_model}`"
        )
    try:
        app_main.embedder.load()
    except Exception as exc:  # pragma: no cover - offline dev box
        pytest.skip(
            f"MiniLM embedding model unavailable: {exc}; "
            "run `ai-engine/scripts/fetch_models.sh`"
        )

    engine = OpenAICompatLLM(
        resolve_base_url(settings),
        settings.llm_model,
        provider=settings.llm_provider,
        timeout_s=_LIVE_TIMEOUT_S,
        connect_timeout_s=5.0,
        stream=settings.llm_stream,
        extra_body=settings.llm_extra_body,
    )
    previous = app_main.pipeline.llm
    app_main.pipeline.llm = engine
    try:
        yield
    finally:
        app_main.pipeline.llm = previous
        engine.close()


def _make_agent(name: str, system_prompt: str) -> dict:
    response = client.post("/agents", json={"name": name, "system_prompt": system_prompt})
    assert response.status_code == 201, response.text
    return response.json()


def _add_document(agent_id: str, text: str) -> None:
    response = client.post(
        f"/agents/{agent_id}/documents", json={"title": "Clinic policy", "text": text}
    )
    assert response.status_code == 201, response.text


def _question_wav() -> bytes:
    """Speak the question with the service's own TTS, so /pipeline hears it.

    This keeps the test on the real audio path: the WAV is produced by Sherpa,
    decoded by `/pipeline`, and transcribed by Whisper -- not handed over as text.
    """
    response = client.post("/tts", json={"text": QUESTION})
    assert response.status_code == 200, response.text
    return response.content


def test_the_knowledge_base_changes_a_real_answer(live_pipeline, capsys) -> None:
    grounded = _make_agent("Grounded clinic", PERSONA)
    _add_document(grounded["id"], POLICY)
    ungrounded = _make_agent("Ungrounded clinic", PERSONA)

    wav = _question_wav()

    grounded_turn = client.post(f"/pipeline?agent_id={grounded['id']}", content=wav)
    bare_turn = client.post(f"/pipeline?agent_id={ungrounded['id']}", content=wav)
    assert grounded_turn.status_code == 200, grounded_turn.text
    assert bare_turn.status_code == 200, bare_turn.text

    grounded_body = grounded_turn.json()
    bare_body = bare_turn.json()

    grounded_answer = grounded_body["reply_text"]
    bare_answer = bare_body["reply_text"]
    # Surface the two generations in the report, not just the assertions.
    with capsys.disabled():
        print("\n--- knowledge-base proof (llama3:8b, live) ---")
        print(f"question            : {grounded_body['transcript']!r}")
        print(f"grounded  chunks    : {grounded_body['knowledge_chunks']}")
        print(f"grounded  answer    : {grounded_answer!r}")
        print(f"ungrounded chunks   : {bare_body['knowledge_chunks']}")
        print(f"ungrounded answer   : {bare_answer!r}")

    # Retrieval ran for one and was skipped for the other: the deterministic half.
    assert grounded_body["knowledge_chunks"] >= 1
    assert bare_body["knowledge_chunks"] == 0

    # Both are real generations from the same model.
    assert grounded_body["model"] == bare_body["model"] == app_main.settings.llm_model
    assert grounded_answer.strip() and bare_answer.strip()

    # The discriminator: the grounded reply states the policy -- that Saturday is
    # closed to *new* patients -- while the bare agent, never told it, does not.
    assert _denies_new_patient_saturdays(grounded_answer), grounded_answer
    assert not _denies_new_patient_saturdays(bare_answer), bare_answer
