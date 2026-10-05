"""The only test that proves the real wire protocol.

Every other LLM test uses ``httpx.MockTransport``, so they prove *our* parsing
against *our* idea of the protocol. This one talks to a real Ollama on
``127.0.0.1:11434`` and proves the idea is right: the SSE framing, the
``stream_options.include_usage`` usage chunk and the ``/v1`` path all behave as
the client expects.

It skips -- cleanly and with an actionable message -- when Ollama is not
running or the model is not pulled, so the suite still passes on a bare
checkout and in CI. A cold model means a multi-gigabyte load on the first call,
which is why the timeout is generous.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.llm.base import ChatMessage, LLMUnavailableError
from app.llm.factory import build_llm, resolve_base_url
from app.llm.openai_compat import OpenAICompatLLM

#: A cold `llama3:8b` is a 4.66 GB load on the first request.
_LIVE_TIMEOUT_S = 180.0

PROMPT = "Reply with exactly the word: pong"


def _ollama_tags(settings: Settings) -> dict | None:
    """Return Ollama's ``/api/tags`` body, or ``None`` if it is not running."""
    root = settings.llm_base_url or "http://127.0.0.1:11434/v1"
    tags_url = root.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        response = httpx.get(tags_url, timeout=2.0)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:  # pragma: no cover - a non-Ollama server on that port
        return None


@pytest.fixture(scope="module")
def live_llm(settings):
    tags = _ollama_tags(settings)
    if tags is None:
        pytest.skip(
            f"no Ollama serving at {settings.llm_base_url or 'http://127.0.0.1:11434'}; "
            "start `ollama serve` to run the live wire-protocol test"
        )
    names = {m.get("name") for m in tags.get("models", [])}
    if settings.llm_model not in names:
        pytest.skip(
            f"model {settings.llm_model!r} is not pulled; "
            f"run `ollama pull {settings.llm_model}` (available: {sorted(names)})"
        )

    engine = build_llm(settings)
    engine.close()
    # Rebuild with a timeout that tolerates a cold multi-gigabyte model load.
    engine = OpenAICompatLLM(
        resolve_base_url(settings),
        settings.llm_model,
        provider=settings.llm_provider,
        timeout_s=_LIVE_TIMEOUT_S,
        connect_timeout_s=5.0,
        stream=settings.llm_stream,
        extra_body=settings.llm_extra_body,
    )
    yield engine
    engine.close()


def test_generate_against_the_real_server(live_llm) -> None:
    result = live_llm.generate([ChatMessage(role="user", content=PROMPT)])

    assert result.text.strip(), "the live model returned an empty reply"
    assert result.model, "the reply must name the model that produced it"
    assert result.finish_reason, "a real completion reports why it stopped"
    assert result.latency_s > 0
    # The default config streams, so a real first-token time must be present.
    assert result.first_token_s is not None
    assert result.first_token_s <= result.latency_s + 1e-6


def test_stream_against_the_real_server(live_llm) -> None:
    deltas = [piece for piece in live_llm.stream([ChatMessage(role="user", content=PROMPT)])]
    assert deltas, "the real SSE stream produced no deltas"
    assert "".join(deltas).strip()


def test_the_system_prompt_reaches_the_model(live_llm) -> None:
    """The headline claim of the change: a prompt steers generation."""
    pirate = live_llm.generate(
        [
            ChatMessage(
                role="system",
                content="You are a pirate. Answer in one short sentence, in character.",
            ),
            ChatMessage(role="user", content="Say hello to the captain."),
        ]
    )
    formal = live_llm.generate(
        [
            ChatMessage(
                role="system",
                content="You are a formal butler. Answer in one short sentence, in character.",
            ),
            ChatMessage(role="user", content="Say hello to the captain."),
        ]
    )
    assert pirate.text.strip() and formal.text.strip()
    assert pirate.text.strip() != formal.text.strip(), (
        "two different personas produced identical text -- the system prompt is "
        "probably not being sent"
    )


def test_an_unpulled_model_is_a_clean_error_not_a_crash(live_llm) -> None:
    """The most likely first-run failure: a tag that was never pulled.

    Builds its own engine rather than mutating the shared one, so the module
    fixture stays pointed at the real model for the tests after it.
    """
    engine = OpenAICompatLLM(
        live_llm.endpoint.removesuffix("/chat/completions"),
        "definitely-not-a-real-model:1b",
        provider=live_llm.provider,
        timeout_s=30.0,
        connect_timeout_s=5.0,
        stream=True,
    )
    try:
        with pytest.raises(LLMUnavailableError) as excinfo:
            engine.generate([ChatMessage(role="user", content="hello")])
        assert "HTTP 404" in str(excinfo.value)
    finally:
        engine.close()


def test_the_usage_chunk_is_parsed_from_a_real_stream(live_llm) -> None:
    """Ollama reports usage in a trailing chunk with an empty ``choices`` list."""
    result = live_llm.generate([ChatMessage(role="user", content=PROMPT)])
    if result.usage is None:  # pragma: no cover - older Ollama builds
        pytest.skip(
            "this Ollama build did not return a usage chunk; "
            f"raw={json.dumps(result.raw) if result.raw else 'n/a'}"
        )
    assert result.usage.total_tokens > 0
