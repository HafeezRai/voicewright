"""The OpenAI-compatible client, driven by ``httpx.MockTransport``.

Every test here runs with no socket and no model. That is the point of using
httpx: the failure modes that matter in production -- a dead endpoint, a model
that was never pulled, a rejected API key, a mid-stream cancel -- are all
reachable from a plain function returning an :class:`httpx.Response`.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.cancellation import CancelToken
from app.llm.base import ChatMessage, LLMTimeoutError, LLMUnavailableError
from app.llm.openai_compat import OpenAICompatLLM, looks_like_reasoning_model

MESSAGES = [ChatMessage(role="user", content="hello there")]


def _llm(handler, **kwargs) -> OpenAICompatLLM:
    kwargs.setdefault("provider", "test")
    kwargs.setdefault("stream", False)
    return OpenAICompatLLM(
        "http://llm.local/v1",
        "test-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _sse(*frames: str) -> bytes:
    return "".join(f"data: {frame}\n\n" for frame in frames).encode()


def _completion(text: str, **extra) -> dict:
    body = {
        "id": "chatcmpl-1",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }
    body.update(extra)
    return body


# ---------------------------------------------------------------------------
# request shape
# ---------------------------------------------------------------------------


def test_endpoint_is_derived_from_the_base_url() -> None:
    llm = _llm(lambda request: httpx.Response(200, json=_completion("hi")))
    assert llm.endpoint == "http://llm.local/v1/chat/completions"
    assert llm.provider == "test" and llm.model == "test-model"


def test_request_body_carries_model_messages_and_limits() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("hi"))

    llm = _llm(handler, temperature=0.25, max_tokens=42, extra_body={"keep_alive": "30m"})
    llm.generate(MESSAGES)

    assert seen["model"] == "test-model"
    assert seen["messages"] == [{"role": "user", "content": "hello there"}]
    assert seen["temperature"] == 0.25
    assert seen["max_tokens"] == 42
    assert seen["stream"] is False
    # extra_body is merged at the top level -- that is where Ollama's keep_alive lives.
    assert seen["keep_alive"] == "30m"


def test_streaming_request_asks_for_usage() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_sse("[DONE]"))

    _llm(handler, stream=True).generate(MESSAGES)
    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


def test_authorization_header_present_only_when_a_key_is_set() -> None:
    captured: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers)
        return httpx.Response(200, json=_completion("hi"))

    _llm(handler).generate(MESSAGES)
    assert "authorization" not in captured[-1]

    _llm(handler, api_key="sk-secret").generate(MESSAGES)
    assert captured[-1]["authorization"] == "Bearer sk-secret"


# ---------------------------------------------------------------------------
# non-streaming path
# ---------------------------------------------------------------------------


def test_generate_once_returns_text_usage_and_finish_reason() -> None:
    llm = _llm(lambda request: httpx.Response(200, json=_completion("Hello there.")))
    result = llm.generate(MESSAGES)

    assert result.text == "Hello there."
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.total_tokens == 10
    assert result.usage.as_dict() == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert result.latency_s >= 0
    # Nothing streamed, so there is no first-token time -- and that is a signal.
    assert result.first_token_s is None


def test_generate_skips_reasoning_content() -> None:
    """A ``message`` carrying chain-of-thought must not become the reply."""
    body = _completion("The answer is four.")
    body["choices"][0]["message"]["reasoning_content"] = "um, let me think..."
    llm = _llm(lambda request: httpx.Response(200, json=body))
    assert llm.generate(MESSAGES).text == "The answer is four."


def test_usage_falls_back_to_the_sum_of_parts() -> None:
    body = _completion("hi")
    body["usage"] = {"prompt_tokens": 4, "completion_tokens": 6}
    llm = _llm(lambda request: httpx.Response(200, json=body))
    assert llm.generate(MESSAGES).usage.total_tokens == 10


# ---------------------------------------------------------------------------
# streaming path
# ---------------------------------------------------------------------------


def test_generate_streaming_accumulates_deltas_and_times_the_first_token() -> None:
    body = _sse(
        '{"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}',
        '{"choices":[{"delta":{"content":"lo "}}]}',
        '{"choices":[{"delta":{"content":"there."},"finish_reason":"stop"}]}',
        '{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}',
        "[DONE]",
    )
    llm = _llm(lambda request: httpx.Response(200, content=body), stream=True)
    result = llm.generate(MESSAGES)

    assert result.text == "Hello there."
    assert result.finish_reason == "stop"
    assert result.first_token_s is not None and result.first_token_s >= 0
    assert result.usage is not None and result.usage.total_tokens == 11


def test_stream_yields_deltas_in_order() -> None:
    body = _sse(
        '{"choices":[{"delta":{"content":"one "}}]}',
        '{"choices":[{"delta":{"content":"two"}}]}',
        "[DONE]",
    )
    llm = _llm(lambda request: httpx.Response(200, content=body))
    assert list(llm.stream(MESSAGES)) == ["one ", "two"]


def test_stream_ignores_keepalives_and_malformed_frames() -> None:
    raw = (
        ": keep-alive comment\n\n"
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        "data: {this is not json}\n\n"
        "data: [DONE]\n\n"
    )
    llm = _llm(lambda request: httpx.Response(200, content=raw.encode()))
    assert list(llm.stream(MESSAGES)) == ["ok"]


def test_stream_stops_when_the_cancel_token_is_set() -> None:
    body = _sse(
        '{"choices":[{"delta":{"content":"first"}}]}',
        '{"choices":[{"delta":{"content":" second"}}]}',
        '{"choices":[{"delta":{"content":" third"}}]}',
        "[DONE]",
    )
    llm = _llm(lambda request: httpx.Response(200, content=body))
    token = CancelToken()

    received = []
    for piece in llm.stream(MESSAGES, cancel=token):
        received.append(piece)
        token.cancel()  # what the barge-in controller does

    assert received == ["first"]


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def test_connect_error_becomes_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError) as excinfo:
        _llm(handler).generate(MESSAGES)
    assert "cannot reach" in str(excinfo.value)


def test_read_timeout_becomes_a_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(LLMTimeoutError):
        _llm(handler).generate(MESSAGES)


def test_ollama_style_404_names_the_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'llama3' not found"})

    with pytest.raises(LLMUnavailableError) as excinfo:
        _llm(handler, stream=True).generate(MESSAGES)
    message = str(excinfo.value)
    assert "HTTP 404" in message
    assert "not found" in message
    assert "ollama pull" in message


def test_openai_style_401_parses_the_nested_error_and_never_echoes_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    with pytest.raises(LLMUnavailableError) as excinfo:
        _llm(handler, api_key="sk-do-not-leak").generate(MESSAGES)
    message = str(excinfo.value)
    assert "Invalid API key" in message
    assert "AI_ENGINE_LLM_API_KEY" in message
    assert "sk-do-not-leak" not in message


def test_a_non_json_error_body_still_produces_a_useful_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"<html>bad gateway</html>")

    with pytest.raises(LLMUnavailableError) as excinfo:
        _llm(handler).generate(MESSAGES)
    assert "HTTP 500" in str(excinfo.value)


def test_a_non_json_success_body_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    with pytest.raises(LLMUnavailableError):
        _llm(handler).generate(MESSAGES)


def test_an_empty_choices_list_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(LLMUnavailableError):
        _llm(handler).generate(MESSAGES)


def test_a_streaming_error_status_raises_before_yielding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'nope' not found"})

    with pytest.raises(LLMUnavailableError):
        list(_llm(handler).stream(MESSAGES))


# ---------------------------------------------------------------------------
# reasoning-model guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["deepseek-r1:latest", "qwq:latest", "gemma-4-26B_q4_0-it:latest", "r1-1776:latest"]
)
def test_reasoning_models_are_recognised(model: str) -> None:
    assert looks_like_reasoning_model(model)


@pytest.mark.parametrize("model", ["llama3:8b", "mistral:latest", "qwen2.5-coder:32b"])
def test_ordinary_models_are_not_flagged(model: str) -> None:
    assert not looks_like_reasoning_model(model)


def test_a_reasoning_model_is_refused_with_an_actionable_message() -> None:
    llm = _llm(
        lambda request: httpx.Response(200, json=_completion("thinking...")),
        provider="ollama",
    )
    llm._model = "deepseek-r1:latest"  # what a misconfiguration would produce

    with pytest.raises(LLMUnavailableError) as excinfo:
        llm.generate(MESSAGES)
    assert "AI_ENGINE_LLM_ALLOW_REASONING_MODELS" in str(excinfo.value)


def test_the_reasoning_guard_can_be_overridden() -> None:
    llm = _llm(
        lambda request: httpx.Response(200, json=_completion("ok")),
        allow_reasoning_models=True,
    )
    llm._model = "deepseek-r1:latest"
    assert llm.generate(MESSAGES).text == "ok"
