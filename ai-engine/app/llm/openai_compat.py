"""The one shipped ``LLMEngine``: any OpenAI-compatible ``/chat/completions``.

Ollama, DeepSeek, vLLM, LM Studio and OpenAI itself all speak the same wire
protocol, so a single client covers the local default and the hosted
alternative with no code change -- only ``AI_ENGINE_LLM_BASE_URL`` and
``AI_ENGINE_LLM_MODEL`` differ.

Why ``httpx`` and not the ``openai`` SDK: the SDK's one real advantage here is
SSE parsing, which is about thirty lines (see :meth:`_iter_sse`), while it
assumes an API key always exists and hides the transport. ``httpx.Client`` is
thread-safe with connection pooling (the orchestrator calls it from
``to_thread`` workers), and ``httpx.MockTransport`` gives the test suite
socket-free coverage of 401s, 404s, timeouts and SSE bodies.

Timeouts are asymmetric on purpose: ``connect`` is short so a dead endpoint
becomes a clean 503 instead of a hang, while ``read`` applies *per chunk* under
streaming, so a long generation never trips it as long as tokens keep flowing.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import httpx

from app.cancellation import CancelToken
from app.llm.base import (
    ChatMessage,
    LLMEngine,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUsage,
)
from app.tools.base import ToolCall, ToolSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool-call parsing
#
# Providers express a function call in two different shapes depending on
# whether the response streamed, and both have to end up as the same
# `ToolCall`. Keeping the parsers at module level (rather than methods) means
# they can be tested against captured payloads with no client at all.
# ---------------------------------------------------------------------------


def tool_calls_from_message(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """Tool calls carried whole on a non-streaming assistant message."""
    raw = message.get("tool_calls")
    if not isinstance(raw, Sequence):
        return ()
    calls: list[ToolCall] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            continue
        function = entry.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        calls.append(
            ToolCall.parse(
                call_id=str(entry.get("id") or f"call_{index}"),
                name=name,
                arguments=function.get("arguments") or "",
            )
        )
    return tuple(calls)


def accumulate_tool_call_deltas(
    partial: dict[int, dict[str, str]], data: Mapping[str, Any]
) -> None:
    """Fold one SSE chunk's tool-call fragments into ``partial``.

    A streamed call arrives in pieces: the first chunk carries the id and the
    name, later ones carry slices of the arguments JSON. They are keyed by
    ``index`` because one reply may ask for several tools at once, and the
    argument fragments must be concatenated in arrival order.
    """
    for choice in data.get("choices") or ():
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta") or choice.get("message")
        if not isinstance(delta, Mapping):
            continue
        for entry in delta.get("tool_calls") or ():
            if not isinstance(entry, Mapping):
                continue
            index = entry.get("index")
            key = index if isinstance(index, int) else 0
            slot = partial.setdefault(key, {"id": "", "name": "", "arguments": ""})
            if entry.get("id"):
                slot["id"] = str(entry["id"])
            function = entry.get("function")
            if isinstance(function, Mapping):
                if function.get("name"):
                    slot["name"] = str(function["name"])
                if function.get("arguments"):
                    slot["arguments"] += str(function["arguments"])


def finish_tool_call_deltas(
    partial: Mapping[int, Mapping[str, str]],
) -> tuple[ToolCall, ...]:
    """Build whole calls from the accumulated fragments, in send order."""
    calls: list[ToolCall] = []
    for index in sorted(partial):
        slot = partial[index]
        name = (slot.get("name") or "").strip()
        if not name:
            continue
        calls.append(
            ToolCall.parse(
                call_id=slot.get("id") or f"call_{index}",
                name=name,
                arguments=slot.get("arguments") or "",
            )
        )
    return tuple(calls)


#: Model-name markers for families that emit chain-of-thought. Ollama advertises
#: this as the ``thinking`` capability, and a reasoning model that streams its
#: reasoning as ordinary ``content`` produces an agent that reads its own
#: deliberation aloud. A name list rather than a capability probe: probing costs
#: an extra round trip on every turn, and this is a guard rail, not a gate.
REASONING_MODEL_MARKERS = (
    "deepseek-r1",
    "qwq",
    "reasoner",
    "r1-1776",
    "-thinking",
    "gemma-4-26b",
)


def looks_like_reasoning_model(model: str) -> bool:
    """True when ``model``'s name matches a known chain-of-thought family."""
    name = (model or "").lower()
    return any(marker in name for marker in REASONING_MODEL_MARKERS)


class OpenAICompatLLM(LLMEngine):
    """Chat completions over ``POST {base_url}/chat/completions``.

    One instance holds one pooled :class:`httpx.Client` and is safe to share
    across threads.
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        provider: str = "openai",
        api_key: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 160,
        timeout_s: float = 30.0,
        connect_timeout_s: float = 2.0,
        stream: bool = True,
        extra_body: Mapping[str, Any] | None = None,
        allow_reasoning_models: bool = False,
        max_retries: int = 0,
        transport: httpx.BaseTransport | None = None,
    ):
        if not base_url:
            raise LLMUnavailableError("an LLM base URL is required")
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._provider = provider
        self._api_key = api_key or None
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stream = stream
        self._extra_body = dict(extra_body or {})
        self._allow_reasoning_models = allow_reasoning_models
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            # httpx's own transport retries *connection* failures only, which is
            # exactly the policy we want: a retried read timeout would double the
            # latency of a turn the caller has already given up on.
            transport=transport
            if transport is not None
            else httpx.HTTPTransport(retries=max(0, int(max_retries))),
            headers=self._headers(),
        )

    # -- identity -----------------------------------------------------------
    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return not self._http.is_closed

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def close(self) -> None:
        self._http.close()

    def load(self) -> None:
        """Warm the model so the first real turn is not a 4.66 GB cold load.

        A one-token request with the same ``extra_body`` (which is where Ollama's
        ``keep_alive`` lives) is enough to put the weights in memory. Local
        providers load from disk; hosted providers just pay the TLS handshake.
        """
        started = time.perf_counter()
        payload = self._payload([ChatMessage(role="user", content="hi")], stream=False)
        payload["max_tokens"] = 1
        try:
            response = self._http.post(self._endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        if response.status_code >= 400:
            raise LLMUnavailableError(self._describe_error(response))
        logger.info(
            "LLM warm-up ok provider=%s model=%s in %.2fs",
            self._provider,
            self._model,
            time.perf_counter() - started,
        )

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Return the reply in full.

        When ``stream`` is enabled this uses the SSE transport and accumulates
        the deltas, which is what makes :attr:`LLMResult.first_token_s` -- the
        single most useful latency signal for a cold local model -- available on
        the batch path too. Same wire format, same parser as :meth:`stream`.
        """
        self._guard_reasoning_model()
        if self._stream:
            return self._generate_streaming(messages, tools, max_tokens)
        return self._generate_once(messages, tools, max_tokens)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        cancel: CancelToken | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int | None = None,
        on_tool_calls: Callable[[tuple[ToolCall, ...]], None] | None = None,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive, stopping between chunks on cancel.

        Tool requests arrive interleaved with (usually before) any text, so they
        are accumulated by index and reported through ``on_tool_calls`` when the
        stream ends. A turn that is going to call a tool produces no text, so
        nothing is spoken while it decides.
        """
        self._guard_reasoning_model()
        started = time.perf_counter()
        payload = self._payload(messages, stream=True, tools=tools, max_tokens=max_tokens)
        partial: dict[int, dict[str, str]] = {}
        try:
            with self._http.stream("POST", self._endpoint, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMUnavailableError(self._describe_error(response))
                for data in self._iter_sse(response):
                    accumulate_tool_call_deltas(partial, data)
                    if cancel is not None and cancel.cancelled:
                        logger.debug("LLM stream cancelled after %.3fs", time.perf_counter() - started)
                        return
                    yield from self._chunk_texts(data)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

        calls = finish_tool_call_deltas(partial)
        if calls and on_tool_calls is not None:
            on_tool_calls(calls)

    # -- internals: request/response ----------------------------------------
    def _headers(self) -> dict[str, str]:
        # No explicit Accept: the same endpoint answers JSON or text/event-stream
        # depending on the `stream` field, and pinning it to one breaks the other.
        headers = {"Content-Type": "application/json"}
        # Sent only when configured: a local Ollama rejects an empty bearer token
        # and a hosted provider needs a real one. The key is never logged or
        # echoed into an error message.
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: bool,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.as_dict() for m in messages],
            "temperature": self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
            "stream": stream,
        }
        if stream:
            # Ollama and OpenAI both accept this and report final usage in a
            # trailing chunk with an empty `choices` list. Verified against
            # Ollama 0.x on 127.0.0.1:11434 before relying on it.
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = [spec.as_openai() for spec in tools]
            # Let the model choose. Forcing a tool call would make every turn
            # call something, including "hello".
            payload["tool_choice"] = "auto"
        # Provider-specific knobs, e.g. Ollama's {"keep_alive": "30m"}.
        payload.update(self._extra_body)
        return payload

    def _generate_once(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        payload = self._payload(messages, stream=False, tools=tools, max_tokens=max_tokens)
        try:
            response = self._http.post(self._endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        if response.status_code >= 400:
            raise LLMUnavailableError(self._describe_error(response))

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMUnavailableError(
                f"{self._provider} returned a non-JSON body (HTTP {response.status_code})"
            ) from exc
        return self._result_from_body(body, started)

    def _generate_streaming(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        first_token_s: float | None = None
        parts: list[str] = []
        usage: LLMUsage | None = None
        finish_reason = ""
        partial: dict[int, dict[str, str]] = {}
        payload = self._payload(messages, stream=True, tools=tools, max_tokens=max_tokens)
        try:
            with self._http.stream("POST", self._endpoint, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMUnavailableError(self._describe_error(response))
                for data in self._iter_sse(response):
                    accumulate_tool_call_deltas(partial, data)
                    parsed_usage = LLMUsage.from_mapping(data.get("usage"))
                    if parsed_usage is not None:
                        usage = parsed_usage
                    for piece in self._chunk_texts(data):
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - started
                        parts.append(piece)
                    for choice in data.get("choices") or ():
                        if isinstance(choice, Mapping) and choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

        return LLMResult(
            text="".join(parts).strip(),
            model=self._model,
            provider=self._provider,
            finish_reason=finish_reason,
            usage=usage,
            latency_s=time.perf_counter() - started,
            first_token_s=first_token_s,
            tool_calls=finish_tool_call_deltas(partial),
        )

    @staticmethod
    def _chunk_texts(data: Mapping[str, Any]) -> Iterator[str]:
        """Yield the assistant text carried by one SSE chunk.

        A chunk is ``{"choices": [{"delta": {"content": ...}}]}``; the final
        usage chunk has an empty ``choices`` list and yields nothing.
        """
        for choice in data.get("choices") or ():
            if not isinstance(choice, Mapping):
                continue
            text = OpenAICompatLLM._delta_text(choice)
            if text:
                yield text

    @staticmethod
    def _delta_text(choice: Mapping[str, Any]) -> str:
        """Pull the assistant text out of one SSE ``choices[0]`` entry.

        Reads ``delta.content`` (streaming) or ``message.content`` (a non-
        streaming body), and deliberately ignores ``reasoning_content``: that is
        chain-of-thought, and speaking it is never what the caller wanted.
        """
        container = choice.get("delta")
        if not isinstance(container, Mapping):
            container = choice.get("message")
        if not isinstance(container, Mapping):
            return ""
        content = container.get("content")
        return content if isinstance(content, str) else ""

    def _result_from_body(self, body: Any, started: float) -> LLMResult:
        if not isinstance(body, Mapping):
            raise LLMUnavailableError(f"{self._provider} returned an unexpected body shape")
        choices = body.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise LLMUnavailableError(f"{self._provider} returned no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LLMUnavailableError(f"{self._provider} returned a malformed choice")
        message = choice.get("message")
        text = ""
        calls: tuple[ToolCall, ...] = ()
        if isinstance(message, Mapping):
            if isinstance(message.get("content"), str):
                text = message["content"]
            calls = tool_calls_from_message(message)
        return LLMResult(
            text=text.strip(),
            tool_calls=calls,
            model=str(body.get("model") or self._model),
            provider=self._provider,
            finish_reason=str(choice.get("finish_reason") or ""),
            usage=LLMUsage.from_mapping(body.get("usage")),
            latency_s=time.perf_counter() - started,
        )

    @staticmethod
    def _iter_sse(response: httpx.Response) -> Iterator[Mapping[str, Any]]:
        """Yield the JSON object from each ``data:`` line of an SSE body.

        Tolerates the noise real providers emit: blank keep-alive lines, comment
        lines, and the terminal ``[DONE]`` sentinel. A malformed JSON chunk is
        skipped rather than fatal -- one bad frame should not cost the turn.
        """
        for raw in response.iter_lines():
            line = raw.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                return
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(data, Mapping):
                yield data

    # -- internals: errors --------------------------------------------------
    def _guard_reasoning_model(self) -> None:
        if not self._allow_reasoning_models and looks_like_reasoning_model(self._model):
            raise LLMUnavailableError(
                f"model {self._model!r} looks like a reasoning model whose "
                "chain-of-thought would be spoken aloud; set "
                "AI_ENGINE_LLM_ALLOW_REASONING_MODELS=true to allow it anyway"
            )

    def _transport_error(self, exc: httpx.HTTPError) -> Exception:
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(
                f"{self._provider} did not respond in time ({type(exc).__name__})"
            )
        if isinstance(exc, httpx.ConnectError):
            return LLMUnavailableError(
                f"cannot reach {self._provider} at {self._base_url}: {exc}"
            )
        return LLMUnavailableError(f"{self._provider} request failed: {exc}")

    def _describe_error(self, response: httpx.Response) -> str:
        """A short, safe description of a failed response.

        Providers shape their error bodies differently -- Ollama uses
        ``{"error": "..."}``, OpenAI and DeepSeek use
        ``{"error": {"message": ...}}`` -- so both are parsed. The request's
        Authorization header is never included.
        """
        detail = ""
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, str):
                detail = error
            elif isinstance(error, Mapping) and isinstance(error.get("message"), str):
                detail = error["message"]
            elif isinstance(body.get("message"), str):
                detail = body["message"]
        hint = self._hint_for(response.status_code)
        parts = [f"{self._provider} returned HTTP {response.status_code}"]
        if detail:
            parts.append(detail)
        if hint:
            parts.append(hint)
        return " -- ".join(parts)

    def _hint_for(self, status: int) -> str:
        if status == 404:
            return (
                f"model {self._model!r} not found; pull it "
                f"(e.g. `ollama pull {self._model}`) or set AI_ENGINE_LLM_MODEL"
            )
        if status in (401, 403):
            return f"set AI_ENGINE_LLM_API_KEY for provider {self._provider!r}"
        if status == 408:
            return "the provider timed out"
        if status >= 500:
            return "the provider is unhealthy"
        return ""
