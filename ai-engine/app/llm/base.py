"""LLM engine interface -- the "think" stage of the pipeline.

The pipeline used to end its understanding stage in a 7-way classifier and a
template table (``app.nlu`` + ``app.pipeline.responder``). It does not any more:
the agent *generates* its reply from a system prompt, which is what makes a
configurable persona and, later, a knowledge base possible at all.

This module is the seam that keeps every provider out of the rest of the
codebase. Callers depend only on :class:`LLMEngine`, :class:`ChatMessage` and
:class:`LLMResult`; no httpx/OpenAI/Ollama type appears in a signature here.

Two shapes, mirroring :class:`app.tts.base.TTSEngine`:

* :meth:`LLMEngine.generate` -- the whole reply at once;
* :meth:`LLMEngine.stream` -- the reply as an iterator of text deltas, which is
  what lets barge-in stop generation rather than only playback.

The base class provides a working ``stream`` (generate-then-yield-one-delta) so
a new provider is usable before it grows true incremental streaming; the shipped
provider overrides it with the real SSE loop.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.cancellation import CancelToken
from app.tools.base import ToolCall, ToolSpec

#: Conventional environment variable holding each provider's key. Someone who
#: has run a DeepSeek example before has DEEPSEEK_API_KEY exported already, and
#: making them copy it into our config as well is a needless step.
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}

#: Roles accepted by the chat protocol. Deliberately the three every
#: OpenAI-compatible host agrees on.
Role = Literal["system", "user", "assistant", "tool"]


class LLMUnavailableError(RuntimeError):
    """Raised when the provider is unreachable, or rejects the request.

    Covers a dead endpoint, a model that was never pulled, a bad API key and a
    provider returning a non-2xx status -- the API layer turns all of them into
    a single clean 503 rather than leaking a stack trace.
    """


class LLMTimeoutError(RuntimeError):
    """Raised when the provider accepted the connection but did not answer."""


@dataclass(frozen=True)
class ChatMessage:
    """One message in a conversation."""

    role: Role
    content: str
    #: Set on an ``assistant`` message that asked for tools to be run.
    tool_calls: tuple[ToolCall, ...] = ()
    #: Set on a ``tool`` message: which call this is the answer to.
    tool_call_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # The provider wants the arguments as a JSON *string*,
                        # which is also why they round-trip through a parse on
                        # the way back in.
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True)
class LLMUsage:
    """Token accounting for one generation, when the provider reports it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> LLMUsage | None:
        """Build from a provider ``usage`` object; ``None`` if unusable.

        Both the non-streaming body and (for providers that honour
        ``stream_options.include_usage``) the final SSE chunk carry this shape.
        """
        if not isinstance(data, Mapping):
            return None
        try:
            prompt = int(data.get("prompt_tokens") or 0)
            completion = int(data.get("completion_tokens") or 0)
            total = int(data.get("total_tokens") or 0)
        except (TypeError, ValueError):
            return None
        if not total:
            total = prompt + completion
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMResult:
    """The generator's output for one call."""

    text: str
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    usage: LLMUsage | None = None
    #: Wall-clock time for the whole call, measured by the engine.
    latency_s: float = 0.0
    #: Time to the first streamed token, when the transport streamed. ``None``
    #: for a non-streaming call -- which is itself the useful signal: it means
    #: the model produced nothing until it had produced everything.
    first_token_s: float | None = None
    #: True when the reply is the configured fallback, not model output.
    fallback: bool = False
    #: Tools the model asked to run, in order. Empty for an ordinary reply.
    tool_calls: tuple[ToolCall, ...] = ()
    #: Provider-specific detail (kept for debugging; never required by callers).
    raw: Any = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary -- what ``POST /llm`` returns."""
        return {
            "reply_text": self.text,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "usage": self.usage.as_dict() if self.usage else None,
            "latency_ms": round(self.latency_s * 1000, 2),
        }


class LLMEngine(abc.ABC):
    """Turns a message list into a reply. Implementations must be thread-safe.

    The orchestrator calls engines from ``asyncio.to_thread`` workers, and
    ``httpx.Client`` is safe to share across them, so one engine instance can
    serve concurrent turns.
    """

    name: str = "llm"

    @property
    def provider(self) -> str:
        """Provider label used on metrics (``ollama``, ``deepseek``, ...)."""
        return ""

    @property
    def model(self) -> str:
        """Model id, for display and for the ``/llm`` response."""
        return ""

    @property
    def is_loaded(self) -> bool:
        return False

    def load(self) -> None:
        """Eagerly warm the model. Optional; generation works without it."""

    def close(self) -> None:
        """Release the client/connection pool. Optional; idempotent."""

    @abc.abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Return the reply to ``messages`` in full.

        With ``tools``, the model may answer by asking for one to be run -- in
        which case ``LLMResult.tool_calls`` is populated and ``text`` is usually
        empty. Running them and calling again is the caller's job, because only
        the caller knows what the tools do.
        """

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        cancel: CancelToken | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int | None = None,
        on_tool_calls: Callable[[tuple[ToolCall, ...]], None] | None = None,
    ) -> Iterator[str]:
        """Yield the reply as text deltas, honouring ``cancel`` between chunks.

        The default implementation generates fully and yields the result as a
        single delta, so a caller that wants incremental output still works
        against a provider that cannot stream -- it just loses the latency win.
        Backends that can stream override this.

        ``on_tool_calls`` is how tool requests come back from a *streaming*
        call. A generator can only yield one thing, and a turn needs both the
        text (to speak as it arrives) and any calls (to run before speaking
        anything) -- an out-parameter keeps the streaming path intact rather
        than forcing every tool-enabled turn back to a blocking call.
        """
        result = self.generate(messages, tools=tools, max_tokens=max_tokens)
        if result.tool_calls and on_tool_calls is not None:
            on_tool_calls(result.tool_calls)
        if result.text:
            yield result.text
