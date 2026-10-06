"""Tools the agent can call.

A voice agent that can only talk is a demo. The thing that makes it a product
is being able to *do* something -- book the slot, look up the order, hand the
caller to a human -- and getting the result into what it says next.

The shape of that exchange is fixed by the model providers, not by us: the
model is given a list of function specifications, it replies with a call
instead of (or before) prose, the application runs it, and the result is
appended to the conversation for the model to use. This module holds the three
pieces that flow through that loop, plus the registry that resolves a name to
something executable.

Nothing here knows about HTTP, the pipeline, or where a specific tool comes
from -- see ``builtin.py`` for the shipped ones. That separation is what makes
the loop testable without a model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: A tool implementation takes keyword arguments and returns text for the model.
ToolFn = Callable[..., str]


class ToolError(RuntimeError):
    """Raised when a tool cannot be run.

    The message is shown to the model, so it should say what was wrong in terms
    it can act on -- "no such order" beats "KeyError: 'ord_9'". A failed tool is
    not a failed turn: the model gets to apologise or ask again.
    """


@dataclass(frozen=True)
class ToolSpec:
    """A function the model is allowed to call."""

    name: str
    description: str
    #: JSON Schema for the arguments, passed through to the provider verbatim.
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_openai(self) -> dict[str, Any]:
        """The provider-facing envelope (OpenAI, Ollama, vLLM, DeepSeek)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe form, for the API and the agent store."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolSpec:
        return cls(
            name=str(data.get("name", "")).strip(),
            description=str(data.get("description", "")).strip(),
            parameters=dict(data.get("parameters") or {}),
        )


@dataclass(frozen=True)
class ToolCall:
    """A model's request to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Raw argument text, kept when it did not parse -- so the error handed back
    #: to the model can quote what it actually sent.
    raw_arguments: str = ""

    @classmethod
    def parse(cls, *, call_id: str, name: str, arguments: str | Mapping[str, Any]) -> ToolCall:
        """Build from a provider payload, tolerating malformed arguments.

        Models emit argument JSON that does not always parse, and one bad call
        must not take down the turn. Unparseable arguments become an empty dict
        plus the raw text, and the executor turns that into a message the model
        can recover from.
        """
        if isinstance(arguments, Mapping):
            return cls(id=call_id, name=name, arguments=dict(arguments))
        text = arguments or ""
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return cls(id=call_id, name=name, arguments={}, raw_arguments=text)
        if not isinstance(parsed, Mapping):
            return cls(id=call_id, name=name, arguments={}, raw_arguments=text)
        return cls(id=call_id, name=name, arguments=dict(parsed))

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ToolResult:
    """What a tool returned, and whether it worked."""

    call_id: str
    name: str
    content: str
    ok: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "name": self.name,
            "ok": self.ok,
            "content": self.content,
        }


@dataclass(frozen=True)
class ToolBinding:
    """A specification paired with the thing that runs it."""

    spec: ToolSpec
    run: ToolFn


@dataclass(frozen=True)
class Toolbox:
    """The tools one turn may use: what to offer the model, and how to run it.

    Passed per-call rather than held on the pipeline, for the same reason
    ``knowledge_provider`` is: the pipeline is stateless and knows nothing about
    agents, and what an agent is allowed to do is decided at the edge.
    """

    registry: ToolRegistry
    specs: tuple[ToolSpec, ...]

    @classmethod
    def build(cls, registry: ToolRegistry, names: Sequence[str]) -> Toolbox:
        return cls(registry=registry, specs=tuple(registry.specs_for(names)))

    def run(self, call: ToolCall) -> ToolResult:
        return self.registry.call(call)

    def __bool__(self) -> bool:
        return bool(self.specs)

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]


class ToolRegistry:
    """The tools an agent may call, by name.

    Deliberately an explicit registry rather than a decorator that discovers
    callables from anywhere in the process: an agent's tools are part of its
    configuration, and what an agent is *allowed* to do should be readable in
    one place rather than inferred from imports.
    """

    def __init__(self, bindings: Iterable[ToolBinding] = ()) -> None:
        self._by_name: dict[str, ToolBinding] = {}
        for binding in bindings:
            self.register(binding)

    def register(self, binding: ToolBinding) -> None:
        self._by_name[binding.spec.name] = binding

    @property
    def names(self) -> list[str]:
        return sorted(self._by_name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def spec(self, name: str) -> ToolSpec | None:
        binding = self._by_name.get(name)
        return binding.spec if binding else None

    def specs_for(self, names: Sequence[str]) -> list[ToolSpec]:
        """Specs for the named tools, skipping any that are not registered.

        An agent referring to a tool this deployment does not have is a
        configuration mistake, not a runtime failure -- the turn should still
        work, minus that capability.
        """
        specs = []
        for name in names:
            spec = self.spec(name)
            if spec is not None:
                specs.append(spec)
        return specs

    def call(self, call: ToolCall) -> ToolResult:
        """Run one tool call, never raising.

        Every failure becomes a ``ToolResult`` with ``ok=False`` and a message
        written for the model, because the alternative -- propagating -- would
        end the turn over something the model can simply apologise for.
        """
        binding = self._by_name.get(call.name)
        if binding is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"No tool named {call.name!r} is available.",
            )
        if call.raw_arguments:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=(
                    f"Could not parse the arguments as JSON: {call.raw_arguments!r}. "
                    "Call it again with valid JSON."
                ),
            )
        try:
            return ToolResult(call_id=call.id, name=call.name, content=str(binding.run(**call.arguments)))
        except ToolError as exc:
            return ToolResult(call_id=call.id, name=call.name, ok=False, content=str(exc))
        except TypeError as exc:
            # Wrong or missing arguments -- say so in a way the model can fix.
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"Bad arguments for {call.name}: {exc}",
            )
        except Exception as exc:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"{call.name} failed: {exc}",
            )
