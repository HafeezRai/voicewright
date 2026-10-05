"""LLM ("think") stage -- generates the agent's reply from a system prompt.

Note what this package deliberately does *not* import: :mod:`app.llm.factory`,
because the factory reads :class:`app.config.Settings` and ``app.config``
imports :data:`app.llm.prompts.DEFAULT_SYSTEM_PROMPT`. Importing the factory
here would close that loop and break ``import app.config``.
``build_llm`` is therefore resolved lazily through :func:`__getattr__`, so
``from app.llm import build_llm`` still works everywhere.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import (
    ChatMessage,
    LLMEngine,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUsage,
    Role,
)
from app.llm.history import ConversationHistory
from app.llm.prompts import DEFAULT_SYSTEM_PROMPT

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ChatMessage",
    "ConversationHistory",
    "LLMEngine",
    "LLMResult",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "LLMUsage",
    "Role",
    "build_llm",
]


def __getattr__(name: str) -> Any:
    """Resolve ``build_llm`` lazily (see the module docstring)."""
    if name == "build_llm":
        from app.llm.factory import build_llm

        return build_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
