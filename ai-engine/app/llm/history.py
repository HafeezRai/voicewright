"""Bounded, snapshot-able conversation history.

History is a *per-call parameter* to the pipeline, never pipeline state. The
orchestrator is deliberately stateless (it is shared by every session in the
process, and ``to_thread`` runs turns concurrently), so whoever owns a session
-- today ``app.main``'s session store -- owns its history and hands the
orchestrator a snapshot.

That is also why this class exists at all rather than a plain list: the store
needs a bound (an unbounded history grows until the model's context window
errors out) and a copy semantics (a generator must not observe a list mutating
underneath it while it is mid-request).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from app.llm.base import ChatMessage


class ConversationHistory:
    """The last ``max_turns`` user/assistant exchanges, oldest first.

    ``max_turns`` counts *exchanges*, so at most ``2 * max_turns`` messages are
    retained. ``max_turns <= 0`` disables retention entirely: nothing is kept,
    and :meth:`snapshot` is always empty. That is the honest encoding of "no
    history" rather than a silently unbounded buffer.
    """

    def __init__(self, max_turns: int = 6):
        self._max_turns = max(0, int(max_turns))
        self._messages: list[ChatMessage] = []
        self._lock = threading.Lock()

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def add(self, role: str, content: str) -> None:
        """Append one message, trimming the oldest if the bound is exceeded."""
        message = ChatMessage(role=role, content=content)  # type: ignore[arg-type]
        with self._lock:
            self._messages.append(message)
            self._trim_locked()

    def add_turn(self, user: str, assistant: str) -> None:
        """Append one complete exchange (the usual case)."""
        with self._lock:
            self._messages.append(ChatMessage(role="user", content=user))
            self._messages.append(ChatMessage(role="assistant", content=assistant))
            self._trim_locked()

    def extend(self, messages: Iterable[ChatMessage]) -> None:
        with self._lock:
            self._messages.extend(messages)
            self._trim_locked()

    def snapshot(self) -> list[ChatMessage]:
        """A point-in-time copy, safe to hand to a request in another thread."""
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    def _trim_locked(self) -> None:
        """Keep the most recent ``2 * max_turns`` messages. Caller holds the lock."""
        limit = self._max_turns * 2
        if limit <= 0:
            self._messages.clear()
        elif len(self._messages) > limit:
            del self._messages[: len(self._messages) - limit]
