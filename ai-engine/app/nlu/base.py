"""NLU engine interface.

This is the seam that keeps Laya (or any other classifier) out of the rest of
the codebase. Callers depend only on :class:`NLUEngine`, :class:`IntentResult`
and :class:`Slot`; no Laya/transformers type appears in a signature here.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class NLUUnavailableError(RuntimeError):
    """Raised when the NLU backend cannot be loaded."""


@dataclass(frozen=True)
class Slot:
    """A single extracted argument, e.g. ``when="next Tuesday"``."""

    name: str
    value: str
    confidence: float = 1.0


@dataclass(frozen=True)
class IntentResult:
    """The classifier's verdict on one utterance."""

    intent: str
    confidence: float = 0.0
    slots: tuple[Slot, ...] = field(default_factory=tuple)
    #: Ranking over all candidate intents, when the backend exposes one.
    scores: Mapping[str, float] = field(default_factory=dict)
    backend: str = ""
    #: Backend-specific detail (kept for debugging; never required by callers).
    raw: Any = None

    def slot(self, name: str) -> str | None:
        """Return the value of slot ``name``, or ``None``."""
        for s in self.slots:
            if s.name == name:
                return s.value
        return None


class NLUEngine(abc.ABC):
    """Classifies an utterance into an intent plus slots."""

    name: str = "nlu"

    @property
    def is_loaded(self) -> bool:
        return False

    def load(self) -> None:
        """Eagerly load weights. Optional; :meth:`classify` loads on demand."""

    def close(self) -> None:
        """Release weights. Optional; safe to call more than once."""

    @abc.abstractmethod
    def classify(self, text: str) -> IntentResult:
        """Return the intent, confidence and slots for ``text``."""
