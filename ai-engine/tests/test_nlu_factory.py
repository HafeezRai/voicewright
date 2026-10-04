"""Backend selection, especially the ``auto`` fallback.

These use fakes -- they are testing control flow, not models -- so they run
everywhere regardless of which weights are cached.
"""

from __future__ import annotations

import pytest

from app.nlu.base import IntentResult, NLUEngine, NLUUnavailableError
from app.nlu.factory import AutoNLU, build_nlu


class _Boom(NLUEngine):
    name = "boom"

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        raise NLUUnavailableError("no weights")

    def classify(self, text: str) -> IntentResult:  # pragma: no cover - never reached
        raise AssertionError("a backend that failed to load must not be used")


class _Works(NLUEngine):
    name = "works"

    def __init__(self, intent: str = "book_appointment") -> None:
        self._intent = intent
        self.classify_calls = 0

    @property
    def is_loaded(self) -> bool:
        return True

    def classify(self, text: str) -> IntentResult:
        self.classify_calls += 1
        return IntentResult(intent=self._intent, confidence=0.9, backend=self.name)


class _Flaky(NLUEngine):
    """Loads fine, then fails on the first classify."""

    name = "flaky"

    def load(self) -> None:
        return None

    def classify(self, text: str) -> IntentResult:
        raise NLUUnavailableError("mid-flight failure")


class _Unavailable(NLUEngine):
    """Cannot load and cannot classify -- a backend that simply is not there."""

    name = "unavailable"

    def load(self) -> None:
        raise NLUUnavailableError("no weights")

    def classify(self, text: str) -> IntentResult:
        raise NLUUnavailableError("no weights")


def test_auto_prefers_the_primary() -> None:
    primary, fallback = _Works("cancel_account"), _Works("out_of_scope")
    auto = AutoNLU(primary, fallback)

    result = auto.classify("anything")

    assert result.intent == "cancel_account"
    assert result.backend == "works"
    assert fallback.classify_calls == 0
    assert auto.name == "works"


def test_auto_falls_back_when_the_primary_cannot_load() -> None:
    boom, fallback = _Boom(), _Works("return_policy")
    auto = AutoNLU(boom, fallback)

    result = auto.classify("anything")

    assert boom.load_calls == 1
    assert result.intent == "return_policy"
    assert auto.name == "works"


def test_auto_resolves_only_once() -> None:
    boom, fallback = _Boom(), _Works()
    auto = AutoNLU(boom, fallback)

    auto.classify("one")
    auto.classify("two")
    auto.classify("three")

    assert boom.load_calls == 1, "the failing load must not be retried every call"
    assert fallback.classify_calls == 3


def test_auto_degrades_on_a_mid_flight_failure() -> None:
    flaky, fallback = _Flaky(), _Works("agent_handoff")
    auto = AutoNLU(flaky, fallback)

    result = auto.classify("anything")

    assert result.intent == "agent_handoff"
    assert auto.name == "works"
    # And it stays degraded for subsequent calls.
    assert auto.classify("again").backend == "works"
    assert fallback.classify_calls == 2


def test_auto_propagates_a_fallback_failure() -> None:
    """When neither backend can answer, the error surfaces rather than hiding."""
    auto = AutoNLU(_Unavailable(), _Unavailable())
    with pytest.raises(NLUUnavailableError):
        auto.classify("anything")


def test_build_nlu_honours_the_setting(settings, monkeypatch) -> None:
    from app.config import Settings

    monkeypatch.setenv("AI_ENGINE_NLU_BACKEND", "hf")
    assert type(build_nlu(Settings())).__name__ == "HFIntentNLU"

    monkeypatch.setenv("AI_ENGINE_NLU_BACKEND", "laya")
    assert type(build_nlu(Settings())).__name__ == "LayaNLU"

    monkeypatch.setenv("AI_ENGINE_NLU_BACKEND", "auto")
    assert isinstance(build_nlu(Settings()), AutoNLU)


def test_build_nlu_does_not_load_anything(monkeypatch) -> None:
    """Selection must not trigger a download or an allocation."""
    from app.config import Settings

    monkeypatch.setenv("AI_ENGINE_NLU_BACKEND", "auto")
    engine = build_nlu(Settings())
    assert engine.is_loaded is False
