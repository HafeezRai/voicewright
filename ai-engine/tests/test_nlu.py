"""Real NLU: the HuggingFace fallback classifier, the shared schema and slots.

These run against the real MiniLM embeddings -- no mocked vectors -- so a broken
centroid build or a mis-ordered label list shows up as a wrong intent here.
"""

from __future__ import annotations

import pytest

from app.nlu import INTENT_NAMES, HFIntentNLU
from app.nlu.schema import INTENTS, intent_spec
from app.nlu.slots import extract_slots

CASES = [
    ("I want to book an appointment", "book_appointment"),
    ("I need to cancel my account", "cancel_account"),
    ("what's your return policy", "return_policy"),
    ("why was I charged twice this month", "billing_question"),
    ("the app keeps crashing", "technical_support"),
    ("let me talk to a real person", "agent_handoff"),
]


def test_schema_is_consistent() -> None:
    names = [spec.name for spec in INTENTS]
    assert names == list(INTENT_NAMES)
    assert len(names) == len(set(names)), "duplicate intent name in the catalogue"
    assert all(spec.exemplars for spec in INTENTS), "every intent needs exemplars for the HF path"
    assert intent_spec("book_appointment") is not None
    assert intent_spec("nope") is None


@pytest.mark.parametrize("utterance,expected", CASES)
def test_hf_classifier_routes_real_utterances(hf_nlu_engine, utterance, expected) -> None:
    result = hf_nlu_engine.classify(utterance)
    assert result.intent == expected, f"{utterance!r} -> {result.intent!r}"
    assert result.backend == "hf"
    assert result.intent in INTENT_NAMES
    assert 0.0 < result.confidence <= 1.0
    # A full ranking is exposed for callers that gate on margin.
    assert set(result.scores) == set(INTENT_NAMES)
    assert abs(sum(result.scores.values()) - 1.0) < 1e-3


def test_confidence_is_not_saturated(hf_nlu_engine) -> None:
    """Confidence must separate a clear utterance from an ambiguous one.

    A softmax that returns 1.0 for everything would make the number useless for
    gating, so this pins that it does not.
    """
    clear = hf_nlu_engine.classify("I want to book an appointment")
    ambiguous = hf_nlu_engine.classify("book me an appointment and also cancel it")
    noise = hf_nlu_engine.classify("hmm")

    assert clear.confidence > ambiguous.confidence > noise.confidence
    assert clear.confidence < 1.0
    assert noise.confidence < 0.5


def test_empty_input_is_handled(hf_nlu_engine) -> None:
    result = hf_nlu_engine.classify("   ")
    assert result.intent == "out_of_scope"
    assert result.confidence == 0.0


def test_nlu_engine_is_configurable_and_lazy(settings) -> None:
    """The engine exposes the interface without having loaded weights."""
    from app.nlu import build_nlu

    engine = build_nlu(settings)  # settings.nlu_backend == "hf" under tests
    assert isinstance(engine, HFIntentNLU)
    assert engine.is_loaded is False, "constructing an engine must not load a model"


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,name,value",
    [
        ("book me in for next Tuesday please", "when", "next tuesday"),
        ("can we do 3pm tomorrow", "when", None),  # two phrases; just check the name
        ("my email is sam@example.com", "email", "sam@example.com"),
        ("it's order 4471 that's wrong", "reference", "4471"),
        ("call me on +44 7911 123456", "phone", None),
    ],
)
def test_extract_slots(text, name, value) -> None:
    slots = {s.name: s.value for s in extract_slots(text)}
    assert name in slots, f"{text!r} did not yield a {name!r} slot: {slots}"
    if value is not None:
        assert slots[name] == value


def test_no_slots_from_plain_text() -> None:
    assert extract_slots("hello there") == ()
    assert extract_slots("") == ()
