"""The intent catalogue, shared by every NLU backend.

One definition drives both backends so they are genuinely swappable:

* ``description`` is the natural-language criterion Laya scores a ``choice``
  question against;
* ``exemplars`` are the utterances the HuggingFace fallback embeds to build a
  per-intent centroid.

Keeping them in one place means adding an intent cannot silently work on one
backend and not the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntentSpec:
    name: str
    description: str
    exemplars: tuple[str, ...] = field(default_factory=tuple)


#: The intents this voice agent recognises. ``out_of_scope`` is intentionally
#: first-class rather than a catch-all failure: a production agent must be able
#: to say "I can't help with that" instead of forcing every utterance into a
#: real intent.
INTENTS: tuple[IntentSpec, ...] = (
    IntentSpec(
        name="book_appointment",
        description="scheduling or booking an appointment, meeting or callback",
        exemplars=(
            "I want to book an appointment",
            "can I schedule a meeting for next week",
            "I'd like to set up a callback please",
            "book me in for a consultation on Friday",
        ),
    ),
    IntentSpec(
        name="cancel_account",
        description="cancelling or closing an account, subscription or service",
        exemplars=(
            "I need to cancel my account",
            "please close my subscription",
            "I want to stop my service",
            "how do I terminate my membership",
        ),
    ),
    IntentSpec(
        name="return_policy",
        description="questions about returns, refunds, shipping or warranty policy",
        exemplars=(
            "what's your return policy",
            "can I get a refund on this item",
            "how long do I have to send something back",
            "do you cover shipping for returns",
        ),
    ),
    IntentSpec(
        name="billing_question",
        description="invoices, charges, payments, pricing or a disputed bill",
        exemplars=(
            "why was I charged twice this month",
            "I have a question about my invoice",
            "my bill is higher than I expected",
            "when will the payment be taken",
        ),
    ),
    IntentSpec(
        name="technical_support",
        description="bugs, errors, outages or something not working",
        exemplars=(
            "the app keeps crashing",
            "I can't log in",
            "your website is down",
            "I'm getting an error message",
        ),
    ),
    IntentSpec(
        name="agent_handoff",
        description="the user asks to speak to a human agent or a person",
        exemplars=(
            "let me talk to a human",
            "can I speak to a real person",
            "transfer me to an agent",
            "get me a manager please",
        ),
    ),
    IntentSpec(
        name="out_of_scope",
        description="anything else, including small talk unrelated to the service",
        exemplars=(
            "what's the weather like today",
            "tell me a joke",
            "who won the game last night",
            "do you like pizza",
        ),
    ),
)

#: Ordered intent names.
INTENT_NAMES: tuple[str, ...] = tuple(spec.name for spec in INTENTS)
#: Fallback intent when a backend cannot decide.
FALLBACK_INTENT = "out_of_scope"


def intent_spec(name: str) -> IntentSpec | None:
    for spec in INTENTS:
        if spec.name == name:
            return spec
    return None
