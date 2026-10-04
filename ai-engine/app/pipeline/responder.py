"""Intent -> reply text.

There is no language model in this phase, and pretending otherwise would be the
wrong kind of shortcut. What the pipeline needs is *something to say* so the TTS
stage has real input and the end-to-end path is exercised honestly. This module
is that: a deterministic, inspectable mapping from intent to a short spoken
reply, plus slots woven in where they help.

It is also the single place a real response generator would slot in later
without touching the orchestrator.
"""

from __future__ import annotations

from app.nlu.base import IntentResult

_RESPONSES: dict[str, str] = {
    "book_appointment": "Sure, I can help you book an appointment.",
    "cancel_account": "I can help you cancel your account. Let me pull up your details.",
    "return_policy": "Our return policy allows returns within thirty days of delivery.",
    "billing_question": "Let me look into that billing question for you.",
    "technical_support": "I'm sorry you're having trouble. Let's get that fixed.",
    "agent_handoff": "Of course, I'll connect you with a human agent right away.",
    "out_of_scope": "I'm not sure I can help with that, but I can pass it to a colleague.",
}

_DEFAULT = "I'm sorry, I didn't quite catch that. Could you say it again?"


def respond(intent: IntentResult) -> str:
    """Build the agent's spoken reply for a classified intent."""
    reply = _RESPONSES.get(intent.intent, _DEFAULT)

    when = intent.slot("when")
    if when and intent.intent == "book_appointment":
        reply = f"{reply} I've noted {when}."
    elif when and intent.intent == "cancel_account":
        reply = f"{reply} Do you want that to take effect today?"

    return reply
