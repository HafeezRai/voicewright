"""Lightweight, backend-agnostic slot extraction.

Both NLU backends share this. It is deliberately rule-based: the intents this
agent handles need a handful of high-precision patterns (a date phrase, an email,
a phone number, an order reference), and regex gets those right *and* explainable,
where a generative extractor would add latency and failure modes for no gain at
this scale. When a real slot-filling model lands, it replaces this module behind
the same function signature.
"""

from __future__ import annotations

import re

from app.nlu.base import Slot

# Order-references we can spot without a schema: "#1234", "order 1234", "ORD-9912".
_ORDER_RE = re.compile(
    r"\b(?:order|ref(?:erence)?|invoice|ticket|case)\s*(?:number|no\.?|#)?\s*[:#]?\s*"
    r"([A-Z]{0,4}-?\d{3,12})\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:(?:\+|00)\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{2,4}")
# Date/time phrases: weekdays, relative days, clock times, explicit dates.
_WHEN_RE = re.compile(
    r"\b(?:"
    r"today|tomorrow|tonight|this (?:morning|afternoon|evening|week|weekend|month)|"
    r"next (?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|at \d{1,2}(?::\d{2})?|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r")\b",
    re.IGNORECASE,
)


def extract_slots(text: str) -> tuple[Slot, ...]:
    """Return every slot this module can recognise in ``text``.

    Only the *first* match of each kind is reported -- for a single-turn voice
    request a second phone number or date is almost always a mis-hearing, and
    guessing which one the user meant is worse than surfacing one candidate.
    """
    slots: list[Slot] = []
    if not text:
        return ()

    when = _WHEN_RE.search(text)
    if when:
        slots.append(Slot(name="when", value=when.group(0).strip().lower(), confidence=0.9))

    email = _EMAIL_RE.search(text)
    if email:
        slots.append(Slot(name="email", value=email.group(0), confidence=0.95))

    order = _ORDER_RE.search(text)
    if order:
        slots.append(Slot(name="reference", value=order.group(1).upper(), confidence=0.85))

    # Skip the phone pattern when it only re-matched an explicit date (a bare
    # "2024-01-02" is not a phone number) -- and require at least 7 digits.
    phone = _PHONE_RE.search(text)
    if phone:
        digits = re.sub(r"\D", "", phone.group(0))
        if len(digits) >= 7 and not _WHEN_RE.fullmatch(phone.group(0).strip()):
            slots.append(Slot(name="phone", value=phone.group(0).strip(), confidence=0.7))

    return tuple(slots)
