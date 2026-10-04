"""Natural-language understanding stage."""

from __future__ import annotations

from app.nlu.base import IntentResult, NLUEngine, NLUUnavailableError, Slot
from app.nlu.factory import AutoNLU, build_nlu
from app.nlu.hf_engine import HFIntentNLU
from app.nlu.laya_engine import LayaNLU
from app.nlu.schema import FALLBACK_INTENT, INTENT_NAMES, INTENTS, IntentSpec

__all__ = [
    "FALLBACK_INTENT",
    "INTENTS",
    "INTENT_NAMES",
    "AutoNLU",
    "HFIntentNLU",
    "IntentResult",
    "IntentSpec",
    "LayaNLU",
    "NLUEngine",
    "NLUUnavailableError",
    "Slot",
    "build_nlu",
]
