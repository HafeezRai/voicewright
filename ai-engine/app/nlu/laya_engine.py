"""Laya NLU backend.

Laya is a non-autoregressive decision model: one forward pass over the state
(the user's utterance) answers every typed question attached to it. We ask it
one ``choice`` question for the intent, a ``noul`` (calibrated P(true)) question
that is its own intent, and keep the door open for further guardrail questions.

The dependency is young and effectively single-author, which is exactly why it
does not appear in any signature outside this module: everything is mapped into
:class:`~app.nlu.base.IntentResult` at the boundary. If Laya is unavailable the
factory falls back to :class:`~app.nlu.hf_engine.HFIntentNLU`.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from app.nlu.base import IntentResult, NLUEngine, NLUUnavailableError
from app.nlu.schema import FALLBACK_INTENT, INTENTS
from app.nlu.slots import extract_slots

logger = logging.getLogger(__name__)

#: Question id for the intent choice.
_INTENT_QID = "intent"
#: Question id for the "does the user want a human?" guardrail.
_HUMAN_QID = "wants_human"


class LayaNLU(NLUEngine):
    """Intent classification via a Laya checkpoint (``choice`` + ``noul``)."""

    name = "laya"

    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        *,
        device: str = "cpu",
        subfolder: str | None = None,
        revision: str | None = None,
        min_confidence: float | None = None,
        hf_home: str | None = None,
    ):
        self._model_id = model
        self._device = device
        self._subfolder = subfolder
        self._revision = revision
        self._min_confidence = min_confidence
        #: Laya calls ``huggingface_hub`` without a cache argument, so the only
        #: way to keep its ~840 MB checkpoint inside the model cache (rather than
        #: ``~/.cache/huggingface``, which a container loses on restart) is
        #: ``HF_HOME``.
        self._hf_home = hf_home
        self._agent: Any = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._agent is not None

    def _load(self) -> Any:
        if self._agent is not None:
            return self._agent
        with self._lock:
            if self._agent is not None:
                return self._agent
            try:
                import laya
            except Exception as exc:  # pragma: no cover - import guard
                raise NLUUnavailableError(
                    "laya is not installed; `pip install laya`"
                ) from exc
            if self._hf_home:
                os.environ.setdefault("HF_HOME", self._hf_home)
            kwargs: dict[str, Any] = {"device": self._device}
            if self._subfolder:
                kwargs["subfolder"] = self._subfolder
            if self._revision:
                kwargs["revision"] = self._revision
            logger.info("loading laya model=%s device=%s", self._model_id, self._device)
            try:
                self._agent = laya.load(self._model_id, **kwargs)
            except Exception as exc:
                raise NLUUnavailableError(
                    f"failed to load laya checkpoint {self._model_id!r}: {exc}"
                ) from exc
            return self._agent

    def load(self) -> None:
        self._load()

    def close(self) -> None:
        self._agent = None

    # -- questions ----------------------------------------------------------
    @staticmethod
    def _questions() -> dict[str, dict[str, Any]]:
        return {
            _INTENT_QID: {
                "type": "choice",
                "instructions": "What does the user want from the voice assistant?",
                "criteria": {spec.name: spec.description for spec in INTENTS},
            },
            _HUMAN_QID: {
                "type": "noul",
                "instructions": "Is the user asking to speak to a human agent?",
            },
        }

    # -- inference ----------------------------------------------------------
    def classify(self, text: str) -> IntentResult:
        agent = self._load()
        if not text or not text.strip():
            return IntentResult(intent=FALLBACK_INTENT, confidence=0.0, backend=self.name)

        result = agent.system_one(text, self._questions())
        answers = (result or {}).get("answers", {}) or {}

        intent_answer = answers.get(_INTENT_QID, {}) or {}
        intent = str(intent_answer.get("choice") or FALLBACK_INTENT)
        confidence = float(intent_answer.get("confidence", 0.0) or 0.0)

        # A confident "wants a human" overrides the choice question: being
        # escalated is what the user asked for, regardless of topic.
        human_answer = answers.get(_HUMAN_QID, {}) or {}
        human_p = float(human_answer.get("noul", 0.0) or 0.0)

        slots = list(extract_slots(text))
        if human_p >= 0.5:
            intent = "agent_handoff"
            confidence = human_p

        # Ranking, when the checkpoint exposes per-option probabilities. The
        # explicit `choice` wins; the ranking is only a fallback for the label
        # itself and is always reported so callers can gate on margin.
        scores = _extract_scores(intent_answer)
        if scores and intent not in scores:
            intent = max(scores, key=scores.__getitem__)

        return IntentResult(
            intent=intent,
            confidence=confidence,
            slots=tuple(slots),
            scores=scores,
            backend=self.name,
            raw={"answers": answers, "usage": (result or {}).get("usage")},
        )


def _extract_scores(answer: dict[str, Any]) -> dict[str, float]:
    """Pull the per-option probability map out of a choice answer, if present."""
    for key in ("probabilities", "probs", "scores"):
        value = answer.get(key)
        if isinstance(value, dict) and value:
            return {str(k): float(v) for k, v in value.items()}
    return {}
