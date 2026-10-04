"""NLU backend selection.

``AI_ENGINE_NLU_BACKEND`` picks the backend:

* ``laya`` -- Laya only; a load failure is an error.
* ``hf``   -- the HuggingFace fallback only.
* ``auto`` -- try Laya on first use, and fall back to the HuggingFace classifier
  *permanently* if it cannot load. The switch is decided lazily (on the first
  real classification) so that neither backend is downloaded at import or at
  ``GET /health`` time, and it is logged loudly so an operator can see which one
  is actually answering.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.nlu.base import IntentResult, NLUEngine, NLUUnavailableError
from app.nlu.hf_engine import HFIntentNLU
from app.nlu.laya_engine import LayaNLU

logger = logging.getLogger(__name__)


class AutoNLU(NLUEngine):
    """Prefers Laya, degrades to the HF classifier if Laya will not load."""

    def __init__(self, primary: NLUEngine, fallback: NLUEngine):
        self._primary = primary
        self._fallback = fallback
        self._active: NLUEngine | None = None
        self._resolved = False

    @property
    def name(self) -> str:  # type: ignore[override]
        if self._active is not None:
            return self._active.name
        return f"auto({self._primary.name}->{self._fallback.name})"

    @property
    def is_loaded(self) -> bool:
        return self._active is not None and self._active.is_loaded

    def _resolve(self) -> NLUEngine:
        if self._resolved and self._active is not None:
            return self._active
        try:
            self._primary.load()
            self._active = self._primary
            logger.info("NLU backend resolved to %s", self._primary.name)
        except NLUUnavailableError as exc:
            logger.warning(
                "Laya unavailable (%s); falling back to %s", exc, self._fallback.name
            )
            self._active = self._fallback
        self._resolved = True
        return self._active

    def load(self) -> None:
        self._resolve()

    def close(self) -> None:
        self._primary.close()
        self._fallback.close()
        self._active = None
        self._resolved = False

    def classify(self, text: str) -> IntentResult:
        engine = self._resolve()
        try:
            return engine.classify(text)
        except NLUUnavailableError as exc:
            if engine is self._fallback:
                raise
            logger.warning("Laya failed mid-flight (%s); degrading to %s", exc, self._fallback.name)
            self._active = self._fallback
            return self._fallback.classify(text)


def build_nlu(settings: Settings) -> NLUEngine:
    """Construct the configured NLU engine (weights load on first use)."""
    backend = settings.nlu_backend
    if backend == "hf":
        return _build_hf(settings)
    if backend == "laya":
        return _build_laya(settings)

    # auto
    return AutoNLU(_build_laya(settings), _build_hf(settings))


def _build_laya(settings: Settings) -> NLUEngine:
    subfolder = None
    if settings.nlu_model.rstrip("/").endswith(("/multilingual", "/typed-decisions")):
        subfolder = settings.nlu_model.rstrip("/").rsplit("/", 1)[-1]
    return LayaNLU(
        settings.nlu_model,
        device=settings.device,
        subfolder=subfolder,
        hf_home=str(settings.hf_home),
    )


def _build_hf(settings: Settings) -> NLUEngine:
    return HFIntentNLU(
        settings.nlu_hf_model,
        device=settings.device,
        cache_dir=str(settings.hf_hub_cache),
    )
