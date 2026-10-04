"""HuggingFace NLU fallback.

A small, permissively licensed sentence-embedding model
(``sentence-transformers/all-MiniLM-L6-v2``, Apache-2.0, ~90 MB) embeds the
utterance and every intent's exemplars; the utterance is assigned to the nearest
intent centroid by cosine similarity, and the similarities are softmaxed into a
confidence.

This is a semantic router rather than a fine-tuned classifier -- it needs no
labelled in-domain data, which is the right trade for a *fallback*: it exists so
the pipeline keeps working (and its tests keep passing) when Laya cannot be
loaded, not to beat it on accuracy. The exemplars live in
:mod:`app.nlu.schema`, shared with the Laya backend.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from app.nlu.base import IntentResult, NLUEngine, NLUUnavailableError
from app.nlu.schema import FALLBACK_INTENT, INTENTS
from app.nlu.slots import extract_slots

logger = logging.getLogger(__name__)

#: Cosine similarities live in [-1, 1], and measured margins on this intent set
#: are ~0.5 for a clear utterance but ~0.09 for an ambiguous one and ~0.01 for
#: noise. Scaling by 6 before the softmax maps those to roughly 0.95 / 0.6 / 0.15
#: -- i.e. the confidence actually tracks how decidable the utterance was. The
#: argmax is scale-invariant, so this only shapes the reported confidence.
_DEFAULT_SCALE = 6.0


class HFIntentNLU(NLUEngine):
    """Nearest-centroid intent classifier over MiniLM embeddings."""

    name = "hf"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        scale: float = _DEFAULT_SCALE,
        cache_dir: str | None = None,
    ):
        self._model_id = model
        self._device = device
        self._scale = scale
        self._cache_dir = cache_dir
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._centroids: np.ndarray | None = None
        self._labels: tuple[str, ...] = ()
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._centroids is not None

    def _load(self) -> None:
        if self._model is not None and self._centroids is not None:
            return
        with self._lock:
            if self._model is not None and self._centroids is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except Exception as exc:  # pragma: no cover - import guard
                raise NLUUnavailableError(
                    "transformers/torch are not installed; the HF NLU fallback needs them"
                ) from exc

            self._torch = torch
            device = "cuda" if (self._device == "cuda" and torch.cuda.is_available()) else "cpu"
            logger.info("loading HF intent model=%s device=%s", self._model_id, device)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, cache_dir=self._cache_dir
                )
                self._model = AutoModel.from_pretrained(self._model_id, cache_dir=self._cache_dir)
            except Exception as exc:
                raise NLUUnavailableError(
                    f"failed to load HF model {self._model_id!r}: {exc}"
                ) from exc
            self._model.to(device)
            self._model.eval()
            self._device = device
            self._centroids, self._labels = self._build_centroids()

    def load(self) -> None:
        self._load()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._centroids = None
        self._torch = None

    # -- embeddings ---------------------------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised sentence embeddings."""
        torch = self._torch
        enc = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def _build_centroids(self) -> tuple[np.ndarray, tuple[str, ...]]:
        labels: list[str] = []
        centroids: list[np.ndarray] = []
        for spec in INTENTS:
            exemplars = list(spec.exemplars) or [spec.description]
            vectors = self._embed(exemplars)
            centroid = vectors.mean(axis=0)
            norm = float(np.linalg.norm(centroid)) or 1.0
            centroids.append(centroid / norm)
            labels.append(spec.name)
        return np.vstack(centroids).astype(np.float32), tuple(labels)

    # -- inference ----------------------------------------------------------
    def classify(self, text: str) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(intent=FALLBACK_INTENT, confidence=0.0, backend=self.name)
        self._load()
        assert self._centroids is not None  # for type checkers; set by _load

        vector = self._embed([text])[0]
        sims = self._centroids @ vector  # both sides are unit vectors

        logits = sims * self._scale
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        best = int(np.argmax(probs))
        scores = {label: float(probs[i]) for i, label in enumerate(self._labels)}
        return IntentResult(
            intent=self._labels[best],
            confidence=float(probs[best]),
            slots=extract_slots(text),
            scores=scores,
            backend=self.name,
            raw={"similarities": {label: float(sims[i]) for i, label in enumerate(self._labels)}},
        )
