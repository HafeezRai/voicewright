"""Sentence embeddings -- the MiniLM primitive that outlived the intent model.

This is the ``_embed`` method of the old ``HFIntentNLU``, lifted out verbatim
(mean-pool over the attention mask, then L2-normalise) and stripped of the
centroid/softmax/intent machinery that existed only to turn a vector into one of
seven labels. Nothing about the vector itself changed.

Why keep it at all: the immediate next step is a knowledge base, and retrieval
needs exactly this -- a cheap, local, permissively licensed encoder
(``sentence-transformers/all-MiniLM-L6-v2``, Apache-2.0, ~90 MB) that turns text
into unit vectors whose dot product is cosine similarity. Deleting it along with
Laya would have meant re-downloading and re-writing it a week later.

That is also why ``torch``/``transformers``/``safetensors`` stay in
``requirements.txt`` while ``laya`` goes.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the embedding model cannot be loaded (missing package/weights)."""


class EmbeddingModel:
    """Lazily-loaded sentence encoder returning L2-normalised vectors.

    Thread-safe: :meth:`encode` guards the one-time load with a lock, and the
    forward pass itself runs under ``torch.no_grad`` on the loaded model. As with
    every other engine in this service, nothing is downloaded at import or at
    ``GET /health`` time.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
        max_length: int = 128,
    ):
        self._model_id = model
        self._device = device
        self._cache_dir = cache_dir
        self._max_length = max_length
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    # -- identity -----------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        """Load the tokenizer and model; idempotent and thread-safe."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except Exception as exc:  # pragma: no cover - import guard
                raise EmbeddingUnavailableError(
                    "transformers/torch are not installed; the embedding model needs them"
                ) from exc

            self._torch = torch
            device = "cuda" if (self._device == "cuda" and torch.cuda.is_available()) else "cpu"
            logger.info("loading embedding model=%s device=%s", self._model_id, device)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, cache_dir=self._cache_dir
                )
                self._model = AutoModel.from_pretrained(self._model_id, cache_dir=self._cache_dir)
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"failed to load embedding model {self._model_id!r}: {exc}"
                ) from exc
            self._model.to(device)
            self._model.eval()
            self._device = device

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None

    # -- inference ----------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` array of mean-pooled, L2-normalised vectors.

        An empty input yields an empty ``(0, 0)`` array rather than raising, so a
        caller that retrieved nothing does not have to special-case it.
        """
        items = [t for t in texts]
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        self.load()
        torch = self._torch
        enc = self._tokenizer(
            items,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        # Mean-pool over the real tokens (the mask excludes padding), then
        # normalise so a dot product is a cosine similarity.
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single string: an ``(dim,)`` unit vector."""
        vectors = self.encode([text])
        return vectors[0] if len(vectors) else np.zeros(0, dtype=np.float32)


def build_embedder(settings: Any) -> EmbeddingModel:
    """Construct the configured retrieval encoder (weights load on first use).

    ``settings`` is typed loosely to keep this module free of an ``app.config``
    import at module scope -- the same reason ``app.llm`` resolves its factory
    lazily. It is an :class:`~app.config.Settings` in practice; only three
    fields are read.
    """
    return EmbeddingModel(
        settings.embedding_model,
        device=settings.device,
        cache_dir=str(settings.hf_hub_cache),
    )
