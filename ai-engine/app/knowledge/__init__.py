"""Knowledge stage -- local embeddings for the upcoming retrieval (RAG) step.

Nothing in the running pipeline reads this package yet. It exists so the
MiniLM encoder survives the removal of ``app.nlu``, where it used to live
bolted onto an intent classifier.
"""

from __future__ import annotations

from app.knowledge.embeddings import EmbeddingModel, EmbeddingUnavailableError

__all__ = ["EmbeddingModel", "EmbeddingUnavailableError"]
