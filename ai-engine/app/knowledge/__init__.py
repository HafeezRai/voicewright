"""Knowledge stage -- local embeddings plus the retrieval (RAG) step built on them.

``app.knowledge.embeddings`` is the MiniLM encoder that outlived the intent
classifier. Step 2 turned it into a knowledge base: :mod:`app.knowledge.chunking`
splits a document into retrievable pieces, :mod:`app.knowledge.retrieval` ranks
those pieces against an utterance, and the pipeline splices the winners into the
system prompt through its one documented seam.
"""

from __future__ import annotations

from app.knowledge.chunking import chunk_text
from app.knowledge.embeddings import (
    EmbeddingModel,
    EmbeddingUnavailableError,
    build_embedder,
)
from app.knowledge.retrieval import RetrievedChunk, Retriever

__all__ = [
    "EmbeddingModel",
    "EmbeddingUnavailableError",
    "RetrievedChunk",
    "Retriever",
    "build_embedder",
    "chunk_text",
]
