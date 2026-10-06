"""Cosine-similarity retrieval over an agent's chunks.

The encoder (:class:`~app.knowledge.embeddings.EmbeddingModel`) already returns
**L2-normalised** vectors, so a dot product *is* the cosine similarity -- no
norm division here, and no chance of dividing by a zero norm.

Scale, stated honestly
----------------------
Retrieval is a brute-force scan: every chunk of the agent is scored against the
query, in Python, on every turn. That is correct and fast for a demo-scale
knowledge base -- a few hundred chunks score in well under a millisecond -- and
it is deliberately *not* pretending to scale. A real deployment moves this to an
ANN index (FAISS, pgvector) or at least a vectorised matrix product; the
interface here (:meth:`Retriever.retrieve` returns ranked chunks) is what a swap
would keep.

Why two cut-offs
----------------
Cosine similarity alone is not enough. On any query, *some* chunk is the most
similar, even when none of them are relevant, and injecting the best three
irrelevant chunks makes the model's answer worse than no grounding at all. So a
chunk is injected only if it clears ``min_score``, and at most ``top_k`` are
kept. When nothing clears the bar the result is an empty list, and the pipeline
adds nothing to the prompt -- the "nothing relevant" case, handled deliberately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.knowledge.embeddings import EmbeddingModel


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk that cleared the threshold, with the score that got it there."""

    chunk_id: str
    document_id: str
    text: str
    score: float


class Retriever:
    """Rank an agent's stored chunks against a query string."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        store,
        *,
        top_k: int = 3,
        min_score: float = 0.4,
    ):
        self._embedder = embedder
        self._store = store
        self._top_k = max(1, int(top_k))
        self._min_score = float(min_score)

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def min_score(self) -> float:
        return self._min_score

    def retrieve(self, org_id: str, agent_id: str, query: str) -> list[RetrievedChunk]:
        """Return the agent's chunks that clear the threshold, best first.

        The store is consulted *before* the encoder: an agent with no documents
        returns ``[]`` without loading MiniLM at all, which is what makes the
        no-knowledge path free. Raises ``EmbeddingUnavailableError`` if the
        encoder cannot load -- the caller decides whether that is fatal (it is
        not: the turn proceeds ungrounded).
        """
        rows = self._store.list_chunk_vectors(org_id, agent_id)
        if not rows:
            return []

        query_vector = np.asarray(self._embedder.encode_one(query), dtype=np.float32)
        if query_vector.size == 0:
            return []

        scored: list[RetrievedChunk] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size != query_vector.size:
                # A chunk embedded by a different model (dimension mismatch)
                # cannot be compared; skip rather than crash the turn.
                continue
            score = float(vector @ query_vector)
            if score >= self._min_score:
                scored.append(
                    RetrievedChunk(
                        chunk_id=row["id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        score=score,
                    )
                )

        # Tie-break on chunk id so the order is total: the same query over the
        # same store always yields the same sequence, which is what makes the
        # retrieval deterministic and testable.
        scored.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return scored[: self._top_k]

    def to_snippets(self, chunks: Sequence[RetrievedChunk]) -> list[str]:
        """The chunk texts, for splicing into the prompt."""
        return [chunk.text for chunk in chunks]
