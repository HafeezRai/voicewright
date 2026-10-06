"""Retrieval ranking: threshold, top-k, ordering, determinism.

Uses a deterministic stand-in encoder -- two-dimensional unit vectors looked up
by exact text -- so every score is known in advance and no model is needed. The
properties under test (cut-offs, ordering, the empty fast path) are properties
of :class:`Retriever`, not of MiniLM.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.agents.store import AgentStore
from app.knowledge.retrieval import Retriever

ORG = "org_retrieval"


class FakeEmbedder:
    """Returns pre-set unit vectors, counting how often it is asked to encode."""

    model_id = "fake"

    def __init__(self, table: dict[str, list[float]]):
        self._table = table
        self.encode_calls = 0

    @staticmethod
    def _unit(vector: list[float]) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        return array / norm if norm else array

    def encode(self, texts):
        self.encode_calls += 1
        return np.array(
            [self._unit(self._table.get(text, [0.0, 0.0])) for text in texts],
            dtype=np.float32,
        )

    def encode_one(self, text):
        return self.encode([text])[0]


#: Query "q" is aligned with the first axis; each chunk's similarity is its
#: first component.
_VECTORS = {
    "q": [1.0, 0.0],
    "alpha": [1.0, 0.0],  # score 1.0
    "beta": [0.9, 0.43589],  # score ~0.9
    "gamma": [0.0, 1.0],  # score 0.0 -- below any sane threshold
}


def _vector(text: str) -> bytes:
    return FakeEmbedder._unit(_VECTORS[text]).tobytes()


@pytest.fixture
def store(tmp_path) -> AgentStore:
    store = AgentStore(tmp_path / "kb.db")
    store.create_org(ORG, "retrieval")
    return store


def _seed(store: AgentStore, agent_id: str, chunks: list[str]) -> None:
    store.add_document(
        ORG,
        agent_id,
        title="doc",
        source="text",
        text=" ".join(chunks),
        chunks=[(name, _vector(name), 2) for name in chunks],
    )


def test_no_documents_returns_empty_without_encoding(store) -> None:
    agent = store.create_agent(ORG, {"name": "Empty"})
    embedder = FakeEmbedder(_VECTORS)
    retriever = Retriever(embedder, store, top_k=3, min_score=0.0)
    assert retriever.retrieve(ORG, agent["id"], "q") == []
    assert embedder.encode_calls == 0, "no chunks must mean no embedding"


def test_threshold_and_top_k_are_both_applied(store) -> None:
    agent = store.create_agent(ORG, {"name": "A"})
    _seed(store, agent["id"], ["alpha", "beta", "gamma"])
    retriever = Retriever(FakeEmbedder(_VECTORS), store, top_k=2, min_score=0.5)

    results = retriever.retrieve(ORG, agent["id"], "q")
    assert [chunk.text for chunk in results] == ["alpha", "beta"]
    assert all(chunk.score >= 0.5 for chunk in results)


def test_results_are_ordered_best_first(store) -> None:
    agent = store.create_agent(ORG, {"name": "A"})
    _seed(store, agent["id"], ["gamma", "beta", "alpha"])  # deliberately unordered
    retriever = Retriever(FakeEmbedder(_VECTORS), store, top_k=3, min_score=0.0)
    scores = [chunk.score for chunk in retriever.retrieve(ORG, agent["id"], "q")]
    assert scores == sorted(scores, reverse=True)


def test_all_below_threshold_yields_nothing(store) -> None:
    agent = store.create_agent(ORG, {"name": "A"})
    _seed(store, agent["id"], ["gamma"])
    retriever = Retriever(FakeEmbedder(_VECTORS), store, top_k=3, min_score=0.5)
    assert retriever.retrieve(ORG, agent["id"], "q") == []


def test_retrieval_is_deterministic(store) -> None:
    agent = store.create_agent(ORG, {"name": "A"})
    _seed(store, agent["id"], ["alpha", "beta", "gamma"])
    retriever = Retriever(FakeEmbedder(_VECTORS), store, top_k=3, min_score=0.0)

    first = [(c.chunk_id, c.score) for c in retriever.retrieve(ORG, agent["id"], "q")]
    second = [(c.chunk_id, c.score) for c in retriever.retrieve(ORG, agent["id"], "q")]
    assert first == second
    assert first, "the seed produced at least one chunk"


def test_a_chunk_of_the_wrong_dimension_is_skipped(store) -> None:
    agent = store.create_agent(ORG, {"name": "A"})
    _seed(store, agent["id"], ["alpha"])
    # A chunk embedded by a different model cannot be compared.
    store.add_document(
        ORG,
        agent["id"],
        title="other",
        source="text",
        text="three-dim",
        chunks=[("three-dim", np.ones(3, dtype=np.float32).tobytes(), 3)],
    )
    retriever = Retriever(FakeEmbedder(_VECTORS), store, top_k=5, min_score=0.0)
    texts = [chunk.text for chunk in retriever.retrieve(ORG, agent["id"], "q")]
    assert texts == ["alpha"]


def test_to_snippets_returns_the_chunk_texts(store) -> None:
    agent = store.create_agent(ORG, {"name": "A"})
    _seed(store, agent["id"], ["alpha", "beta"])
    retriever = Retriever(FakeEmbedder(_VECTORS), store, top_k=2, min_score=0.0)
    chunks = retriever.retrieve(ORG, agent["id"], "q")
    assert retriever.to_snippets(chunks) == [chunk.text for chunk in chunks]


def test_retrieval_never_reads_another_agents_chunks(store) -> None:
    a = store.create_agent(ORG, {"name": "A"})
    b = store.create_agent(ORG, {"name": "B"})
    _seed(store, a["id"], ["alpha"])
    _seed(store, b["id"], ["gamma"])
    retriever = Retriever(FakeEmbedder(_VECTORS), store, top_k=5, min_score=0.0)
    assert [c.text for c in retriever.retrieve(ORG, a["id"], "q")] == ["alpha"]
