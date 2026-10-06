"""Document/chunk persistence and its tenant scoping.

Store-level, with hand-made vector blobs -- no embedder, so this runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.agents.store import DEFAULT_ORG_ID, AgentStore

ORG_B = "org_kb_b"


@pytest.fixture
def store(tmp_path) -> AgentStore:
    store = AgentStore(tmp_path / "agents.db")
    store.create_org(ORG_B, "kb-tenant-b")
    return store


def _chunks(n: int, dim: int = 3) -> list[tuple[str, bytes, int]]:
    return [
        (f"chunk {i}", np.ones(dim, dtype=np.float32).tobytes(), dim) for i in range(n)
    ]


def test_add_document_stores_chunks_and_reports_the_count(store) -> None:
    agent = store.create_agent(DEFAULT_ORG_ID, {"name": "A"})
    doc = store.add_document(
        DEFAULT_ORG_ID, agent["id"], title="Policy", source="text", text="hi", chunks=_chunks(3)
    )
    assert doc["chunk_count"] == 3
    assert doc["char_count"] == 2
    assert store.agent_has_documents(DEFAULT_ORG_ID, agent["id"]) is True


def test_list_and_get_documents(store) -> None:
    agent = store.create_agent(DEFAULT_ORG_ID, {"name": "A"})
    doc = store.add_document(
        DEFAULT_ORG_ID, agent["id"], title="T", source="text", text="x", chunks=_chunks(1)
    )
    docs = store.list_documents(DEFAULT_ORG_ID, agent["id"])
    assert [d["id"] for d in docs] == [doc["id"]]
    assert store.get_document(DEFAULT_ORG_ID, doc["id"])["title"] == "T"
    assert store.get_document(DEFAULT_ORG_ID, "missing") is None


def test_delete_document_removes_its_chunks(store) -> None:
    agent = store.create_agent(DEFAULT_ORG_ID, {"name": "A"})
    doc = store.add_document(
        DEFAULT_ORG_ID, agent["id"], title="T", source="text", text="x", chunks=_chunks(2)
    )
    assert store.delete_document(DEFAULT_ORG_ID, doc["id"]) is True
    assert store.agent_has_documents(DEFAULT_ORG_ID, agent["id"]) is False
    assert store.list_chunk_vectors(DEFAULT_ORG_ID, agent["id"]) == []
    assert store.delete_document(DEFAULT_ORG_ID, doc["id"]) is False


def test_deleting_an_agent_cascades_to_documents_and_chunks(store) -> None:
    agent = store.create_agent(DEFAULT_ORG_ID, {"name": "A"})
    store.add_document(
        DEFAULT_ORG_ID, agent["id"], title="T", source="text", text="x", chunks=_chunks(2)
    )
    assert store.delete_agent(DEFAULT_ORG_ID, agent["id"]) is True
    assert store.list_chunk_vectors(DEFAULT_ORG_ID, agent["id"]) == []


def test_chunk_vectors_are_scoped_by_agent(store) -> None:
    a = store.create_agent(DEFAULT_ORG_ID, {"name": "A"})
    b = store.create_agent(DEFAULT_ORG_ID, {"name": "B"})
    store.add_document(
        DEFAULT_ORG_ID, a["id"], title="T", source="text", text="x", chunks=_chunks(2)
    )
    store.add_document(
        DEFAULT_ORG_ID, b["id"], title="T", source="text", text="y", chunks=_chunks(5)
    )
    assert len(store.list_chunk_vectors(DEFAULT_ORG_ID, a["id"])) == 2
    assert len(store.list_chunk_vectors(DEFAULT_ORG_ID, b["id"])) == 5


def test_documents_are_scoped_by_org(store) -> None:
    """Org B must not reach org A's documents at all."""
    agent = store.create_agent(DEFAULT_ORG_ID, {"name": "A"})
    doc = store.add_document(
        DEFAULT_ORG_ID, agent["id"], title="Secret", source="text", text="x", chunks=_chunks(1)
    )
    assert store.get_document(ORG_B, doc["id"]) is None
    assert store.list_documents(ORG_B, agent["id"]) == []
    assert store.delete_document(ORG_B, doc["id"]) is False
    # Still there for the real owner.
    assert store.get_document(DEFAULT_ORG_ID, doc["id"]) is not None
