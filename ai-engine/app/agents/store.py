"""SQLite-backed agent configuration store.

Design notes that matter
------------------------

**Tenant scoping is not optional.** Every table carries ``org_id`` and every
public method takes it as the first argument. There is no method that reads or
writes a row without an org, so a caller cannot forget to scope a query -- the
type of leak that turns "we'll add tenancy later" into an incident.

**Connection per call, not a shared connection.** ``sqlite3.Connection`` objects
are not safe to share across threads, and FastAPI runs synchronous handlers on
``to_thread`` workers, so two requests really can hit the store at once. A
process-wide lock would serialise *reads* too, which is the wrong trade for a
read-mostly store. Instead every operation opens a short-lived connection,
does its work, and closes it -- SQLite connections to a local file are cheap,
and WAL journaling lets readers proceed while a writer holds the file lock. A
``timeout`` on connect turns the rare write/write collision into a brief wait
rather than an immediate ``database is locked``.

**The schema is created once, at construction**, including the ``documents``
and ``chunks`` tables that the knowledge base (step 2B) fills in. Creating them
here means there is exactly one place that knows the shape of the store.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The org every request falls back to when it names none. Seeded on init so
#: the service works with no setup, and so the API has something to scope to
#: before org management exists.
DEFAULT_ORG_ID = "org_default"
DEFAULT_ORG_NAME = "default"

#: Seconds to wait for a competing writer before raising. Generous relative to
#: the work done here (single-row writes) but short enough that a wedged writer
#: surfaces as an error rather than a hang.
_BUSY_TIMEOUT_S = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    id            TEXT PRIMARY KEY,
    org_id        TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    voice         TEXT NOT NULL DEFAULT '',
    language      TEXT NOT NULL DEFAULT '',
    greeting      TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agents_org ON agents(org_id);

CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    org_id     TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    agent_id   TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title      TEXT NOT NULL DEFAULT '',
    source     TEXT NOT NULL DEFAULT 'text',
    char_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_agent ON documents(agent_id);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    -- Denormalised on purpose: retrieval filters by (org_id, agent_id) and
    -- never needs to join back to documents, so the hot path is one indexed
    -- scan of one table.
    org_id      TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_agent ON chunks(org_id, agent_id);
"""

#: Columns of ``agents`` the API may set, in schema order. Keeps the create and
#: update statements honest about what an agent *is*.
_AGENT_FIELDS = ("name", "role", "system_prompt", "voice", "language", "greeting")


def _now() -> str:
    """UTC timestamp as ISO 8601, for storage in a TEXT column."""
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class AgentStore:
    """CRUD over orgs, agents, documents and chunk vectors, scoped by org."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        parent = Path(self._path).parent
        parent.mkdir(parents=True, exist_ok=True)
        with self._write() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO orgs (id, name, created_at) VALUES (?, ?, ?)",
                (DEFAULT_ORG_ID, DEFAULT_ORG_NAME, _now()),
            )

    @property
    def path(self) -> str:
        return self._path

    # -- plumbing -----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """A fresh, configured connection. Caller owns and must close it."""
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        # ON DELETE CASCADE is per-connection in SQLite, so it must be set on
        # every connection or deleting an agent would silently orphan its
        # documents and chunks.
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL persists in the file, but setting it is idempotent and cheap;
        # doing it here means a file created outside this class still gets it.
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """A connection inside a transaction: commit on success, rollback on error."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    # -- orgs ---------------------------------------------------------------
    def org_exists(self, org_id: str) -> bool:
        with self._read() as conn:
            row = conn.execute("SELECT 1 FROM orgs WHERE id = ?", (org_id,)).fetchone()
        return row is not None

    def create_org(self, org_id: str, name: str = "") -> str:
        """Provision an org. Idempotent.

        There is no org API -- orgs are provisioned out of band (the default one
        is seeded at construction). This exists so tests and operators can stand
        up a second tenant without one, which is what the isolation test needs.
        """
        with self._write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO orgs (id, name, created_at) VALUES (?, ?, ?)",
                (org_id, name or org_id, _now()),
            )
        return org_id

    def list_orgs(self) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM orgs ORDER BY created_at, id").fetchall()
        return [dict(row) for row in rows]

    # -- agents -------------------------------------------------------------
    def create_agent(self, org_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Insert an agent. Caller must have checked the org exists."""
        agent_id = _new_id()
        now = _now()
        values = {field: str(data.get(field) or "") for field in _AGENT_FIELDS}
        values["name"] = str(data["name"])
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO agents
                    (id, org_id, name, role, system_prompt, voice, language,
                     greeting, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    org_id,
                    values["name"],
                    values["role"],
                    values["system_prompt"],
                    values["voice"],
                    values["language"],
                    values["greeting"],
                    now,
                    now,
                ),
            )
        created = self.get_agent(org_id, agent_id)
        assert created is not None  # just written
        return created

    def get_agent(self, org_id: str, agent_id: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE id = ? AND org_id = ?",
                (agent_id, org_id),
            ).fetchone()
        return dict(row) if row else None

    def list_agents(self, org_id: str) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM agents WHERE org_id = ? ORDER BY created_at, id",
                (org_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_agent(
        self, org_id: str, agent_id: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply a partial update. Returns the updated row, or None if not found.

        Only keys present in ``changes`` are written; ``None`` values are
        dropped by the caller (``AgentUpdate`` uses None for "unchanged").
        """
        updates = {k: v for k, v in changes.items() if k in _AGENT_FIELDS and v is not None}
        if not updates:
            return self.get_agent(org_id, agent_id)
        assignments = ", ".join(f"{field} = ?" for field in updates)
        with self._write() as conn:
            cursor = conn.execute(
                f"UPDATE agents SET {assignments}, updated_at = ? WHERE id = ? AND org_id = ?",
                (*updates.values(), _now(), agent_id, org_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_agent(org_id, agent_id)

    def delete_agent(self, org_id: str, agent_id: str) -> bool:
        """Delete an agent and (by cascade) its documents and chunks."""
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM agents WHERE id = ? AND org_id = ?", (agent_id, org_id)
            )
            deleted = cursor.rowcount > 0
        return deleted

    # -- documents and chunks ----------------------------------------------
    def add_document(
        self,
        org_id: str,
        agent_id: str,
        *,
        title: str,
        source: str,
        text: str,
        chunks: Sequence[tuple[str, bytes, int]],
    ) -> dict[str, Any]:
        """Store a document and its embedded chunks in one transaction.

        ``chunks`` is a sequence of ``(chunk_text, embedding_bytes, dim)``. The
        whole insert is atomic: a document is never left half-indexed, which
        would show up as a document that lists but never retrieves.
        """
        doc_id = _new_id()
        now = _now()
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (id, org_id, agent_id, title, source, char_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, org_id, agent_id, title, source, len(text), now),
            )
            if chunks:
                conn.executemany(
                    """
                    INSERT INTO chunks
                        (id, org_id, agent_id, document_id, ordinal, text,
                         embedding, dim, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (_new_id(), org_id, agent_id, doc_id, ordinal, chunk_text, blob, dim, now)
                        for ordinal, (chunk_text, blob, dim) in enumerate(chunks)
                    ],
                )
        stored = self.get_document(org_id, doc_id)
        assert stored is not None  # just written
        return stored

    def get_document(self, org_id: str, document_id: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute(
                """
                SELECT d.*,
                       (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
                FROM documents d
                WHERE d.id = ? AND d.org_id = ?
                """,
                (document_id, org_id),
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self, org_id: str, agent_id: str) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                """
                SELECT d.*,
                       (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
                FROM documents d
                WHERE d.org_id = ? AND d.agent_id = ?
                ORDER BY d.created_at, d.id
                """,
                (org_id, agent_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, org_id: str, document_id: str) -> bool:
        """Delete a document and (by cascade) its chunks."""
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE id = ? AND org_id = ?",
                (document_id, org_id),
            )
            deleted = cursor.rowcount > 0
        return deleted

    def agent_has_documents(self, org_id: str, agent_id: str) -> bool:
        """Whether the agent has at least one chunk to retrieve from.

        The retrieval fast path: an agent with no chunks is answered from the
        prompt alone, with the encoder never consulted.
        """
        with self._read() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE org_id = ? AND agent_id = ? LIMIT 1",
                (org_id, agent_id),
            ).fetchone()
        return row is not None

    def list_chunk_vectors(self, org_id: str, agent_id: str) -> list[sqlite3.Row]:
        """Every chunk of an agent, with its raw vector blob, for scoring.

        Scoped by ``org_id`` and ``agent_id`` so retrieval for one agent can
        never read another agent's -- or another org's -- chunks.
        """
        with self._read() as conn:
            return conn.execute(
                """
                SELECT id, document_id, text, embedding
                FROM chunks
                WHERE org_id = ? AND agent_id = ?
                """,
                (org_id, agent_id),
            ).fetchall()
