"""Tenant-aware agent configuration store and its HTTP schema.

An *agent* is a configured persona -- name, role, system prompt, voice,
language, greeting -- owned by an *org*. Every row in every table carries
``org_id`` from day one, and every query is scoped by it, because retrofitting
tenancy onto a store whose reads are unscoped is how cross-tenant leaks happen.

The store is SQLite (stdlib ``sqlite3``): one file, no server, no new
dependency. It is deliberately small -- this is a demo-scale control plane, not
a multi-tenant database service.
"""

from __future__ import annotations

from app.agents.models import (
    AgentCreate,
    AgentOut,
    AgentUpdate,
    DocumentCreate,
    DocumentOut,
)
from app.agents.store import DEFAULT_ORG_ID, DEFAULT_ORG_NAME, AgentStore

__all__ = [
    "DEFAULT_ORG_ID",
    "DEFAULT_ORG_NAME",
    "AgentCreate",
    "AgentOut",
    "AgentStore",
    "AgentUpdate",
    "DocumentCreate",
    "DocumentOut",
]
