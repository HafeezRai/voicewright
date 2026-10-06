"""Pydantic request/response models for the agent and document API.

Separate from :mod:`app.agents.store` on purpose: the store speaks plain dicts
(rows from SQLite), and these are the wire contract. Keeping them apart means a
schema change to the table does not silently change the API, and vice versa.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentCreate(BaseModel):
    """Body of ``POST /agents``. Only ``name`` is required."""

    name: str = Field(min_length=1, max_length=120)
    #: Free-form label ("receptionist", "billing") -- not an enum, so an org can
    #: name its own roles without a code change.
    role: str = Field(default="", max_length=200)
    #: The persona handed to the LLM. Blank means the pipeline's configured
    #: default prompt is used for this agent.
    system_prompt: str = Field(default="", max_length=8000)
    #: A voice identifier. Free-form because TTS backends name voices
    #: differently; the pipeline does not yet switch voices per agent.
    voice: str = Field(default="", max_length=120)
    #: BCP-47-ish language tag ("en", "en-US"). Advisory for now.
    language: str = Field(default="", max_length=32)
    #: Spoken before the conversation starts, if the client chooses to play it.
    greeting: str = Field(default="", max_length=2000)
    #: LLM id for this agent. Blank uses the service default -- but note that
    #: only some models can call tools, and a tool-enabled agent on a model that
    #: cannot is a configuration error the API reports rather than hides.
    model: str = Field(default="", max_length=120)
    #: Host for that model ("ollama", "deepseek", ...). Blank uses the
    #: deployment's provider.
    provider: str = Field(default="", max_length=40)
    #: Tools this agent may call, as ToolSpec objects (name/description/params).
    tools: list[dict] = Field(default_factory=list, max_length=20)


class AgentUpdate(BaseModel):
    """Body of ``PUT /agents/{id}``. Every field optional; unset means unchanged.

    ``None`` is the sentinel for "leave as is", which is why the fields are
    ``| None`` rather than defaulting to ``""`` -- otherwise a partial update
    could not tell "clear the greeting" from "do not touch it".
    """

    name: str | None = Field(default=None, min_length=1, max_length=120)
    role: str | None = Field(default=None, max_length=200)
    system_prompt: str | None = Field(default=None, max_length=8000)
    voice: str | None = Field(default=None, max_length=120)
    language: str | None = Field(default=None, max_length=32)
    greeting: str | None = Field(default=None, max_length=2000)
    model: str | None = Field(default=None, max_length=120)
    provider: str | None = Field(default=None, max_length=40)
    tools: list[dict] | None = Field(default=None, max_length=20)


class AgentOut(BaseModel):
    """An agent as returned by the API."""

    id: str
    org_id: str
    name: str
    role: str
    system_prompt: str
    voice: str
    language: str
    greeting: str
    model: str
    provider: str
    tools: list[dict]
    created_at: str
    updated_at: str


class DocumentCreate(BaseModel):
    """Body of the JSON form of ``POST /agents/{id}/documents``."""

    title: str = Field(default="", max_length=200)
    #: The document body. The configured ``knowledge_max_document_chars`` cap is
    #: enforced by the route (a 422), because the bound lives in settings, not
    #: in this model. ``min_length=1`` only rejects the trivially empty body.
    text: str = Field(min_length=1)


class DocumentOut(BaseModel):
    """A stored document as returned by the API (never its chunks or vectors)."""

    id: str
    org_id: str
    agent_id: str
    title: str
    source: str
    char_count: int
    chunk_count: int
    created_at: str
