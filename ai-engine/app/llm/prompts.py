"""The default system prompt.

This is the agent's *persona* slot. ``AI_ENGINE_LLM_SYSTEM_PROMPT`` overrides
it and ``POST /pipeline?system_prompt=...`` overrides that per call, which is the
whole point of the change: a role is now configured by writing a prompt, not by
writing code.

The wording is load-bearing, not decoration. The reply is spoken by a TTS engine
that faithfully reads whatever it is given, so a model that answers with a
bulleted markdown list produces an agent that says "asterisk asterisk". The
prompt therefore asks for plain spoken prose and an explicit short length -- and
``app.pipeline.segmentation`` strips the markdown that models emit anyway.
"""

from __future__ import annotations

#: Used when neither the config nor the request supplies a system prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant speaking aloud to a caller. "
    "Answer in plain spoken prose, using one or two short sentences. "
    "Never use markdown, bullet points, numbered lists, headings, emoji, code "
    "or stage directions -- everything you write is read aloud by a text to "
    "speech engine. If you do not know something, say so briefly."
)
