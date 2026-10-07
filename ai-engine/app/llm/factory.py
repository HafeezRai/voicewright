"""LLM backend selection.

``AI_ENGINE_LLM_BACKEND`` picks the implementation; today there is exactly one
(``openai_compat``), so this is mostly about resolving *where* to talk to.

``AI_ENGINE_LLM_BASE_URL`` wins when set. Otherwise the base URL is derived from
``AI_ENGINE_LLM_PROVIDER``, which is what makes the provider swap a two-line
config change: DeepSeek, vLLM and LM Studio are the same client pointed
elsewhere. An unknown provider with no explicit base URL is an error rather than
a silent fall back to Ollama -- a typo in the provider name should not quietly
send the demo to the wrong host.
"""

from __future__ import annotations

import os

from app.config import Settings
from app.llm.base import PROVIDER_API_KEY_ENV, LLMEngine, LLMUnavailableError
from app.llm.openai_compat import OpenAICompatLLM

#: Default endpoints per provider, used when ``llm_base_url`` is blank.
#: Every one of these exposes ``POST {base}/chat/completions``.
PROVIDER_BASE_URLS: dict[str, str] = {
    "ollama": "http://127.0.0.1:11434/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}


def resolve_base_url(settings: Settings, provider: str | None = None) -> str:
    """Explicit base URL if configured, else the provider's default.

    ``provider`` overrides the service default, which is how one agent can talk
    to a hosted model while the rest of the deployment stays on a local one.
    """
    name = (provider or settings.llm_provider or "").strip().lower()
    configured = (settings.llm_base_url or "").strip()
    # An explicit base URL wins, but only for the *default* provider -- it is a
    # deployment-level statement, and applying it to a per-agent override would
    # silently send that agent to the wrong host.
    if configured and not provider:
        return configured
    if name in PROVIDER_BASE_URLS:
        return PROVIDER_BASE_URLS[name]
    known = ", ".join(sorted(PROVIDER_BASE_URLS))
    raise LLMUnavailableError(
        f"unknown LLM provider {name!r} and no AI_ENGINE_LLM_BASE_URL set; "
        f"known providers: {known}"
    )


def resolve_api_key(settings: Settings, provider: str | None = None) -> str | None:
    """The key for a provider: the configured one, or its conventional env var.

    Never logged, never echoed into an error.
    """
    name = (provider or settings.llm_provider or "").strip().lower()
    # Precedence, most specific first.
    #
    # 1. AI_ENGINE_LLM_API_KEY is *this deployment's* key for its own provider.
    #    It is the most deliberate statement anyone can make, and it must beat
    #    an inherited DEEPSEEK_API_KEY -- otherwise exporting a key for one
    #    experiment silently overrides the configured one.
    if name == (settings.llm_provider or "").strip().lower() and settings.llm_api_key:
        value = settings.llm_api_key.get_secret_value().strip()
        if value:
            return value
    # 2. The provider's own key, declared as a setting (so `.env` can carry it)
    #    or exported under its conventional name. This is what lets one agent
    #    use a hosted provider the deployment is not otherwise configured for.
    declared = getattr(settings, f"{name}_api_key", None)
    if declared is not None:
        value = declared.get_secret_value().strip()
        if value:
            return value
    env_name = PROVIDER_API_KEY_ENV.get(name)
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    # 3. Anything else configured, as a last resort.
    if settings.llm_api_key:
        return settings.llm_api_key.get_secret_value()
    return None


def build_llm(
    settings: Settings, *, model: str | None = None, provider: str | None = None
) -> LLMEngine:
    """Construct an LLM engine (nothing is contacted until first use).

    ``model`` and ``provider`` override the service defaults, which is how an
    agent that needs a hosted, tool-capable model gets one without changing the
    deployment's default.
    """
    if settings.llm_backend != "openai_compat":  # pragma: no cover - Literal-typed
        raise LLMUnavailableError(f"unknown LLM backend: {settings.llm_backend!r}")

    name = (provider or settings.llm_provider or "").strip().lower()
    return OpenAICompatLLM(
        resolve_base_url(settings, provider),
        model or settings.llm_model,
        provider=name,
        api_key=resolve_api_key(settings, provider),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        connect_timeout_s=settings.llm_connect_timeout_s,
        stream=settings.llm_stream,
        extra_body=settings.llm_extra_body,
        allow_reasoning_models=settings.llm_allow_reasoning_models,
        max_retries=settings.llm_max_retries,
    )
