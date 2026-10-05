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

from app.config import Settings
from app.llm.base import LLMEngine, LLMUnavailableError
from app.llm.openai_compat import OpenAICompatLLM

#: Default endpoints per provider, used when ``llm_base_url`` is blank.
#: Every one of these exposes ``POST {base}/chat/completions``.
PROVIDER_BASE_URLS: dict[str, str] = {
    "ollama": "http://127.0.0.1:11434/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}


def resolve_base_url(settings: Settings) -> str:
    """Explicit base URL if configured, else the provider's default."""
    configured = (settings.llm_base_url or "").strip()
    if configured:
        return configured
    provider = (settings.llm_provider or "").strip().lower()
    if provider in PROVIDER_BASE_URLS:
        return PROVIDER_BASE_URLS[provider]
    known = ", ".join(sorted(PROVIDER_BASE_URLS))
    raise LLMUnavailableError(
        f"unknown LLM provider {settings.llm_provider!r} and no "
        f"AI_ENGINE_LLM_BASE_URL set; known providers: {known}"
    )


def build_llm(settings: Settings) -> LLMEngine:
    """Construct the configured LLM engine (nothing is contacted until first use)."""
    if settings.llm_backend != "openai_compat":  # pragma: no cover - Literal-typed
        raise LLMUnavailableError(f"unknown LLM backend: {settings.llm_backend!r}")

    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return OpenAICompatLLM(
        resolve_base_url(settings),
        settings.llm_model,
        provider=settings.llm_provider,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        connect_timeout_s=settings.llm_connect_timeout_s,
        stream=settings.llm_stream,
        extra_body=settings.llm_extra_body,
        allow_reasoning_models=settings.llm_allow_reasoning_models,
        max_retries=settings.llm_max_retries,
    )
