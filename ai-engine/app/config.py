"""Configuration layer for ai-engine.

Settings are loaded from the process environment and validated on construction,
with defaults chosen so the service boots from an entirely empty environment.

Phase 1 declared the model/VAD fields so downstream phases had one place to read
them from. Phase 3 filled those fields in with the real STT/NLU/TTS stack. The
intent-classification stage has since been replaced by a generative LLM, so the
``AI_ENGINE_NLU_*`` fields are gone: `AI_ENGINE_LLM_*` configures the think
stage and `AI_ENGINE_EMBEDDING_MODEL` the encoder that outlived it.

Nothing here imports an ML runtime at import time: the model names below are
plain strings that are only resolved to weights when an engine is first used (see
``app/stt``, ``app/llm``, ``app/tts``). Device auto-detection probes torch, and
does so lazily -- inside the ``device`` field validator, only when settings are
constructed, and never fatally (see :func:`_auto_device`).

Note that ``AI_ENGINE_DEVICE`` does **not** affect the LLM: it selects the
torch device for STT and the embedder, while Ollama (or whichever provider is
configured) picks its own backend and does its own offload.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.prompts import DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["development", "staging", "production"]
# `auto` (the default) is resolved to a concrete runtime device at load time by
# `_normalise_device`: MPS where torch reports it, otherwise CPU.
Device = Literal["auto", "cpu", "cuda", "mps"]
STTBackend = Literal["faster-whisper"]
TTSBackend = Literal["sherpa"]
#: There is exactly one shipped LLM implementation: every provider we support
#: (Ollama, DeepSeek, vLLM, LM Studio, OpenAI) speaks the same OpenAI-compatible
#: ``/chat/completions`` protocol, so the provider is a *base URL and model*
#: choice, not a code path.
LLMBackend = Literal["openai_compat"]

#: Environment variables the retired NLU stage used to read. ``extra="ignore"``
#: means a stale one in an ``.env`` or a compose file is *silently* dropped,
#: which is the worst possible outcome: the deployment looks configured and
#: quietly is not. They are detected and logged loudly instead.
_LEGACY_NLU_PREFIX = "AI_ENGINE_NLU_"

#: Directory (relative to the ai-engine package root) used when the configured
#: model cache is absent or not writable -- the common case on a dev laptop,
#: where the container path ``/models`` does not exist.
_LOCAL_MODELS_DIRNAME = "models"

#: Directory (relative to the ai-engine package root) holding runtime state --
#: today the agent/knowledge SQLite file. Separate from ``models`` so wiping
#: downloaded weights does not delete the user's agents, and vice versa.
_LOCAL_DATA_DIRNAME = "data"


def _auto_device() -> Device:
    """Pick the fastest device available when none is configured explicitly.

    Apple Silicon's MPS backend when torch reports it usable, otherwise CPU.
    torch is imported lazily and the probe is fully guarded: an absent or broken
    torch (a slim container, an odd wheel) must degrade silently to ``"cpu"``
    rather than stop the settings -- and therefore the service -- from loading.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch absent or misconfigured
        pass
    return "cpu"


class Settings(BaseSettings):
    """Validated runtime settings for the ai-engine service."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        # `.env.example` has always said "copy this to .env", but nothing read
        # it -- the documented workflow silently did nothing, and the only way
        # to set a variable was to export it into the process. It is gitignored.
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- shared / cross-service ---------------------------------------------
    service_name: str = "ai-engine"
    config_prefix: str = Field(default="voicewright", alias="VOICEWRIGHT_CONFIG_PREFIX")
    env: Environment = Field(default="development", alias="VOICEWRIGHT_ENV")
    log_level: LogLevel = Field(default="INFO", alias="VOICEWRIGHT_LOG_LEVEL")

    # --- HTTP server ---------------------------------------------------------
    host: str = Field(default="0.0.0.0", alias="AI_ENGINE_HOST")
    port: int = Field(default=8001, ge=1, le=65535, alias="AI_ENGINE_PORT")

    # --- gRPC server (VoiceGateway contract) --------------------------------
    grpc_enabled: bool = Field(default=True, alias="AI_ENGINE_GRPC_ENABLED")
    grpc_port: int = Field(default=50052, ge=1, le=65535, alias="AI_ENGINE_GRPC_PORT")

    # --- CORS (Phase 4) ------------------------------------------------------
    # Comma-separated list of browser origins allowed to call the HTTP API.
    # Defaults to the local dashboard only and never to "*" — a wildcard would
    # let any page a user visits drive STT/LLM/TTS on their behalf.
    cors_allow_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="VOICEWRIGHT_CORS_ALLOW_ORIGINS",
    )

    # --- models --------------------------------------------------------------
    # In-container path for cached weights. On a dev host this usually does not
    # exist, in which case `models_root` falls back to `<repo>/ai-engine/models`.
    model_cache: str = Field(default="/models", alias="AI_ENGINE_MODEL_CACHE")
    #: When true (default) weights are loaded on first use, so the process -- and
    #: ``GET /health`` -- starts without downloading or allocating any model.
    lazy_models: bool = Field(default=True, alias="AI_ENGINE_LAZY_MODELS")
    #: ``auto`` (default) -> MPS on Apple Silicon, CPU elsewhere; an explicit
    #: ``cpu``/``cuda``/``mps`` always wins. See :func:`_auto_device`.
    device: Device = Field(default="auto", alias="AI_ENGINE_DEVICE")

    # --- STT --------------------------------------------------------------
    stt_backend: STTBackend = Field(default="faster-whisper", alias="AI_ENGINE_STT_BACKEND")
    stt_model: str = Field(default="base.en", alias="AI_ENGINE_STT_MODEL")
    stt_compute_type: str = Field(default="int8", alias="AI_ENGINE_STT_COMPUTE_TYPE")
    stt_beam_size: int = Field(default=1, ge=1, le=10, alias="AI_ENGINE_STT_BEAM_SIZE")

    # --- agent config store / knowledge base (step 2) -----------------------
    #: SQLite file holding orgs, agents, documents and chunk vectors. Blank
    #: (the default) resolves to ``<repo>/ai-engine/data/agents.db``; tests
    #: point this at a temp file. See :attr:`Settings.agent_db_path`.
    agent_db: str = Field(default="", alias="AI_ENGINE_AGENT_DB")
    #: Master switch for retrieval grounding. Off means every turn runs
    #: ungrounded even when the agent has documents, with no other change.
    knowledge_enabled: bool = Field(default=True, alias="AI_ENGINE_KNOWLEDGE_ENABLED")
    #: Target chunk size in *characters*. 500 chars is ~125 tokens of English
    #: (roughly 4 chars/token), which fits inside MiniLM's 128-token window --
    #: a longer chunk would be silently truncated by the tokenizer at embed
    #: time and so would half-exist in the index. See ``app.knowledge.chunking``.
    knowledge_chunk_chars: int = Field(
        default=500, ge=100, le=4000, alias="AI_ENGINE_KNOWLEDGE_CHUNK_CHARS"
    )
    #: Characters of the previous chunk repeated at the head of the next, so a
    #: sentence split across a boundary is still retrievable from either side.
    knowledge_chunk_overlap_chars: int = Field(
        default=80, ge=0, le=2000, alias="AI_ENGINE_KNOWLEDGE_CHUNK_OVERLAP_CHARS"
    )
    #: Maximum chunks injected into the system message per turn.
    knowledge_top_k: int = Field(default=3, ge=1, le=20, alias="AI_ENGINE_KNOWLEDGE_TOP_K")
    #: Minimum cosine similarity a chunk must reach to be injected. The point is
    #: the *empty* case: injecting the three best chunks on a query they do not
    #: answer makes the reply worse, not better. Measured with MiniLM on
    #: question-vs-statement pairs (not sentence-vs-sentence): a question the
    #: chunk answers scores ~0.85, an unrelated question still scores ~0.35 --
    #: MiniLM's cosine floor between two English sentences is high, so the old
    #: "unrelated is below 0.2" rule of thumb does not hold here. 0.4 clears the
    #: unrelated band with margin while leaving the relevant band untouched.
    knowledge_min_score: float = Field(
        default=0.4, ge=-1.0, le=1.0, alias="AI_ENGINE_KNOWLEDGE_MIN_SCORE"
    )
    #: Hard cap on a single pasted/uploaded document, so one request cannot
    #: fill the process's memory or the store.
    knowledge_max_document_chars: int = Field(
        default=200_000, ge=1, alias="AI_ENGINE_KNOWLEDGE_MAX_DOCUMENT_CHARS"
    )

    # --- knowledge (embeddings) ---------------------------------------------
    #: The MiniLM encoder that outlived the intent model: it is the retrieval
    #: encoder for the knowledge base (step 2). It used to be
    #: ``AI_ENGINE_NLU_HF_MODEL``, named for a fallback classifier that no longer
    #: exists.
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="AI_ENGINE_EMBEDDING_MODEL"
    )

    # --- LLM (the "think" stage) --------------------------------------------
    #: Generation replaces intent classification. One OpenAI-compatible client
    #: covers every provider; ``llm_provider`` only selects a default base URL.
    llm_backend: LLMBackend = Field(default="openai_compat", alias="AI_ENGINE_LLM_BACKEND")
    #: Free-form so a new OpenAI-compatible host needs no code change. Known
    #: names (``ollama``, ``deepseek``, ``openai``) get a default base URL.
    llm_provider: str = Field(default="ollama", alias="AI_ENGINE_LLM_PROVIDER")
    #: Blank -> derived from ``llm_provider`` (see ``app.llm.factory``).
    llm_base_url: str = Field(default="", alias="AI_ENGINE_LLM_BASE_URL")
    #: ``llama3:8b`` verbatim. The bare tag ``llama3`` does not exist in this
    #: environment and 404s on the very first request -- verified against
    #: Ollama's ``/api/tags``.
    llm_model: str = Field(default="llama3:8b", alias="AI_ENGINE_LLM_MODEL")
    #: Sent as a bearer token only when set; blank for a local Ollama.
    llm_api_key: SecretStr | None = Field(default=None, alias="AI_ENGINE_LLM_API_KEY")
    #: Per-provider keys, declared rather than read ad hoc from the environment.
    #: `extra="ignore"` means an undeclared variable in `.env` is dropped
    #: silently, so a key that is not a field here would appear to be set and
    #: never be seen -- which is exactly the confusing failure to avoid.
    deepseek_api_key: SecretStr | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_temperature: float = Field(
        default=0.4, ge=0.0, le=2.0, alias="AI_ENGINE_LLM_TEMPERATURE"
    )
    #: The cheapest latency lever there is: a demo reply is one or two sentences.
    llm_max_tokens: int = Field(default=160, ge=1, le=4096, alias="AI_ENGINE_LLM_MAX_TOKENS")
    #: Token budget for a turn that can call tools.
    #:
    #: Much larger than ``llm_max_tokens``, and it has to be. A spoken reply is
    #: two or three sentences, but a *thinking* model -- which is what the
    #: tool-capable models on this machine are -- spends its budget on hidden
    #: reasoning before it emits anything. Measured on qwen3.5:0.8b: at 160
    #: tokens it called a tool 1 time in 5 and returned ``finish_reason=length``
    #: the rest; at 512 it called one 4 times in 5.
    llm_tool_max_tokens: int = Field(default=1024, ge=64, le=16384, alias="AI_ENGINE_LLM_TOOL_MAX_TOKENS")
    #: How many tool rounds one turn may take before it must answer.
    llm_tool_rounds: int = Field(default=3, ge=1, le=8, alias="AI_ENGINE_LLM_TOOL_ROUNDS")
    #: Models offered in the agent editor. Free-form on the server; this is
    #: purely what the picker lists.
    #: "provider:model" pairs the editor offers, comma separated.
    llm_options_hint: str = Field(
        default="ollama:llama3:8b,ollama:qwen3.5:0.8b,ollama:mistral:latest,deepseek:deepseek-chat",
        alias="AI_ENGINE_LLM_OPTIONS_HINT",
    )
    #: Which of those can call tools. Verified by probing, not assumed: Ollama
    #: answers "does not support tools" for llama3 and mistral, and works for
    #: qwen3.5. Surfaced so the editor can warn instead of letting you build a
    #: tool-enabled agent that quietly never calls anything.
    llm_tool_capable_hint: str = Field(
        default="ollama:qwen3.5:0.8b,ollama:qwen3-coder:30b,deepseek:deepseek-chat",
        alias="AI_ENGINE_LLM_TOOL_CAPABLE_HINT",
    )
    #: Which of those can call tools. Verified by probing, not assumed: Ollama
    #: answers "does not support tools" for llama3 and mistral, and works for
    #: qwen3.5. Surfaced so the editor can warn instead of letting you build a
    #: tool-enabled agent that quietly never calls anything.
    llm_tool_capable_models_hint: str = Field(
        default="qwen3.5:0.8b,qwen3-coder:30b",
        alias="AI_ENGINE_LLM_TOOL_CAPABLE_MODELS",
    )
    #: Per-chunk read timeout under streaming, so a long generation is fine as
    #: long as tokens keep arriving.
    llm_timeout_s: float = Field(default=30.0, gt=0.0, alias="AI_ENGINE_LLM_TIMEOUT_S")
    #: Short on purpose: a dead endpoint must become a clean 503, not a hang.
    llm_connect_timeout_s: float = Field(
        default=2.0, gt=0.0, alias="AI_ENGINE_LLM_CONNECT_TIMEOUT_S"
    )
    #: 0 = never retry. A retry doubles the latency of a turn the caller has
    #: already abandoned. Connection failures only; read timeouts are not retried.
    llm_max_retries: int = Field(default=0, ge=0, le=5, alias="AI_ENGINE_LLM_MAX_RETRIES")
    #: Stream the transport even for a whole-reply call, so first-token latency
    #: is measurable (and a cold model load is distinguishable from a slow one).
    llm_stream: bool = Field(default=True, alias="AI_ENGINE_LLM_STREAM")
    #: The agent's persona. Overridable per request via ``POST /pipeline``.
    llm_system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT, alias="AI_ENGINE_LLM_SYSTEM_PROMPT"
    )
    #: How many prior user/assistant exchanges to replay. 0 disables history.
    llm_history_turns: int = Field(
        default=6, ge=0, le=50, alias="AI_ENGINE_LLM_HISTORY_TURNS"
    )
    #: Warm the model in a background thread at startup. The first turn on a
    #: cold Ollama pays a multi-GB load; this moves that cost off the caller.
    llm_warmup: bool = Field(default=True, alias="AI_ENGINE_LLM_WARMUP")
    #: Extra top-level fields merged into the request body. Ollama's
    #: ``keep_alive`` keeps the model resident between turns.
    llm_extra_body: dict[str, Any] = Field(
        default_factory=lambda: {"keep_alive": "30m"}, alias="AI_ENGINE_LLM_EXTRA_BODY"
    )
    #: Spoken when the provider is unreachable. Empty (the default) fails loudly
    #: with a 503; setting it keeps a demo audible when the model is down.
    llm_fallback_reply: str = Field(default="", alias="AI_ENGINE_LLM_FALLBACK_REPLY")
    #: Reasoning models stream chain-of-thought that the TTS would read aloud.
    #: Off by default; see ``app.llm.openai_compat.REASONING_MODEL_MARKERS``.
    llm_allow_reasoning_models: bool = Field(
        default=False, alias="AI_ENGINE_LLM_ALLOW_REASONING_MODELS"
    )

    # --- TTS --------------------------------------------------------------
    tts_backend: TTSBackend = Field(default="sherpa", alias="AI_ENGINE_TTS_BACKEND")
    #: Model directory name under ``<models_root>/tts`` (a Piper/VITS voice).
    tts_model: str = Field(
        default="vits-piper-en_US-amy-low", alias="AI_ENGINE_TTS_MODEL"
    )
    tts_speaker_id: int = Field(default=0, ge=0, alias="AI_ENGINE_TTS_SPEAKER_ID")
    tts_speed: float = Field(default=1.0, gt=0.0, le=4.0, alias="AI_ENGINE_TTS_SPEED")
    #: Size of the PCM chunks the pipeline emits for a synthesised utterance.
    tts_chunk_samples: int = Field(
        default=3200, ge=160, le=160000, alias="AI_ENGINE_TTS_CHUNK_SAMPLES"
    )

    # --- voice activity detection (declared Phase 1; owned by backend-core) ---
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0, alias="AI_ENGINE_VAD_THRESHOLD")
    sample_rate_hz: int = Field(default=16000, gt=0, alias="AI_ENGINE_SAMPLE_RATE_HZ")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("device", mode="before")
    @classmethod
    def _normalise_device(cls, value: object) -> object:
        if isinstance(value, str):
            normalised = value.lower()
            # `auto` (also the field default) is resolved here, so the stored
            # value is always a concrete device and an explicit choice is never
            # overridden by detection.
            return _auto_device() if normalised == "auto" else normalised
        return value

    @field_validator("stt_backend", "tts_backend", "llm_backend", mode="before")
    @classmethod
    def _normalise_choice(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        """Lower-case the provider name: it keys the default base-URL table."""
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _warn_on_retired_nlu_config(self) -> Settings:
        """Shout about ``AI_ENGINE_NLU_*`` instead of silently ignoring it.

        ``extra="ignore"`` is deliberate -- an unknown variable must not stop the
        service from booting -- but silence is exactly the wrong response to a
        *retired* one: the deployment would look configured while the setting it
        is counting on does nothing. Nothing here raises; the service still
        starts, it just says so.
        """
        legacy = sorted(
            key for key in os.environ if key.upper().startswith(_LEGACY_NLU_PREFIX)
        )
        if legacy:
            logger.warning(
                "ignoring retired NLU configuration %s: intent classification was "
                "replaced by the LLM stage. Use AI_ENGINE_LLM_* instead, and "
                "AI_ENGINE_EMBEDDING_MODEL for the encoder.",
                ", ".join(legacy),
            )
        return self

    @property
    def llm_options(self) -> list[tuple[str, str]]:
        """``(provider, model)`` pairs the editor should offer."""
        pairs: list[tuple[str, str]] = []
        for entry in self.llm_options_hint.split(","):
            entry = entry.strip()
            if not entry:
                continue
            provider, _, model = entry.partition(":")
            if provider and model:
                pairs.append((provider.strip(), model.strip()))
        return pairs

    @property
    def llm_tool_capable(self) -> set[tuple[str, str]]:
        capable: set[tuple[str, str]] = set()
        for entry in self.llm_tool_capable_hint.split(","):
            provider, _, model = entry.strip().partition(":")
            if provider and model:
                capable.add((provider.strip(), model.strip()))
        return capable

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def cors_origins(self) -> list[str]:
        """Parsed CORS allow-list (comma-separated; blank entries dropped)."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def models_root(self) -> Path:
        """Writable directory for cached model weights.

        Prefers the configured ``model_cache``; if that path is missing or not
        writable (the usual case on a dev laptop, where ``/models`` is the
        container path) it falls back to ``<repo>/ai-engine/models``.
        """
        configured = Path(self.model_cache)
        if configured.is_dir() and os.access(configured, os.W_OK):
            return configured
        fallback = Path(__file__).resolve().parents[1] / _LOCAL_MODELS_DIRNAME
        return fallback

    @property
    def hf_home(self) -> Path:
        """Value exported as ``HF_HOME``: keeps HuggingFace off ``~/.cache``.

        Everything HuggingFace downloads must land inside the model cache, or a
        container re-downloads it on every restart and the ``/models`` volume
        does not do its job.
        """
        return self.models_root / "hf"

    @property
    def hf_hub_cache(self) -> Path:
        """The HuggingFace *hub* cache (``HF_HOME/hub``).

        This is where ``huggingface_hub`` puts ``models--<org>--<name>``
        directories when it is driven by ``HF_HOME``, and it is what the STT and
        engines are pointed at explicitly so every stage shares one
        layout instead of three.
        """
        return self.hf_home / "hub"

    @property
    def tts_model_dir(self) -> Path:
        """Directory holding the Piper/VITS voice referenced by ``tts_model``."""
        return self.models_root / "tts" / self.tts_model

    @property
    def data_dir(self) -> Path:
        """Writable directory for service state (the agent store)."""
        return Path(__file__).resolve().parents[1] / _LOCAL_DATA_DIRNAME

    @property
    def agent_db_path(self) -> Path:
        """The SQLite file backing the agent store and knowledge base.

        ``AI_ENGINE_AGENT_DB`` when set (a temp file under pytest), otherwise
        ``<repo>/ai-engine/data/agents.db``. The parent directory is created on
        demand by the store, not here -- a property must stay side-effect free.
        """
        return Path(self.agent_db) if self.agent_db else self.data_dir / "agents.db"


def configure_hf_home(settings: Settings) -> None:
    """Point HuggingFace at the model cache for this process.

    ``setdefault`` so an operator who has deliberately exported ``HF_HOME``
    keeps control; otherwise the first engine to load pins it to the model cache.
    Engines that accept an explicit cache directory are given
    :attr:`Settings.hf_hub_cache` regardless, so the layout is the same either way.
    """
    os.environ.setdefault("HF_HOME", str(settings.hf_home))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
