"""Configuration layer for ai-engine.

Settings are loaded from the process environment and validated on construction,
with defaults chosen so the service boots from an entirely empty environment.

Phase 1 declared the model/VAD fields so downstream phases had one place to read
them from. Phase 3 fills those fields in with the *real* STT/NLU/TTS stack and
adds the knobs the pipeline needs (backend selection, compute type, gRPC port).

Nothing here imports an ML runtime at import time: the model names below are
plain strings that are only resolved to weights when an engine is first used (see
``app/stt``, ``app/nlu``, ``app/tts``). Device auto-detection probes torch, and
does so lazily -- inside the ``device`` field validator, only when settings are
constructed, and never fatally (see :func:`_auto_device`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["development", "staging", "production"]
# `auto` (the default) is resolved to a concrete runtime device at load time by
# `_normalise_device`: MPS where torch reports it, otherwise CPU.
Device = Literal["auto", "cpu", "cuda", "mps"]
STTBackend = Literal["faster-whisper"]
TTSBackend = Literal["sherpa"]
# `auto` prefers Laya and falls back to the HuggingFace classifier if Laya is
# unavailable (not installed, checkpoint unreachable, or failed to load).
NLUBackend = Literal["auto", "laya", "hf"]

#: Directory (relative to the ai-engine package root) used when the configured
#: model cache is absent or not writable -- the common case on a dev laptop,
#: where the container path ``/models`` does not exist.
_LOCAL_MODELS_DIRNAME = "models"


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
    )

    # --- shared / cross-service ---------------------------------------------
    service_name: str = "ai-engine"
    config_prefix: str = Field(default="cloudpso", alias="CLOUDEPSO_CONFIG_PREFIX")
    env: Environment = Field(default="development", alias="CLOUDEPSO_ENV")
    log_level: LogLevel = Field(default="INFO", alias="CLOUDEPSO_LOG_LEVEL")

    # --- HTTP server ---------------------------------------------------------
    host: str = Field(default="0.0.0.0", alias="AI_ENGINE_HOST")
    port: int = Field(default=8001, ge=1, le=65535, alias="AI_ENGINE_PORT")

    # --- gRPC server (VoiceGateway contract) --------------------------------
    grpc_enabled: bool = Field(default=True, alias="AI_ENGINE_GRPC_ENABLED")
    grpc_port: int = Field(default=50052, ge=1, le=65535, alias="AI_ENGINE_GRPC_PORT")

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

    # --- NLU --------------------------------------------------------------
    nlu_backend: NLUBackend = Field(default="auto", alias="AI_ENGINE_NLU_BACKEND")
    #: Laya checkpoint id (see https://huggingface.co/convaiinnovations/laya).
    nlu_model: str = Field(default="convaiinnovations/laya", alias="AI_ENGINE_NLU_MODEL")
    #: HuggingFace embedding model used by the fallback classifier.
    nlu_hf_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="AI_ENGINE_NLU_HF_MODEL"
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

    @field_validator("nlu_backend", "stt_backend", "tts_backend", mode="before")
    @classmethod
    def _normalise_choice(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @property
    def is_production(self) -> bool:
        return self.env == "production"

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
        HF-NLU engines are pointed at explicitly so all three stages share one
        layout instead of three.
        """
        return self.hf_home / "hub"

    @property
    def tts_model_dir(self) -> Path:
        """Directory holding the Piper/VITS voice referenced by ``tts_model``."""
        return self.models_root / "tts" / self.tts_model


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
