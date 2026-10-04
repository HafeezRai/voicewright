"""Configuration layer for backend-core.

Settings are loaded from the process environment (populated by docker-compose,
a .env file, or the shell) and validated on construction. Defaults are chosen so
the service boots with an entirely empty environment.

`Settings()` reads the environment at *instantiation* time, which keeps it
trivially testable: mutate the environment, construct a fresh instance, assert.
`get_settings()` adds process-wide caching for request-path use.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    """Validated runtime settings for the backend-core service."""

    model_config = SettingsConfigDict(
        # No env prefix: each field declares its full environment variable name
        # via `alias`, matching the names documented in .env.example.
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- shared / cross-service ---------------------------------------------
    service_name: str = "backend-core"
    config_prefix: str = Field(default="cloudpso", alias="CLOUDEPSO_CONFIG_PREFIX")
    env: Environment = Field(default="development", alias="CLOUDEPSO_ENV")
    log_level: LogLevel = Field(default="INFO", alias="CLOUDEPSO_LOG_LEVEL")

    # --- HTTP / WebSocket server --------------------------------------------
    host: str = Field(default="0.0.0.0", alias="BACKEND_CORE_HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="BACKEND_CORE_PORT")

    # --- gRPC ----------------------------------------------------------------
    grpc_port: int = Field(default=50051, ge=1, le=65535, alias="BACKEND_CORE_GRPC_PORT")

    # --- downstream services -------------------------------------------------
    ai_engine_url: str = Field(default="http://ai-engine:8001", alias="AI_ENGINE_URL")

    # --- Phase 2: audio / VAD / barge-in ------------------------------------
    # Audio is normalised to 16 kHz mono before it reaches the VAD; Silero VAD
    # requires frames of exactly 512 samples (32 ms) at that rate.
    audio_sample_rate: int = Field(
        default=16000, ge=8000, le=48000, alias="BACKEND_CORE_AUDIO_SAMPLE_RATE"
    )
    vad_frame_samples: int = Field(
        default=512, ge=64, le=8192, alias="BACKEND_CORE_VAD_FRAME_SAMPLES"
    )
    # "silero" = real ONNX speech model; "energy" = deterministic RMS fallback.
    vad_backend: Literal["silero", "energy"] = Field(
        default="silero", alias="BACKEND_CORE_VAD_BACKEND"
    )
    # Empty => use the model packaged at app/audio/models/silero_vad.onnx.
    vad_model_path: str = Field(default="", alias="BACKEND_CORE_VAD_MODEL_PATH")
    vad_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, alias="BACKEND_CORE_VAD_THRESHOLD"
    )
    bargein_enabled: bool = Field(default=True, alias="BACKEND_CORE_BARGEIN_ENABLED")
    # Minimum *continuous* user speech before the agent's playback is interrupted;
    # rejects clicks/coughs and (together with WebRTC AEC — see bargein.py) the
    # agent's own audio bleeding back into the mic.
    bargein_min_speech_ms: int = Field(
        default=200, ge=0, alias="BACKEND_CORE_BARGEIN_MIN_SPEECH_MS"
    )
    # Sustained silence that ends the user's turn and re-arms barge-in.
    bargein_min_silence_ms: int = Field(
        default=400, ge=0, alias="BACKEND_CORE_BARGEIN_MIN_SILENCE_MS"
    )

    # --- Phase 2: WebRTC / gRPC transport -----------------------------------
    # Empty => aiortc's default (host candidates only), suitable for local dev.
    webrtc_stun_server: str = Field(default="", alias="BACKEND_CORE_WEBRTC_STUN_SERVER")
    grpc_enabled: bool = Field(default=True, alias="BACKEND_CORE_GRPC_ENABLED")

    # --- Phase 4: CORS -------------------------------------------------------
    # Comma-separated list of browser origins allowed to call the HTTP API. The
    # default permits only the local dashboard; a deployment must set this
    # explicitly. Deliberately never "*": a credential-less wildcard is one
    # config typo away from exposing the service to any page a user visits.
    cors_allow_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CLOUDEPSO_CORS_ALLOW_ORIGINS",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        # Accept "info" / "Info" and normalise to the canonical upper-case form.
        return value.upper() if isinstance(value, str) else value

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def cors_origins(self) -> list[str]:
        """Parsed CORS allow-list (comma-separated; blank entries dropped)."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def frame_duration_ms(self) -> int:
        """Wall-clock duration of one VAD frame (e.g. 512 samples @16k -> 32 ms)."""
        return int(round(1000 * self.vad_frame_samples / self.audio_sample_rate))

    @property
    def ice_servers(self) -> list[dict[str, list[str]]]:
        """aiortc ICE server configuration derived from the single STUN URL."""
        if self.webrtc_stun_server:
            return [{"urls": [self.webrtc_stun_server]}]
        return []


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
