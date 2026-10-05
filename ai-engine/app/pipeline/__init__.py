"""STT -> LLM -> TTS pipeline."""

from __future__ import annotations

from app.config import Settings, configure_hf_home
from app.llm import build_llm
from app.pipeline.orchestrator import (
    EventKind,
    PipelineEvent,
    TurnResult,
    VoicePipeline,
)
from app.stt import build_stt
from app.tts import build_tts

__all__ = [
    "EventKind",
    "PipelineEvent",
    "TurnResult",
    "VoicePipeline",
    "build_pipeline",
]


def build_pipeline(settings: Settings) -> VoicePipeline:
    """Construct the full pipeline (no model is loaded until first use)."""
    # Keep every stage's HuggingFace downloads inside the model cache.
    configure_hf_home(settings)
    return VoicePipeline(
        build_stt(settings),
        build_llm(settings),
        build_tts(settings),
        sample_rate=settings.sample_rate_hz,
        chunk_samples=settings.tts_chunk_samples,
        system_prompt=settings.llm_system_prompt,
        fallback_reply=settings.llm_fallback_reply,
    )
