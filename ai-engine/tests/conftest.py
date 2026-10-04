"""Shared fixtures for the ai-engine test suite.

The suite runs **real inference** -- real Whisper transcription, real VITS
synthesis, real NLU. That is the point of Phase 3, so nothing here is mocked.
What this file does control is *which* real things run by default:

* ``AI_ENGINE_NLU_BACKEND=hf`` -- the fast, deterministic path (a ~90 MB
  embedding model) is the default for the suite, so a plain ``pytest`` does not
  pull the 842 MB Laya checkpoint. The Laya backend has its own test
  (``test_laya.py``) which skips with a clear reason when the checkpoint is not
  cached, and runs for real when it is.
* ``AI_ENGINE_GRPC_ENABLED=false`` -- no port is bound during the HTTP tests; the
  gRPC test builds its own server on an ephemeral port.

Model weights are expected under ``ai-engine/models`` (the gitignored fallback
of ``Settings.models_root``). Tests that need a model which has not been
downloaded skip with a message telling you to run ``scripts/fetch_models.sh``
rather than failing -- a bare checkout stays green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# --- environment defaults (must precede any `app` import) -------------------
os.environ.setdefault("AI_ENGINE_NLU_BACKEND", "hf")
os.environ.setdefault("AI_ENGINE_GRPC_ENABLED", "false")
os.environ.setdefault("AI_ENGINE_LOG_LEVEL", "WARNING")

from app.config import configure_hf_home, get_settings
from app.nlu import HFIntentNLU
from app.pipeline import VoicePipeline
from app.stt import FasterWhisperSTT
from app.tts import SherpaTTS

TESTS_DIR = Path(__file__).resolve().parent
AI_ENGINE_DIR = TESTS_DIR.parent
REPO_ROOT = AI_ENGINE_DIR.parent
#: The Phase 2 speech fixture, reused rather than duplicated.
SPEECH_WAV = REPO_ROOT / "backend-core" / "tests" / "data" / "speech_16k.wav"

FETCH_HINT = "run `ai-engine/scripts/fetch_models.sh` to download model weights"

_settings = get_settings()
# Same cache layout the service uses, so a test run warms the container's volume.
configure_hf_home(_settings)


def _stt_available() -> bool:
    return any(
        p.is_dir() and "whisper" in p.name for p in _settings.hf_hub_cache.glob("models--*")
    )


def _tts_available() -> bool:
    return _settings.tts_model_dir.is_dir() and any(
        _settings.tts_model_dir.glob("*.onnx")
    )


@pytest.fixture(scope="session")
def settings():
    return _settings


@pytest.fixture(scope="session")
def speech_wav() -> Path:
    if not SPEECH_WAV.is_file():
        pytest.skip(f"Phase 2 speech fixture not found at {SPEECH_WAV}")
    return SPEECH_WAV


@pytest.fixture(scope="session")
def stt_engine() -> FasterWhisperSTT:
    if not _stt_available():
        pytest.skip(f"faster-whisper weights not cached under {_settings.hf_hub_cache}; {FETCH_HINT}")
    engine = FasterWhisperSTT(
        _settings.stt_model,
        device="cpu",
        compute_type=_settings.stt_compute_type,
        download_root=str(_settings.hf_hub_cache),
        hf_cache_dir=str(_settings.hf_home),
    )
    yield engine
    engine.close()


@pytest.fixture(scope="session")
def tts_engine() -> SherpaTTS:
    if not _tts_available():
        pytest.skip(f"Piper voice not cached at {_settings.tts_model_dir}; {FETCH_HINT}")
    engine = SherpaTTS(_settings.tts_model_dir)
    yield engine
    engine.close()


@pytest.fixture(scope="session")
def hf_nlu_engine() -> HFIntentNLU:
    engine = HFIntentNLU(_settings.nlu_hf_model, cache_dir=str(_settings.hf_hub_cache))
    # Loading downloads ~90 MB on first use; treat an unreachable Hub as a skip.
    try:
        engine.load()
    except Exception as exc:  # pragma: no cover - offline dev box
        pytest.skip(f"HF NLU model unavailable: {exc}; {FETCH_HINT}")
    yield engine
    engine.close()


@pytest.fixture(scope="session")
def pipeline_engine(stt_engine, hf_nlu_engine, tts_engine) -> VoicePipeline:
    """The full real pipeline, backed by the fast NLU path."""
    return VoicePipeline(
        stt_engine,
        hf_nlu_engine,
        tts_engine,
        sample_rate=_settings.sample_rate_hz,
        chunk_samples=_settings.tts_chunk_samples,
    )
