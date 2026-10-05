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
import time
from pathlib import Path

import pytest

# --- environment defaults (must precede any `app` import) -------------------
os.environ.setdefault("AI_ENGINE_NLU_BACKEND", "hf")
os.environ.setdefault("AI_ENGINE_GRPC_ENABLED", "false")
os.environ.setdefault("AI_ENGINE_LOG_LEVEL", "WARNING")

from app.cancellation import CancelToken
from app.config import configure_hf_home, get_settings
from app.llm.base import ChatMessage, LLMEngine, LLMResult, LLMUsage
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

#: Deterministic stand-in for the LLM stage. Long enough that the TTS produces
#: several chunks (so barge-in has something to interrupt), and stable so tests
#: can assert on structure without a model.
FAKE_REPLY = "This is a deterministic test reply from the fake language model."


class FakeLLM(LLMEngine):
    """A scripted LLM engine: no network, no model, fully deterministic.

    It emits *real* deltas through the same ``stream`` contract the shipped
    client implements, honours ``cancel`` between them, and can be told to fail,
    so the whole pipeline suite runs with no Ollama and no cold-start wait --
    while still exercising the incremental path rather than a stub that returns
    one string.

    Every call records the exact message list it received, which is how tests
    assert that a system prompt actually reached the model.
    """

    name = "fake"

    def __init__(
        self,
        reply: str = FAKE_REPLY,
        *,
        provider: str = "fake",
        model: str = "fake-1",
        delta_delay_s: float = 0.0,
        fail: Exception | None = None,
    ):
        self._reply = reply
        self._provider = provider
        self._model = model
        self._delta_delay_s = delta_delay_s
        self._fail = fail
        self.calls: list[list[ChatMessage]] = []
        self.closed = False

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return not self.closed

    @property
    def last_messages(self) -> list[ChatMessage]:
        return self.calls[-1] if self.calls else []

    def close(self) -> None:
        self.closed = True

    def _check(self, messages) -> None:
        self.calls.append(list(messages))
        if self._fail is not None:
            raise self._fail

    def generate(self, messages) -> LLMResult:
        started = time.perf_counter()
        self._check(messages)
        if self._delta_delay_s:
            time.sleep(self._delta_delay_s)
        return LLMResult(
            text=self._reply,
            model=self._model,
            provider=self._provider,
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=len(self._reply.split()),
                           total_tokens=5 + len(self._reply.split())),
            latency_s=time.perf_counter() - started,
            first_token_s=0.0,
        )

    def stream(self, messages, *, cancel: CancelToken | None = None):
        self._check(messages)
        for word in self._reply.split():
            if cancel is not None and cancel.cancelled:
                return
            if self._delta_delay_s:
                time.sleep(self._delta_delay_s)
            yield word + " "

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


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A fresh scripted LLM per test, so recorded calls cannot leak between them."""
    return FakeLLM()
