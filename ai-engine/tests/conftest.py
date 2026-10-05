"""Shared fixtures for the ai-engine test suite.

The suite runs **real inference** where inference is ours -- real Whisper
transcription, real VITS synthesis. The point of Phase 3 was that nothing is
mocked, and that still holds for the two stages this service owns.

The middle stage is different, and is the one deliberate substitution here. The
LLM is a *remote* provider, so "real" would mean "requires Ollama to be running
with the right model pulled" -- a test suite that fails on a bare checkout for
reasons that have nothing to do with this code. :class:`FakeLLM` therefore stands
in: deterministic, no network, emitting real deltas through the same contract the
shipped client implements, and able to fail on demand. The live wire protocol
has its own test (``test_llm_ollama_live.py``), which skips cleanly when Ollama
or the model is absent.

Environment the suite pins, before any ``app`` import:

* ``AI_ENGINE_LLM_WARMUP=false`` -- no background model load during the HTTP
  tests. Without this, ``TestClient`` would start a warm-up thread that reaches
  for a real Ollama on every run.
* ``AI_ENGINE_GRPC_ENABLED=false`` -- no port is bound during the HTTP tests; the
  gRPC test builds its own server on an ephemeral port.

Model weights are expected under ``ai-engine/models`` (the gitignored fallback
of ``Settings.models_root``). Tests that need a model which has not been
downloaded skip with a message telling you to run ``scripts/fetch_models.sh``
rather than failing -- a bare checkout stays green.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

# --- environment defaults (must precede any `app` import) -------------------
os.environ.setdefault("AI_ENGINE_LLM_WARMUP", "false")
os.environ.setdefault("AI_ENGINE_GRPC_ENABLED", "false")
os.environ.setdefault("AI_ENGINE_LOG_LEVEL", "WARNING")
#: The agent store is a SQLite file the served app opens at import. Point it at
#: a throwaway file so the suite never touches a developer's real agents, and so
#: the app-under-test and ad-hoc stores in tests share nothing.
os.environ.setdefault(
    "AI_ENGINE_AGENT_DB", str(Path(tempfile.mkdtemp(prefix="cloudpso-agents-")) / "agents.db")
)

from app.cancellation import CancelToken
from app.config import configure_hf_home, get_settings
from app.llm.base import ChatMessage, LLMEngine, LLMResult, LLMUsage
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


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A fresh scripted LLM per test, so recorded calls cannot leak between them."""
    return FakeLLM()


@pytest.fixture
def pipeline_engine(stt_engine, tts_engine, fake_llm) -> VoicePipeline:
    """The real pipeline: real Whisper and real VITS, scripted generation.

    Function-scoped rather than session-scoped because the LLM changes per test;
    constructing a :class:`VoicePipeline` loads nothing, so this costs nothing --
    the two expensive engines are still session-scoped.
    """
    return VoicePipeline(
        stt_engine,
        fake_llm,
        tts_engine,
        sample_rate=_settings.sample_rate_hz,
        chunk_samples=_settings.tts_chunk_samples,
    )


# ---------------------------------------------------------------------------
# The served application
#
# `app.main` builds its pipeline once, at import, from settings -- i.e. pointed
# at a real Ollama. The HTTP tests drive that exact pipeline object (it is what
# `GET /metrics` observes), so the fake has to be installed *there*: a fixture
# the app cannot see would leave `POST /llm` reaching for a model that may not
# be running. Done once, here, rather than per-test, so every ASGI test sees the
# same deterministic engine.
# ---------------------------------------------------------------------------
from app import main as _app_main  # noqa: E402  (must follow the env defaults)

_app_main.pipeline.llm.close()
_app_main.pipeline.llm = FakeLLM()
