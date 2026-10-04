"""The Laya NLU backend, against the real checkpoint.

Skipped -- with an explanation, not a failure -- when ``laya`` is not installed or
the 842 MB checkpoint is not cached, so a bare checkout stays green. Where the
checkpoint *is* present (as it was for the Phase 3 spike) these run for real.

Measured on the spike hardware (Apple M-series, MPS): ~31 ms for a single
``choice`` decision, ~136 ms on CPU. The CPU-only path is still well inside a
conversational budget once STT is accounted for, so the default device remains
``cpu``; set ``AI_ENGINE_DEVICE=mps`` to use the GPU.
"""

from __future__ import annotations

import pytest

from app.config import get_settings

pytest.importorskip("laya", reason="laya is not installed")

from app.nlu import LayaNLU
from app.nlu.base import NLUUnavailableError

_settings = get_settings()
_CHECKPOINT_DIR = _settings.hf_hub_cache / "models--convaiinnovations--laya"


def _device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


@pytest.fixture(scope="session")
def laya_engine():
    if not _CHECKPOINT_DIR.is_dir():
        pytest.skip(
            f"Laya checkpoint not cached under {_CHECKPOINT_DIR}; "
            "run `ai-engine/scripts/fetch_models.sh` to download it"
        )
    engine = LayaNLU(_settings.nlu_model, device=_device())
    try:
        engine.load()
    except NLUUnavailableError as exc:
        pytest.skip(f"Laya unavailable: {exc}")
    yield engine
    engine.close()


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("I want to book an appointment", "book_appointment"),
        ("I need to cancel my account", "cancel_account"),
        ("what's your return policy", "return_policy"),
    ],
)
def test_laya_decides_real_utterances(laya_engine, utterance, expected) -> None:
    result = laya_engine.classify(utterance)
    assert result.backend == "laya"
    assert result.intent == expected, f"{utterance!r} -> {result.intent!r}"
    assert 0.0 < result.confidence <= 1.0


def test_laya_reports_a_ranking(laya_engine) -> None:
    result = laya_engine.classify("I want to book an appointment")
    assert result.scores, "expected a per-intent ranking from the choice head"
    assert result.intent in result.scores


def test_laya_handles_empty_input_without_a_model_call(laya_engine) -> None:
    result = laya_engine.classify("   ")
    assert result.intent == "out_of_scope"
    assert result.confidence == 0.0


def test_laya_is_the_active_auto_backend(laya_engine) -> None:
    """With Laya loadable, `auto` must select it rather than the fallback."""
    from app.nlu.factory import AutoNLU
    from app.nlu.hf_engine import HFIntentNLU

    auto = AutoNLU(laya_engine, HFIntentNLU(_settings.nlu_hf_model))
    result = auto.classify("I want to book an appointment")
    assert result.backend == "laya"
    assert auto.name == "laya"
