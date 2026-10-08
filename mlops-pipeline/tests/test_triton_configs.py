"""Triton model-repository config validation tests.

These parse every ``config.pbtxt`` against the real Triton ``ModelConfig``
protobuf and assert the repository is internally consistent. They are STATIC
validation: no Triton server is started (this host has no NVIDIA GPU, so it
cannot run one) and these tests do not claim a model would serve correctly.

The validator lives at ``mlops-pipeline/triton_repo/validate_configs.py`` and is
loaded *by file path* rather than as ``triton_repo.validate_configs``: it is a
standalone script, not an importable package, and loading it by path keeps the
tests independent of whatever else is on ``sys.path``. See ``conftest.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from google.protobuf import text_format

_VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "triton_repo" / "validate_configs.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("voicewright_validate_configs", _VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_configs = _load_validator()
ModelConfig = validate_configs.model_config_pb2.ModelConfig

EXPECTED_MODELS = {
    "minilm_nlu",
    "piper_tts",
    "whisper_ct2",
    "tts_frontend",
}


def _config(name: str) -> ModelConfig:
    config = ModelConfig()
    text_format.Parse((validate_configs.REPO_ROOT / name / "config.pbtxt").read_text(), config)
    return config


def test_all_configs_parse_and_are_valid():
    results = validate_configs.validate_all()
    bad = {r.model: r.errors for r in results if not r.ok}
    assert not bad, f"invalid Triton configs: {bad}"


def test_expected_models_present():
    results = validate_configs.validate_all()
    assert {r.model for r in results} == EXPECTED_MODELS


def test_stt_uses_ctranslate2_not_onnx():
    """STT must target the CTranslate2 backend; ONNX would discard quantisation."""
    assert _config("whisper_ct2").backend == "ctranslate2"


def test_onnx_models_declare_dynamic_batching_or_explain_why_not():
    """onnxruntime models enable batching, except piper (documented exception)."""
    for name in ("minilm_nlu",):
        config = _config(name)
        assert config.max_batch_size > 0
        assert config.dynamic_batching.preferred_batch_size, f"{name} has no preferred_batch_size"
    assert _config("piper_tts").max_batch_size == 0


def test_validation_catches_a_malformed_config(tmp_path):
    """A config with a name/dir mismatch must be rejected (Triton fails silently)."""
    bad_dir = tmp_path / "broken_repo"
    bad_dir.mkdir()
    (bad_dir / "config.pbtxt").write_text('name: "not_the_dir_name"\nbackend: "onnxruntime"\n')
    result = validate_configs.validate_one(bad_dir / "config.pbtxt")
    assert not result.ok
    assert any("directory name" in e for e in result.errors)


def test_validation_catches_dynamic_batching_with_zero_batch(tmp_path):
    bad_dir = tmp_path / "broken_repo2"
    bad_dir.mkdir()
    (bad_dir / "config.pbtxt").write_text(
        'name: "broken_repo2"\n'
        'backend: "onnxruntime"\n'
        "max_batch_size: 0\n"
        'input [{ name: "x" data_type: TYPE_FP32 dims: [ -1 ] }]\n'
        'output [{ name: "y" data_type: TYPE_FP32 dims: [ -1 ] }]\n'
        "dynamic_batching { max_queue_delay_microseconds: 5000 }\n"
    )
    result = validate_configs.validate_one(bad_dir / "config.pbtxt")
    assert not result.ok
    assert any("max_batch_size is 0" in e for e in result.errors)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
