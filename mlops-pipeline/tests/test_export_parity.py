"""Export + numerical-parity tests (Phase 5 headline evidence).

These tests *export the graphs and run real inference on both sides*. They are
not schema checks. Each test asserts the max-abs-diff between the source model
and its ONNX Runtime execution is within :data:`export.parity.ABS_TOL`.

Heavy models:
    The Laya encoder is a 28-layer, 1024-hidden transformer; exporting and
    parity-checking it costs ~1.6 GiB of disk and tens of seconds of CPU. It is
    on by default (so a plain ``pytest`` produces the real evidence) and can be
    skipped in a resource-constrained runner with ``CLOUDEPSO_SKIP_HEAVY_ONNX=1``.
"""

from __future__ import annotations

import importlib
import os

import pytest

import export_onnx
import parity

_SKIP_HEAVY = os.environ.get("CLOUDEPSO_SKIP_HEAVY_ONNX") == "1"

#: Third-party modules the export path imports. If any is missing the host
#: genuinely cannot export -- that is a legitimate skip, and we name the module.
_EXPORT_DEPS = ("torch", "transformers", "safetensors", "onnxruntime", "onnx")


def _missing_export_deps() -> list[str]:
    missing: list[str] = []
    for name in _EXPORT_DEPS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report every missing dep, don't stop at the first
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
    return missing


def _missing_model_sources(models_root) -> list[str]:
    """Checkpoints the exporters read; absent means ``fetch_models.sh`` was never run."""
    missing: list[str] = []
    for repo in (export_onnx.MINILM_REPO, export_onnx.LAYA_REPO, export_onnx.WHISPER_CT2_REPO):
        try:
            export_onnx._snapshot(models_root, repo)
        except FileNotFoundError:
            missing.append(repo)
    voice = models_root / "tts" / export_onnx.PIPER_VOICE_DIRNAME
    if not sorted(voice.glob("*.onnx")):
        missing.append(f"tts/{export_onnx.PIPER_VOICE_DIRNAME}/*.onnx")
    return missing


@pytest.fixture(scope="session")
def artifacts_dir():
    """Export every model once per session and reuse the artifacts.

    This skips ONLY for a genuine host limitation, and says which one:

    * a missing export dependency (torch/transformers/onnxruntime/...), or
    * a checkpoint that was never fetched into the ai-engine cache.

    Any *other* export failure -- i.e. one that happens with every dependency
    installed and every checkpoint present -- is a bug in our own export
    tooling, so it FAILS the run loudly. Collapsing all failures into one
    ``if failed: skip`` is what once let a module-shadowing bug hide behind the
    innocent-looking message "export(s) unavailable on this host".
    """
    models_root = export_onnx.DEFAULT_MODELS_ROOT
    out = export_onnx.DEFAULT_ARTIFACTS_DIR

    missing_deps = _missing_export_deps()
    if missing_deps:
        pytest.skip("export dependencies not installed in this interpreter: " + ", ".join(missing_deps))

    missing_sources = _missing_model_sources(models_root)
    if missing_sources:
        pytest.skip(
            "model checkpoints absent from the ai-engine cache "
            "(run ai-engine/scripts/fetch_models.sh): " + ", ".join(missing_sources)
        )

    results = export_onnx.run(list(export_onnx.EXPORTERS), models_root, out)
    by_model = {r.model: r for r in results}
    failed = [r for r in results if r.status == "failed"]
    if failed:
        detail = "\n  ".join(f"{r.model}: {r.notes}" for r in failed)
        pytest.fail(
            "export failed even though every dependency and checkpoint is present -- "
            "this is a bug in the export tooling, not a host limitation:\n  " + detail
        )
    return out, by_model


def test_minilm_exports_and_matches_torch(artifacts_dir):
    out, by_model = artifacts_dir
    art = by_model["minilm_nlu"]
    assert art.status == "exported"
    assert art.size_bytes > 0 and len(art.sha256) == 64
    assert [i["name"] for i in art.inputs] == ["input_ids", "attention_mask", "token_type_ids"]
    # Exactly one output: the unused BERT pooler output is dropped by the
    # encoder-only wrapper so the Triton config and the graph agree.
    assert [o["name"] for o in art.outputs] == ["last_hidden_state"]
    assert art.outputs[0]["shape"] == ["batch", "sequence", "384"]

    report = parity.parity_minilm(out / "minilm_nlu.onnx", export_onnx.DEFAULT_MODELS_ROOT)
    assert report["passed"], report
    assert report["hidden_max_abs_diff"] <= parity.ABS_TOL
    assert report["embedding_max_abs_diff"] <= parity.ABS_TOL
    print("minilm parity:", report["hidden_max_abs_diff"], report["embedding_max_abs_diff"])


@pytest.mark.skipif(_SKIP_HEAVY, reason="CLOUDEPSO_SKIP_HEAVY_ONNX=1")
def test_laya_encoder_exports_and_matches_torch(artifacts_dir):
    out, by_model = artifacts_dir
    art = by_model["laya_encoder"]
    assert art.status == "exported"
    assert [i["name"] for i in art.inputs] == ["input_ids", "attention_mask"]

    report = parity.parity_laya(out / "laya_encoder.onnx", export_onnx.DEFAULT_MODELS_ROOT)
    assert report["passed"], report
    # Dynamic axes are only real if parity holds at lengths other than the trace
    # length -- the harness re-checks at 16, 11 and 3.
    assert {r["seq_len"] for r in report["per_seq_len"]} == {16, 11, 3}
    print("laya parity:", report["hidden_max_abs_diff"])


def test_piper_graph_is_deterministic_and_finite(artifacts_dir):
    _, by_model = artifacts_dir
    art = by_model["piper_tts"]
    assert art.status == "already-onnx"
    report = parity.check_piper(export_onnx.DEFAULT_MODELS_ROOT
                                / "tts" / export_onnx.PIPER_VOICE_DIRNAME
                                / "en_US-amy-low.onnx")
    assert report["passed"], report
    assert report["samples"] > 0 and report["finite"]
    print("piper determinism diff:", report["determinism_max_abs_diff"])


def test_whisper_is_native_ctranslate2_not_onnx(artifacts_dir):
    _, by_model = artifacts_dir
    art = by_model["whisper_ct2"]
    # The whole point: STT is served by the CTranslate2 backend, not exported.
    assert art.status == "native-ct2"
    assert art.triton_backend == "ctranslate2"
    assert art.path.endswith("model.bin")
