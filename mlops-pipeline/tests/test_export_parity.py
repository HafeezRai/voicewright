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

import os

import pytest

import export_onnx
import parity

_SKIP_HEAVY = os.environ.get("CLOUDEPSO_SKIP_HEAVY_ONNX") == "1"


@pytest.fixture(scope="session")
def artifacts_dir():
    """Export every model once per session and reuse the artifacts."""
    out = export_onnx.DEFAULT_ARTIFACTS_DIR
    results = export_onnx.run(list(export_onnx.EXPORTERS), export_onnx.DEFAULT_MODELS_ROOT, out)
    by_model = {r.model: r for r in results}
    failed = [r.model for r in results if r.status == "failed"]
    if failed:
        pytest.skip(f"export(s) unavailable on this host: {failed}")
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
