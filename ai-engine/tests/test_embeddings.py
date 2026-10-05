"""The surviving MiniLM primitive, extracted from the old HF NLU fallback.

These run real inference (~90 MB model) when it is cached and skip with an
actionable message otherwise, exactly like the STT and TTS engine tests. The
point of the file is that the encoder still works *after* ``app.nlu`` is gone:
it is the piece step 2 (retrieval) is built on.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.knowledge.embeddings import EmbeddingModel

FETCH_HINT = "run `ai-engine/scripts/fetch_models.sh` to download model weights"


@pytest.fixture(scope="module")
def embedder(settings) -> EmbeddingModel:
    model = EmbeddingModel(settings.embedding_model, cache_dir=str(settings.hf_hub_cache))
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - offline dev box
        pytest.skip(f"MiniLM embedding model unavailable: {exc}; {FETCH_HINT}")
    yield model
    model.close()


def test_loads_lazily_and_reports_state(settings) -> None:
    model = EmbeddingModel(settings.embedding_model, cache_dir=str(settings.hf_hub_cache))
    assert model.is_loaded is False
    assert model.model_id == settings.embedding_model
    model.close()  # safe before load


def test_vectors_are_unit_length_and_float32(embedder) -> None:
    vectors = embedder.encode(["hello world", "a completely different sentence"])
    assert vectors.dtype == np.float32
    assert vectors.shape[0] == 2
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "vectors must be L2-normalised"


def test_dot_product_is_cosine_similarity(embedder) -> None:
    """The property retrieval depends on: related text scores higher."""
    vectors = embedder.encode(
        [
            "I would like to book an appointment",
            "I want to schedule a meeting",
            "the weather in Berlin tomorrow",
        ]
    )
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated, (related, unrelated)


def test_encode_one_returns_a_single_vector(embedder) -> None:
    vector = embedder.encode_one("hello world")
    assert vector.ndim == 1
    assert np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-4)


def test_encode_one_matches_encode(embedder) -> None:
    single = embedder.encode_one("consistency matters")
    batched = embedder.encode(["consistency matters"])[0]
    assert np.allclose(single, batched, atol=1e-5)


def test_an_empty_input_is_an_empty_array(embedder) -> None:
    """A retrieval that found nothing must not raise."""
    vectors = embedder.encode([])
    assert vectors.shape == (0, 0)


def test_close_releases_the_model(embedder) -> None:
    """close() is idempotent and puts the engine back in the unloaded state."""
    assert embedder.is_loaded is True
    embedder.close()
    assert embedder.is_loaded is False
    embedder.close()
    # Reload so later tests in the module still have a working encoder.
    embedder.load()
    assert embedder.is_loaded is True
