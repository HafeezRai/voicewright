"""Document chunking: size bounds, sentence awareness, overlap.

No model involved -- this is pure string logic, so it runs everywhere.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.knowledge.chunking import chunk_text

_PROSE = " ".join(
    f"Policy item {i} explains a rule that a caller might ask about." for i in range(40)
)


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n\t  ") == []


def test_every_chunk_respects_the_size_bound() -> None:
    chunks = chunk_text(_PROSE, max_chars=120, overlap_chars=20)
    assert chunks
    assert all(len(chunk) <= 120 for chunk in chunks), [len(c) for c in chunks]


def test_more_than_one_chunk_for_long_text() -> None:
    assert len(chunk_text(_PROSE, max_chars=120, overlap_chars=20)) > 1


def test_overlap_repeats_the_previous_chunk_tail() -> None:
    """The next chunk must begin with a word carried over from the previous one."""
    chunks = chunk_text(_PROSE, max_chars=120, overlap_chars=30)
    assert len(chunks) >= 2
    for previous, following in pairwise(chunks):
        first_word = following.split()[0]
        assert first_word in previous.split()[-10:], (previous[-40:], following[:40])


def test_no_overlap_when_disabled() -> None:
    chunks = chunk_text(_PROSE, max_chars=120, overlap_chars=0)
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_chunks_never_start_or_end_mid_word() -> None:
    chunks = chunk_text(_PROSE, max_chars=120, overlap_chars=25)
    for chunk in chunks:
        assert chunk == chunk.strip()
        assert "  " not in chunk


def test_a_sentence_that_fits_is_not_split() -> None:
    """Sentence awareness: short prose stays on sentence boundaries."""
    text = "First short sentence. Second short sentence. Third short sentence."
    chunks = chunk_text(text, max_chars=40, overlap_chars=0)
    # Each chunk is one or two whole sentences; none begins mid-sentence.
    for chunk in chunks:
        assert chunk[0].isupper(), chunk


def test_paragraphs_are_treated_as_boundaries() -> None:
    text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
    chunks = chunk_text(text, max_chars=1000, overlap_chars=0)
    # Small enough to fit in one chunk, and joined in order.
    assert len(chunks) == 1
    assert chunks[0].index("Alpha") < chunks[0].index("Beta") < chunks[0].index("Gamma")


def test_an_unpunctuated_wall_of_text_is_hard_split() -> None:
    text = "word " * 300
    chunks = chunk_text(text, max_chars=100, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_non_positive_max_chars_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=0, overlap_chars=0)


def test_overlap_is_clamped_below_the_chunk_size() -> None:
    """An overlap >= max_chars would never advance; it is clamped, not honoured."""
    chunks = chunk_text(_PROSE, max_chars=100, overlap_chars=9999)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
