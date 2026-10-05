"""Sentence splitting: the cases that break ``text.split(".")``."""

from __future__ import annotations

from app.pipeline.segmentation import SentenceSegmenter, iter_sentences, normalise

# ---------------------------------------------------------------------------
# plain prose
# ---------------------------------------------------------------------------


def test_splits_on_sentence_terminators() -> None:
    assert list(iter_sentences("Hello there. How are you? Fine!")) == [
        "Hello there.",
        "How are you?",
        "Fine!",
    ]


def test_flushes_a_trailing_fragment_without_punctuation() -> None:
    assert list(iter_sentences("First sentence. And then a tail")) == [
        "First sentence.",
        "And then a tail",
    ]


def test_a_single_sentence_without_punctuation_is_still_yielded() -> None:
    assert list(iter_sentences("no punctuation here")) == ["no punctuation here"]


def test_empty_and_whitespace_input_yield_nothing() -> None:
    assert list(iter_sentences("")) == []
    assert list(iter_sentences("   \n  ")) == []


# ---------------------------------------------------------------------------
# abbreviations, decimals, initials
# ---------------------------------------------------------------------------


def test_title_abbreviations_do_not_end_a_sentence() -> None:
    assert list(iter_sentences("Dr. Smith is here. Please sit.")) == [
        "Dr. Smith is here.",
        "Please sit.",
    ]


def test_etc_and_dotted_abbreviations_do_not_end_a_sentence() -> None:
    assert list(iter_sentences("Bring pens, paper, etc. and then leave.")) == [
        "Bring pens, paper, etc. and then leave."
    ]
    assert list(iter_sentences("Use e.g. this one. It works.")) == [
        "Use e.g. this one.",
        "It works.",
    ]


def test_decimals_do_not_end_a_sentence() -> None:
    assert list(iter_sentences("Pi is 3.14 and change. That is all.")) == [
        "Pi is 3.14 and change.",
        "That is all.",
    ]


def test_initials_do_not_end_a_sentence() -> None:
    assert list(iter_sentences("J. R. R. Tolkien wrote it. I read it.")) == [
        "J. R. R. Tolkien wrote it.",
        "I read it.",
    ]


def test_a_lone_i_or_a_still_ends_a_sentence() -> None:
    """The "is it an initial?" heuristic must not swallow real words."""
    assert list(iter_sentences("So do I. Then we left.")) == [
        "So do I.",
        "Then we left.",
    ]


def test_stacked_punctuation_and_ellipses_are_one_boundary() -> None:
    assert list(iter_sentences("Wait... Really?! Yes.")) == ["Wait...", "Really?!", "Yes."]


def test_a_question_inside_quotes_keeps_the_quote_with_the_sentence() -> None:
    assert list(iter_sentences('He said "go home." Then he left.')) == [
        'He said "go home."',
        "Then he left.",
    ]


# ---------------------------------------------------------------------------
# markdown the model emits anyway
# ---------------------------------------------------------------------------


def test_markdown_emphasis_is_removed_not_spoken() -> None:
    assert list(iter_sentences("That is **very** important.")) == [
        "That is very important."
    ]


def test_headings_and_list_items_become_separate_utterances() -> None:
    """A list spoken as one run-on utterance has no pause anywhere in it."""
    text = "# Heading\n- first item\n- second item\n\nDone."
    assert list(iter_sentences(text)) == [
        "Heading.",
        "first item.",
        "second item.",
        "Done.",
    ]


def test_numbered_lists_are_recognised() -> None:
    assert list(iter_sentences("1. First\n2. Second\n\nOk.")) == [
        "First.",
        "Second.",
        "Ok.",
    ]


def test_hard_wrapped_prose_is_not_split() -> None:
    """Only *block* lines get a terminator; wrapped prose must stay one sentence."""
    assert list(iter_sentences("line one\nline two wrapped\n\nDone.")) == [
        "line one line two wrapped Done."
    ]


def test_links_keep_their_text_and_drop_the_url() -> None:
    assert list(iter_sentences("See [the docs](https://x.test/y) now.")) == [
        "See the docs now."
    ]


def test_fenced_code_is_removed_entirely() -> None:
    text = "Here is how:\n```python\nprint('hi')\n```\nThat is all."
    spoken = list(iter_sentences(text))
    joined = " ".join(spoken)
    assert "```" not in joined
    assert "print" not in joined
    assert joined.endswith("That is all.")


def test_snake_case_identifiers_survive_emphasis_stripping() -> None:
    """``_max_tokens_`` is emphasis; ``llm_max_tokens`` is an identifier."""
    assert list(iter_sentences("Set llm_max_tokens to 160. It works.")) == [
        "Set llm_max_tokens to 160.",
        "It works.",
    ]
    assert list(iter_sentences("This is _italic_ text. Ok.")) == [
        "This is italic text.",
        "Ok.",
    ]


def test_normalise_collapses_whitespace() -> None:
    assert normalise("  lots \n\n of\tspace ") == "lots of space"


# ---------------------------------------------------------------------------
# incremental segmentation
# ---------------------------------------------------------------------------


def test_segmenter_emits_a_sentence_once_it_is_complete() -> None:
    segmenter = SentenceSegmenter()
    assert segmenter.feed("Hello wor") == []
    assert segmenter.feed("ld. How ") == ["Hello world."]
    assert segmenter.feed("are you?") == ["How are you?"]


def test_segmenter_waits_for_lookahead_before_committing_a_decimal() -> None:
    """``3.`` at a fragment boundary could still become ``3.14``."""
    segmenter = SentenceSegmenter()
    assert segmenter.feed("Pi is 3.") == []
    assert segmenter.feed("14 exactly.") == ["Pi is 3.14 exactly."]


def test_segmenter_flush_returns_the_tail() -> None:
    segmenter = SentenceSegmenter()
    assert segmenter.feed("A complete one. and a tail") == ["A complete one."]
    assert segmenter.flush() == ["and a tail"]
    assert segmenter.flush() == []


def test_segmenter_flush_on_a_complete_sentence_yields_nothing_extra() -> None:
    segmenter = SentenceSegmenter()
    assert segmenter.feed("All done.") == ["All done."]
    assert segmenter.flush() == []


def test_segmenter_holds_back_an_unclosed_code_fence() -> None:
    segmenter = SentenceSegmenter()
    emitted = segmenter.feed("Look: ```code\nmore code``` done.")
    assert all("```" not in sentence for sentence in emitted)
    assert all("```" not in sentence for sentence in segmenter.flush())


def test_segmenter_pending_exposes_the_unemitted_buffer() -> None:
    segmenter = SentenceSegmenter()
    segmenter.feed("Half a sentence")
    assert segmenter.pending == "Half a sentence"


def test_segmenter_over_a_whole_string_matches_iter_sentences() -> None:
    text = "Dr. Smith went to Washington. It cost 3.14 dollars! Really?"
    segmenter = SentenceSegmenter()
    streamed = segmenter.feed(text) + segmenter.flush()
    assert streamed == list(iter_sentences(text))
