"""Split a document into retrievable chunks.

Naive fixed-width slicing (``text[i:i+500]``) cuts mid-sentence and mid-word,
which produces chunks that embed poorly: half a sentence is not a unit of
meaning, and a chunk that begins inside a word ("...appointments are only for
existing pat|ients on Saturdays") embeds as noise. So the split is
hierarchical -- paragraphs, then sentences, then words only as a last resort --
and greedy packing reassembles whole sentences into chunks near the target size.

Parameters, and why these numbers
---------------------------------
* **500 characters** (~125 English tokens at ~4 chars/token). MiniLM truncates
  at 128 tokens, so a longer chunk would be *silently* half-embedded: the tail
  would never reach the vector, and a query matching the tail would not find it.
  A chunk that fits the window is a chunk whose every word is searchable.
* **80 characters of overlap** (~16%). A fact can straddle a sentence boundary;
  repeating the tail of the previous chunk means it is retrievable from either
  side. Overlap costs storage and a little redundancy in the prompt, so it is
  deliberately a small fraction of the chunk, not half of it.

Both are :class:`~app.config.Settings` values, not constants, so an operator can
retune them without a code change.
"""

from __future__ import annotations

import re

#: A blank line (or a line of only whitespace) separates paragraphs, which is
#: where the author put the strongest boundary in their document.
_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n+")
#: A sentence ends at terminal punctuation followed by whitespace. Not a real
#: sentence tokeniser -- ``Dr. Smith`` will split -- but conservative about the
#: common case and cheap. Over-splitting costs a slightly smaller chunk, which
#: greedy packing immediately rejoins; under-splitting costs a chunk over the
#: token window, which is worse.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    *,
    max_chars: int = 500,
    overlap_chars: int = 80,
) -> list[str]:
    """Split ``text`` into overlapping, sentence-aware chunks.

    Returns ``[]`` for empty/whitespace-only input. Every returned chunk is at
    most ``max_chars`` long. Paragraph breaks are treated as sentence
    boundaries -- the packer is free to join two short paragraphs into one
    chunk, which is the right call for retrieval (a two-line policy and its
    one-line exception belong together).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    # An overlap >= the chunk size would never advance; clamp it just below.
    overlap = max(0, min(overlap_chars, max_chars - 1))

    normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalised:
        return []

    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(normalised):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.extend(_hard_split(sentence, max_chars))

    return _pack(units, max_chars, overlap)


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split a single sentence longer than ``max_chars`` on word boundaries.

    Only reached by pathological input (a wall of text with no punctuation), but
    it must be handled or the packer would emit an over-long chunk that the
    embedder truncates.
    """
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def _pack(units: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack sentence units into chunks, carrying a tail for overlap."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        if len(current) + 1 + len(unit) <= max_chars:
            current = f"{current} {unit}"
            continue

        chunks.append(current)
        carry = _carry(current, overlap_chars)
        candidate = f"{carry} {unit}".strip() if carry else unit
        # The carry plus the next unit can exceed the budget when the unit is
        # itself near max_chars; drop the carry rather than emit an over-long
        # chunk. Overlap is best-effort, the size bound is not.
        current = candidate if len(candidate) <= max_chars else unit

    if current.strip():
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _carry(previous: str, overlap_chars: int) -> str:
    """The tail of ``previous`` to repeat at the head of the next chunk.

    Trimmed to a word boundary so the next chunk does not begin mid-word.
    """
    if overlap_chars <= 0:
        return ""
    tail = previous[-overlap_chars:].strip()
    if " " in tail:
        tail = tail[tail.index(" ") + 1 :].strip()
    return tail
