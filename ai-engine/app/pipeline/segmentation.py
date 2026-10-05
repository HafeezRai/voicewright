"""Split generated text into speakable sentences.

The LLM emits one continuous string. Feeding the whole thing to the TTS at once
means the caller hears nothing until the *last* token has been generated, which
defeats the point of streaming. Feeding it sentence by sentence is what gets the
first audio out in about a second.

Splitting prose correctly is not ``text.split(".")``. The cases that actually
show up in generated English and break the naive version:

* abbreviations -- ``Dr. Smith``, ``etc.``, ``e.g.``, ``U.S.``
* initials -- ``J. R. R. Tolkien``
* decimals -- ``3.14``, ``pi is 3.14``
* ellipses and stacked punctuation -- ``wait...``, ``Really?!``
* markdown the model emits despite being asked not to -- ``**bold**``,
  ``- item``, ```code```, ``[text](url)``

Markdown is stripped rather than split, because a TTS engine reads an asterisk as
"asterisk". Everything else is about *where* the boundaries are. When the
heuristic is unsure it errs toward joining two sentences (one slightly long
utterance) rather than splitting one in half (two nonsense ones).

:class:`SentenceSegmenter` is the incremental form, for a token stream that
arrives in arbitrary fragments; :func:`iter_sentences` is the same logic over a
string that has already arrived.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

#: Characters that end a sentence.
_TERMINATORS = ".!?…"
#: Characters that may trail a terminator and still belong to the same sentence
#: -- including the typographic closing quotes a model emits when it writes
#: prose. Escaped rather than literal so the source file stays ASCII.
_CLOSERS = "\"'\u201d\u2019)]}"

#: Word abbreviations that end in a period but not a sentence ("Dr. Smith").
#: Compared lower-cased, without the trailing period.
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "miss", "prof", "rev", "hon", "sr", "jr",
        "st", "mt", "vs", "etc", "no", "fig", "figs", "al", "approx", "dept",
        "inc", "ltd", "co", "corp", "est", "min", "max", "avg", "vol", "pp",
        "sec", "chap", "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "ph.d",
    }
)

#: A dotted two-letter abbreviation: ``e.g``, ``i.e``, ``U.S``, ``a.m``, ``B.A``.
_DOTTED_ABBREVIATION = re.compile(r"[a-z]\.[a-z]")

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
#: A line that opens a block: a heading, or a bullet/numbered list item.
_BLOCK_MARKER = re.compile(r"^[ \t]{0,3}(?:#{1,6}[ \t]*|[-*+][ \t]+|\d+[.)][ \t]+)")
#: Paired emphasis, unwrapped to its text. Applied as *pairs* rather than by
#: blanket-stripping ``*``/``_``, and the underscore pairs require a word
#: boundary on both sides -- otherwise ``llm_max_tokens`` is read as an emphasis
#: span and comes out as ``llmmaxtokens``, which a TTS then mispronounces.
_MD_EMPHASIS_PAIRS = (
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9])__(.+?)__(?![A-Za-z0-9])", re.DOTALL), r"\1"),
    (re.compile(r"\*(.+?)\*", re.DOTALL), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9])_(.+?)_(?![A-Za-z0-9])", re.DOTALL), r"\1"),
)
#: Whatever asterisks survive pairing are stray; an asterisk is never prose.
_MD_STRAY_ASTERISK = re.compile(r"\*+")
_WHITESPACE = re.compile(r"\s+")


def _blocks_to_prose(text: str) -> str:
    """Turn heading and list *lines* into separate spoken sentences.

    Joining every line with a space would make a three-item list one run-on
    utterance with no pause. A block line is instead closed with a terminator,
    so the list is spoken item by item. Lines that open no block (ordinary
    prose, including hard-wrapped prose) are left alone -- adding a period to a
    wrapped line would split a sentence in the middle.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        marker = _BLOCK_MARKER.match(raw)
        if marker is None:
            lines.append(raw)
            continue
        content = raw[marker.end():].strip()
        if content and not content.endswith(tuple(_TERMINATORS)):
            content += "."
        lines.append(content)
    return " ".join(lines)


def normalise(text: str) -> str:
    """Strip markdown formatting and collapse whitespace.

    Markdown is *removed*, not escaped: the goal is a string that reads the same
    aloud as it looks on screen.
    """
    if not text:
        return ""
    cleaned = _CODE_FENCE.sub(" ", text)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)   # keep the code text, drop the ticks
    cleaned = _MD_LINK.sub(r"\1", cleaned)       # keep the link text, drop the URL
    cleaned = _blocks_to_prose(cleaned)
    for pattern, replacement in _MD_EMPHASIS_PAIRS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = _MD_STRAY_ASTERISK.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def _token_before(text: str, index: int) -> str:
    """The word-ish token immediately preceding ``index``."""
    start = index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ".-_'"):
        start -= 1
    return text[start:index]


def _is_decimal_point(text: str, index: int) -> bool:
    """True when the ``.`` at ``index`` sits between two digits (``3.14``)."""
    return (
        0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _is_abbreviation(text: str, index: int) -> bool:
    """True when the ``.`` at ``index`` closes an abbreviation, not a sentence."""
    if (
        0 < index < len(text) - 1
        and text[index - 1].isalpha()
        and text[index + 1].isalpha()
    ):
        # A period *inside* a letter run: the first dot of "e.g", "U.S", "B.A".
        # Ordinary prose never puts a period tight between two letters.
        return True

    token = _token_before(text, index).lower()
    if not token:
        return False
    if token in _ABBREVIATIONS:
        return True
    if _DOTTED_ABBREVIATION.fullmatch(token):
        # The last dot of "e.g" / "U.S", where the token is now complete.
        return True
    if len(token) == 1 and token.isalpha():
        # An initial ("J. Smith"). Single-letter *words* are excluded, because
        # "so do I." and "a." really do end sentences.
        if token in {"a", "i"}:
            return False
        return index + 1 < len(text) and text[index + 1].isspace()
    return False


def _consume_terminators(text: str, index: int) -> int:
    """Advance past a run of terminators: ``...``, ``?!``, ``?..``."""
    end = index
    while end < len(text) and text[end] in _TERMINATORS:
        end += 1
    return end


def _find_boundary(text: str, *, final: bool) -> int | None:
    """Index just past the first complete sentence, or ``None`` if there is none.

    With ``final=False`` a period at the very end of the buffer is held back when
    the character before it is a digit, because the next fragment could still
    turn ``"3."`` into ``"3.14"``. Every other terminator is unambiguous once the
    token before it has been checked: an abbreviation at a fragment boundary has
    already been ruled out above, since that check only needs what precedes it.
    """
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char not in _TERMINATORS:
            index += 1
            continue
        if char == "." and _is_decimal_point(text, index):
            index += 1
            continue
        if char == "." and _is_abbreviation(text, index):
            index += 1
            continue

        end = _consume_terminators(text, index)
        if end < length:
            while end < length and (text[end] in _CLOSERS or text[end].isspace()):
                end += 1
            return end
        if final:
            return length
        if char == "." and text[index - 1].isdigit():
            return None  # could still become a decimal
        return length

    if final and text.strip():
        return length
    return None


def iter_sentences(text: str) -> Iterator[str]:
    """Yield ``text`` one speakable sentence at a time.

    The final fragment is always yielded even without terminal punctuation, so a
    model that stops mid-sentence still gets spoken.
    """
    buffer = normalise(text)
    while buffer:
        boundary = _find_boundary(buffer, final=True)
        if boundary is None:
            break
        sentence = buffer[:boundary].strip()
        buffer = buffer[boundary:].strip()
        if sentence:
            yield sentence
    if buffer.strip():
        yield buffer.strip()


class SentenceSegmenter:
    """Incremental :func:`iter_sentences` for a stream of text fragments.

    Feed each delta, get back whatever sentences are now complete; call
    :meth:`flush` at the end of the stream for the trailing fragment. The buffer
    is only consumed up to a boundary it can *prove*, so a fragment arriving as
    ``"3."`` does not become a sentence before ``"14"`` shows up.
    """

    def __init__(self) -> None:
        self._buffer = ""
        #: Markdown stripping is applied per completed sentence, but a code fence
        #: or emphasis span can straddle a fragment boundary. Tracking fenced
        #: state keeps a stray "```" from being read aloud.
        self._in_fence = False

    @property
    def pending(self) -> str:
        """Text held back because no sentence boundary has been proven yet."""
        return self._buffer

    def feed(self, delta: str) -> list[str]:
        """Add a fragment; return any sentences it completed."""
        if not delta:
            return []
        self._buffer += delta
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """Return the trailing fragment as a final sentence (usually 0 or 1)."""
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        sentences: list[str] = []
        while self._buffer:
            boundary = _find_boundary(self._buffer, final=final)
            if boundary is None:
                break
            raw = self._buffer[:boundary]
            self._buffer = self._buffer[boundary:]
            # A fenced block may be split across fragments; while inside one,
            # hold the text back rather than speaking code.
            raw, self._in_fence = _strip_fences(raw, self._in_fence)
            sentence = normalise(raw)
            if sentence:
                sentences.append(sentence)
        return sentences


def _strip_fences(text: str, in_fence: bool) -> tuple[str, bool]:
    """Drop fenced-code spans, tracking a fence left open across fragments."""
    out: list[str] = []
    position = 0
    while position < len(text):
        marker = text.find("```", position)
        if marker == -1:
            if not in_fence:
                out.append(text[position:])
            break
        if not in_fence:
            out.append(text[position:marker])
        in_fence = not in_fence
        position = marker + 3
    return "".join(out), in_fence
