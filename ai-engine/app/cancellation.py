"""A thread-safe one-way cancel flag, shared by the TTS and LLM stages.

This lived in :mod:`app.tts.base` while TTS was the only stage that could be
interrupted. The LLM stage needs the same token -- barge-in must be able to stop
generation, not just playback -- and importing the TTS package to get a
three-method flag would make the LLM stage depend on synthesis for no reason.
So the token lives here, in a module with no dependencies at all, and
:mod:`app.tts.base` re-exports it so every existing import path keeps working.

One token covers one utterance. :meth:`CancelToken.cancel` is called from
whichever thread owns the barge-in controller; the generation/playback loops
running in another thread observe :attr:`CancelToken.cancelled` between chunks.
"""

from __future__ import annotations

import threading


class CancelToken:
    """A thread-safe one-way cancel flag.

    Set with :meth:`cancel`, tested with :attr:`cancelled`. One token covers one
    utterance; :meth:`reset` re-arms it for the next.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()
