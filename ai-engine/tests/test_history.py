"""ConversationHistory: the bound, and the copy semantics the pipeline relies on.

No model and no network -- this is pure bookkeeping, and the two things that
matter are that it is *bounded* (an unbounded transcript eventually exceeds the
model's context window) and that a snapshot cannot change underneath a request
running in another thread.
"""

from __future__ import annotations

import threading

from app.llm.history import ConversationHistory


def test_an_empty_history_has_no_messages() -> None:
    history = ConversationHistory(max_turns=3)
    assert len(history) == 0
    assert history.snapshot() == []


def test_add_turn_records_user_then_assistant() -> None:
    history = ConversationHistory(max_turns=3)
    history.add_turn("hello", "hi there")
    messages = history.snapshot()
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]


def test_history_is_bounded_by_max_turns() -> None:
    history = ConversationHistory(max_turns=2)
    for index in range(5):
        history.add_turn(f"u{index}", f"a{index}")

    messages = history.snapshot()
    assert len(messages) == 4  # 2 turns == 2 * max_turns messages
    assert [m.content for m in messages] == ["u3", "a3", "u4", "a4"]


def test_zero_max_turns_disables_retention() -> None:
    """0 means "no history", not "unbounded"."""
    history = ConversationHistory(max_turns=0)
    history.add_turn("hello", "hi")
    assert len(history) == 0
    assert history.snapshot() == []


def test_a_negative_max_turns_is_treated_as_zero() -> None:
    history = ConversationHistory(max_turns=-4)
    history.add_turn("hello", "hi")
    assert history.snapshot() == []


def test_snapshot_is_a_copy() -> None:
    """A generator must not see the list mutate while it is mid-request."""
    history = ConversationHistory(max_turns=3)
    history.add_turn("first", "reply")
    snapshot = history.snapshot()

    history.add_turn("second", "reply two")
    assert len(snapshot) == 2
    assert len(history) == 4
    assert snapshot[0].content == "first"


def test_clear_empties_the_history() -> None:
    history = ConversationHistory(max_turns=3)
    history.add_turn("hello", "hi")
    history.clear()
    assert history.snapshot() == []


def test_extend_appends_messages() -> None:
    from app.llm.base import ChatMessage

    history = ConversationHistory(max_turns=2)
    history.extend([ChatMessage(role="user", content="a")])
    history.extend([ChatMessage(role="assistant", content="b")])
    assert [(m.role, m.content) for m in history.snapshot()] == [
        ("user", "a"),
        ("assistant", "b"),
    ]


def test_concurrent_adds_do_not_lose_messages() -> None:
    """The bound must be the only thing that drops messages, never a race."""
    history = ConversationHistory(max_turns=200)  # bound above what we add

    def worker(tag: str) -> None:
        for index in range(20):
            history.add_turn(f"{tag}-{index}", f"{tag}-reply-{index}")

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in "abcd"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(history.snapshot()) == 4 * 20 * 2
