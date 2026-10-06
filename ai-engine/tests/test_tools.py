"""Tools, and the loop that runs them.

The loop is tested against a *scripted* engine -- one that returns whatever
tool calls it was told to, in order -- rather than a real model. That is the
only way to assert the thing that actually matters: that a call is executed,
its result is fed back into the conversation, and the model is asked again.
A real model tests the model; a scripted one tests the loop.
"""

from __future__ import annotations

from app.llm.base import ChatMessage, LLMEngine, LLMResult
from app.tools import (
    ToolBinding,
    Toolbox,
    ToolCall,
    ToolError,
    ToolRegistry,
    ToolSpec,
    build_registry,
)

# ---------------------------------------------------------------------------
# Specs and calls
# ---------------------------------------------------------------------------


def test_spec_reaches_the_provider_in_the_expected_envelope() -> None:
    spec = ToolSpec(name="book", description="Book it", parameters={"type": "object"})
    assert spec.as_openai() == {
        "type": "function",
        "function": {
            "name": "book",
            "description": "Book it",
            "parameters": {"type": "object"},
        },
    }


def test_spec_without_parameters_still_declares_an_object() -> None:
    # Providers reject a missing `parameters`; an argument-less tool still has
    # to say so explicitly.
    assert ToolSpec(name="now", description="").as_openai()["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_spec_round_trips_through_the_store_format() -> None:
    spec = ToolSpec(name="a", description="b", parameters={"type": "object"})
    assert ToolSpec.from_dict(spec.as_dict()) == spec


def test_call_parses_an_arguments_string() -> None:
    call = ToolCall.parse(call_id="c1", name="book", arguments='{"date": "Tuesday"}')
    assert call.arguments == {"date": "Tuesday"}
    assert call.raw_arguments == ""


def test_call_survives_malformed_arguments() -> None:
    # Models emit broken JSON. It must not raise here -- the executor turns it
    # into something the model can recover from.
    call = ToolCall.parse(call_id="c1", name="book", arguments='{"date": ')
    assert call.arguments == {}
    assert call.raw_arguments == '{"date": '


def test_call_rejects_arguments_that_are_json_but_not_an_object() -> None:
    call = ToolCall.parse(call_id="c1", name="book", arguments='["a", "b"]')
    assert call.arguments == {}
    assert call.raw_arguments


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolBinding(ToolSpec(name="ok", description=""), lambda **_: "fine"),
            ToolBinding(
                ToolSpec(name="broken", description=""),
                lambda **_: (_ for _ in ()).throw(ToolError("the slot is taken")),
            ),
            ToolBinding(
                ToolSpec(name="needs_date", description=""),
                lambda date: f"got {date}",
            ),
        ]
    )


def test_registry_runs_a_tool() -> None:
    result = _registry().call(ToolCall(id="c1", name="ok"))
    assert result.ok and result.content == "fine"


def test_registry_reports_an_unknown_tool_instead_of_raising() -> None:
    result = _registry().call(ToolCall(id="c1", name="nope"))
    assert not result.ok
    assert "No tool named" in result.content


def test_registry_turns_a_tool_error_into_a_result_the_model_can_read() -> None:
    result = _registry().call(ToolCall(id="c1", name="broken"))
    assert not result.ok
    assert result.content == "the slot is taken"


def test_registry_reports_bad_arguments_without_dying() -> None:
    # Missing a required keyword: a TypeError, which must become a message.
    result = _registry().call(ToolCall(id="c1", name="needs_date", arguments={}))
    assert not result.ok
    assert "Bad arguments" in result.content


def test_registry_refuses_unparseable_arguments_before_calling() -> None:
    result = _registry().call(
        ToolCall(id="c1", name="ok", arguments={}, raw_arguments='{"broken')
    )
    assert not result.ok
    assert "valid JSON" in result.content


def test_specs_for_skips_tools_this_deployment_does_not_have() -> None:
    # An agent naming a tool that is not deployed is a configuration mistake,
    # not a reason to fail the turn.
    assert [s.name for s in _registry().specs_for(["ok", "missing"])] == ["ok"]


def test_toolbox_is_falsy_when_it_resolves_to_nothing() -> None:
    registry = _registry()
    assert bool(Toolbox.build(registry, ["ok"]))
    assert not Toolbox.build(registry, ["missing"])


# ---------------------------------------------------------------------------
# The shipped tools
# ---------------------------------------------------------------------------


def test_booking_then_listing_sees_the_booking() -> None:
    from app.tools import builtin

    builtin.reset()
    registry = build_registry()
    assert registry.call(
        ToolCall(id="1", name="book_appointment", arguments={"date": "Tue", "time": "10:00", "name": "Sam"})
    ).ok
    listed = registry.call(ToolCall(id="2", name="list_appointments", arguments={"name": "Sam"}))
    assert "10:00 on Tue" in listed.content


def test_double_booking_is_refused() -> None:
    from app.tools import builtin

    builtin.reset()
    registry = build_registry()
    first = registry.call(
        ToolCall(id="1", name="book_appointment", arguments={"date": "Tue", "time": "10:00", "name": "Sam"})
    )
    second = registry.call(
        ToolCall(id="2", name="book_appointment", arguments={"date": "Tue", "time": "10:00", "name": "Alex"})
    )
    assert first.ok
    assert not second.ok
    assert "already booked" in second.content


def test_availability_reflects_what_is_booked() -> None:
    from app.tools import builtin

    builtin.reset()
    registry = build_registry()
    registry.call(
        ToolCall(id="1", name="book_appointment", arguments={"date": "Tue", "time": "10:00", "name": "Sam"})
    )
    free = registry.call(ToolCall(id="2", name="check_availability", arguments={"date": "Tue"}))
    assert "10:00" not in free.content
    assert "09:00" in free.content


def test_order_lookup_reports_unknown_references_rather_than_inventing_one() -> None:
    result = build_registry().call(
        ToolCall(id="1", name="lookup_order", arguments={"order_id": "NOPE"})
    )
    assert not result.ok
    assert "No order found" in result.content


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class ScriptedLLM(LLMEngine):
    """Returns whatever it was told to, in order, and records what it was asked."""

    name = "scripted"

    def __init__(self, *results: LLMResult) -> None:
        self._results = list(results)
        self.calls: list[tuple[list[ChatMessage], tuple[ToolSpec, ...], int | None]] = []

    @property
    def model(self) -> str:
        return "scripted-1"

    def generate(
        self,
        messages,
        *,
        tools=None,
        max_tokens=None,
    ) -> LLMResult:
        self.calls.append((list(messages), tuple(tools or ()), max_tokens))
        return self._results.pop(0) if self._results else LLMResult(text="done")


def test_loop_runs_a_tool_and_feeds_the_result_back(pipeline_engine) -> None:
    engine = ScriptedLLM(
        LLMResult(
            text="",
            tool_calls=(ToolCall(id="c1", name="check_availability", arguments={"date": "Tue"}),),
        ),
        LLMResult(text="Tuesday has slots from nine."),
    )
    toolbox = Toolbox.build(build_registry(), ["check_availability"])

    start = [ChatMessage("user", "anything free Tuesday?")]
    result, calls, results = pipeline_engine._complete_with_tools(start, toolbox, engine)

    assert result.text == "Tuesday has slots from nine."
    assert [c.name for c in calls] == ["check_availability"]
    assert results and results[0].ok

    # The model was asked twice, and the second time it was told what the tool
    # said -- that round trip is the entire feature.
    assert len(engine.calls) == 2
    second_messages = engine.calls[1][0]
    assert any(m.role == "tool" and "Free on Tue" in m.content for m in second_messages)
    assert any(m.role == "assistant" and m.tool_calls for m in second_messages)


def test_loop_chains_several_tools_in_one_turn(pipeline_engine) -> None:
    engine = ScriptedLLM(
        LLMResult(text="", tool_calls=(ToolCall(id="c1", name="check_availability", arguments={"date": "Tue"}),)),
        LLMResult(text="", tool_calls=(ToolCall(id="c2", name="book_appointment", arguments={"date": "Tue", "time": "10:00", "name": "Sam"}),)),
        LLMResult(text="Booked."),
    )
    toolbox = Toolbox.build(build_registry(), ["check_availability", "book_appointment"])
    from app.tools import builtin

    builtin.reset()

    result, calls, _ = pipeline_engine._complete_with_tools(
        [ChatMessage("user", "book Tuesday at 10 for Sam")], toolbox, engine
    )
    assert result.text == "Booked."
    assert [c.name for c in calls] == ["check_availability", "book_appointment"]


def test_loop_withdraws_the_tools_on_the_final_round(pipeline_engine) -> None:
    # A model that keeps asking for tools has to be made to answer, or the
    # caller waits forever.
    engine = ScriptedLLM(
        *[
            LLMResult(text="", tool_calls=(ToolCall(id=f"c{i}", name="check_availability", arguments={"date": "Tue"}),))
            for i in range(pipeline_engine.tool_rounds + 2)
        ]
    )
    toolbox = Toolbox.build(build_registry(), ["check_availability"])
    pipeline_engine._complete_with_tools([ChatMessage("user", "hi")], toolbox, engine)

    last_tools = engine.calls[-1][1]
    assert last_tools == (), "the final round must offer no tools"


def test_loop_uses_the_tool_token_budget(pipeline_engine) -> None:
    # Thinking models spend the budget on reasoning before they call anything,
    # so the planning pass cannot use the short spoken-reply allowance.
    engine = ScriptedLLM(
        LLMResult(text="", tool_calls=(ToolCall(id="c1", name="check_availability", arguments={"date": "Tue"}),)),
        LLMResult(text="ok"),
    )
    pipeline_engine._complete_with_tools(
        [ChatMessage("user", "hi")], Toolbox.build(build_registry(), ["check_availability"]), engine
    )
    assert engine.calls[0][2] == pipeline_engine.tool_max_tokens
    assert pipeline_engine.tool_max_tokens > 160


def test_loop_continues_after_a_failing_tool(pipeline_engine) -> None:
    engine = ScriptedLLM(
        LLMResult(text="", tool_calls=(ToolCall(id="c1", name="lookup_order", arguments={"order_id": "NOPE"}),)),
        LLMResult(text="I couldn't find that order."),
    )
    toolbox = Toolbox.build(build_registry(), ["lookup_order"])
    result, _, results = pipeline_engine._complete_with_tools([ChatMessage("user", "where is it")], toolbox, engine)

    assert not results[0].ok
    assert result.text == "I couldn't find that order."


def test_no_toolbox_means_a_single_call(pipeline_engine) -> None:
    engine = ScriptedLLM(LLMResult(text="hello"))
    result, calls, results = pipeline_engine._complete_with_tools([ChatMessage("user", "hi")], None, engine)
    assert result.text == "hello"
    assert calls == () and results == ()
    assert len(engine.calls) == 1
