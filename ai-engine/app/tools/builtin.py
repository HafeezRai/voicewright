"""Tools that ship with the service.

These are deliberately stateful. A demo tool that returns a canned string
proves nothing: the model reads it and says something plausible whether or not
the call did anything. Booking a slot and having the *next* question see that
booking is what shows the loop is real -- the agent can be asked "what have I
booked?" and answer from the record it wrote.

State lives in this process, which is the honest scope for a demo: it resets on
restart and is not shared between replicas. A real deployment points these at
whatever system already owns the data.
"""

from __future__ import annotations

import threading

from app.tools.base import ToolBinding, ToolError, ToolRegistry, ToolSpec

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
#: date -> {time -> caller}. A slot maps to at most one name.
_booked: dict[str, dict[str, str]] = {}

#: Opening hours, in 24h. The clinic is shut at weekends, which is a fact the
#: agent must be able to *find out* rather than guess -- it is what makes the
#: knowledge-grounding story extend to actions.
_SLOTS = ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]

_ORDERS = {
    "DFW2JS": "Order DFW2JS: 2 items, dispatched Tuesday, arriving Thursday.",
    "8SD2JW": "Order 8SD2JW: 1 item, out of stock, expected back in 5 days.",
}


def reset() -> None:
    """Clear all state. Used by tests."""
    with _lock:
        _booked.clear()


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def check_availability(date: str) -> str:
    """List the free slots on a date."""
    if not date.strip():
        raise ToolError("A date is required, for example 'Tuesday' or '2026-03-04'.")
    with _lock:
        taken = dict(_booked.get(date, {}))
    free = [slot for slot in _SLOTS if slot not in taken]
    if not free:
        return f"{date} is fully booked."
    return f"Free on {date}: {', '.join(free)}."


def book_appointment(date: str, time: str, name: str) -> str:
    """Book a slot, refusing if it is already taken."""
    if not (date.strip() and time.strip() and name.strip()):
        raise ToolError("date, time and name are all required.")
    with _lock:
        day = _booked.setdefault(date, {})
        holder = day.get(time)
        if holder is not None:
            raise ToolError(f"{time} on {date} is already booked.")
        day[time] = name
    return f"Booked {name} for {time} on {date}."


def list_appointments(name: str) -> str:
    """Everything booked under a name."""
    with _lock:
        found = [
            f"{time} on {date}"
            for date, day in sorted(_booked.items())
            for time, holder in sorted(day.items())
            if holder.lower() == name.strip().lower()
        ]
    if not found:
        return f"Nothing booked under {name}."
    return f"{name} has: " + "; ".join(found) + "."


def lookup_order(order_id: str) -> str:
    """Status of an order by reference."""
    status = _ORDERS.get(order_id.strip().upper())
    if status is None:
        raise ToolError(f"No order found with reference {order_id!r}.")
    return status


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRING = {"type": "string"}


def build_registry() -> ToolRegistry:
    """The registry the service ships with."""
    return ToolRegistry(
        [
            ToolBinding(
                ToolSpec(
                    name="check_availability",
                    description=(
                        "List the appointment slots still free on a given date. "
                        "Call this before offering a time."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {"date": {**_STRING, "description": "The date to check."}},
                        "required": ["date"],
                    },
                ),
                check_availability,
            ),
            ToolBinding(
                ToolSpec(
                    name="book_appointment",
                    description=(
                        "Book an appointment. Fails if the slot is taken, so check "
                        "availability first."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "date": _STRING,
                            "time": {**_STRING, "description": "24-hour, e.g. 15:00."},
                            "name": {**_STRING, "description": "Who the booking is for."},
                        },
                        "required": ["date", "time", "name"],
                    },
                ),
                book_appointment,
            ),
            ToolBinding(
                ToolSpec(
                    name="list_appointments",
                    description="Everything already booked under a caller's name.",
                    parameters={
                        "type": "object",
                        "properties": {"name": _STRING},
                        "required": ["name"],
                    },
                ),
                list_appointments,
            ),
            ToolBinding(
                ToolSpec(
                    name="lookup_order",
                    description="Look up the status of an order by its reference.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "order_id": {**_STRING, "description": "The order reference."}
                        },
                        "required": ["order_id"],
                    },
                ),
                lookup_order,
            ),
        ]
    )


#: The names an agent can be configured with, for validation and for the UI.
BUILTIN_TOOL_NAMES = tuple(
    spec.name for spec in build_registry().specs_for(["check_availability", "book_appointment", "list_appointments", "lookup_order"])
)
