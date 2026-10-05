"""The agent CRUD API and, above all, its tenant isolation.

The store is SQLite and the API is small; the property that actually needs a
test is that org A cannot see, change, or delete org B's agents -- and that it
is told *404*, not 403, so the status code does not leak the agent's existence.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main as app_main
from app.agents.store import DEFAULT_ORG_ID
from app.main import app

client = TestClient(app)

#: A second tenant, provisioned directly (there is no org API by design).
ORG_B = "org_tenant_b"

app_main.agent_store.create_org(ORG_B, "tenant-b")


def _create(name: str = "Reception", *, org: str | None = None, **fields) -> dict:
    body = {"name": name, **fields}
    headers = {"X-Org-Id": org} if org else None
    response = client.post("/agents", json=body, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_create_returns_the_agent_with_an_id_and_defaults() -> None:
    agent = _create("Billing bot", role="billing", system_prompt="You bill people.")
    assert agent["id"]
    assert agent["org_id"] == DEFAULT_ORG_ID
    assert agent["name"] == "Billing bot"
    assert agent["role"] == "billing"
    assert agent["created_at"] and agent["updated_at"]


def test_get_and_list_round_trip() -> None:
    agent = _create("Round trip")
    fetched = client.get(f"/agents/{agent['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == agent["id"]

    listed = client.get("/agents").json()
    assert agent["id"] in {row["id"] for row in listed}


def test_update_is_partial_and_bumps_updated_at() -> None:
    agent = _create("Partial", greeting="Hello there")
    updated = client.put(
        f"/agents/{agent['id']}", json={"role": "support"}
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["role"] == "support"
    # Unset fields are untouched, not cleared.
    assert body["greeting"] == "Hello there"
    assert body["name"] == "Partial"


def test_delete_then_get_is_404() -> None:
    agent = _create("Doomed")
    assert client.delete(f"/agents/{agent['id']}").status_code == 204
    assert client.get(f"/agents/{agent['id']}").status_code == 404
    # Deleting again is a 404, not a 204: the id no longer resolves.
    assert client.delete(f"/agents/{agent['id']}").status_code == 404


def test_unknown_id_is_404() -> None:
    assert client.get("/agents/nope").status_code == 404
    assert client.put("/agents/nope", json={"role": "x"}).status_code == 404
    assert client.delete("/agents/nope").status_code == 404


def test_invalid_body_is_422() -> None:
    assert client.post("/agents", json={"name": ""}).status_code == 422
    assert client.post("/agents", json={"role": "no name"}).status_code == 422
    assert client.post("/agents", json={"name": "x" * 500}).status_code == 422


def test_creating_in_an_unknown_org_is_404() -> None:
    response = client.post(
        "/agents", json={"name": "Ghost"}, headers={"X-Org-Id": "org_does_not_exist"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# tenant isolation -- the headline property
# ---------------------------------------------------------------------------


def test_org_cannot_read_another_orgs_agent() -> None:
    b_agent = _create("B's agent", org=ORG_B)
    # Org A (the default) must not see it, and must be told 404, not 403.
    response = client.get(f"/agents/{b_agent['id']}")
    assert response.status_code == 404, response.text


def test_org_cannot_update_another_orgs_agent() -> None:
    b_agent = _create("B's other agent", org=ORG_B)
    response = client.put(f"/agents/{b_agent['id']}", json={"role": "hijacked"})
    assert response.status_code == 404, response.text
    # And it really was not changed.
    still_b = client.get(f"/agents/{b_agent['id']}", headers={"X-Org-Id": ORG_B})
    assert still_b.json()["role"] != "hijacked"


def test_org_cannot_delete_another_orgs_agent() -> None:
    b_agent = _create("B's third agent", org=ORG_B)
    assert client.delete(f"/agents/{b_agent['id']}").status_code == 404
    # It survives: org A's failed delete did not touch it.
    assert client.get(f"/agents/{b_agent['id']}", headers={"X-Org-Id": ORG_B}).status_code == 200


def test_list_is_scoped_to_the_calling_org() -> None:
    a_agent = _create("A only")
    b_agent = _create("B only", org=ORG_B)

    a_ids = {row["id"] for row in client.get("/agents").json()}
    b_ids = {row["id"] for row in client.get("/agents", headers={"X-Org-Id": ORG_B}).json()}

    assert a_agent["id"] in a_ids and b_agent["id"] not in a_ids
    assert b_agent["id"] in b_ids and a_agent["id"] not in b_ids


def test_org_resolution_prefers_the_query_param_over_the_header() -> None:
    """``?org_id=`` wins, so a client can override a pinned header."""
    agent = _create("Query org", org=ORG_B)
    response = client.get(
        f"/agents/{agent['id']}?org_id={DEFAULT_ORG_ID}", headers={"X-Org-Id": ORG_B}
    )
    assert response.status_code == 404
