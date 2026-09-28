"""API-adapter integration tests — entry auth + HITL /resume ownership.

Exercises src/api/server.py over HTTP (FastAPI TestClient). Guards the retrofit
invariants: unauthenticated → 401 (enum/string ANONYMOUS); the D6 plan gate
suspends with awaiting_human; /resume ownership fail-closed (403 foreign / 404
unknown / 409 replay); a valid owner reaches success.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from framework.schemas.trust_level import TrustLevel
from src.graph.graph import Graph
from tests.fakes import FakeLLMClient, FakeResearchService, graph_config

_TOKEN = "test-invoke-token-cmn249"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    import importlib

    import src.api.server as server

    importlib.reload(server)
    cfg = graph_config(service=FakeResearchService(), llm=FakeLLMClient())
    cfg.update({"memory_enabled": True, "hitl": {"enabled": True, "max_hitl": 8}})
    agent = Graph(config=cfg)
    agent.compile(checkpointer=InMemorySaver())
    server.agent = agent
    return TestClient(server.app), server


def _auth(token: str = _TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _body(session_id="sess-owner"):
    return {"input": "What is X?", "session_id": session_id}


def test_invoke_without_token_is_401(client):
    c, _ = client
    assert c.post("/invoke", json=_body()).status_code == 401


def test_invoke_wrong_token_is_401(client):
    c, _ = client
    assert c.post("/invoke", json=_body(), headers=_auth("nope")).status_code == 401


@pytest.mark.parametrize("stamp", [TrustLevel.ANONYMOUS, "ANONYMOUS"])
def test_invoke_explicit_anonymous_middleware_is_401(client, stamp):
    c, server = client

    @server.app.middleware("http")
    async def _stamp(request, call_next):
        request.state.trust_level = stamp
        return await call_next(request)

    fresh = TestClient(server.app)
    assert fresh.post("/invoke", json=_body()).status_code == 401


def test_invoke_good_token_suspends_at_plan_gate(client):
    c, _ = client
    r = c.post("/invoke", json=_body(), headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "awaiting_human"
    assert body["thread_id"]
    assert body.get("session_id") == "sess-owner"


def test_resume_without_token_is_401(client):
    c, _ = client
    th = c.post("/invoke", json=_body(), headers=_auth()).json()["thread_id"]
    r = c.post("/resume", json={"thread_id": th, "feedback": "Go", "session_id": "sess-owner"})
    assert r.status_code == 401


def test_resume_foreign_session_is_403(client):
    c, _ = client
    th = c.post("/invoke", json=_body(), headers=_auth()).json()["thread_id"]
    r = c.post("/resume", json={"thread_id": th, "feedback": "Go", "session_id": "attacker"}, headers=_auth())
    assert r.status_code == 403


def test_resume_unknown_thread_is_404(client):
    c, _ = client
    c.post("/invoke", json=_body(), headers=_auth())
    r = c.post("/resume", json={"thread_id": "00000000-0000-0000-0000-000000000000",
                                "feedback": "Go", "session_id": "sess-owner"}, headers=_auth())
    assert r.status_code == 404


def test_resume_correct_owner_reaches_success(client):
    c, _ = client
    th = c.post("/invoke", json=_body(), headers=_auth()).json()["thread_id"]
    r = c.post("/resume", json={"thread_id": th, "feedback": "Go", "session_id": "sess-owner"}, headers=_auth())
    assert r.status_code == 200
    assert r.json()["status"] == "success"


def test_resume_replay_after_completion_is_409(client):
    c, _ = client
    th = c.post("/invoke", json=_body(), headers=_auth()).json()["thread_id"]
    first = c.post("/resume", json={"thread_id": th, "feedback": "Go", "session_id": "sess-owner"}, headers=_auth())
    assert first.status_code == 200 and first.json()["status"] == "success"
    replay = c.post("/resume", json={"thread_id": th, "feedback": "Go", "session_id": "sess-owner"}, headers=_auth())
    assert replay.status_code == 409


def test_resume_echoes_session_id(client):
    """The /resume response must echo session_id like /invoke so a multi-round
    resume can supply the ownership key on the next call."""
    c, _ = client
    inv = c.post("/invoke", json=_body(), headers=_auth()).json()
    th, sid = inv["thread_id"], inv["session_id"]
    r = c.post("/resume", json={"thread_id": th, "feedback": "Go", "session_id": sid}, headers=_auth())
    assert r.status_code == 200
    assert r.json().get("session_id") == sid
