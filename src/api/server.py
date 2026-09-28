"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly.
#
# HITL (D6): the inner plan node calls interrupt() at the plan gate. /invoke may
# return status "awaiting_human" with a thread_id; the reviewer answers via
# /resume with that thread_id and feedback (Go, or an edited research_plan).

from __future__ import annotations

import threading
import weakref
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory

from src.entry_adapter.auth import _require_verified_trust, _resolve_trust
from src.graph.graph import Graph

app = FastAPI(title="Agent")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _load_runtime_config() -> dict:
    """Load config/config.yaml — the runtime-parameter home.

    Without this the graph received only the HITL literals below, so every
    runtime parameter (max_retry, timeout_seconds, the source catalog, the
    llm_* provider options read by the synthesize node) silently fell back to a
    framework default in the deployed agent while the tests supplied their own.
    """
    try:
        with _CONFIG_PATH.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    except FileNotFoundError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


# HITL requires memory_enabled/hitl so the framework threads a thread_id into the
# run config, and an EXPLICIT outer checkpointer (the 1.0.0 wheel does not derive
# one). The inner graph gets its own checkpointer in ResearchWorkflowNode. These
# two keys stay explicit so the HITL plan gate cannot be switched off by an
# incomplete config file.
_CONFIG = {
    **_load_runtime_config(),
    "memory_enabled": True,
    "hitl": {"enabled": True, "max_hitl": 8},
}

# The LLM itself is NOT built here: the provider credential is resolved per
# invocation from ctx.secrets inside the synthesize node, and
# provision_secrets() below runs after register_nodes() has already built every
# node. A caller that owns process wiring may still inject config["llm"].
agent = Graph(config=_CONFIG)
agent.compile(checkpointer=InMemorySaver())
# Replace namespace/agent_name to match the agent's manifest values.
agent.provision_secrets(secrets_factory(namespace="cmn", agent_name="cmn_c2_249"))


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Optional structured context: {"source_spec": {...}, "plan_confirmed": bool}.
    input_context: dict = {}


class ResumeRequest(BaseModel):
    thread_id: str
    # feedback: "Go"/{} to approve the plan as presented, or
    # {"research_plan": {...}} to approve an edited plan.
    feedback: Any = None
    session_id: str = ""


def _authenticated_ctx(request: Request, session_id: str) -> InvocationContext:
    """Build the InvocationContext AFTER entry authentication.

    Entry auth is implemented by entry_adapter.auth:
    verified middleware trust is used as-is; INVOKE_AUTH_TOKEN Bearer auth
    promotes ANONYMOUS → VERIFIED_EXTERNAL (401 on mismatch). An unauthenticated
    caller (still ANONYMOUS, no token promotion) → 401 — replacing the old
    fallback that defaulted such callers to ANONYMOUS. This template has no
    tenant/org request-scope model, so only the auth core is wired.
    """
    trust, token_authenticated = _resolve_trust(request)
    if trust is TrustLevel.ANONYMOUS and not token_authenticated:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    _require_verified_trust(trust)
    return InvocationContext(
        session_id=session_id or str(uuid4()),
        caller_trust_level=trust,
        caller_id=getattr(request.state, "caller_id", ""),
    )


def _require_resume_ownership(thread_id: str, ctx: InvocationContext) -> None:
    """Fail-closed checkpoint ownership check for HITL resume.

    Verify the suspended checkpoint exists (404) and its recorded session_id
    matches (403); a checkpoint with no session_id is un-resumable. Require a
    pending HITL wait (snapshot.next non-empty AND subgraph_thread_id set —
    propagate_hitl=True) so a completed/replayed resume is rejected (409).

    Ownership model & limitation: session_id is the primary owner key. Under a
    single global INVOKE_AUTH_TOKEN deployment, token callers share caller_id="",
    so this defends against a caller who does NOT know the victim's session_id;
    binding to a verified principal is the robust fix and engages automatically
    in a real deployment (platform middleware stamps request.state.caller_id →
    the caller_id check enforces it). The check + resume are serialized per
    thread by a keyed lock in /resume; a durable multi-process deployment must
    use a per-thread lease/CAS instead.
    """
    compiled = getattr(agent, "_compiled", None)
    if compiled is None:
        raise HTTPException(status_code=409, detail="No suspended session for this thread.")
    snapshot = compiled.get_state({"configurable": {"thread_id": thread_id}})
    values = getattr(snapshot, "values", None) or {}
    if not values:
        raise HTTPException(status_code=404, detail="No suspended session for this thread.")
    owner_session = values.get("session_id")
    if not owner_session or owner_session != ctx.session_id:
        raise HTTPException(status_code=403, detail="Session does not own this thread.")
    owner_caller = values.get("caller_id") or ""
    if owner_caller and owner_caller != (ctx.caller_id or ""):
        raise HTTPException(status_code=403, detail="Caller does not own this thread.")
    if not getattr(snapshot, "next", None) or not values.get("subgraph_thread_id"):
        raise HTTPException(status_code=409, detail="Thread is not awaiting human input.")


_RESUME_LOCKS: "weakref.WeakValueDictionary[str, threading.Lock]" = weakref.WeakValueDictionary()
_RESUME_LOCKS_GUARD = threading.Lock()


def _resume_lock(thread_id: str) -> threading.Lock:
    with _RESUME_LOCKS_GUARD:
        lock = _RESUME_LOCKS.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _RESUME_LOCKS[thread_id] = lock
        return lock


def _with_session_id(result: Any, ctx: InvocationContext) -> Any:
    if isinstance(result, dict) and "session_id" not in result:
        return {**result, "session_id": ctx.session_id}
    return result


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request):
    ctx = _authenticated_ctx(request, req.session_id)
    with bound_secrets(agent._secrets_provider):
        return _with_session_id(agent.invoke(req.input, ctx=ctx, input_context=req.input_context), ctx)


@app.post("/resume")
async def resume(req: ResumeRequest, request: Request):
    """Continue a run paused at the D6 HITL plan gate.

    The caller MUST echo the session_id returned by the awaiting /invoke response
    (ownership key). feedback approves the plan (Go) or supplies an edited
    research_plan.
    """
    ctx = _authenticated_ctx(request, req.session_id)
    with _resume_lock(req.thread_id):
        _require_resume_ownership(req.thread_id, ctx)
        with bound_secrets(agent._secrets_provider):
            # Echo session_id like /invoke so a multi-round resume can
            # still supply the ownership key on the next call.
            return _with_session_id(agent.resume(thread_id=req.thread_id, feedback=req.feedback), ctx)


@app.get("/health")
def health():
    return {"status": "ok", "agent": "cmn_c2_249"}
