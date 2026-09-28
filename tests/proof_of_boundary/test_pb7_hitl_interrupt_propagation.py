# tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py
#
# PB-7: HITL interrupt propagation for templates that enable human review.
#
# This template enables HITL: the inner plan node calls interrupt() at the plan
# gate (migrated from the prior state-flag pseudo-HITL to the framework D6
# mechanism). The assertions below run for real.

from __future__ import annotations

import pathlib
import warnings

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from tests.fakes import FakeLLMClient, FakeResearchService, graph_config

_CONFIG_PATH = pathlib.Path(__file__).parents[2] / "config" / "config.yaml"


def _hitl_enabled() -> bool:
    """Return True when config/config.yaml declares hitl.enabled: true.

    An absent or unreadable config.yaml warns instead of skipping silently —
    either case is ambiguous (never shipped / broken
    vs. genuinely non-HITL) and should not look like a clean auto-waiver.
    """
    if not _CONFIG_PATH.exists():
        warnings.warn(
            f"{_CONFIG_PATH} not found — PB-7 skipped without verifying hitl.enabled. "
            "If this template calls interrupt(), ship config/config.yaml before release.",
            stacklevel=2,
        )
        return False
    try:
        import yaml  # pyyaml==6.0.1 — pinned in pyproject.toml

        data = yaml.safe_load(_CONFIG_PATH.read_text())
    except Exception as exc:
        warnings.warn(
            f"{_CONFIG_PATH} could not be read as YAML ({exc}) — PB-7 skipped without "
            "verifying hitl.enabled. If this template calls interrupt(), fix config/config.yaml "
            "before release.",
            stacklevel=2,
        )
        return False
    hitl = (data or {}).get("hitl", {}) if isinstance(data, dict) else None
    if not isinstance(hitl, dict):
        warnings.warn(
            f"{_CONFIG_PATH} does not have the expected 'hitl:' mapping shape — PB-7 skipped "
            "without verifying hitl.enabled. If this template calls interrupt(), fix "
            "config/config.yaml before release.",
            stacklevel=2,
        )
        return False
    if bool(hitl.get("enabled", False)):
        return True
    # Retain compatibility with legacy configurable and top-level locations.
    if (data.get("configurable") or {}).get("hitl_enabled") is True:
        return True
    return data.get("hitl_enabled") is True


pytestmark = pytest.mark.skipif(
    not _hitl_enabled(),
    reason="config/agent.yaml does not enable HITL — PB-7 not applicable",
)


def _hitl_agent():
    cfg = graph_config(service=FakeResearchService(), llm=FakeLLMClient())
    cfg.update({"memory_enabled": True, "hitl": {"enabled": True, "max_hitl": 8}})
    agent = Graph(config=cfg)
    agent.compile(checkpointer=InMemorySaver())
    return agent


def _ctx(session, thread):
    return InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
                             session_id=session, thread_id=thread)


def test_pb7_interrupt_surfaces_as_awaiting_human_not_error():
    """interrupt() at the plan gate propagates through BaseNode.__call__() →
    awaiting_human, not swallowed by the error boundary."""
    out = _hitl_agent().invoke("What is X?", ctx=_ctx("pb7-s", "pb7-t"), input_context={})
    assert out["status"] == AgentStatus.AWAITING_HUMAN.value
    assert not out.get("error_log")


def test_pb7_resume_reaches_success():
    """resume() must reach SUCCESS — guards the fresh-subgraph-per-call bug where
    a new inner graph+saver loses the suspend checkpoint."""
    agent = _hitl_agent()
    out = agent.invoke("What is X?", ctx=_ctx("pb7-r", "pb7-rt"), input_context={})
    assert out["status"] == AgentStatus.AWAITING_HUMAN.value
    final = agent.resume(thread_id=out["thread_id"], feedback="Go")
    assert final["status"] == AgentStatus.SUCCESS.value
    assert final["formatted_output"]


def test_pb7_bare_graph_compile_suspends_and_resumes():
    """A bare Graph().compile() (no explicit config/checkpointer —
    direct / AgentGateway construction) must still suspend at the plan gate and
    resume to success. Graph defaults memory/hitl ON and attaches a default
    saver, so interrupt() does not degrade to status=error."""
    cfg = graph_config(service=FakeResearchService(), llm=FakeLLMClient())
    agent = Graph(config=cfg)
    agent.compile()  # NO checkpointer arg
    out = agent.invoke("What is X?", ctx=_ctx("bare-s", "bare-t"), input_context={})
    assert out["status"] == AgentStatus.AWAITING_HUMAN.value
    final = agent.resume(thread_id=out["thread_id"], feedback="Go")
    assert final["status"] == AgentStatus.SUCCESS.value


def test_pb7_edited_quality_bar_takes_effect():
    """A human-edited quality_bar on resume must sync to the top-level
    state field that retrieve/synthesize read (not just research_plan)."""
    agent = _hitl_agent()
    out = agent.invoke("What is X?", ctx=_ctx("qb-s", "qb-t"), input_context={})
    assert out["status"] == AgentStatus.AWAITING_HUMAN.value
    agent.resume(thread_id=out["thread_id"],
                 feedback={"research_plan": {"proposed_sources": ["arxiv"],
                                             "quality_bar": "peer_reviewed_only"}})
    snap = agent._compiled.get_state({"configurable": {"thread_id": "qb-t"}})
    assert (snap.values or {}).get("quality_bar") == "peer_reviewed_only"


@pytest.mark.parametrize("opt_out", [
    {"hitl": {"enabled": False}},                  # standard flag
    {"hitl_enabled": False},                       # inner flag
    {"configurable": {"hitl_enabled": False}},     # legacy configurable location
])
def test_pb7_hitl_opt_out_completes_without_interrupt(opt_out):
    """Disabling HITL via ANY supported location — the
    standard hitl.enabled=false, the inner hitl_enabled=false, or the manifest
    configurable.hitl_enabled=false — must actually opt out: an unconfirmed
    invoke runs to completion instead of suspending. Guards the precedence bug
    where the injected default hitl block overrode the manifest opt-out."""
    cfg = graph_config(service=FakeResearchService(), llm=FakeLLMClient())
    cfg.update(opt_out)
    agent = Graph(config=cfg)
    agent.compile()
    out = agent.invoke("What is X?", ctx=_ctx("opt-s", "opt-t"), input_context={})
    assert out["status"] == AgentStatus.SUCCESS.value
    assert out["status"] != AgentStatus.AWAITING_HUMAN.value


def test_pb7_hitl_allowed_false_skips_interrupt():
    """hitl_allowed=False must skip interrupt() and confirm the plan — no
    deadlock. Exercised through the compiled graph (bare node call can't run
    interrupt())."""
    cfg = graph_config(service=FakeResearchService(), llm=FakeLLMClient())
    cfg.update({"memory_enabled": True, "hitl": {"enabled": True, "max_hitl": 8}})
    agent = Graph(config=cfg)
    agent.compile(checkpointer=InMemorySaver())
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
                            session_id="pb7-auto", thread_id="pb7-at", hitl_allowed=False)
    out = agent.invoke("What is X?", ctx=ctx, input_context={})
    # No suspend: the guard skips interrupt and the pipeline runs to completion.
    assert out["status"] == AgentStatus.SUCCESS.value
    assert out["status"] != AgentStatus.AWAITING_HUMAN.value
