# CMN-C2-249 — Integration Tests: full backbone + inner research workflow
# (real agenticstar-agentcore wheel — no framework stubs)

from __future__ import annotations

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from tests.fakes import FakeLLMClient, FakeResearchService, graph_config


def _ctx(level=TrustLevel.VERIFIED_EXTERNAL):
    return InvocationContext(caller_trust_level=level, session_id="s1")


def _agent(service=None, llm=None, hitl_enabled=None):
    agent = Graph(config=graph_config(
        service=service or FakeResearchService(),
        llm=llm or FakeLLMClient(),
        hitl_enabled=hitl_enabled,
    ))
    agent.compile()
    return agent


def _hitl_agent(service=None, llm=None):
    """A HITL-capable agent: memory_enabled/hitl in config (so the framework
    threads a thread_id) + an EXPLICIT checkpointer, matching src/api/server.py.
    Required for the D6 interrupt()/resume round-trip."""
    from langgraph.checkpoint.memory import InMemorySaver

    cfg = graph_config(service=service or FakeResearchService(), llm=llm or FakeLLMClient())
    cfg.update({"memory_enabled": True, "hitl": {"enabled": True, "max_hitl": 8}})
    agent = Graph(config=cfg)
    agent.compile(checkpointer=InMemorySaver())
    return agent


def _hitl_ctx(level=TrustLevel.VERIFIED_EXTERNAL):
    return InvocationContext(caller_trust_level=level, session_id="hitl-s1", thread_id="hitl-t1")


_CONFIRMED_CONTEXT = {
    "plan_confirmed": True,
    "source_spec": {"sources": ["arxiv"], "quality_bar": "peer_reviewed_only"},
}


class TestWorkflow:
    def test_happy_path_produces_report(self):
        out = _agent().invoke("What is X?", ctx=_ctx(), input_context=dict(_CONFIRMED_CONTEXT))
        assert out["formatted_output"]
        assert out["synthesized_findings"]
        assert out["evidence_report"]["citations"]
        # full 5-node backbone executed
        assert out["node_history"][0] == "InitializeNode"
        assert out["node_history"][-1] == "FinalizeNode"
        assert "PostProcessNode" in out["node_history"]

    def test_hitl_pause_when_plan_unconfirmed(self):
        """D6 migration: an unconfirmed plan now pauses via interrupt() and
        surfaces status=awaiting_human (was a state-flag hitl_required that
        ended the run). resume(Go) then reaches success — this catches the
        subgraph-cache / lost-checkpoint failure mode."""
        agent = _hitl_agent()
        out = agent.invoke("What is X?", ctx=_hitl_ctx(), input_context={})
        assert out["status"] == AgentStatus.AWAITING_HUMAN.value
        assert not out.get("formatted_output")   # paused before retrieval/report

        final = agent.resume(thread_id=out["thread_id"], feedback="Go")
        assert final["status"] == AgentStatus.SUCCESS.value
        assert final["formatted_output"]
        assert final["evidence_report"]["citations"]

    def test_empty_question_early_terminates(self):
        out = _agent().invoke("   ", ctx=_ctx(), input_context=dict(_CONFIRMED_CONTEXT))
        assert out.get("error")
        assert not out.get("formatted_output")
        assert "PostProcessNode" not in out["node_history"]

    def test_partial_source_failure_degrades_to_gaps(self):
        svc = FakeResearchService(fail_sources={"pubmed"})
        out = _agent(service=svc).invoke(
            "What is X?",
            ctx=_ctx(),
            input_context={
                "plan_confirmed": True,
                "source_spec": {"sources": ["arxiv", "pubmed"],
                                "quality_bar": "peer_reviewed_only"},
            },
        )
        assert out["formatted_output"]
        assert any("pubmed" in g for g in out["identified_gaps"])
