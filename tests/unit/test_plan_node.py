# CMN-C2-249 — Unit Tests: plan_node (HITL plan gate)

from __future__ import annotations

from framework.schemas.agent_status import AgentStatus

from src.nodes.plan_node import PlanNode


class TestPlanNode:
    def setup_method(self):
        self.plan = {"proposed_sources": ["arxiv"], "quality_bar": "peer_reviewed_only", "depth": "deep"}

    def test_unconfirmed_plan_hitl_allowed_false_confirms(self):
        """D6 migration: with hitl_allowed=False (automated pipeline) the plan
        gate skips interrupt() and confirms the plan as-is (no deadlock). The
        interrupt() path itself is exercised through the compiled graph in the
        integration + PB-7 tests (interrupt() cannot run in a bare node call)."""
        state = {"research_plan": self.plan, "plan_confirmed": False, "hitl_allowed": False}
        r = PlanNode(hitl_enabled=True).execute(state)
        assert r["status"] == AgentStatus.SUCCESS.value
        assert r["hitl_required"] is False
        assert r["research_plan"] == self.plan

    def test_confirmed_plan_skips_hitl(self):
        state = {"research_plan": self.plan, "plan_confirmed": True}
        r = PlanNode(hitl_enabled=True).execute(state)
        assert r["status"] == AgentStatus.SUCCESS.value
        assert r["hitl_required"] is False

    def test_hitl_disabled_proceeds(self):
        state = {"research_plan": self.plan, "plan_confirmed": False}
        r = PlanNode(hitl_enabled=False).execute(state)
        assert r["status"] == AgentStatus.SUCCESS.value
        assert r["hitl_required"] is False
