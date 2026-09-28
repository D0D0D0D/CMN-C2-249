# CMN-C2-249 — Unit Tests: pre_process node (S-2 input gate)

from __future__ import annotations

from framework.schemas.agent_status import AgentStatus
from src.nodes.pre_process_node import PreProcessNode


class TestPreProcessNode:
    def setup_method(self):
        self.node = PreProcessNode()

    def test_valid_question_builds_plan(self):
        state = {"user_input": "  What are the effects of X?  ", "input_context": {}}
        r = self.node.execute(state)
        assert r["status"] == AgentStatus.SUCCESS
        assert r["research_question"] == "What are the effects of X?"
        assert "proposed_sources" in r["research_plan"]
        assert r["plan_confirmed"] is False  # no explicit user spec

    def test_empty_input_early_terminates(self):
        r = self.node.execute({"user_input": "   "})
        assert r["status"] == AgentStatus.ERROR
        assert r["error"]  # state["error"] triggers backbone hard stop

    def test_overlong_input_rejected(self):
        r = self.node.execute({"user_input": "x" * 5000})
        assert r["status"] == AgentStatus.ERROR

    def test_credential_pattern_rejected(self):
        r = self.node.execute({"user_input": "summarize sk-abcdef1234567890abcdefXY"})
        assert r["status"] == AgentStatus.ERROR
        assert "credential" in r["error"].lower()

    def test_invalid_quality_bar_rejected(self):
        state = {
            "user_input": "q",
            "input_context": {"source_spec": {"quality_bar": "bogus"}},
        }
        r = self.node.execute(state)
        assert r["status"] == AgentStatus.ERROR

    def test_out_of_catalog_source_rejected(self):
        state = {
            "user_input": "q",
            "input_context": {"source_spec": {"sources": ["evil_endpoint"]}},
        }
        r = self.node.execute(state)
        assert r["status"] == AgentStatus.ERROR
        assert "catalog" in r["error"].lower()

    def test_fully_specified_plan_is_preconfirmed(self):
        state = {
            "user_input": "q",
            "input_context": {
                "source_spec": {"sources": ["arxiv"], "quality_bar": "peer_reviewed_only"}
            },
        }
        r = self.node.execute(state)
        assert r["status"] == AgentStatus.SUCCESS
        assert r["plan_confirmed"] is True
        assert r["quality_bar"] == "peer_reviewed_only"
