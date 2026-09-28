# CMN-C2-249 — Proof-of-Boundary: S-1 / S-2 / S-3 enforcement
# (real agenticstar-agentcore wheel — no framework stubs)

from __future__ import annotations

import pytest

from framework.errors import SecurityViolationError
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from tests.fakes import FakeLLMClient, FakeResearchService, graph_config

_CONFIRMED_CONTEXT = {
    "plan_confirmed": True,  # skip HITL so retrieval is reached
    "source_spec": {"sources": ["arxiv"], "quality_bar": "peer_reviewed_only"},
}


def _agent():
    agent = Graph(config=graph_config(
        service=FakeResearchService(), llm=FakeLLMClient(),
    ))
    agent.compile()
    return agent


class TestS1TrustBoundary:
    """retrieve_node requires VERIFIED_EXTERNAL; ANONYMOUS callers are denied."""

    def test_anonymous_caller_denied_at_retrieval(self):
        out = _agent().invoke(
            "What is X?",
            ctx=InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS),
            input_context=dict(_CONFIRMED_CONTEXT),
        )
        assert out.get("error")
        assert "S-1" in out["error"]
        assert not out.get("formatted_output")  # no report under denied trust

    def test_verified_external_caller_allowed(self):
        out = _agent().invoke(
            "What is X?",
            ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
            input_context=dict(_CONFIRMED_CONTEXT),
        )
        assert out.get("formatted_output")


class TestS2InputBoundary:
    """pre_process rejects out-of-catalog source selection and credential input."""

    def test_out_of_catalog_source_rejected(self):
        node = PreProcessNode()
        r = node.execute({
            "user_input": "q",
            "input_context": {"source_spec": {"sources": ["http://attacker.test"]}},
        })
        assert r["status"] == "error"
        assert r["error"]

    def test_credential_in_input_rejected(self):
        node = PreProcessNode()
        r = node.execute({"user_input": "AKIAABCDEFGHIJKLMNOP leak"})
        assert r["status"] == "error"


class TestS3OutputBoundary:
    """post_process rejects any finding lacking a source or confidence (no-hallucination)."""

    def test_unsourced_claim_blocks_report(self):
        node = PostProcessNode()
        with pytest.raises(SecurityViolationError):
            node.execute({
                "research_question": "q",
                "synthesized_findings": [
                    {"claim": "fabricated", "supporting_source_ids": [], "confidence": 0.99}
                ],
                "retrieved_evidence": [],
                "identified_gaps": [],
            })
