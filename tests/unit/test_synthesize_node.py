# CMN-C2-249 — Unit Tests: synthesize_node (claim->source binding, no-hallucination)

from __future__ import annotations

from framework.schemas.agent_status import AgentStatus

from src.nodes.synthesize_node import SynthesizeNode
from tests.fakes import FakeLLMClient


def _state():
    return {
        "research_question": "q",
        "retrieved_evidence": [
            {"evidence_id": "arxiv#1", "source_id": "arxiv", "title": "t", "url": "u"}],
        "quality_bar": "news_acceptable",
        "identified_gaps": ["prior gap"],
    }


class TestSynthesizeNode:
    def test_wellformed_findings_preserved(self):
        llm = FakeLLMClient(findings=[
            {"claim": "C1", "supporting_source_ids": ["arxiv#1"], "confidence": 0.7},
        ])
        r = SynthesizeNode(llm=llm).execute(_state())
        assert r["status"] == AgentStatus.SUCCESS.value
        assert len(r["synthesized_findings"]) == 1
        assert r["identified_gaps"][0] == "prior gap"  # prior gaps preserved

    def test_unsourced_claim_dropped_to_gap(self):
        llm = FakeLLMClient(findings=[
            {"claim": "no source", "supporting_source_ids": [], "confidence": 0.9},
            {"claim": "good", "supporting_source_ids": ["arxiv#1"], "confidence": 0.6},
        ])
        r = SynthesizeNode(llm=llm).execute(_state())
        assert len(r["synthesized_findings"]) == 1
        assert any("unsupported" in g for g in r["identified_gaps"])

    def test_missing_confidence_dropped(self):
        llm = FakeLLMClient(findings=[
            {"claim": "no conf", "supporting_source_ids": ["arxiv#1"], "confidence": None},
        ])
        r = SynthesizeNode(llm=llm).execute(_state())
        assert r["synthesized_findings"] == []
