# CMN-C2-249 — Unit Tests: post_process node (S-3 preservation / report rendering)

from __future__ import annotations

import pytest

from framework.errors import SecurityViolationError
from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import PostProcessNode


def _state(findings):
    return {
        "research_question": "q",
        "synthesized_findings": findings,
        "retrieved_evidence": [
            {"evidence_id": "arxiv#1", "source_id": "arxiv", "source_tier": "peer_reviewed",
             "title": "T", "url": "https://x/1", "published_date": "2026-01-01"},
        ],
        "identified_gaps": ["a gap"],
    }


class TestPostProcessNode:
    def setup_method(self):
        self.node = PostProcessNode()

    def test_valid_findings_render_report(self):
        r = self.node.execute(_state(
            [{"claim": "C", "supporting_source_ids": ["arxiv#1"], "confidence": 0.8}]
        ))
        assert r["status"] == AgentStatus.SUCCESS
        assert r["evidence_report"]["citations"]
        assert "Evidence Report" in r["formatted_output"]
        assert r["evidence_report"]["confidence_summary"]["n"] == 1

    def test_unsourced_finding_rejected_s3(self):
        with pytest.raises(SecurityViolationError):
            self.node.execute(_state(
                [{"claim": "C", "supporting_source_ids": [], "confidence": 0.8}]
            ))

    def test_no_confidence_finding_rejected_s3(self):
        with pytest.raises(SecurityViolationError):
            self.node.execute(_state(
                [{"claim": "C", "supporting_source_ids": ["arxiv#1"], "confidence": None}]
            ))

    def test_gaps_surface_in_report(self):
        r = self.node.execute(_state(
            [{"claim": "C", "supporting_source_ids": ["arxiv#1"], "confidence": 0.8}]
        ))
        assert r["evidence_report"]["gaps"] == ["a gap"]
