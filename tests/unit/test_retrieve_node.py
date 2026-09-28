# CMN-C2-249 — Unit Tests: retrieve_node (multi-source, S-1 declared, gap degrade)

from __future__ import annotations

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from tests.fakes import FakeResearchService


def _state(sources, quality="news_acceptable"):
    return {
        "research_question": "q",
        "research_plan": {"proposed_sources": sources, "quality_bar": quality},
        "quality_bar": quality,
    }


class TestRetrieveNode:
    def test_declares_s1_trust_level(self):
        assert RetrieveNode().required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_retrieves_from_allowed_sources(self):
        node = RetrieveNode(service=FakeResearchService())
        r = node.execute(_state(["arxiv", "news"]))
        assert r["status"] == AgentStatus.SUCCESS.value
        # Two connectors x two documents each. Ids are written out literally —
        # deriving them from make_evidence_id() would make this a self-
        # referential oracle that passes whatever the production code emits.
        assert [e["evidence_id"] for e in r["retrieved_evidence"]] == [
            "arxiv#1", "arxiv#2", "news#1", "news#2",
        ]
        assert set(r["source_whitelist"]) == {"arxiv", "news"}

    def test_quality_bar_filters_low_tier_sources(self):
        node = RetrieveNode(service=FakeResearchService())
        # peer_reviewed_only must exclude the news source
        r = node.execute(_state(["arxiv", "news"], "peer_reviewed_only"))
        assert r["source_whitelist"] == ["arxiv"]

    def test_failed_source_degrades_to_gap(self):
        node = RetrieveNode(service=FakeResearchService(fail_sources={"news"}))
        r = node.execute(_state(["arxiv", "news"]))
        assert r["status"] == AgentStatus.SUCCESS.value
        assert [e["evidence_id"] for e in r["retrieved_evidence"]] == ["arxiv#1", "arxiv#2"]
        assert any("news" in g for g in r["identified_gaps"])

    def test_connector_supplied_evidence_id_is_overwritten(self):
        # evidence_id is a server-issued handle. A connector that returns its own
        # value must not be able to impersonate another document's handle.
        class _Forging:
            def retrieve(self, source_id, query, quality_bar):
                return [{"evidence_id": "arxiv#1", "title": "forged",
                         "url": "https://example.test/forged/1"}]

        r = RetrieveNode(service=_Forging()).execute(_state(["pubmed"]))
        assert [e["evidence_id"] for e in r["retrieved_evidence"]] == ["pubmed#1"]

    def test_numbering_restarts_per_connector(self):
        r = RetrieveNode(service=FakeResearchService()).execute(_state(["arxiv", "pubmed"]))
        assert [e["evidence_id"] for e in r["retrieved_evidence"]] == [
            "arxiv#1", "arxiv#2", "pubmed#1", "pubmed#2",
        ]

    def test_no_evidence_is_retryable(self):
        node = RetrieveNode(service=FakeResearchService(empty=True))
        r = node.execute(_state(["arxiv"]))
        # SDK v1.0 backbone: retryable = status RETRY (route() -> pre_process,
        # max_retry 3); hard errors use status ERROR + state["error"].
        assert r["status"] == AgentStatus.RETRY.value
        assert "error" not in r  # no hard-stop marker on the retryable path
