"""Test doubles for CMN-C2-249 — constructor-injected (SDK v1.0 wheel contract)."""

from __future__ import annotations

from typing import Any


class FakeResearchService:
    """Deterministic retrieval. `fail_sources` raise to exercise gap degradation.

    Returns TWO documents per source by default. That is not padding: a fake
    that returns one document per connector makes `source_id` and "the document"
    indistinguishable, which is exactly why the citation-granularity defect was
    invisible to the whole suite while reproducing on the first real corpus.
    The second document is deliberately off-topic, so any test whose claim cites
    one document will fail if the report drags in its connector-mate.
    """

    def __init__(self, fail_sources: set | None = None, empty: bool = False,
                 docs_per_source: int = 2):
        self.fail_sources = fail_sources or set()
        self.empty = empty
        self.docs_per_source = docs_per_source
        self.calls: list[str] = []

    def retrieve(self, source_id: str, query: str, quality_bar: str) -> list[dict[str, Any]]:
        self.calls.append(source_id)
        if source_id in self.fail_sources:
            raise RuntimeError(f"connector {source_id} down")
        if self.empty:
            return []
        titles = [f"{source_id} result for {query}", f"{source_id} unrelated paper"]
        return [
            {
                "title": titles[i] if i < len(titles) else f"{source_id} doc {i + 1}",
                "snippet": "evidence snippet" if i == 0 else "unrelated snippet",
                "url": f"https://example.test/{source_id}/{i + 1}",
                "published_date": "2026-01-01",
                "score": 0.9 - (0.1 * i),
            }
            for i in range(self.docs_per_source)
        ]


class FakeLLMClient:
    """Returns well-formed findings by default. Configurable for S-3 tests."""

    def __init__(self, findings: list | None = None, gaps: list | None = None):
        self._findings = findings
        self._gaps = gaps or []
        self.calls = 0

    def synthesize(self, question: str, evidence: list, quality_bar: str) -> dict:
        self.calls += 1
        if self._findings is not None:
            return {"findings": self._findings, "gaps": self._gaps}
        # Default: ONE finding citing exactly the FIRST document — never the
        # whole connector. A claim must name the document that supports it.
        from src.services.research_service import evidence_reference_id

        ref = evidence_reference_id(evidence[0]) if evidence else None
        findings = (
            [{"claim": f"Answer to: {question}", "supporting_source_ids": [ref], "confidence": 0.82}]
            if ref
            else []
        )
        return {"findings": findings, "gaps": self._gaps}


def graph_config(service=None, llm=None, hitl_enabled=None, allowed_sources=None) -> dict:
    """Build a Graph(config=...) dict wiring test doubles into the inner workflow.

    `llm` goes into the standard `config["llm"]` seam — the same key the
    live-LLM harness and a composed caller use. There is no separate test-only
    key, so a test that passes here exercises the production wiring.
    """
    cfg: dict = {}
    if service is not None:
        cfg["research_service"] = service
    if llm is not None:
        cfg["llm"] = llm
    if hitl_enabled is not None:
        cfg["hitl_enabled"] = hitl_enabled
    if allowed_sources is not None:
        cfg["allowed_sources"] = allowed_sources
    return cfg
