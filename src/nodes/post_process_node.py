"""AgentCore Platform v1.0 — CMN-C2-249 post_process node.

Renders the evidence report (tiered citations, per-claim confidence, gap list).
Citations are resolved per DOCUMENT via the `evidence_id` handle issued by
retrieve_node, so a claim cites the documents that support it rather than every
document its connector happened to return.
and enforces the S-3 Output Gate (preservation variant / no-hallucination-as-
fact): a report is rejected if any finding lacks a supporting source or an
explicit confidence score. Credential scanning of the output is performed by
the framework @final _security_gate_output.

Returns only changed fields. Runs only on the SUCCESS path (HITL pause and hard
errors route straight to finalize, so findings are guaranteed present here).
"""

from __future__ import annotations

from typing import ClassVar

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.security_gates import assert_no_nested_credentials
from src.services.research_service import evidence_reference_id


class PostProcessNode(FunctionNode):
    """Format the evidence report and enforce citation integrity (S-3)."""

    # S-1: explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict) -> dict:
        question = state.get("research_question", "")
        findings = state.get("synthesized_findings", []) or []
        evidence = state.get("retrieved_evidence", []) or []
        gaps = state.get("identified_gaps", []) or []
        # S-4: at least one domain event per node — the framework audit contract
        emit_trace_event(
            "report_render",
            {"findings": len(findings), "evidence_items": len(evidence), "gaps": len(gaps)},
            state,
        )

        # ── S-3 preservation: no claim without a source / confidence ────
        for f in findings:
            if not f.get("supporting_source_ids"):
                raise SecurityViolationError(
                    "S-3: finding emitted without a supporting source "
                    "(no-hallucination-as-fact gate)"
                )
            if not isinstance(f.get("confidence"), (int, float)):
                raise SecurityViolationError(
                    "S-3: finding emitted without an explicit confidence score"
                )

        evidence_by_id = self._index_evidence(evidence)
        citations = self._build_citations(findings, evidence_by_id)
        report = {
            "sections": self._build_sections(question, findings),
            "citations": citations,
            "confidence_summary": self._confidence_summary(findings),
            "gaps": gaps,
        }

        return {
            "evidence_report": report,
            "formatted_output": self._render_markdown(question, report, findings, gaps),
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_output(self, result: dict) -> dict:
        """S-3 extension — see src/security_gates.py.

        The pinned 1.0.0 `@final` gate scans top-level strings only; this node
        returns its payload as nested lists/dicts, so the nested values need the
        same fail-closed credential check.
        """
        return assert_no_nested_credentials(result, "PostProcessNode")

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _index_evidence(evidence: list) -> dict:
        """Index retrieval records by their per-DOCUMENT citation handle.

        Indexing by `source_id` (the connector) was the citation-granularity
        defect: a claim citing "pubmed" resolved to EVERY pubmed document, so an
        unrelated paper from the same connector was rendered as a citation for
        that claim. The handle is one document, so the lookup is now exact.
        """
        idx: dict = {}
        for e in evidence:
            ref = evidence_reference_id(e)
            if ref:
                idx.setdefault(ref, []).append(e)
        return idx

    @staticmethod
    def _build_citations(findings: list, evidence_by_id: dict) -> list:
        """Cite exactly the documents the findings reference — nothing else.

        Every field below is read from the retrieval RECORD; the model supplies
        only the handle that selects it.
        """
        cited: dict = {}
        for f in findings:
            for ref in f.get("supporting_source_ids", []):
                for e in evidence_by_id.get(ref, []):
                    cited[ref] = {
                        "evidence_id": ref,
                        "source_id": e.get("source_id"),
                        "source_tier": e.get("source_tier"),
                        "title": e.get("title"),
                        "url": e.get("url"),
                        "published_date": e.get("published_date"),
                    }
        return list(cited.values())

    @staticmethod
    def _build_sections(question: str, findings: list) -> list:
        return [
            {
                "heading": "Findings",
                "claims": [
                    {
                        "claim": f["claim"],
                        "supporting_source_ids": f["supporting_source_ids"],
                        "confidence": f["confidence"],
                    }
                    for f in findings
                ],
            }
        ]

    @staticmethod
    def _confidence_summary(findings: list) -> dict:
        if not findings:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        scores = [float(f["confidence"]) for f in findings]
        return {
            "mean": round(sum(scores) / len(scores), 3),
            "min": min(scores),
            "max": max(scores),
            "n": len(scores),
        }

    @staticmethod
    def _render_markdown(question: str, report: dict, findings: list, gaps: list) -> str:
        lines = [f"# Evidence Report: {question}", ""]
        lines.append("## Findings")
        for f in findings:
            srcs = ", ".join(f["supporting_source_ids"])
            lines.append(f"- {f['claim']} _(confidence: {f['confidence']}; sources: {srcs})_")
        lines.append("")
        lines.append("## Citations")
        for c in report["citations"]:
            lines.append(
                f"- [{c.get('evidence_id')}] [{c.get('source_tier')}] "
                f"{c.get('title')} — {c.get('url')}"
            )
        lines.append("")
        lines.append("## Evidence Gaps")
        if gaps:
            for g in gaps:
                lines.append(f"- {g}")
        else:
            lines.append("- None identified")
        return "\n".join(lines)
