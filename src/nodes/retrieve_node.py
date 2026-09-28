"""AgentCore Platform v1.0 — CMN-C2-249 retrieve_node (inner workflow).

Fetches evidence from the effective source set = (deploy catalog) ∩ (user/plan
selection), filtered by the citation quality bar. Internal KB is index search;
external sources are live connectors. Hits are deduplicated and aggregated.

Each accepted hit is stamped with a per-document `evidence_id` (`{source_id}#{n}`)
issued here — this is the handle a finding cites, so a claim binds to the one
document that supports it rather than to every document the connector returned.

S-1: required_trust_level = VERIFIED_EXTERNAL — the framework BaseNode.__call__
     enforces this before execute() runs (internal KB + external connectors).
S-4: emits one domain audit event per source (source_retrieval).
Error strategy: a per-source failure is caught and the source is degraded to an
     identified_gap; a total failure (no source returned anything) returns
     status RETRY so the outer backbone route() can re-attempt (max_retry 3).
Dependencies (ResearchService / catalog) are constructor-injected — the SDK
v1.0 wheel passes state only into execute().
"""

from __future__ import annotations

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.policy import DEFAULT_SOURCE_CATALOG, effective_sources
from src.security_gates import assert_no_nested_credentials
from src.services.research_service import EVIDENCE_ID_FIELD, ResearchService, make_evidence_id


class RetrieveNode(FunctionNode):
    """Multi-source evidence retrieval (S-1 gated, S-4 audited)."""

    # S-1: declared trust requirement, auto-enforced by BaseNode.__call__.
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, service: ResearchService | None = None,
                 catalog: dict | None = None) -> None:
        # DI seams: ctor arg > fail-loud default (unconfigured service raises).
        self._service = service or ResearchService()
        self._catalog = catalog or DEFAULT_SOURCE_CATALOG

    def execute(self, state: dict) -> dict:
        question = state.get("research_question", "")
        plan = state.get("research_plan", {}) or {}
        quality_bar = state.get("quality_bar", "news_acceptable")
        catalog = self._catalog
        service = self._service

        whitelist = effective_sources(
            list(plan.get("proposed_sources", [])), catalog, quality_bar
        )

        evidence: list[dict] = []
        gaps: list[str] = []
        seen: set = set()

        for source_id in whitelist:
            meta = catalog[source_id]
            try:
                hits = service.retrieve(source_id, question, quality_bar)
                count = 0
                for hit in hits:
                    key = hit.get("url") or (source_id, hit.get("title"))
                    if key in seen:
                        continue
                    seen.add(key)
                    # Assigned, never setdefault: the connector we actually
                    # called and the deploy catalog are authoritative for
                    # provenance. A connector allowed to keep its own
                    # source_id/source_tier could label its hit as coming from a
                    # different source, or claim a higher citation tier than the
                    # quality bar admits.
                    hit["source_id"] = source_id
                    hit["source_type"] = meta["type"]
                    hit["source_tier"] = meta["tier"]
                    count += 1
                    # Per-DOCUMENT citation handle, issued here. Assigned, never
                    # setdefault: a connector-supplied `evidence_id` is untrusted
                    # payload that could collide with (or impersonate) another
                    # document's handle, so it is overwritten unconditionally.
                    hit[EVIDENCE_ID_FIELD] = make_evidence_id(source_id, count)
                    evidence.append(hit)
                emit_trace_event(
                    "source_retrieval",
                    {"source_id": source_id, "hits": count, "status": "ok"},
                    state,
                )
            except Exception as exc:  # transient/per-source failure -> gap
                gaps.append(f"source '{source_id}' unavailable: {type(exc).__name__}")
                emit_trace_event(
                    "source_retrieval",
                    {"source_id": source_id, "status": "error",
                     "error": type(exc).__name__},
                    state,
                )

        # Fail closed on a duplicate handle. Two documents sharing one
        # evidence_id would make a citation ambiguous, and the report would
        # silently render whichever record the index happened to keep — the
        # claim would then cite a document the model never pointed at.
        issued = [e[EVIDENCE_ID_FIELD] for e in evidence]
        if len(set(issued)) != len(issued):
            raise SecurityViolationError(
                "S-3: duplicate evidence_id issued; citations would be ambiguous"
            )

        if not evidence:
            # Nothing retrieved at all -> retryable (backbone route: RETRY -> pre_process).
            return {
                "source_whitelist": whitelist,
                "retrieved_evidence": [],
                "identified_gaps": gaps or ["no evidence retrieved for the question"],
                "status": AgentStatus.RETRY.value,
                "error_log": ["RetrieveNode: no evidence retrieved (retryable)"],
            }

        return {
            "source_whitelist": whitelist,
            "retrieved_evidence": evidence,
            "identified_gaps": gaps,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_output(self, result: dict) -> dict:
        """S-3 extension — see src/security_gates.py.

        The pinned 1.0.0 `@final` gate scans top-level strings only; this node
        returns its payload as nested lists/dicts, so the nested values need the
        same fail-closed credential check.
        """
        return assert_no_nested_credentials(result, "RetrieveNode")
