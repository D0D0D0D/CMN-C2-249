"""AgentCore Platform v1.0 — CMN-C2-249 synthesize_node (inner workflow).

Cross-source synthesis: binds each claim to its supporting source_ids with an
explicit confidence score, and aggregates evidence shortfalls into
identified_gaps. The actual generation is delegated to an injected LLMClient so
keys stay out of state and the node stays testable.

Provenance ownership: the model authors prose and *reference*
ids only. Every supporting_source_id it returns is a per-DOCUMENT `evidence_id`
issued by retrieve_node, resolved against retrieved_evidence here — an id absent
from the retrieval records is discarded and recorded as a gap, so neither a
fabricated citation nor a connector-wide one can reach the report. The
citation metadata itself (title / URL / tier / date) is read from the retrieval
records in post_process_node, never from the model.

Degradation: synthesis IS this agent's core capability and there is no
deterministic substitute for it, so an LLM failure returns status=error rather
than a plausible-looking empty report. Only the exception TYPE NAME is recorded
— framework BaseNode.__call__ would otherwise place str(exc) plus a traceback
into error_log, which this template propagates to the caller through
merge_output() -> get_output().

S-1: required_trust_level = VERIFIED_EXTERNAL — the node may resolve a billable
     provider credential, which is not an anonymous-caller operation.
S-4: emits one domain audit event per synthesis call (llm_synthesis).
Dependencies (LLMClient) are constructor-injected — the SDK v1.0 wheel passes
state only into execute().
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.security_gates import assert_no_nested_credentials
from src.services.llm_client import LLMClient, build_default_llm_client, coerce_llm_client
from src.services.research_service import evidence_reference_ids

# Secret declared in config/agent.yaml requires.secrets.
LLM_API_KEY_SECRET = "LLM_API_KEY"

_MAX_ID_CHARS = 64
_MAX_CLAIM_CHARS = 120


def _valid_confidence(value: object) -> bool:
    """A confidence score must be a finite number in [0, 1].

    The prompt asks for that range, but a model is free to ignore it. An
    out-of-range score (`7`) would be rendered as a real confidence, and `NaN`
    — which `json.loads` accepts — propagates into the report and then fails
    JSON serialisation at the HTTP boundary. `bool` is excluded explicitly
    because `isinstance(True, int)` is True in Python.
    """
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        # float() raises OverflowError on a huge int (e.g. 10**400). This runs
        # outside the provider try/except, so an unguarded conversion would turn
        # a malformed score into a framework-level traceback instead of an
        # ordinary gap demotion.
        score = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(score) and 0.0 <= score <= 1.0


class SynthesizeNode(FunctionNode):
    """Cross-source synthesis with claim->source binding and confidence."""

    # S-1: explicit by design, not inherited implicitly. The node resolves a
    # paid provider credential on the un-injected path, so ANONYMOUS is not
    # appropriate.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm: Any = None, llm_options: dict[str, Any] | None = None) -> None:
        # DI seam: an injected client (config["llm"], normalised by the graph)
        # wins; otherwise the provider client is built per invocation inside
        # execute() from ctx.secrets.
        self._llm: LLMClient | None = coerce_llm_client(llm)
        # Static, caller-independent configuration only — never a client built
        # from one caller's secrets (node instances are shared across
        # invocations).
        self._llm_options: dict[str, Any] = dict(llm_options or {})

    # ── client resolution ───────────────────────────────────────────────
    def _resolve_client(self, state: dict) -> LLMClient:
        """Return the synthesis client for THIS invocation.

        A caller-independent injected client is reused. Otherwise the provider
        client is constructed fresh from this invocation's secrets and
        deliberately NOT cached on self — node instances are shared across every
        invocation, so caching would expose one caller's credential to the next.
        """
        if self._llm is not None:
            return self._llm
        ctx = InvocationContext.from_state(state)
        return build_default_llm_client(
            self._llm_options, ctx.secrets.require(LLM_API_KEY_SECRET)
        )

    def execute(self, state: dict) -> dict:
        question = state.get("research_question", "")
        evidence = state.get("retrieved_evidence", []) or []
        quality_bar = state.get("quality_bar", "news_acceptable")
        prior_gaps = list(state.get("identified_gaps", []) or [])

        try:
            client = self._resolve_client(state)
            result = client.synthesize(question, evidence, quality_bar)
            if not isinstance(result, dict):
                raise TypeError("synthesis client did not return a mapping")
        except Exception as exc:  # noqa: BLE001 — provider/secret/parse failures
            # TYPE NAME ONLY. str(exc) can quote the request back or carry the
            # credential used to sign it, and this template surfaces error_log
            # to the caller.
            reason = type(exc).__name__
            emit_trace_event("llm_synthesis_failed", {"error_type": reason}, state)
            return {
                "synthesized_findings": [],
                "identified_gaps": prior_gaps + [f"synthesis unavailable ({reason})"],
                "status": AgentStatus.ERROR.value,
                "error": f"SynthesizeNode: synthesis failed ({reason})",
                "error_log": [f"SynthesizeNode: synthesis failed ({reason})"],
            }

        raw_findings = result.get("findings", []) or []
        new_gaps = [str(g) for g in (result.get("gaps", []) or [])]

        # Provenance resolution: only ids that exist in the retrieval records
        # may be cited. Ids the model invented are dropped, never rendered.
        # These are per-DOCUMENT evidence_id handles: a bare connector name
        # ("pubmed") no longer resolves, because it does not identify which of
        # that connector's documents supports the claim.
        known_ids = set(evidence_reference_ids(evidence))

        findings: list[dict] = []
        for f in raw_findings:
            if not isinstance(f, dict):
                continue
            claim = (f.get("claim") or "").strip()
            requested = [str(s) for s in (f.get("supporting_source_ids", []) or [])]
            resolved = [s for s in requested if s in known_ids]
            unresolved = [s for s in requested if s not in known_ids]
            confidence = f.get("confidence")

            if unresolved:
                new_gaps.append(
                    "citation to unknown source id discarded: "
                    + ", ".join(s[:_MAX_ID_CHARS] for s in unresolved)
                )

            # Keep only well-formed findings (claim + >=1 RESOLVED supporting
            # source + confidence). Malformed findings are NOT silently dropped
            # into the report — they are recorded as gaps so the evidence trail
            # stays honest.
            if claim and resolved and _valid_confidence(confidence):
                findings.append(
                    {
                        "claim": claim,
                        "supporting_source_ids": resolved,
                        "confidence": float(confidence),
                    }
                )
            elif claim:
                new_gaps.append(
                    f"unsupported/low-confidence claim dropped: {claim[:_MAX_CLAIM_CHARS]}"
                )

        # Never return an empty report that also reports no gaps — "no findings,
        # no gaps" reads as "nothing was missing", which is a false conclusion.
        if not findings and not new_gaps:
            new_gaps.append("no supportable finding was produced from the retrieved evidence")

        emit_trace_event(
            "llm_synthesis",
            {
                "findings": len(findings),
                "evidence_items": len(evidence),
                # Canonical response model when the client reports one.
                "model": str(result.get("model") or "unreported"),
            },
            state,
        )

        return {
            "synthesized_findings": findings,
            "identified_gaps": prior_gaps + new_gaps,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_output(self, result: dict) -> dict:
        """S-3 extension — see src/security_gates.py.

        The pinned 1.0.0 `@final` gate scans top-level strings only; this node
        returns its payload as nested lists/dicts, so the nested values need the
        same fail-closed credential check.
        """
        return assert_no_nested_credentials(result, "SynthesizeNode")
