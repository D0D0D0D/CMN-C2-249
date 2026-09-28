"""AgentCore Platform v1.0 — CMN-C2-249 State schema."""

# ADR-005: State must be a flat TypedDict (see ADR-005 for the prohibited
# object kinds). LangGraph checkpoints use msgpack serialization; non-primitive model objects
# cause silent corruption. Extend AgentState with agent-specific
# fields only. Do NOT add credentials, secrets, or such model objects.
#
# All shared fields (user_input, status, session_id, node_history,
# error_log, error, hitl_*, caller_trust_level, etc.) are inherited
# from AgentState. Source/LLM API keys are NEVER stored in state —
# they are resolved at call time via ctx.secrets.require(...).

from __future__ import annotations

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Deep Research Synthesis & Evidence Report Agent state.

    Cat 2 multi-step workflow: parse question -> (plan) -> HITL confirm ->
    multi-source retrieval -> cross-source synthesis -> evidence report.
    Only agent-specific fields are declared below.
    """

    # ── pre_process output ──────────────────────────────────────────────
    research_question: str          # validated / normalized research question
    user_source_spec: dict          # user-given sources/quality/depth (optional, {} if none)
    research_plan: dict             # {proposed_sources: list[str], quality_bar, depth}
    plan_confirmed: bool            # True -> HITL plan gate skipped
    quality_bar: str                # citation quality tier (peer_reviewed_only / news_acceptable / internal_only)

    # ── main (inner research workflow) output ───────────────────────────
    source_whitelist: list          # effective sources = (deploy catalog) ∩ (user/plan selection)
    retrieved_evidence: list        # [{evidence_id, source_id, source_type, source_tier, title, snippet, url, published_date, score}]
                                    # evidence_id = '{source_id}#{n}', issued per document by retrieve_node
    synthesized_findings: list      # [{claim, supporting_source_ids: list[str], confidence: float}]
                                    # supporting_source_ids holds evidence_id values (per DOCUMENT,
                                    # not connector ids); confidence is finite and in [0, 1]
    identified_gaps: list           # sub-topics/questions lacking sufficient evidence

    # ── HITL plan gate (D6) ─────────────────────────────────────────────
    # In production this is wired to the framework D6 interrupt() mechanism.
    hitl_required: bool             # plan needs human Go before retrieval
    hitl_payload: dict              # proposed plan presented to the human

    # ── post_process output ─────────────────────────────────────────────
    evidence_report: dict           # {sections, citations, confidence_summary, gaps}
    formatted_output: str           # rendered 2-5 page markdown report returned to caller

    # ── hard-stop marker ────────────────────────────────────────────────
    error: str                      # S-2 rejection / subgraph failure message (main passes through)
