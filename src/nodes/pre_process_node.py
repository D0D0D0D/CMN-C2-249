"""AgentCore Platform v1.0 — CMN-C2-249 pre_process node.

Node contract (SDK v1.0.0 wheel):
  - Extend FunctionNode; implement execute(self, state) -> dict
    (BaseNode.__call__ passes state only — dependencies are constructor-injected)
  - Return ONLY the fields this node changes (never full state)
  - Return AgentStatus values (str enum .value)
  - Read input_context via state.get("input_context", {}) — read-only [C1]
  - Never import from mediator/, api/, or other agents
  - Do NOT override __call__ / _invoke_impl / the @final security gates

S-2 Input Gate (inline): reject empty / over-long questions, inputs containing
credential patterns, and user source selections outside the deploy catalog.
PII scanning is performed by the framework @final _security_gate_input. Hard
early-termination uses state["error"]; the main GraphNode passes an upstream
error through without running the subgraph.
"""

from __future__ import annotations

from typing import ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event

from src.policy import DEFAULT_SOURCE_CATALOG, VALID_QUALITY_BARS, out_of_catalog

MAX_QUESTION_CHARS = 4000


class PreProcessNode(FunctionNode):
    """Validate the research question and build the (candidate) research plan."""

    # S-1: explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, catalog: dict | None = None) -> None:
        # Deploy-time source catalog (DI seam): ctor arg > default catalog.
        self._catalog = catalog or DEFAULT_SOURCE_CATALOG

    def execute(self, state: dict) -> dict:
        user_input = state.get("user_input", "") or ""
        input_context = state.get("input_context", {}) or {}  # read-only [C1]
        question = user_input.strip()
        # S-4: at least one domain event per node — the framework audit contract
        emit_trace_event("pre_process_validate", {"input_chars": len(question)}, state)

        # ── S-2: empty / over-long ──────────────────────────────────────
        if not question:
            return self._reject("S-2: research question is empty or missing")
        if len(question) > MAX_QUESTION_CHARS:
            return self._reject(
                f"S-2: research question exceeds {MAX_QUESTION_CHARS} chars"
            )

        # ── S-2: credential leakage in the question ─────────────────────
        if detect_credentials(question):
            return self._reject("S-2: credential-like pattern detected in input")

        # ── user-provided source spec (optional) ────────────────────────
        spec = input_context.get("source_spec", {}) or {}
        requested_sources = list(spec.get("sources", []) or [])
        quality_bar = spec.get("quality_bar", "news_acceptable")
        depth = spec.get("depth", "standard")

        if quality_bar not in VALID_QUALITY_BARS:
            return self._reject(f"S-2: invalid quality_bar '{quality_bar}'")

        # ── S-2: user-selected sources must be inside the deploy catalog ─
        outside = out_of_catalog(requested_sources, self._catalog)
        if outside:
            return self._reject(
                f"S-2: source(s) outside allowed catalog rejected: {outside}"
            )

        # ── build candidate plan; confirm if the user fully specified it ─
        research_plan = {
            "proposed_sources": requested_sources or list(self._catalog.keys()),
            "quality_bar": quality_bar,
            "depth": depth,
        }
        # A plan is pre-confirmed only when the user explicitly pinned sources
        # AND quality (so the HITL plan gate can be safely skipped), or when an
        # upstream caller marks plan_confirmed in the context.
        plan_confirmed = bool(input_context.get("plan_confirmed")) or (
            bool(requested_sources) and "quality_bar" in spec
        )

        return {
            "research_question": question,
            "user_source_spec": spec,
            "research_plan": research_plan,
            "quality_bar": quality_bar,
            "plan_confirmed": plan_confirmed,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _reject(message: str) -> dict:
        # state["error"] marks a hard S-2 early-termination; the main GraphNode
        # passes it through and route() sends the run to finalize.
        return {
            "error": message,
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {message}"],
        }
