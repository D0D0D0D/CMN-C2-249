"""AgentCore Platform v1.0 — CMN-C2-249 plan_node (inner workflow).

Finalizes the research_plan (candidate sources + citation quality tier + depth).
When the plan is not pre-confirmed and HITL is enabled, it raises the plan
gate: the proposed plan is surfaced to the human and the workflow pauses for a
Go before any retrieval happens.

HITL modeling: expressed via explicit state flags (hitl_required / hitl_payload
+ status PENDING) so the gate is deterministic and unit-testable. The inner
graph short-circuits on hitl_required and the run resumes on the next invoke
with plan_confirmed=True (and any user edits applied to research_plan).
Migrating this to the framework D6 interrupt() mechanism is a future design
decision (review guide 🔴 item).

S-1: no external access — ANONYMOUS is sufficient. Returns only changed fields.
The hitl_enabled switch is constructor-injected (SDK v1.0 wheel passes state
only into execute()).
"""

from __future__ import annotations

from typing import ClassVar

from langgraph.types import interrupt

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event


class PlanNode(FunctionNode):
    """Confirm the research plan, gating on human approval when needed."""

    # S-1: explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, hitl_enabled: bool = True) -> None:
        # default ON: never silently retrieve under an unreviewed plan
        self._hitl_enabled = bool(hitl_enabled)

    def execute(self, state: dict) -> dict:
        plan = dict(state.get("research_plan", {}) or {})
        plan_confirmed = bool(state.get("plan_confirmed"))
        # S-4: at least one domain event per node — the framework audit contract
        emit_trace_event(
            "plan_gate",
            {"plan_confirmed": plan_confirmed, "hitl_enabled": self._hitl_enabled},
            state,
        )

        # Already confirmed upstream (or a caller pre-confirmed the plan) → skip
        # the gate and proceed with the finalized plan.
        if plan_confirmed or not self._hitl_enabled:
            return self._finalize(plan)

        # D6 HITL plan gate: present the plan and pause via the
        # framework interrupt() mechanism (replaces the prior state-flag pseudo-
        # HITL that ended the run and required a fresh invoke to resume).
        # MANDATORY hitl_allowed guard: inside an automated pipeline
        # (hitl_allowed=False) skip the interrupt and confirm the plan as-is so
        # the composition does not deadlock.
        if not state.get("hitl_allowed", True):
            emit_trace_event("plan_gate_skipped_not_allowed",
                             {"reason": "hitl_allowed=false"}, state)
            return self._finalize(plan)

        feedback = interrupt({
            "type": "research_plan_approval",
            "research_plan": plan,
            "message": (
                "Review the proposed research plan (sources / citation quality / "
                "depth). Reply Go to start retrieval, or edit the plan before "
                "approving."
            ),
        })
        # On resume, feedback carries the human decision. A dict with an edited
        # "research_plan" overrides the proposal; anything else is treated as an
        # approval of the plan as presented.
        if isinstance(feedback, dict) and isinstance(feedback.get("research_plan"), dict):
            plan = dict(feedback["research_plan"])
        emit_trace_event("plan_confirmed_via_hitl", {"edited": isinstance(feedback, dict)}, state)
        return self._finalize(plan, state)

    @staticmethod
    def _finalize(plan: dict, state: dict | None = None) -> dict:
        # Clear any prior payload so state only carries a payload while a HITL
        # pause is actually in effect (F-2).
        delta = {
            "research_plan": plan,
            "plan_confirmed": True,
            "hitl_required": False,
            "hitl_payload": {},
            "status": AgentStatus.SUCCESS.value,
        }
        # Sync a human-edited quality_bar back into the top-level state field that
        # retrieve/synthesize actually read: they read
        # state["quality_bar"], not research_plan["quality_bar"], so an edited
        # plan's stricter/looser bar would otherwise be silently ignored (and a
        # too-strict bar would find no evidence → outer retry re-enters the gate).
        # Validate against the policy allowlist so a bad edit cannot inject an
        # unknown bar into retrieval; an invalid value keeps the prior bar.
        from src.policy import VALID_QUALITY_BARS

        edited_bar = plan.get("quality_bar")
        if edited_bar in VALID_QUALITY_BARS:
            delta["quality_bar"] = edited_bar
        return delta
