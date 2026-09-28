"""AgentCore Platform v1.0 — CMN-C2-249 outer graph.

Cat 2: outer AgentBaseGraph (fixed 5-node backbone) + a GraphNode in the `main`
slot wrapping the inner ResearchWorkflowGraph (src/graph/domain_workflow_graph.py).

  START → initialize → pre_process → main → {route} → post_process → finalize → END

Do NOT override add_edges() — backbone wiring belongs to the framework.
"""

from __future__ import annotations

import json
import threading

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from shared.utils.audit_logger import emit_trace_event

from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.policy import resolve_catalog
from src.schemas.state import State

# Keys produced by the inner workflow that propagate to the outer state.
_PROPAGATE = (
    "research_plan",
    "plan_confirmed",
    "quality_bar",
    "source_whitelist",
    "retrieved_evidence",
    "synthesized_findings",
    "identified_gaps",
    "hitl_required",
    "hitl_payload",
    "status",
)

# Domain fields surfaced to the caller in addition to the default projection.
_OUTPUT_FIELDS = (
    "research_question",
    "research_plan",
    "plan_confirmed",
    "quality_bar",
    "source_whitelist",
    "retrieved_evidence",
    "synthesized_findings",
    "identified_gaps",
    "hitl_required",
    "hitl_payload",
    "evidence_report",
    "formatted_output",
    "error",
)


class ResearchWorkflowNode(GraphNode):
    """`main` slot — wraps the inner ResearchWorkflowGraph."""

    # Inner ERROR outcomes are handled (degrade to finalize with the inner
    # error_log surfaced) rather than re-raised — S-1 denials and hard errors
    # must reach the caller as a structured error, not a SubgraphError trace.
    error_strategy = "handle"
    # D6 HITL: the inner plan node calls interrupt() at the plan gate; surface
    # that pause to the outer backbone so /invoke returns awaiting_human and
    # /resume can continue it (was propagate_hitl=False under the old state-flag
    # pseudo-HITL that ended the run instead of suspending).
    propagate_hitl = True

    def __init__(self, config: dict | None = None) -> None:
        # config-only: the inner graph resolves its service / LLM seams from
        # config keys (a bespoke `llm=` keyword is unreachable
        # from a caller following BaseGraph's real constructor contract).
        self._config = config or {}
        self._subgraph = None
        self._subgraph_lock = threading.Lock()

    def get_subgraph(self):
        """Return the cached, compiled inner graph.

        CRITICAL (HITL resume): GraphNode.execute() calls get_subgraph() on BOTH
        the suspend invoke AND the resume, so it MUST return the SAME compiled
        instance+saver — a fresh graph+saver per call loses the interrupt
        checkpoint. Built once under a double-checked lock (race-safe on
        concurrent cold-start), with its OWN checkpointer distinct from the outer
        graph's saver (inner/outer must not share one, or thread namespaces
        collide). Ref: the Cat 2 HITL contract.
        """
        if self._subgraph is None:
            with self._subgraph_lock:
                if self._subgraph is None:
                    from langgraph.checkpoint.memory import InMemorySaver

                    from src.graph.domain_workflow_graph import ResearchWorkflowGraph

                    sub = ResearchWorkflowGraph(config=self._config)
                    sub.compile(checkpointer=InMemorySaver())
                    self._subgraph = sub
        return self._subgraph

    def execute(self, state: dict) -> dict:
        # pre_process -> main is an unconditional backbone edge: pass an
        # upstream S-2 rejection / hard error through without running the
        # subgraph (route() then finalizes).
        if state.get("error") or state.get("status") == AgentStatus.ERROR.value:
            emit_trace_event(
                "research_workflow_skipped", {"reason": "upstream_error"}, state
            )
            return {}
        # S-4: at least one domain event per node — the framework audit contract
        emit_trace_event(
            "research_workflow_start",
            {"question_chars": len(state.get("research_question", "") or "")},
            state,
        )
        return super().execute(state)

    def extract_input(self, state: AgentState) -> str:
        """Pack the research context as JSON into the string user_input.

        The GraphNode contract passes only a string into inner invoke(); the
        inner _IngestNode unpacks it.
        """
        payload = {
            "research_question": state.get("research_question", ""),
            "research_plan": state.get("research_plan", {}),
            "plan_confirmed": bool(state.get("plan_confirmed")),
            "quality_bar": state.get("quality_bar", "news_acceptable"),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def merge_output(self, state: AgentState, sub_result: dict) -> dict:
        """Surface only the keys present in sub_result (path-dependent).

        HITL pause  -> {hitl_required, hitl_payload, status pending, ...}
        retryable   -> {status retry, retry_count+1, ...}
        success     -> {retrieved_evidence, synthesized_findings, ...}
        """
        delta = {k: sub_result[k] for k in _PROPAGATE if k in sub_result}
        if sub_result.get("error_log"):
            # inner error_log entries are all new (fresh inner state) — the
            # outer reducer appends them (no double-count).
            delta["error_log"] = list(sub_result["error_log"])
        if sub_result.get("status") == AgentStatus.RETRY.value:
            # route() consumes retry_count; the framework does not increment it.
            delta["retry_count"] = int(state.get("retry_count", 0) or 0) + 1
        return delta

    def on_subgraph_error(self, state: AgentState, error: Exception) -> dict:
        # error_strategy="handle": surface the inner error_log (e.g. the S-1
        # trust-gate denial from RetrieveNode) as a structured outer error.
        logs = list(getattr(error, "error_log", []) or [str(error)])
        return {
            "status": AgentStatus.ERROR.value,
            "error": logs[0] if logs else str(error),
            "error_log": [f"ResearchWorkflowNode: subgraph error — {m}" for m in logs],
        }


class Graph(AgentBaseGraph):
    """CMN-C2-249 outer graph — Deep Research Synthesis & Evidence Report Agent."""

    def __init__(self, config: dict | None = None, **kwargs) -> None:
        # HITL-by-default: the plan gate calls interrupt(), which
        # requires the framework to thread a thread_id into the run config —
        # which only happens when memory_enabled/hitl is set. Enable it by
        # default so a bare Graph() (direct / AgentGateway construction) suspends
        # correctly instead of turning the interrupt into status=error.
        cfg = dict(config or {})
        # Resolve the inner plan-gate flag (`hitl_enabled`) from any EXPLICIT
        # caller opt-out FIRST, before injecting defaults — otherwise the default
        # hitl block below would win over a manifest/standard opt-out.
        # Precedence: explicit `hitl_enabled` > standard
        # `hitl.enabled` > manifest `configurable.hitl_enabled`. Only these
        # caller-supplied values count here; the default hitl block added
        # afterwards must NOT feed back into this resolution.
        if "hitl_enabled" not in cfg:
            standard = cfg.get("hitl")
            configurable = (cfg.get("configurable") or {}).get("hitl_enabled")
            if isinstance(standard, dict) and "enabled" in standard:
                cfg["hitl_enabled"] = bool(standard["enabled"])
            elif configurable is not None:
                cfg["hitl_enabled"] = bool(configurable)
        # HITL-by-default: the plan gate calls interrupt(), which needs
        # the framework to thread a thread_id (only when memory_enabled/hitl is
        # set). Default it ON so a bare Graph() suspends correctly instead of
        # erroring. setdefault leaves any explicit caller value untouched, and
        # the opt-out above has already been captured into hitl_enabled.
        cfg.setdefault("memory_enabled", True)
        cfg.setdefault("hitl", {"enabled": True, "max_hitl": 8})
        super().__init__(config=cfg, **kwargs)

    def compile(self, checkpointer=None) -> None:
        # Attach a default in-memory checkpointer when none is supplied:
        # the plan-gate interrupt() cannot suspend/resume without one, so a
        # bare Graph().compile() must not silently produce an unresumable graph.
        # Production injects a durable saver via the same argument.
        if checkpointer is None:
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()
        super().compile(checkpointer=checkpointer)

    @property
    def name(self) -> str:
        return "cmn_c2_249"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        super().register_nodes()  # injects initialize + finalize

        # Domain pipeline slots (required — fill all three).
        # Cat 2: main is a GraphNode wrapping the inner research workflow
        # (plan -> HITL gate -> retrieve -> synthesize). route()/add_edges()
        # are owned by AgentBaseGraph and MUST NOT be overridden here.
        cfg = self.config or {}
        self._nodes["pre_process"] = PreProcessNode(catalog=resolve_catalog(cfg))
        self._nodes["main"] = ResearchWorkflowNode(config=cfg)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.

    def get_output(self, state: State) -> dict:
        """Extend the default projection with the domain fields.

        Surfaces the evidence report, findings, gaps and the HITL plan-gate
        payload so callers (and a traceability review) can consume them,
        in addition to the default {output, status, trace_id, correlation_id,
        node_history}.
        """
        out = super().get_output(state)
        for k in _OUTPUT_FIELDS:
            if k in state:
                out[k] = state[k]
        out["error_log"] = list(state.get("error_log", []) or [])
        return out
