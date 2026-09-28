"""AgentCore Platform v1.0 — CMN-C2-249 inner domain workflow graph.

The Cat 2 domain complexity lives here, wrapped by the `main` GraphNode of the
outer 5-node backbone (src/graph/graph.py). Topology:

    START → ingest → plan ─┬─[pause: HITL plan gate]────────────→ END
                           ├─[stop: upstream error]─────────────→ END
                           └─[continue]→ retrieve ─┬─[stop/retry]→ END
                                                   └─[continue]─→ synthesize → END

- The GraphNode contract passes only a string user_input into invoke();
  research_question / research_plan / plan_confirmed / quality_bar are packed
  as JSON by the outer extract_input() and unpacked here in _IngestNode.
- plan may pause for human Go (state-flag HITL gate). When it does, the
  workflow short-circuits before any retrieval (status PENDING surfaced to the
  backbone, which finalizes and returns the plan payload to the caller).
- retrieve is S-1 gated (VERIFIED_EXTERNAL) and may return status RETRY
  (no evidence) which the outer backbone handles via route() (max_retry 3).
- get_output() projects the fields the outer merge_output() propagates.
"""

from __future__ import annotations

import json
from typing import ClassVar

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.nodes.plan_node import PlanNode
from src.nodes.retrieve_node import RetrieveNode
from src.nodes.synthesize_node import SynthesizeNode
from src.policy import resolve_catalog
from src.schemas.state import State
from src.services.corpus_research_service import build_research_service
from src.services.llm_client import coerce_llm_client

# Fields the inner workflow surfaces to the outer graph via get_output().
_OUTPUT_FIELDS = (
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
    "error",
)


class _IngestNode(FunctionNode):
    """Unpack the JSON-packed research context from the string user_input.

    The GraphNode contract passes only user_input (str) into the inner graph,
    so the outer extract_input() JSON-encodes the research context and this
    node restores it into the inner state.
    """

    # S-1: explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict) -> dict:
        raw = state.get("user_input", "") or ""
        payload: dict = {}
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                payload = decoded
        except (ValueError, TypeError):
            payload = {}
        # S-4: at least one domain event per node — the framework audit contract
        emit_trace_event(
            "ingest_unpack",
            {"has_research_question": bool(payload.get("research_question"))},
            state,
        )
        return {
            "research_question": payload.get("research_question", ""),
            "research_plan": payload.get("research_plan", {}) or {},
            "plan_confirmed": bool(payload.get("plan_confirmed")),
            "quality_bar": payload.get("quality_bar", "news_acceptable"),
            "status": AgentStatus.SUCCESS.value,
        }


def _cfg_get(cfg: dict, key: str):
    """Config key lookup supporting flat and agent.yaml `configurable:` shapes."""
    if key in cfg:
        return cfg[key]
    return (cfg.get("configurable") or {}).get(key)


# Runtime parameters for the lazily-built provider client (config/config.yaml).
# `model` is intentionally absent by default — no model id is pinned in src/.
_LLM_OPTION_KEYS = ("llm_provider", "llm_model", "llm_temperature", "llm_max_tokens", "llm_timeout")


def _llm_options(cfg: dict) -> dict:
    """Collect `llm_*` runtime parameters, stripped of their prefix."""
    options: dict = {}
    for key in _LLM_OPTION_KEYS:
        value = _cfg_get(cfg, key)
        if value is not None:
            options[key[len("llm_"):]] = value
    return options


class ResearchWorkflowGraph(BaseGraph):
    """Inner BaseGraph: ingest -> plan -> [HITL gate] -> retrieve -> synthesize."""

    def __init__(self, config: dict | None = None) -> None:
        # DI seam: config keys ONLY. BaseGraph.__init__(self, config=None) is the
        # single constructor signature the framework defines, so a bespoke
        # `llm=` / `service=` keyword is unreachable from every caller that
        # follows that contract — including src/api/server.py — and would leave
        # the LLM branch permanently unwired.
        #
        # config["llm"] is the standard seam; coerce_llm_client() accepts
        # both a framework BaseLLM and this template's own synthesize() protocol.
        cfg = config or {}
        # Explicit connector injection wins; the configured sample corpus is
        # the deployable fallback, while no configuration keeps the loud seam.
        self._service = _cfg_get(cfg, "research_service") or build_research_service(cfg)
        self._llm = coerce_llm_client(_cfg_get(cfg, "llm"))
        self._llm_options = _llm_options(cfg)
        resolved_hitl = _cfg_get(cfg, "hitl_enabled")
        # default ON: never silently retrieve under an unreviewed plan
        self._hitl_enabled = True if resolved_hitl is None else bool(resolved_hitl)
        self._catalog = resolve_catalog(cfg)
        super().__init__(config=config)

    @property
    def name(self) -> str:
        return "ResearchWorkflowGraph"

    @property
    def state_schema(self) -> type:
        return State

    def _validate_config(self) -> None:
        pass

    def register_nodes(self) -> None:
        # No super() — BaseGraph.register_nodes() is abstract. Do NOT register
        # initialize/finalize (outer backbone concern).
        self._nodes["ingest"] = _IngestNode()
        self._nodes["plan"] = PlanNode(hitl_enabled=self._hitl_enabled)
        self._nodes["retrieve"] = RetrieveNode(service=self._service, catalog=self._catalog)
        self._nodes["synthesize"] = SynthesizeNode(llm=self._llm, llm_options=self._llm_options)

    def add_edges(self) -> None:
        self._sg.add_edge(START, "ingest")
        self._sg.add_edge("ingest", "plan")
        self._sg.add_conditional_edges(
            "plan",
            self.route,
            {"pause": END, "stop": END, "continue": "retrieve"},
        )
        self._sg.add_conditional_edges(
            "retrieve",
            self.route,
            {"pause": END, "stop": END, "continue": "synthesize"},
        )
        self._sg.add_edge("synthesize", END)

    def route(self, state: State) -> str:
        # NOTE: annotate with State (not AgentState) so LangGraph projects the
        # FULL State schema into the branch — hitl_required is a State field,
        # not an AgentState field; annotating AgentState hides it and mis-routes.
        if state.get("hitl_required"):
            return "pause"  # HITL plan gate — do not retrieve under an unreviewed plan
        status = state.get("status")
        if state.get("error") or status in (AgentStatus.ERROR.value, AgentStatus.RETRY.value):
            return "stop"  # S-1 denial / hard error / retryable no-evidence
        return "continue"

    def get_output(self, state: State) -> dict:
        out = {k: state[k] for k in _OUTPUT_FIELDS if k in state}
        if state.get("error_log"):
            out["error_log"] = list(state["error_log"])
        return out
