from __future__ import annotations

from pathlib import Path

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.utils.config_loader import load_config

from src.graph.graph import Graph
from tests.fakes import FakeLLMClient

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_real_corpus_flows_through_retrieval_synthesis_and_report():
    config = load_config(str(_REPO_ROOT / "config" / "config.yaml"))
    llm = FakeLLMClient()
    config.update({"llm": llm, "hitl_enabled": False})
    graph = Graph(config=config)
    graph.compile()

    result = graph.invoke(
        "ハイブリッドワークは生産性にどう影響するか",
        ctx=InvocationContext(
            caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
            session_id="mock-corpus-session",
        ),
        input_context={},
    )

    assert result["status"] == AgentStatus.SUCCESS.value
    assert result["output"]
    assert llm.calls == 1
    assert len({item["source_id"] for item in result["retrieved_evidence"]}) >= 2
    assert any("[SAMPLE]" in item["title"] for item in result["evidence_report"]["citations"])
    assert any("sample://" in item["url"] for item in result["evidence_report"]["citations"])
    assert "[SAMPLE]" in result["formatted_output"]
    assert "sample://" in result["formatted_output"]
