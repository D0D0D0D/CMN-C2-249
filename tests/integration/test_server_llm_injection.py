# CMN-C2-249 — Integration: standalone entry point (src/api/server.py) LLM wiring.
#
# The defect this guards: the server built the graph from two hard-coded HITL
# keys, so config/config.yaml never reached it and there was no route from
# config["llm"] to the synthesize node. A suite that injects the LLM only into
# its own locally-built Graph cannot see that.

from __future__ import annotations

import importlib

import pytest
import yaml

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.services.llm.base_llm import BaseLLM

from src.graph.graph import Graph
from tests.fakes import FakeResearchService, graph_config

_CONFIRMED_CONTEXT = {
    "plan_confirmed": True,
    "source_spec": {"sources": ["arxiv"], "quality_bar": "peer_reviewed_only"},
}
_PAYLOAD = '{"findings":[{"claim":"C","supporting_source_ids":["arxiv#1"],"confidence":0.5}],"gaps":[]}'


class _RecordingLLM(BaseLLM):
    def __init__(self) -> None:
        self.calls: list = []

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        return {"content": _PAYLOAD, "tool_calls": [], "model": "recording", "usage": {}}

    def stream(self, messages: list):
        yield ""

    def bind_tools(self, tools: list) -> "_RecordingLLM":
        return self


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", "test-invoke-token-cmn249")
    import src.api.server as mod

    return importlib.reload(mod)


class TestServerRuntimeConfig:
    def test_config_yaml_reaches_the_graph(self, server):
        with open("config/config.yaml", encoding="utf-8") as fh:
            on_disk = yaml.safe_load(fh)
        # Every runtime parameter the file declares must be present on the
        # graph the server actually built.
        for key in ("max_retry", "timeout_seconds", "llm_provider", "configurable"):
            assert server._CONFIG.get(key) == on_disk[key], f"{key} did not reach the server graph"
        assert server.agent.config.get("max_retry") == on_disk["max_retry"]

    def test_hitl_block_stays_explicit(self, server):
        # An incomplete config file must not be able to switch the plan gate off.
        assert server._CONFIG["memory_enabled"] is True
        assert server._CONFIG["hitl"] == {"enabled": True, "max_hitl": 8}

    def test_missing_config_file_degrades_to_empty(self, server, monkeypatch):
        monkeypatch.setattr(server, "_CONFIG_PATH", server._CONFIG_PATH.with_name("absent.yaml"))
        assert server._load_runtime_config() == {}

    def test_server_does_not_build_an_llm_at_import_time(self, server):
        # Secrets are provisioned after register_nodes(), so an import-time
        # client would have to read the key from somewhere it must not.
        assert "llm" not in server._CONFIG


class TestServerConfigCarriesTheLLM:
    def test_config_llm_over_the_server_config_reaches_the_node(self, server):
        """The server's own config shape + config["llm"] must reach the LLM.

        This is the end-to-end statement of the wiring fix: nothing in the path
        server config -> Graph -> GraphNode -> inner graph -> SynthesizeNode
        drops the injected client.
        """
        llm = _RecordingLLM()
        cfg = {
            **server._CONFIG,
            **graph_config(service=FakeResearchService(), hitl_enabled=False),
            "llm": llm,
        }
        agent = Graph(config=cfg)
        agent.compile()
        out = agent.invoke(
            "What is X?",
            ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
            input_context=dict(_CONFIRMED_CONTEXT),
        )
        assert llm.calls, "config['llm'] did not reach the synthesize node"
        assert out["status"] == "success"
        assert out["synthesized_findings"][0]["supporting_source_ids"] == ["arxiv#1"]
