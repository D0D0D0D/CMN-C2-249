from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import cli
from framework.schemas.invocation_context import InvocationContext
from framework.secrets.base import SecretProvider
from framework.utils.config_loader import load_config
from src.graph.graph import Graph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"
_AZURE_VALUES = {
    "AZURE_OPENAI_API_KEY": "mock-azure-api-key",
    "AZURE_OPENAI_ENDPOINT": "mock-azure-endpoint",
    "AZURE_OPENAI_DEPLOYMENT": "mock-azure-deployment",
}


class _UnusedProvider(SecretProvider):
    def get(self, key: str, default: str | None = None) -> str | None:
        return default


def _runtime_config() -> dict:
    return load_config(str(_CONFIG_PATH))


def _set_clean_azure_environment(monkeypatch) -> None:
    for key, value in _AZURE_VALUES.items():
        monkeypatch.setenv(key, value)


def test_marketplace_graph_injects_azure_llm_into_real_runtime_config():
    graph = cli.MarketplaceGraph(config=_runtime_config())

    assert isinstance(graph.config["llm"], cli.AzureLLM)
    assert vars(cli.AzureLLM()) == {}


def test_marketplace_graph_preserves_non_none_caller_llm():
    supplied = object()
    config = _runtime_config()
    config["llm"] = supplied

    graph = cli.MarketplaceGraph(config=config)

    assert graph.config["llm"] is supplied


def test_marketplace_graph_replaces_explicit_none_llm():
    config = _runtime_config()
    config["llm"] = None

    graph = cli.MarketplaceGraph(config=config)

    assert isinstance(graph.config["llm"], cli.AzureLLM)


def test_bind_tools_returns_a_new_azure_llm():
    llm = cli.AzureLLM()

    bound = llm.bind_tools([{"name": "mock-tool"}])

    assert isinstance(bound, cli.AzureLLM)
    assert bound is not llm
    assert vars(llm) == {}


def test_invoke_rebuilds_context_and_disables_hitl(monkeypatch):
    recorded: dict = {}

    def recorder(self, user_input, session_id="", ctx=None, input_context=None):
        recorded.update(
            user_input=user_input,
            session_id=session_id,
            ctx=ctx,
            input_context=input_context,
        )
        return {"status": "mock-success"}

    monkeypatch.setattr(Graph, "invoke", recorder)
    graph = cli.MarketplaceGraph(config=_runtime_config())
    incoming = {"conversation_history": [{"role": "user", "content": "CANARY"}]}

    result = graph.invoke(
        "research question",
        session_id="mock-session",
        ctx=InvocationContext(hitl_allowed=True),
        input_context=incoming,
    )

    assert result == {"status": "mock-success"}
    assert recorded["input_context"] == {}
    assert "conversation_history" not in recorded["input_context"]
    assert "plan_confirmed" not in recorded["input_context"]
    assert "source_spec" not in recorded["input_context"]
    assert recorded["ctx"].hitl_allowed is False


@pytest.mark.parametrize("missing_key", list(_AZURE_VALUES))
def test_provision_secrets_rejects_each_missing_azure_value(monkeypatch, missing_key):
    _set_clean_azure_environment(monkeypatch)
    monkeypatch.delenv(missing_key)
    graph = cli.MarketplaceGraph(config=_runtime_config())

    with pytest.raises(Exception):
        graph.provision_secrets(_UnusedProvider())


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
        "https://example.services.ai.azure.example/openai",
        "https://example.services.ai.azure.example/v1",
        "https://example.services.ai.azure.example?api-version=2024-10-21",
        "example.services.ai.azure.example",
    ],
    ids=["api-path", "openai-path", "base-path", "query-only", "missing-scheme"],
)
def test_unusable_azure_endpoint_shape_fails_during_startup(monkeypatch, endpoint):
    """AzureOpenAIClient appends its own path, so anything but a bare endpoint breaks.

    The resulting error surfaces inside complete(); a node that swallows it
    degrades silently while the invocation still reports status: success, so the
    adapter has to reject the shape at startup instead.
    """

    _set_clean_azure_environment(monkeypatch)
    graph = cli.MarketplaceGraph(config=_runtime_config())
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)

    with pytest.raises(ValueError, match="bare"):
        graph.provision_secrets(_UnusedProvider())


@pytest.mark.parametrize("invalid", ["", " mock-value", "mock-value ", "mock\nvalue"])
@pytest.mark.parametrize("key", list(_AZURE_VALUES))
def test_provision_secrets_rejects_unclean_azure_values(monkeypatch, key, invalid):
    _set_clean_azure_environment(monkeypatch)
    monkeypatch.setenv(key, invalid)
    graph = cli.MarketplaceGraph(config=_runtime_config())

    with pytest.raises(Exception):
        graph.provision_secrets(_UnusedProvider())


def test_provision_secrets_rejects_missing_corpus(monkeypatch):
    _set_clean_azure_environment(monkeypatch)
    config = _runtime_config()
    config["research_corpus_path"] = "config/missing-corpus.sample.json"
    graph = cli.MarketplaceGraph(config=config)

    with pytest.raises(ValueError, match="missing-corpus.sample.json"):
        graph.provision_secrets(_UnusedProvider())


def test_allowlisted_provider_does_not_expose_legacy_llm_key(monkeypatch):
    _set_clean_azure_environment(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "mock-legacy-key")

    provider = cli._AllowlistedEnvironmentProvider()

    assert provider.get("LLM_API_KEY") is None


def test_main_calls_marketplace_runner_with_graph_and_nonempty_config(monkeypatch):
    recorded: dict = {}
    bootstrap = types.ModuleType("shared.bootstrap")

    def recorder(graph_class, **kwargs):
        recorded["graph_class"] = graph_class
        recorded.update(kwargs)

    bootstrap.run_agent_marketplace = recorder
    monkeypatch.setitem(sys.modules, "shared.bootstrap", bootstrap)

    cli.main()

    assert recorded["graph_class"] is cli.MarketplaceGraph
    assert recorded["agent_name"] == "CMN-C2-249"
    assert recorded["namespace"] == "cmn"
    assert recorded["config"]
    assert recorded["config"]["research_corpus_path"]
