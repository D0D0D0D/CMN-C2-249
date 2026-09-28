"""AGENTIC STAR Marketplace container entry adapter for CMN-C2-249."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from framework.schemas.invocation_context import InvocationContext
from framework.secrets.base import SecretProvider
from framework.secrets.context import current_secrets

try:
    from framework.utils.config_loader import load_agent_config
except ImportError:  # AgentCore 1.0.0 / 1.0.1 compatibility
    from framework.utils.config_loader import load_config

    def load_agent_config(agent_dir: Path) -> dict[str, Any]:
        config_path = agent_dir / "config" / "config.yaml"
        return load_config(str(config_path)) if config_path.exists() else {}


try:
    from shared.secrets.env_provider import EnvProvider
except ImportError:  # AgentCore 1.0.0 / 1.0.1 compatibility

    class EnvProvider(SecretProvider):
        """AgentCore 1.0.2-compatible process environment provider."""

        def __init__(self, *, namespace: str = "", agent_name: str = "") -> None:
            self._namespace = namespace
            self._agent_name = agent_name

        def get(self, key: str, default: str | None = None) -> str | None:
            return os.environ.get(key, default)


from shared.services.llm.base_llm import BaseLLM

from src.graph.graph import Graph
from src.services.corpus_research_service import build_research_service

_NAMESPACE = "cmn"
_AGENT_NAME = "CMN-C2-249"
_PROVIDER_AGENT_NAME = "cmn_c2_249"

_AZURE_OPENAI_SECRET_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)
_ALLOWED_SECRET_KEYS = frozenset(_AZURE_OPENAI_SECRET_KEYS)


def _cfg_get(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    configurable = config.get("configurable") or {}
    return configurable.get(key) if isinstance(configurable, dict) else None


def _azure_options(config: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for config_key, client_key in (
        ("llm_temperature", "temperature"),
        ("llm_max_tokens", "max_tokens"),
    ):
        value = _cfg_get(config, config_key)
        if value is not None:
            options[client_key] = value
    return options


class _AllowlistedEnvironmentProvider(EnvProvider):  # type: ignore[misc]
    """Expose only this agent's three Azure connection values."""

    def __init__(self) -> None:
        super().__init__(namespace=_NAMESPACE, agent_name=_PROVIDER_AGENT_NAME)

    def get(self, key: str, default: str | None = None) -> str | None:
        if key not in _ALLOWED_SECRET_KEYS:
            return default
        value = super().get(key)
        return value if value else default


def _require_clean_secret(provider: SecretProvider, key: str) -> str:
    value = provider.require(key)
    if value != value.strip() or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"Marketplace secret {key!r} must be a clean non-empty single-line value")
    return value


def _require_azure_endpoint(provider: SecretProvider) -> str:
    """Reject ``AZURE_OPENAI_ENDPOINT`` values that are not a bare Azure resource endpoint.

    ``AzureOpenAIClient`` appends its own ``/openai/v1/`` path, so anything
    beyond a scheme and host silently produces a broken request URL, and the
    failure surfaces inside ``complete()``. A node that swallows that exception
    degrades quietly while the invocation still reports ``status: success``,
    which is why this has to be a startup check. The value is reported here,
    never repaired.
    """

    endpoint = _require_clean_secret(provider, "AZURE_OPENAI_ENDPOINT")
    parts = urlsplit(endpoint)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or port == -1
        or parts.path.strip("/")
        or parts.query
        or parts.fragment
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError(
            "Marketplace environment variable 'AZURE_OPENAI_ENDPOINT' must be a bare Azure "
            "resource endpoint -- scheme and host only, with no path (such as "
            "'/openai'), query or fragment"
        )
    return endpoint


class AzureLLM(BaseLLM):  # type: ignore[misc]
    """Build a fresh Azure client from invocation-bound secrets for every call."""

    def __init__(
        self,
        *,
        tools: tuple[Any, ...] = (),
        client_options: dict[str, Any] | None = None,
    ) -> None:
        if tools:
            self._tools = tools
        if client_options:
            self._client_options = dict(client_options)

    def _client(self) -> BaseLLM:
        secrets = current_secrets()
        client_config = {
            "api_key": _require_clean_secret(secrets, "AZURE_OPENAI_API_KEY"),
            "azure_endpoint": _require_clean_secret(secrets, "AZURE_OPENAI_ENDPOINT"),
            "azure_deployment": _require_clean_secret(secrets, "AZURE_OPENAI_DEPLOYMENT"),
            **getattr(self, "_client_options", {}),
        }
        # This provider integration is not present in AgentCore 1.0.0, so the
        # import must remain lazy and after connection-value validation.
        from shared.services.llm.azure_openai_client import AzureOpenAIClient

        client = AzureOpenAIClient(client_config)
        tools = getattr(self, "_tools", ())
        return client.bind_tools(list(tools)) if tools else client

    def complete(self, messages: list[Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self._client().complete(messages))

    def stream(self, messages: list[Any]) -> Iterator[str]:
        yield from self._client().stream(messages)

    def bind_tools(self, tools: list[Any]) -> "AzureLLM":
        return type(self)(
            tools=tuple(tools),
            client_options=getattr(self, "_client_options", None),
        )


class MarketplaceGraph(Graph):
    """Marketplace context hardening and Azure dependency wiring."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        merged = dict(config or {})
        if merged.get("llm") is None:
            merged["llm"] = AzureLLM(client_options=_azure_options(merged))
        super().__init__(config=merged, **kwargs)

    def invoke(
        self,
        user_input: str,
        session_id: str = "",
        ctx: InvocationContext | None = None,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Runner conversation history is free text outside this agent's S-2
        # scan. Rebuild the context and do not claim that a human approved it.
        del input_context
        marketplace_ctx = replace(ctx, hitl_allowed=False) if ctx is not None else ctx
        return cast(
            dict[str, Any],
            super().invoke(
                user_input,
                session_id=session_id,
                ctx=marketplace_ctx,
                input_context={},
            ),
        )

    def provision_secrets(self, provider: SecretProvider) -> None:
        # Ignore the runner's wider provider and expose only consumed values.
        del provider
        environment_provider = _AllowlistedEnvironmentProvider()
        for key in _AZURE_OPENAI_SECRET_KEYS:
            _require_clean_secret(environment_provider, key)
        # Validate a configured corpus at startup, before the first invocation.
        build_research_service(self.config)
        _require_azure_endpoint(environment_provider)
        super().provision_secrets(environment_provider)


def main() -> None:
    # Marketplace support is absent from AgentCore 1.0.0 and belongs only on
    # the executable path so importing this module remains compatible.
    from shared.bootstrap import run_agent_marketplace

    run_agent_marketplace(
        MarketplaceGraph,
        agent_name=_AGENT_NAME,
        namespace=_NAMESPACE,
        config=load_agent_config(Path(__file__).resolve().parent),
    )


if __name__ == "__main__":
    main()
