"""AgentCore Platform v1.0 — LLM injection seam (S-5).

`LLMClient` is the boundary between synthesize_node and the underlying model
provider. API keys are resolved via `ctx.secrets.require(...)` by the caller and
passed in here — NEVER read from os.environ and NEVER stored in agent state.

Two shapes reach this module:

* the standard `config["llm"]` seam, carrying a framework `BaseLLM`
  (`complete(messages) -> {"content", "tool_calls", "model", "usage"}`), and
* this template's own `synthesize(question, evidence, quality_bar)` protocol,
  which tests and composed callers may inject directly.

`coerce_llm_client()` normalises both onto the `LLMClient` protocol.
`BaseLLMSynthesisAdapter` is the production adapter for the former — it lives in
`src/` (not in tests/) so the standalone server, AgentGateway and the live-LLM
harness all traverse the identical code path.

Provenance discipline: the model authors prose (`claim`) and *reference* ids
into the retrieved evidence set. Those ids are per-DOCUMENT `evidence_id`
handles issued by retrieve_node — not connector names — so a claim binds to the
documents that actually support it. The model never authors citation metadata:
title, URL, tier and publication date are read from the retrieval records in
post_process_node. synthesize_node additionally resolves every model-supplied id
against the retrieved evidence, so an id the model invented cannot reach the
report.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from shared.services.llm.base_llm import BaseLLM

from src.services.research_service import evidence_reference_ids

# Evidence fields handed to the model. Projecting explicitly keeps the prompt
# deterministic and stops incidental retrieval internals from being echoed back.
_EVIDENCE_FIELDS = (
    "evidence_id", "source_id", "source_tier", "title", "snippet", "url", "published_date",
)

# Provider selection for the lazy default client. Values are the concrete
# BaseLLM implementations shipped by the framework's substrate catalog —
# this template does not hand-roll a provider client, and it does not
# pin a model id in src/ (the model is a config/config.yaml runtime parameter,
# or the client's own default when unset).
# API-key providers only. BedrockClient is deliberately absent: it requires
# aws_access_key_id / aws_secret_access_key / aws_region and raises ValueError
# when given only an api_key, so offering it here would be a promise this
# builder cannot keep. A Bedrock deployment injects its own client via
# config["llm"], which this template accepts unchanged.
_PROVIDERS = {
    "anthropic": ("shared.services.llm.anthropic_client", "AnthropicClient"),
    "openai": ("shared.services.llm.openai_client", "OpenAIClient"),
}

_PROVIDER_OPTION_KEYS = ("temperature", "max_tokens", "timeout")


class LLMSynthesisError(RuntimeError):
    """The injected LLM did not yield a usable synthesis payload.

    Carries only a static, caller-independent description. Provider exception
    text may quote the request back (or the credential used to sign it), so it
    is never embedded here and never reaches agent state — synthesize_node
    reports the exception *type name* alone.
    """


class LLMClient(Protocol):
    """Structural type for an injectable synthesis client."""

    def synthesize(
        self, question: str, evidence: list[dict[str, Any]], quality_bar: str
    ) -> dict[str, Any]:
        """Return {"findings": [{claim, supporting_source_ids, confidence}], "gaps": [str]}."""
        ...


class UnconfiguredLLMClient:
    """Fail-closed default. Replace via dependency injection in production."""

    def synthesize(
        self, question: str, evidence: list[dict[str, Any]], quality_bar: str
    ) -> dict[str, Any]:
        raise LLMSynthesisError(
            "No LLMClient configured. Inject one via config['llm'] "
            "(keys via ctx.secrets.require) before invoking synthesize_node."
        )


class BaseLLMSynthesisAdapter:
    """Adapt a framework `BaseLLM` onto this template's `synthesize()` protocol.

    Production adapter — deliberately in `src/`, not in `tests/`: an adapter that
    exists only in the test tree makes the suite green while the deployed agent
    has no path from `config["llm"]` to the node.
    """

    def __init__(self, base_llm: Any) -> None:
        self._base_llm = base_llm

    def synthesize(
        self, question: str, evidence: list[dict[str, Any]], quality_bar: str
    ) -> dict[str, Any]:
        # Per-DOCUMENT handles, in retrieval order — NOT connector ids. A model
        # offered only 'pubmed' cannot say which pubmed document it means.
        allowed = evidence_reference_ids(evidence)
        prompt = self._build_prompt(question, evidence, quality_bar, allowed)
        response = self._base_llm.complete([{"role": "user", "content": prompt}])
        if not isinstance(response, dict):
            raise LLMSynthesisError("LLM response was not the canonical BaseLLM dict")

        # `.get("content", "")` still yields None when the key is present and
        # null; str(None) would render the literal "None" as the answer.
        content = response.get("content") or ""
        if not isinstance(content, str):
            content = str(content)

        payload = _parse_json_object(content)
        findings = payload.get("findings")
        gaps = payload.get("gaps")
        return {
            "findings": list(findings) if isinstance(findings, list) else [],
            "gaps": [str(g) for g in gaps] if isinstance(gaps, list) else [],
            # Canonical response model — the audit event must report the model
            # the injected client actually used, not a config string that does
            # not decide the client's selection.
            "model": str(response.get("model") or ""),
        }

    @staticmethod
    def _build_prompt(
        question: str, evidence: list[dict[str, Any]], quality_bar: str, allowed: list[str]
    ) -> str:
        projected = [
            {k: e.get(k) for k in _EVIDENCE_FIELDS if e.get(k) is not None}
            for e in evidence
        ]
        return (
            "You synthesise findings across retrieved research evidence.\n"
            "Respond with JSON only — no prose, no markdown fence — in exactly this shape:\n"
            '{"findings":[{"claim":str,"supporting_source_ids":[str],"confidence":number}],'
            '"gaps":[str]}\n'
            "Rules:\n"
            "- supporting_source_ids holds evidence_id values from the allowed list below.\n"
            "- Each id identifies ONE document. Cite exactly the documents that support\n"
            "  the claim — never a whole source, and never every id you were given.\n"
            "- Never invent an id. Ids outside the allowed list are discarded.\n"
            "- confidence is a number between 0 and 1.\n"
            "- Record what the evidence does NOT establish in \"gaps\".\n"
            f"Question: {question}\n"
            f"Citation quality bar: {quality_bar}\n"
            f"Allowed evidence_ids: {json.dumps(allowed, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(projected, ensure_ascii=False, default=str)}"
        )


def _extract_fenced_block(text: str) -> str | None:
    """Return the body of a leading ```…``` block, if the response opens with one.

    Extracting the block is what makes a fenced answer parse *exactly*; the
    brace-slice fallback below is a heuristic that breaks the moment the model
    appends a friendly closing sentence containing a brace — a very common real
    provider shape that the live-check double never produces, because it strips
    its own fences before returning.
    """
    if not text.startswith("```"):
        return None
    body = text.split("\n", 1)[1] if "\n" in text else ""
    end = body.find("```")
    return (body[:end] if end != -1 else body).strip() or None


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object out of a raw model response.

    Tolerates the ```json fence real providers emit and a JSON object wrapped in
    stray prose, in that order of precision. Anything else fails loudly rather
    than degrading to an empty report.
    """
    text = content.strip()
    if not text:
        raise LLMSynthesisError("LLM returned an empty response")

    candidates: list[str] = []
    fenced = _extract_fenced_block(text)
    if fenced is not None:
        candidates.append(fenced)
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    saw_non_object = False
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
        saw_non_object = True

    if saw_non_object:
        raise LLMSynthesisError("LLM response was not a JSON object")
    raise LLMSynthesisError("LLM response was not valid JSON")


def _has_callable(obj: Any, name: str) -> bool:
    """Probe for a method WITHOUT letting the probe itself become a failure path.

    `getattr` executes a property/`__getattr__` hook, and a proxy client can
    raise from one. Left unguarded, the probe would raise during graph
    construction, where the framework's generic handler writes `str(exc)` plus a
    traceback into `error_log` — which this template returns to the caller. A
    descriptor that cannot be read is simply "not this protocol".
    """
    try:
        return callable(getattr(obj, name, None))
    except Exception:  # noqa: BLE001 — a raising descriptor is not a usable seam
        return False


def coerce_llm_client(obj: Any) -> LLMClient | None:
    """Normalise an injected object onto the `LLMClient` protocol.

    Order matters. `isinstance(BaseLLM)` is tested FIRST, before any duck-typing
    probe: a concrete SDK client may delegate unknown attributes to an internal
    LangChain client, so `getattr(obj, "synthesize")` can resolve to an unrelated
    callable with a different signature and silently mis-bind the seam.
    """
    if obj is None:
        return None
    if isinstance(obj, BaseLLM):
        return BaseLLMSynthesisAdapter(obj)
    if _has_callable(obj, "synthesize"):
        return obj  # type: ignore[no-any-return]
    if _has_callable(obj, "complete"):
        return BaseLLMSynthesisAdapter(obj)
    raise LLMSynthesisError(
        "injected llm implements neither synthesize() nor BaseLLM.complete()"
    )


def build_default_llm_client(options: dict[str, Any], api_key: str) -> LLMClient:
    """Build the provider-backed client when nothing was injected via config.

    `api_key` is resolved by the caller from `ctx.secrets.require(...)` on each
    invocation — it is never read here from the environment,
    never cached, and never written to state. The provider module is imported
    lazily so a template that always injects `config["llm"]` does not require
    the provider extra to be installed.
    """
    provider = str(options.get("provider") or "anthropic").lower()
    try:
        module_name, class_name = _PROVIDERS[provider]
    except KeyError:
        raise LLMSynthesisError(f"unsupported llm_provider: {provider!r}") from None

    import importlib

    try:
        client_cls = getattr(importlib.import_module(module_name), class_name)
    except Exception:  # noqa: BLE001 — missing provider extra, import-time failure
        raise LLMSynthesisError(
            f"llm provider {provider!r} is unavailable; declare its extra in agent.yaml"
        ) from None

    client_config: dict[str, Any] = {"api_key": api_key}
    # No model id is written in src/. When config/config.yaml does not set
    # llm_model the provider client's own default applies.
    model = options.get("model")
    if model:
        client_config["model"] = str(model)
    for key in _PROVIDER_OPTION_KEYS:
        if options.get(key) is not None:
            client_config[key] = options[key]
    return BaseLLMSynthesisAdapter(client_cls(client_config))
