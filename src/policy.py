"""AgentCore Platform v1.0 — research governance policy (no business logic, no secrets).

Central source-catalog / citation-quality policy shared by pre_process_node and
retrieve_node. Pure functions + constants only.

Governance boundary (S-1 / ToS / SSRF):
  effective sources = (deploy-time allowed catalog) ∩ (user / plan selection)
The deploy catalog is the hard outer bound; users can only pick from within it,
never inject arbitrary endpoints at runtime.
"""

from __future__ import annotations

from typing import Any

# Default deploy-time source catalog. Overridable per-deploy via
# config["configurable"]["allowed_sources"]. Each entry maps a source_id to its
# {type, tier}. tier feeds citation-quality enforcement.
DEFAULT_SOURCE_CATALOG: dict[str, dict[str, str]] = {
    "internal_kb": {"type": "index", "tier": "internal_kb"},
    "arxiv": {"type": "connector", "tier": "peer_reviewed"},
    "pubmed": {"type": "connector", "tier": "peer_reviewed"},
    "news": {"type": "connector", "tier": "news"},
}

# Citation quality tiers -> the minimum source tier they admit.
QUALITY_BAR_MIN_TIER: dict[str, str] = {
    "peer_reviewed_only": "peer_reviewed",
    "internal_only": "internal_kb",
    "news_acceptable": "news",
}

_TIER_RANK: dict[str, int] = {
    "peer_reviewed": 3,
    "internal_kb": 2,
    "news": 1,
    "web": 0,
}

VALID_QUALITY_BARS = tuple(QUALITY_BAR_MIN_TIER.keys())


def resolve_catalog(config: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    """Return the effective deploy source catalog (config override or default).

    Accepts both the flat graph-config shape ({"allowed_sources": ...}) and the
    agent.yaml `configurable:`-nested shape for backwards compatibility.
    """
    cfg = config or {}
    configurable = cfg.get("configurable", {}) or {}
    return (
        cfg.get("allowed_sources")
        or configurable.get("allowed_sources")
        or DEFAULT_SOURCE_CATALOG
    )


def tier_rank(tier: str) -> int:
    return _TIER_RANK.get(tier, -1)


def out_of_catalog(requested: list[str], catalog: dict[str, dict[str, str]]) -> list[str]:
    """Return requested source_ids that are NOT in the allowed catalog (S-2 reject set)."""
    return [s for s in requested if s not in catalog]


def effective_sources(
    requested: list[str],
    catalog: dict[str, dict[str, str]],
    quality_bar: str,
) -> list[str]:
    """(deploy catalog) ∩ (user/plan selection), filtered by quality_bar tier.

    If `requested` is empty, fall back to all catalog sources meeting the bar.

    De-duplicated, order preserved. A caller may legitimately repeat a source in
    `source_spec.sources`; retrieving from the same connector twice not only
    doubles the external calls but restarts that connector's per-document
    numbering, which can issue one `evidence_id` for two different documents.
    """
    min_tier = QUALITY_BAR_MIN_TIER.get(quality_bar, "web")
    min_rank = tier_rank(min_tier)
    pool = requested if requested else list(catalog.keys())
    selected: list[str] = []
    for s in pool:
        if s in catalog and tier_rank(catalog[s]["tier"]) >= min_rank and s not in selected:
            selected.append(s)
    return selected
