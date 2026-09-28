"""AgentCore Platform v1.0 — multi-source retrieval seam.

ResearchService is the boundary between retrieve_node and concrete source
connectors (internal KB index search + external live connectors such as
arXiv / PubMed / news). Connector credentials are resolved via
ctx.secrets.require(...) by the deploy wiring — never read here from environ
and never stored in agent state.

The default service raises NotImplementedError so an unconfigured deploy fails
loudly. Tests inject a fake service implementing retrieve().

Governance: the deploy-time source catalog (allowed_sources) is the outer
bound. The effective source set is (deploy catalog) ∩ (user/plan selection),
computed in retrieve_node — this service only fetches from sources it is asked
for, never discovers new endpoints at runtime (SSRF / ToS protection, keeps the
agent Cat 2 rather than autonomous Cat 3).
"""

from __future__ import annotations

from typing import Any


# ── evidence identity ───────────────────────────────────────────────
# Citations are bound per DOCUMENT, not per connector. `source_id` names the
# connector a hit came from (arxiv / pubmed / ...); a connector routinely
# returns several documents, so a claim that cites only the connector cannot say
# WHICH of them supports it — and the report then lists all of them, including
# ones the claim has nothing to do with.
#
# `evidence_id` is that missing per-document handle. It is issued by
# retrieve_node from the retrieval record itself and is never accepted from the
# connector payload or authored by the model: it is a server-issued token, so a
# connector cannot forge a collision and the LLM can only *reference* it.
EVIDENCE_ID_FIELD = "evidence_id"


def make_evidence_id(source_id: str, ordinal: int) -> str:
    """The citation handle for the `ordinal`-th accepted hit from `source_id`.

    1-based so the rendered report reads `arxiv#1`, not `arxiv#0`.
    """
    return f"{source_id}#{ordinal}"


def evidence_reference_id(record: dict[str, Any]) -> str:
    """The id a finding must cite to bind itself to THIS document.

    Returns "" for a record carrying no server-issued handle — such a record is
    NOT citable. There is deliberately no fallback to `source_id`: falling back
    would put the connector name back into the citable set, and a claim citing
    it would once again resolve to every document that connector returned. That
    is the exact defect the per-document handle exists to remove, so the
    "compatible" fallback quietly reintroduced it whenever stamped and unstamped
    records were mixed.
    """
    return str(record.get(EVIDENCE_ID_FIELD) or "")


def evidence_reference_ids(evidence: list[dict[str, Any]]) -> list[str]:
    """Ordered, de-duplicated citation handles for a retrieved evidence set."""
    ids: list[str] = []
    for record in evidence:
        ref = evidence_reference_id(record)
        if ref and ref not in ids:
            ids.append(ref)
    return ids


# Source tier ranking used for citation quality enforcement.
SOURCE_TIERS: dict[str, int] = {
    "peer_reviewed": 3,
    "internal_kb": 2,
    "news": 1,
    "web": 0,
}


class ResearchService:
    """Domain service fetching evidence from a single allowed source."""

    def retrieve(
        self, source_id: str, query: str, quality_bar: str
    ) -> list[dict[str, Any]]:
        """Fetch evidence hits for `query` from `source_id`.

        Returns a list of evidence dicts:
            {source_id, source_type, source_tier, title, snippet, url,
             published_date, score}
        `evidence_id` is NOT supplied here — retrieve_node issues it per
        accepted document and overwrites anything a connector puts in that key.
        Raises on transient failure so retrieve_node can record the source as a
        gap (degrade) after backbone RETRY exhaustion.
        """
        raise NotImplementedError("Inject a connector-backed ResearchService for retrieve().")
