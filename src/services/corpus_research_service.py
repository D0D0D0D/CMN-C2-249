"""Deterministic retrieval over a configured JSON research corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.services.research_service import ResearchService

_ASCII_TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,64}")
_CJK_SEQUENCE_PATTERN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uff66-\uff9f]{1,4000}")
_ASCII_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "for",
        "and",
        "or",
        "is",
        "are",
        "what",
        "how",
        "does",
        "do",
        "in",
        "on",
        "to",
        "with",
        "by",
    }
)
_RETURN_KEYS = ("title", "snippet", "url", "published_date")
_REQUIRED_DOCUMENT_KEYS = frozenset({"source_id", "doc_id", "title", "url"})
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _tokens(text: str) -> set[str]:
    """Return bounded ASCII words and CJK character 2-grams."""

    lowered = str(text or "").lower()
    tokens = {token for token in _ASCII_TOKEN_PATTERN.findall(lowered) if token not in _ASCII_STOPWORDS}
    for sequence in _CJK_SEQUENCE_PATTERN.findall(lowered):
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _cfg_get(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    configurable = config.get("configurable") or {}
    return configurable.get(key) if isinstance(configurable, dict) else None


class CorpusResearchService(ResearchService):
    """Retrieve the highest-overlap documents from an in-process corpus."""

    def __init__(self, documents: list[dict[str, Any]], *, top_k: int = 3) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("research_corpus_top_k must be a positive integer")
        self._documents = documents
        self._top_k = top_k

    def retrieve(self, source_id: str, query: str, quality_bar: str) -> list[dict[str, Any]]:
        del quality_bar
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for document in self._documents:
            if document.get("source_id") != source_id:
                continue
            keywords = document.get("keywords", [])
            if isinstance(keywords, list):
                keyword_text = " ".join(str(value) for value in keywords)
            else:
                keyword_text = str(keywords or "")
            searchable = " ".join((str(document.get("title") or ""), str(document.get("snippet") or ""), keyword_text))
            overlap = query_tokens & _tokens(searchable)
            if not overlap:
                continue
            score = min(1.0, len(overlap) / len(query_tokens))
            ranked.append((score, str(document.get("doc_id") or ""), document))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                **{key: document.get(key) for key in _RETURN_KEYS},
                "score": float(score),
            }
            for score, _doc_id, document in ranked[: self._top_k]
        ]


def build_research_service(
    config: dict[str, Any] | None,
) -> CorpusResearchService | None:
    """Build a corpus service from flat or ``configurable`` config."""

    cfg = config or {}
    configured_path = _cfg_get(cfg, "research_corpus_path")
    if configured_path is None:
        return None

    display_path = str(configured_path)
    path = Path(display_path)
    resolved = path if path.is_absolute() else _REPO_ROOT / path
    if not resolved.is_file():
        raise ValueError(f"research corpus {display_path!r} does not exist")

    try:
        with resolved.open(encoding="utf-8") as handle:
            documents = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"research corpus {display_path!r} could not be loaded ({type(exc).__name__})") from None

    if not isinstance(documents, list):
        raise ValueError(f"research corpus {display_path!r} must contain a JSON list")
    if not documents:
        raise ValueError(f"research corpus {display_path!r} must contain at least one document")
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"research corpus {display_path!r} document {index} must be an object")
        missing = sorted(_REQUIRED_DOCUMENT_KEYS - document.keys())
        if missing:
            raise ValueError(f"research corpus {display_path!r} document {index} missing keys: {missing}")

    top_k = _cfg_get(cfg, "research_corpus_top_k")
    if top_k is None:
        top_k = 3
    return CorpusResearchService(documents, top_k=top_k)
