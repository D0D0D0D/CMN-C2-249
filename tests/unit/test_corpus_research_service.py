from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.services.corpus_research_service import (
    CorpusResearchService,
    build_research_service,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_PATH = _REPO_ROOT / "config" / "research_corpus.sample.json"


def _documents() -> list[dict]:
    return [
        {
            "source_id": "arxiv",
            "doc_id": "SAMPLE-ARXIV-002",
            "title": "[SAMPLE] Hybrid work productivity",
            "snippet": "[SAMPLE CORPUS] Remote teams and focus time.",
            "url": "sample://corpus/arxiv/002",
            "published_date": "2026-01-02",
            "keywords": ["hybrid work", "productivity", "ハイブリッドワーク", "生産性"],
        },
        {
            "source_id": "arxiv",
            "doc_id": "SAMPLE-ARXIV-001",
            "title": "[SAMPLE] Hybrid work productivity",
            "snippet": "[SAMPLE CORPUS] Coordinated office days.",
            "url": "sample://corpus/arxiv/001",
            "published_date": "2026-01-01",
            "keywords": ["hybrid work", "productivity", "ハイブリッドワーク", "生産性"],
        },
        {
            "source_id": "news",
            "doc_id": "SAMPLE-NEWS-001",
            "title": "[SAMPLE] Shift scheduling",
            "snippet": "[SAMPLE CORPUS] Sleep opportunity after night work.",
            "url": "sample://corpus/news/001",
            "published_date": "2026-01-03",
            "keywords": ["shift work", "sleep", "交代勤務", "睡眠"],
        },
    ]


def test_english_retrieval_respects_top_k_and_return_schema():
    service = CorpusResearchService(_documents(), top_k=1)

    results = service.retrieve("arxiv", "How does hybrid work affect productivity?", "any")

    assert len(results) == 1
    assert set(results[0]) == {"title", "snippet", "url", "published_date", "score"}
    assert 0.0 < results[0]["score"] <= 1.0


def test_japanese_query_retrieves_matching_document():
    service = CorpusResearchService(_documents())

    results = service.retrieve("arxiv", "ハイブリッドワークは生産性にどう影響するか", "any")

    assert results
    assert "Hybrid work" in results[0]["title"]


def test_no_overlap_and_unknown_source_return_empty_lists():
    service = CorpusResearchService(_documents())

    assert service.retrieve("arxiv", "xylophone nebula", "any") == []
    assert service.retrieve("unknown", "hybrid work productivity", "any") == []


def test_results_are_fresh_and_do_not_mutate_corpus():
    documents = _documents()
    service = CorpusResearchService(documents)

    first = service.retrieve("arxiv", "hybrid work productivity", "any")
    original = first[0]["title"]
    first[0]["title"] = "changed"
    first[0]["source_id"] = "forged"
    second = service.retrieve("arxiv", "hybrid work productivity", "any")

    assert second[0]["title"] == original
    assert documents[1]["title"] == original
    assert "source_id" not in second[0]
    assert "evidence_id" not in second[0]


def test_equal_scores_are_ordered_by_doc_id():
    service = CorpusResearchService(_documents())

    results = service.retrieve("arxiv", "hybrid work productivity", "any")

    assert [result["url"] for result in results] == [
        "sample://corpus/arxiv/001",
        "sample://corpus/arxiv/002",
    ]


def test_factory_without_path_returns_none():
    assert build_research_service({}) is None
    assert build_research_service({"configurable": {}}) is None


def test_factory_resolves_relative_path_independently_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    service = build_research_service(
        {
            "configurable": {
                "research_corpus_path": "config/research_corpus.sample.json",
                "research_corpus_top_k": 2,
            }
        }
    )

    assert isinstance(service, CorpusResearchService)
    assert len(service.retrieve("news", "生成AIエージェントの導入効果", "any")) <= 2


def test_factory_rejects_missing_file_without_resolved_path_in_message():
    relative = "config/missing-research-corpus.sample.json"

    with pytest.raises(ValueError) as error:
        build_research_service({"research_corpus_path": relative})

    assert relative in str(error.value)
    assert str(_REPO_ROOT / relative) not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"documents": []},
        [{"source_id": "arxiv", "doc_id": "SAMPLE-001", "title": "missing URL"}],
    ],
    ids=["empty-list", "not-list", "missing-required-key"],
)
def test_factory_rejects_invalid_corpus(tmp_path, payload):
    path = tmp_path / "invalid.sample.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        build_research_service({"research_corpus_path": str(path)})


def test_factory_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.sample.json"
    path.write_text("[not json", encoding="utf-8")

    with pytest.raises(ValueError):
        build_research_service({"research_corpus_path": str(path)})


def test_shipped_sample_corpus_has_explicit_provenance_markers():
    documents = json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))

    assert 16 <= len(documents) <= 24
    assert Counter(document["source_id"] for document in documents) == {
        "internal_kb": 5,
        "arxiv": 5,
        "pubmed": 5,
        "news": 5,
    }
    assert all(document["title"].startswith("[SAMPLE] ") for document in documents)
    assert all(document["snippet"].startswith("[SAMPLE CORPUS] ") for document in documents)
    assert all(document["url"].startswith(f"sample://corpus/{document['source_id']}/") for document in documents)
    assert all(document["doc_id"].startswith("SAMPLE-") for document in documents)
    assert all(any(ord(char) > 127 for word in document["keywords"] for char in word) for document in documents)
