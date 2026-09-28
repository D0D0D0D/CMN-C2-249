"""tests/conftest.py — CMN-C2-249 test bootstrap (wheel-era).

The L1 framework is provided by the published `agenticstar-agentcore==1.0.0`
wheel (installed locally into .venv and in CI by the dual-mode run-tests job).
The legacy sys.modules framework stubs were removed in the SDK v1.0 migration —
tests now run against the real framework contract.

This file only makes the repository root importable so `import src...` and
`import tests...` resolve when pytest is invoked from the repo root, and
supplies the fixtures the live-LLM harness looks for.

NOTE: there is deliberately no test-side LLM adapter here any more. The
`BaseLLM` -> `synthesize()` bridge lives in `src/services/llm_client.py`, so the
live harness (which only sets `config["llm"]`) exercises the same code path the
deployed agent uses. A bridge that exists only in `tests/` makes the suite green
while production has no route from `config["llm"]` to the node.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from tests.fakes import FakeResearchService, graph_config


@pytest.fixture
def agent_config() -> dict:
    """実 LLM 1 回転チェック用 config。

    `tests/integration/test_workflow.py::_agent()` と同じく `FakeResearchService`
    を seed する。LLM は harness が `config["llm"]` に入れるので、ここでは触らない
    （本番と同じシームを通す）。
    """
    return graph_config(service=FakeResearchService(), hitl_enabled=False)


@pytest.fixture
def case_payload() -> str:
    """`tests/integration/test_workflow.py::test_happy_path_produces_report` と同じ質問。"""
    return "What is X?"


@pytest.fixture
def live_input_context() -> dict:
    """統合テストの `_CONFIRMED_CONTEXT` と同じ、HITL を越えて中核を走らせる入力。"""
    return {
        "plan_confirmed": True,
        "source_spec": {"sources": ["arxiv"], "quality_bar": "peer_reviewed_only"},
    }
