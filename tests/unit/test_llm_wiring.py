# CMN-C2-249 — Unit Tests: LLM injection seam (config["llm"]) and its adapter.
#
# These cover the wiring defect class the fix addresses: the deployed agent must
# reach the LLM through the same seam the tests use, an invented source id must
# never reach the report, and a provider exception must not put its message into
# caller-visible state.

from __future__ import annotations

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from shared.services.llm.base_llm import BaseLLM

from src.graph.domain_workflow_graph import ResearchWorkflowGraph
from src.nodes.synthesize_node import SynthesizeNode
from src.services.llm_client import (
    BaseLLMSynthesisAdapter,
    LLMSynthesisError,
    coerce_llm_client,
)

_PAYLOAD = '{"findings":[{"claim":"C","supporting_source_ids":["arxiv#1"],"confidence":0.5}],"gaps":["g"]}'


class _StubBaseLLM(BaseLLM):
    """A real BaseLLM subclass — the shape config["llm"] carries in production."""

    def __init__(self, content: object = _PAYLOAD, model: str = "stub-model") -> None:
        self.calls: list = []
        self._content = content
        self._model = model

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        return {"content": self._content, "tool_calls": [], "model": self._model, "usage": {}}

    def stream(self, messages: list):
        yield ""

    def bind_tools(self, tools: list) -> "_StubBaseLLM":
        return self


class _NativeClient:
    """This template's own protocol — no complete(), only synthesize()."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, question, evidence, quality_bar) -> dict:
        self.calls += 1
        return {"findings": [], "gaps": ["native"]}


class _BothProtocols(BaseLLM):
    """A BaseLLM that ALSO exposes synthesize().

    Stands in for an SDK client that delegates unknown attributes to an internal
    LangChain client: a duck-typing probe finds `synthesize` and mis-binds the
    seam. isinstance(BaseLLM) must be decided first.
    """

    def complete(self, messages: list) -> dict:
        return {"content": _PAYLOAD, "tool_calls": [], "model": "m", "usage": {}}

    def synthesize(self, *a, **k):  # noqa: D102 - delegated attribute, wrong signature in practice
        raise AssertionError("duck-typed synthesize() was preferred over isinstance(BaseLLM)")

    def stream(self, messages: list):
        yield ""

    def bind_tools(self, tools: list) -> "_BothProtocols":
        return self


def _state():
    return {
        "research_question": "q",
        "retrieved_evidence": [
            {"evidence_id": "arxiv#1", "source_id": "arxiv", "title": "t", "url": "u"}],
        "quality_bar": "news_acceptable",
        "identified_gaps": [],
    }


class TestCoerceLLMClient:
    def test_base_llm_is_wrapped_in_the_production_adapter(self):
        client = coerce_llm_client(_StubBaseLLM())
        assert isinstance(client, BaseLLMSynthesisAdapter)

    def test_isinstance_wins_over_duck_typed_synthesize(self):
        # If the order were reversed, _BothProtocols.synthesize() would run and
        # raise AssertionError instead of going through complete().
        client = coerce_llm_client(_BothProtocols())
        assert isinstance(client, BaseLLMSynthesisAdapter)
        assert client.synthesize("q", [{"evidence_id": "arxiv#1"}], "news_acceptable")["findings"]

    def test_native_protocol_passes_through_unwrapped(self):
        native = _NativeClient()
        assert coerce_llm_client(native) is native

    def test_none_stays_none(self):
        assert coerce_llm_client(None) is None

    def test_incompatible_object_fails_loudly(self):
        with pytest.raises(LLMSynthesisError):
            coerce_llm_client(object())


class TestBaseLLMSynthesisAdapter:
    def test_prompt_offers_per_document_ids_not_connector_names(self):
        stub = _StubBaseLLM()
        BaseLLMSynthesisAdapter(stub).synthesize(
            "q",
            [{"evidence_id": "pubmed#1", "source_id": "pubmed", "title": "A"},
             {"evidence_id": "pubmed#2", "source_id": "pubmed", "title": "B"}],
            "news_acceptable",
        )
        prompt = stub.calls[0][0]["content"]
        assert 'Allowed evidence_ids: ["pubmed#1", "pubmed#2"]' in prompt
        # The bare connector name must not be offered as a citable id — that is
        # what let one claim drag in every document of its connector.
        assert 'Allowed evidence_ids: ["pubmed"]' not in prompt

    def test_json_fence_is_stripped(self):
        stub = _StubBaseLLM(content="```json\n" + _PAYLOAD + "\n```")
        out = BaseLLMSynthesisAdapter(stub).synthesize("q", [], "news_acceptable")
        assert out["findings"][0]["claim"] == "C"

    def test_fenced_block_wins_over_the_brace_heuristic(self):
        # A fence followed by a closing sentence that contains a brace: the
        # first-{ .. last-} slice spans past the JSON and cannot parse, so only
        # real fence extraction recovers this. Real providers emit this shape;
        # the live-check double strips its own fence and never does.
        stub = _StubBaseLLM(
            content="```json\n" + _PAYLOAD + "\n```\n\nLet me know if you want {more} detail."
        )
        out = BaseLLMSynthesisAdapter(stub).synthesize("q", [], "news_acceptable")
        assert out["findings"][0]["claim"] == "C"

    def test_non_canonical_response_is_rejected(self):
        class _RawString(BaseLLM):
            def complete(self, messages: list):
                return _PAYLOAD  # not the canonical dict

            def stream(self, messages: list):
                yield ""

            def bind_tools(self, tools: list):
                return self

        with pytest.raises(LLMSynthesisError):
            BaseLLMSynthesisAdapter(_RawString()).synthesize("q", [], "news_acceptable")

    def test_json_object_embedded_in_prose_is_recovered(self):
        stub = _StubBaseLLM(content="Sure, here you go:\n" + _PAYLOAD + "\nHope that helps.")
        out = BaseLLMSynthesisAdapter(stub).synthesize("q", [], "news_acceptable")
        assert out["gaps"] == ["g"]

    def test_null_content_is_not_stringified_into_None(self):
        stub = _StubBaseLLM(content=None)
        with pytest.raises(LLMSynthesisError):
            BaseLLMSynthesisAdapter(stub).synthesize("q", [], "news_acceptable")

    def test_non_json_response_fails_loudly(self):
        stub = _StubBaseLLM(content="I am unable to answer that.")
        with pytest.raises(LLMSynthesisError):
            BaseLLMSynthesisAdapter(stub).synthesize("q", [], "news_acceptable")

    def test_canonical_model_is_surfaced_for_the_audit_event(self):
        stub = _StubBaseLLM(model="claude-from-the-client")
        out = BaseLLMSynthesisAdapter(stub).synthesize("q", [], "news_acceptable")
        assert out["model"] == "claude-from-the-client"


class TestInnerGraphSeam:
    def test_config_llm_reaches_the_synthesize_node(self):
        stub = _StubBaseLLM()
        graph = ResearchWorkflowGraph(config={"llm": stub})
        graph.register_nodes()
        node = graph._nodes["synthesize"]
        node.execute(_state())
        assert stub.calls, "config['llm'] did not reach SynthesizeNode"

    def test_configurable_nested_llm_also_reaches_the_node(self):
        stub = _StubBaseLLM()
        graph = ResearchWorkflowGraph(config={"configurable": {"llm": stub}})
        graph.register_nodes()
        graph._nodes["synthesize"].execute(_state())
        assert stub.calls

    def test_graph_takes_no_bespoke_llm_keyword(self):
        # BaseGraph.__init__(self, config=None) is the only
        # constructor contract; a bespoke keyword is unreachable in production.
        import inspect

        params = inspect.signature(ResearchWorkflowGraph.__init__).parameters
        assert set(params) == {"self", "config"}

    def test_llm_runtime_options_are_collected_from_config(self):
        graph = ResearchWorkflowGraph(config={"llm_provider": "openai", "llm_model": "some-model"})
        graph.register_nodes()
        assert graph._nodes["synthesize"]._llm_options == {
            "provider": "openai",
            "model": "some-model",
        }


class TestProvenanceResolution:
    def test_invented_source_id_never_reaches_a_finding(self):
        stub = _StubBaseLLM(content=json.dumps({
            "findings": [{"claim": "Fabricated", "supporting_source_ids": ["ghost_999"],
                          "confidence": 0.95}],
            "gaps": [],
        }))
        r = SynthesizeNode(llm=stub).execute(_state())
        assert r["synthesized_findings"] == []
        assert any("unknown source id" in g for g in r["identified_gaps"])
        assert any("ghost_999" in g for g in r["identified_gaps"])

    def test_partially_invented_citation_keeps_only_the_resolvable_id(self):
        stub = _StubBaseLLM(content=json.dumps({
            "findings": [{"claim": "Real", "supporting_source_ids": ["arxiv#1", "ghost_999"],
                          "confidence": 0.7}],
            "gaps": [],
        }))
        r = SynthesizeNode(llm=stub).execute(_state())
        assert r["synthesized_findings"][0]["supporting_source_ids"] == ["arxiv#1"]
        assert any("unknown source id" in g for g in r["identified_gaps"])

    def test_empty_result_still_records_why(self):
        # "no findings AND no gaps" renders as "Evidence Gaps: None identified",
        # which reads as "nothing was missing" — a false conclusion.
        stub = _StubBaseLLM(content='{"findings":[],"gaps":[]}')
        r = SynthesizeNode(llm=stub).execute(_state())
        assert r["status"] == AgentStatus.SUCCESS.value
        assert r["identified_gaps"], "an empty report must state that it is empty"

    def test_boolean_confidence_is_not_accepted_as_a_score(self):
        stub = _StubBaseLLM(content=json.dumps({
            "findings": [{"claim": "C", "supporting_source_ids": ["arxiv#1"], "confidence": True}],
            "gaps": [],
        }))
        r = SynthesizeNode(llm=stub).execute(_state())
        assert r["synthesized_findings"] == []


class TestFailureContainment:
    def test_provider_exception_message_never_enters_state(self):
        secret = "sk-ant-DO-NOT-LEAK-0123456789"

        class _Boom(BaseLLM):
            def complete(self, messages: list) -> dict:
                raise RuntimeError(f"401 unauthorized api_key={secret} request={messages}")

            def stream(self, messages: list):
                yield ""

            def bind_tools(self, tools: list):
                return self

        r = SynthesizeNode(llm=_Boom()).execute(_state())
        blob = json.dumps(r, ensure_ascii=False, default=str)
        assert r["status"] == AgentStatus.ERROR.value
        assert secret not in blob
        assert "401 unauthorized" not in blob
        assert "Traceback" not in blob
        # The exception TYPE is the whole reason that is recorded.
        assert "RuntimeError" in r["error"]

    def test_failure_is_error_not_a_plausible_empty_report(self):
        class _Boom(BaseLLM):
            def complete(self, messages: list) -> dict:
                raise TimeoutError("upstream timed out")

            def stream(self, messages: list):
                yield ""

            def bind_tools(self, tools: list):
                return self

        r = SynthesizeNode(llm=_Boom()).execute(_state())
        # No deterministic substitute for synthesis exists, so this must not
        # degrade to a success carrying an empty-but-plausible report.
        assert r["status"] == AgentStatus.ERROR.value
        assert r["synthesized_findings"] == []

    def test_uninjected_node_reports_only_the_secret_error_type(self):
        # No config["llm"] and no provisioned LLM_API_KEY: the node resolves the
        # secret per invocation and fails closed with a type name only.
        r = SynthesizeNode().execute(_state())
        assert r["status"] == AgentStatus.ERROR.value
        assert "SynthesizeNode: synthesis failed (" in r["error"]
        assert "LLM_API_KEY" not in json.dumps(r, default=str)


class TestNestedOutputCredentialGate:
    """Review finding (High): an older `@final` S-3 gate scans TOP-LEVEL
    strings only, so credentials inside the nested report payload reached the
    caller even while the run was blocked on `formatted_output`."""

    # Shapes the framework detector recognises; assembled at run time so the
    # literals are not committed (the credential scan grades tests/ too).
    _OPENAI_KEY = "sk-" + "A" * 32

    def test_credential_in_a_model_authored_claim_is_blocked(self):
        stub = _StubBaseLLM(content=json.dumps({
            "findings": [{"claim": f"the token is {self._OPENAI_KEY}",
                          "supporting_source_ids": ["arxiv#1"], "confidence": 0.9}],
            "gaps": [],
        }))
        node = SynthesizeNode(llm=stub)
        with pytest.raises(RuntimeError, match="S-3 output gate"):
            node._extra_security_gate_output(node.execute(_state()))

    def test_credential_in_a_model_authored_gap_is_blocked(self):
        stub = _StubBaseLLM(content=json.dumps({
            "findings": [], "gaps": [f"leaked {self._OPENAI_KEY}"],
        }))
        node = SynthesizeNode(llm=stub)
        with pytest.raises(RuntimeError, match="S-3 output gate"):
            node._extra_security_gate_output(node.execute(_state()))

    def test_credential_in_retrieved_source_text_is_blocked(self):
        from src.nodes.retrieve_node import RetrieveNode

        class _LeakySource:
            def retrieve(self, source_id, query, quality_bar):
                return [{"title": "t", "snippet": f"key {TestNestedOutputCredentialGate._OPENAI_KEY}",
                         "url": "https://x/1"}]

        node = RetrieveNode(service=_LeakySource())
        result = node.execute({"research_question": "q", "quality_bar": "news_acceptable",
                               "research_plan": {"proposed_sources": ["arxiv"]}})
        with pytest.raises(RuntimeError, match="S-3 output gate"):
            node._extra_security_gate_output(result)

    def test_the_block_message_never_repeats_the_matched_value(self):
        stub = _StubBaseLLM(content=json.dumps({
            "findings": [], "gaps": [f"leaked {self._OPENAI_KEY}"],
        }))
        node = SynthesizeNode(llm=stub)
        with pytest.raises(RuntimeError) as excinfo:
            node._extra_security_gate_output(node.execute(_state()))
        assert self._OPENAI_KEY not in str(excinfo.value)

    def test_a_clean_nested_result_passes_through_unchanged(self):
        stub = _StubBaseLLM()
        node = SynthesizeNode(llm=stub)
        result = node.execute(_state())
        assert node._extra_security_gate_output(result) == result


class TestConfidenceValidation:
    """Review finding (Medium): the prompt asks for 0..1 but the model may ignore it."""

    def _finding(self, confidence):
        return json.dumps({
            "findings": [{"claim": "C", "supporting_source_ids": ["arxiv#1"],
                          "confidence": confidence}],
            "gaps": [],
        })

    @pytest.mark.parametrize("bad", [7, -0.5, 1.5])
    def test_out_of_range_confidence_is_not_reported_as_a_score(self, bad):
        r = SynthesizeNode(llm=_StubBaseLLM(content=self._finding(bad))).execute(_state())
        assert r["synthesized_findings"] == []
        assert any("unsupported/low-confidence" in g for g in r["identified_gaps"])

    def test_nan_confidence_is_rejected(self):
        # json.loads accepts bare NaN; it would then break JSON serialisation at
        # the HTTP boundary and render as a meaningless score.
        raw = '{"findings":[{"claim":"C","supporting_source_ids":["arxiv#1"],"confidence":NaN}],"gaps":[]}'
        r = SynthesizeNode(llm=_StubBaseLLM(content=raw)).execute(_state())
        assert r["synthesized_findings"] == []

    def test_the_model_supplied_score_is_preserved_exactly(self):
        # Guards a mutation that pins every confidence to a constant.
        r = SynthesizeNode(llm=_StubBaseLLM(content=self._finding(0.37))).execute(_state())
        assert r["synthesized_findings"][0]["confidence"] == 0.37


class TestProbeSafety:
    """Review finding (Medium): getattr() executes descriptors, and a raising one
    turned the seam probe itself into a leak path through the generic handler."""

    def test_a_raising_synthesize_property_does_not_propagate(self):
        secret = "probe-leaked-" + "B" * 20

        class _Proxy:
            @property
            def synthesize(self):
                raise RuntimeError(f"internal state: {secret}")

            def complete(self, messages):
                return {"content": _PAYLOAD, "tool_calls": [], "model": "m", "usage": {}}

        client = coerce_llm_client(_Proxy())
        assert isinstance(client, BaseLLMSynthesisAdapter)
        assert client.synthesize("q", [{"evidence_id": "arxiv#1"}], "news_acceptable")["findings"]

    def test_an_object_whose_every_attribute_raises_is_rejected_cleanly(self):
        class _Hostile:
            def __getattr__(self, name):
                raise RuntimeError("secret-bearing failure")

        with pytest.raises(LLMSynthesisError):
            coerce_llm_client(_Hostile())


class TestDefaultProviderBuilder:
    """Review finding (Medium) + mutation blind spots around the un-injected path."""

    def test_the_declared_secret_name_is_the_one_requested(self):
        # Guards a typo'd secret constant: without this, only "it errors" is
        # asserted anywhere and any wrong key name still errors.
        from src.nodes.synthesize_node import LLM_API_KEY_SECRET

        requested: list[str] = []

        class _Secrets:
            def require(self, key):
                requested.append(key)
                return "unit-test-not-a-real-key"

        class _Ctx:
            secrets = _Secrets()

        node = SynthesizeNode()
        import src.nodes.synthesize_node as mod

        built: list[tuple] = []
        original_ic = mod.InvocationContext
        original_build = mod.build_default_llm_client
        try:
            mod.InvocationContext = type("IC", (), {"from_state": staticmethod(lambda s: _Ctx())})
            mod.build_default_llm_client = lambda opts, key: built.append((opts, key)) or _NativeClient()
            node.execute(_state())
        finally:
            mod.InvocationContext = original_ic
            mod.build_default_llm_client = original_build

        assert requested == [LLM_API_KEY_SECRET] == ["LLM_API_KEY"]
        assert built and built[0][1] == "unit-test-not-a-real-key"

    def test_only_api_key_providers_are_offered(self):
        # BedrockClient needs aws_* keys and raises on api_key alone; offering
        # it from a builder that only has an api key is a promise it can't keep.
        from src.services.llm_client import _PROVIDERS

        assert set(_PROVIDERS) == {"anthropic", "openai"}

    def test_an_unknown_provider_fails_loudly(self):
        from src.services.llm_client import build_default_llm_client

        with pytest.raises(LLMSynthesisError):
            build_default_llm_client({"provider": "nope"}, "k")

    def test_no_model_id_is_pinned_in_src(self):
        import pathlib
        import re

        offenders = []
        for path in pathlib.Path("src").rglob("*.py"):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r'["\'](claude-|gpt-|anthropic\.claude)', line):
                    offenders.append(f"{path}:{n}")
        assert not offenders, f"model id hardcoded in src/: {offenders}"


class TestPromptCarriesTheEvidence:
    def test_the_evidence_body_is_actually_sent(self):
        # Guards a mutation that drops `snippet` from the projection: every
        # integration fake still passes with an evidence-free prompt.
        stub = _StubBaseLLM()
        BaseLLMSynthesisAdapter(stub).synthesize(
            "q",
            [{"evidence_id": "arxiv#1", "source_id": "arxiv", "title": "T",
              "snippet": "the decisive finding text",
              "url": "https://x/1", "published_date": "2026-01-01"}],
            "news_acceptable",
        )
        prompt = stub.calls[0][0]["content"]
        for expected in ("the decisive finding text", "T", "https://x/1", "2026-01-01"):
            assert expected in prompt, f"{expected!r} was not sent to the model"


class TestNestedScanRobustness:
    """Second adversarial round: holes in the traversal itself."""

    _KEY = "sk-" + "C" * 32

    def _gate(self, result):
        from src.security_gates import assert_no_nested_credentials

        return assert_no_nested_credentials(result, "TestNode")

    def test_a_credential_used_as_a_dict_key_is_caught(self):
        with pytest.raises(RuntimeError, match="S-3 output gate"):
            self._gate({"evidence": [{self._KEY: "safe value"}]})

    def test_the_message_never_names_the_offending_key(self):
        with pytest.raises(RuntimeError) as excinfo:
            self._gate({"evidence": [{self._KEY: self._KEY}]})
        # The message is copied verbatim into error_log and the S-4 audit event.
        assert self._KEY not in str(excinfo.value)

    def test_a_credential_inside_a_tuple_is_caught(self):
        with pytest.raises(RuntimeError, match="S-3 output gate"):
            self._gate({"evidence": ({"snippet": self._KEY},)})

    def test_an_untraversable_container_is_blocked_not_skipped(self):
        # A set is not JSON/msgpack-serialisable (so it already breaks the State
        # contract) and FastAPI renders it into the response as a list.
        with pytest.raises(RuntimeError, match="cannot be proven credential-free"):
            self._gate({"evidence": [{"snippet": {self._KEY}}]})

    def test_an_object_hiding_text_in_dunder_str_is_blocked(self):
        class _Sneaky:
            def __str__(self):
                return TestNestedScanRobustness._KEY

        with pytest.raises(RuntimeError, match="cannot be proven credential-free"):
            self._gate({"evidence": [_Sneaky()]})

    def test_nesting_beyond_the_depth_limit_is_blocked_not_skipped(self):
        from src.security_gates import _MAX_DEPTH

        deep: object = self._KEY
        for _ in range(_MAX_DEPTH + 4):
            deep = [deep]
        with pytest.raises(RuntimeError, match="S-3 output gate"):
            self._gate({"evidence": deep})

    def test_a_payload_at_the_depth_limit_still_scans(self):
        # Guards a mutation that shrinks _MAX_DEPTH: a legitimately nested
        # payload must still be walked all the way down.
        from src.security_gates import _MAX_DEPTH

        deep: object = self._KEY
        for _ in range(_MAX_DEPTH - 2):
            deep = [deep]
        with pytest.raises(RuntimeError, match="credential pattern"):
            self._gate({"evidence": deep})

    def test_a_shared_reference_graph_does_not_explode(self):
        # Without identity tracking a diamond of shared references is re-walked
        # once per path and the scan never returns.
        import time

        leaf = {"snippet": "clean"}
        level: object = leaf
        for _ in range(20):
            level = [level, level, level]
        start = time.perf_counter()
        self._gate({"evidence": level})
        assert time.perf_counter() - start < 2.0

    def test_a_self_referential_structure_terminates(self):
        loop: list = ["clean"]
        loop.append(loop)
        self._gate({"evidence": loop})

    def test_scalars_and_none_are_accepted(self):
        assert self._gate({"n": 1, "f": 0.5, "b": True, "none": None, "empty": []})


class TestPostProcessNestedGate:
    """PostProcess owns evidence_report; its hook needs its own coverage —
    published_date never appears in the markdown, so a credential there is
    invisible to the flattened formatted_output scan."""

    _KEY = "sk-" + "D" * 32

    def test_credential_in_citation_metadata_is_blocked(self):
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        result = node.execute({
            "research_question": "q",
            "synthesized_findings": [
                {"claim": "C", "supporting_source_ids": ["arxiv#1"], "confidence": 0.8}],
            "retrieved_evidence": [
                {"evidence_id": "arxiv#1", "source_id": "arxiv",
                 "source_tier": "peer_reviewed", "title": "T",
                 "url": "https://x/1", "published_date": self._KEY}],
            "identified_gaps": [],
        })
        assert self._KEY not in result["formatted_output"], "precondition: invisible in markdown"
        with pytest.raises(RuntimeError, match="S-3 output gate"):
            node._extra_security_gate_output(result)


class TestPromptCompleteness:
    def test_the_question_and_quality_bar_reach_the_model(self):
        stub = _StubBaseLLM()
        BaseLLMSynthesisAdapter(stub).synthesize(
            "Does treatment T reduce outcome O?",
            [{"evidence_id": "arxiv#1", "snippet": "s"}],
            "peer_reviewed_only",
        )
        prompt = stub.calls[0][0]["content"]
        assert "Does treatment T reduce outcome O?" in prompt
        assert "peer_reviewed_only" in prompt

    def test_the_provider_map_points_at_the_matching_client(self):
        # Guards a mutation that swaps a provider's module/class target.
        from src.services.llm_client import _PROVIDERS

        assert _PROVIDERS["anthropic"] == (
            "shared.services.llm.anthropic_client", "AnthropicClient")
        assert _PROVIDERS["openai"] == (
            "shared.services.llm.openai_client", "OpenAIClient")


class TestConfidenceOverflow:
    def test_a_huge_integer_score_degrades_to_a_gap_not_a_crash(self):
        # float(10**400) raises OverflowError, outside the provider try/except.
        raw = '{"findings":[{"claim":"C","supporting_source_ids":["arxiv#1"],"confidence":1e400}],"gaps":[]}'
        r = SynthesizeNode(llm=_StubBaseLLM(content=raw)).execute(_state())
        assert r["status"] == AgentStatus.SUCCESS.value
        assert r["synthesized_findings"] == []

    def test_a_huge_python_int_score_does_not_raise(self):
        from src.nodes.synthesize_node import _valid_confidence

        assert _valid_confidence(10 ** 400) is False


class TestTupleTraversal:
    """A tuple must be WALKED, not blocked as an unknown type — blocking it
    would also 'catch' a credential inside one, which is why the positive case
    needs its own assertion."""

    def test_a_clean_tuple_passes_through(self):
        from src.security_gates import assert_no_nested_credentials

        result = {"evidence": ({"snippet": "clean text"}, {"snippet": "also clean"})}
        assert assert_no_nested_credentials(result, "TestNode") == result


class TestCitationGranularity:
    """The defect this MR's second half fixes: citations were indexed by
    `source_id` (the CONNECTOR), so a claim citing "pubmed" pulled in every
    pubmed document — including one it had nothing to do with. Caught on a real
    corpus by the live-LLM run; reproduces deterministically with no LLM.

    All expected values are literals. Deriving them from make_evidence_id() or
    from _index_evidence() would make these self-referential.
    """

    # Two documents from ONE connector: the supporting trial and an unrelated paper.
    _EVIDENCE = [
        {"evidence_id": "pubmed#1", "source_id": "pubmed", "source_tier": "peer_reviewed",
         "title": "Semaglutide cardiovascular outcome trial", "url": "https://pubmed.test/1",
         "published_date": "2025-11-02"},
        {"evidence_id": "pubmed#2", "source_id": "pubmed", "source_tier": "peer_reviewed",
         "title": "Soil microbial diversity gradients", "url": "https://pubmed.test/2",
         "published_date": "2025-04-18"},
    ]

    def _report(self, supporting):
        from src.nodes.post_process_node import PostProcessNode

        return PostProcessNode().execute({
            "research_question": "Do GLP-1 agonists reduce MACE?",
            "synthesized_findings": [
                {"claim": "GLP-1 agonists reduce MACE",
                 "supporting_source_ids": supporting, "confidence": 0.96}],
            "retrieved_evidence": self._EVIDENCE,
            "identified_gaps": [],
        })

    def test_a_claim_cites_only_the_document_it_names(self):
        report = self._report(["pubmed#1"])["evidence_report"]
        assert [c["title"] for c in report["citations"]] == [
            "Semaglutide cardiovascular outcome trial"]
        assert [c["evidence_id"] for c in report["citations"]] == ["pubmed#1"]

    def test_the_connector_mate_never_appears(self):
        report = self._report(["pubmed#1"])["evidence_report"]
        blob = json.dumps(report, ensure_ascii=False)
        assert "Soil microbial diversity gradients" not in blob
        assert "https://pubmed.test/2" not in blob

    def test_the_unrelated_paper_is_absent_from_the_rendered_markdown(self):
        rendered = self._report(["pubmed#1"])["formatted_output"]
        assert "Semaglutide cardiovascular outcome trial" in rendered
        assert "Soil microbial diversity gradients" not in rendered

    def test_citing_two_documents_cites_exactly_those_two(self):
        report = self._report(["pubmed#1", "pubmed#2"])["evidence_report"]
        assert sorted(c["evidence_id"] for c in report["citations"]) == ["pubmed#1", "pubmed#2"]

    def test_a_bare_connector_name_no_longer_resolves(self):
        # "pubmed" identifies a connector, not a document. Under the old
        # indexing this cited BOTH documents; it must now cite neither.
        report = self._report(["pubmed"])["evidence_report"]
        assert report["citations"] == []

    def test_citation_metadata_comes_from_the_record_not_the_finding(self):
        report = self._report(["pubmed#1"])["evidence_report"]
        assert report["citations"][0] == {
            "evidence_id": "pubmed#1",
            "source_id": "pubmed",
            "source_tier": "peer_reviewed",
            "title": "Semaglutide cardiovascular outcome trial",
            "url": "https://pubmed.test/1",
            "published_date": "2025-11-02",
        }


class TestConnectorWideCitationIsRejectedAtSynthesis:
    """A connector-wide citation is discarded before it can reach the report,
    with the reason recorded — the same treatment an invented id gets."""

    _EVIDENCE = [
        {"evidence_id": "pubmed#1", "source_id": "pubmed", "title": "trial"},
        {"evidence_id": "pubmed#2", "source_id": "pubmed", "title": "unrelated"},
    ]

    def _synth(self, supporting):
        raw = json.dumps({
            "findings": [{"claim": "C", "supporting_source_ids": supporting, "confidence": 0.9}],
            "gaps": [],
        })
        return SynthesizeNode(llm=_StubBaseLLM(content=raw)).execute({
            "research_question": "q",
            "retrieved_evidence": self._EVIDENCE,
            "quality_bar": "peer_reviewed_only",
            "identified_gaps": [],
        })

    def test_a_bare_connector_id_is_discarded_and_recorded(self):
        r = self._synth(["pubmed"])
        assert r["synthesized_findings"] == []
        assert any("unknown source id" in g and "pubmed" in g for g in r["identified_gaps"])

    def test_a_real_document_id_survives(self):
        r = self._synth(["pubmed#2"])
        assert r["synthesized_findings"][0]["supporting_source_ids"] == ["pubmed#2"]

    def test_a_mix_keeps_only_the_document_id(self):
        r = self._synth(["pubmed", "pubmed#1"])
        assert r["synthesized_findings"][0]["supporting_source_ids"] == ["pubmed#1"]
        assert any("unknown source id" in g for g in r["identified_gaps"])


class TestEndToEndCitationGranularity:
    """The whole graph, with a connector returning two documents."""

    def test_the_report_cites_one_document_not_the_whole_connector(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel
        from src.graph.graph import Graph
        from tests.fakes import FakeLLMClient, FakeResearchService, graph_config

        agent = Graph(config=graph_config(
            service=FakeResearchService(), llm=FakeLLMClient(), hitl_enabled=False))
        agent.compile(checkpointer=InMemorySaver())
        out = agent.invoke(
            "What is X?", session_id="s",
            ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
            input_context={"plan_confirmed": True,
                           "source_spec": {"sources": ["arxiv"],
                                           "quality_bar": "peer_reviewed_only"}},
        )
        assert out["status"] == "success"
        # arxiv returned two documents; the claim cites the first.
        assert [e["evidence_id"] for e in out["retrieved_evidence"]] == ["arxiv#1", "arxiv#2"]
        assert out["synthesized_findings"][0]["supporting_source_ids"] == ["arxiv#1"]
        assert [c["evidence_id"] for c in out["evidence_report"]["citations"]] == ["arxiv#1"]
        assert "unrelated paper" not in out["output"]


class TestEvidenceIdUniqueness:
    """Third adversarial round: the per-document handle is only worth anything
    if it identifies exactly one document."""

    def _retrieve(self, service, sources):
        from src.nodes.retrieve_node import RetrieveNode

        return RetrieveNode(service=service).execute({
            "research_question": "q", "quality_bar": "news_acceptable",
            "research_plan": {"proposed_sources": sources},
        })

    def test_a_repeated_connector_is_only_called_once(self):
        class _Varying:
            def __init__(self):
                self.calls = 0

            def retrieve(self, source_id, query, quality_bar):
                self.calls += 1
                return [{"title": f"doc{self.calls}", "url": f"https://x/{self.calls}"}]

        svc = _Varying()
        r = self._retrieve(svc, ["pubmed", "pubmed"])
        assert svc.calls == 1, "a repeated source must not double the external calls"
        assert [e["evidence_id"] for e in r["retrieved_evidence"]] == ["pubmed#1"]

    def test_effective_sources_dedupes_preserving_order(self):
        from src.policy import DEFAULT_SOURCE_CATALOG, effective_sources

        assert effective_sources(
            ["pubmed", "arxiv", "pubmed"], DEFAULT_SOURCE_CATALOG, "peer_reviewed_only"
        ) == ["pubmed", "arxiv"]

    def test_a_duplicate_handle_is_blocked_not_silently_resolved(self):
        from framework.errors import SecurityViolationError

        import src.nodes.retrieve_node as rn

        class _TwoDocs:
            def retrieve(self, source_id, query, quality_bar):
                return [{"title": "a", "url": "https://x/1"}, {"title": "b", "url": "https://x/2"}]

        original = rn.make_evidence_id
        rn.make_evidence_id = lambda sid, n: f"{sid}#1"
        try:
            with pytest.raises(SecurityViolationError, match="duplicate evidence_id"):
                self._retrieve(_TwoDocs(), ["pubmed"])
        finally:
            rn.make_evidence_id = original

    def test_provenance_fields_are_authoritative_not_connector_supplied(self):
        class _Forging:
            def retrieve(self, source_id, query, quality_bar):
                return [{"source_id": "arxiv", "source_type": "index",
                         "source_tier": "internal_kb", "title": "t", "url": "https://x/9"}]

        e = self._retrieve(_Forging(), ["pubmed"])["retrieved_evidence"][0]
        # A connector must not be able to relabel its hit as another source, nor
        # claim a citation tier the quality bar would have excluded.
        assert e["source_id"] == "pubmed"
        assert e["source_type"] == "connector"
        assert e["source_tier"] == "peer_reviewed"


class TestNoConnectorFallback:
    """A record with no server-issued handle is NOT citable. A fallback to
    `source_id` put the connector name back in the citable set, which is the
    original defect wearing a compatibility hat."""

    _MIXED = [
        {"evidence_id": "pubmed#1", "source_id": "pubmed", "title": "stamped"},
        {"source_id": "pubmed", "title": "legacy-a"},
        {"source_id": "pubmed", "title": "legacy-b"},
    ]

    def test_an_unstamped_record_has_no_citable_handle(self):
        from src.services.research_service import evidence_reference_id

        assert evidence_reference_id({"source_id": "pubmed", "title": "x"}) == ""

    def test_mixed_data_never_offers_the_connector_name(self):
        from src.services.research_service import evidence_reference_ids

        assert evidence_reference_ids(self._MIXED) == ["pubmed#1"]

    def test_reference_ids_are_deduplicated(self):
        from src.services.research_service import evidence_reference_ids

        dupes = [{"evidence_id": "a#1"}, {"evidence_id": "a#1"}, {"evidence_id": "a#2"}]
        assert evidence_reference_ids(dupes) == ["a#1", "a#2"]

    def test_citing_the_connector_name_in_mixed_data_cites_nothing(self):
        from src.nodes.post_process_node import PostProcessNode

        report = PostProcessNode().execute({
            "research_question": "q",
            "synthesized_findings": [
                {"claim": "C", "supporting_source_ids": ["pubmed"], "confidence": 0.9}],
            "retrieved_evidence": self._MIXED,
            "identified_gaps": [],
        })["evidence_report"]
        assert report["citations"] == []


class TestRenderedCitationIdentifiesTheDocument:
    def test_the_markdown_citation_line_carries_the_handle(self):
        from src.nodes.post_process_node import PostProcessNode

        rendered = PostProcessNode().execute({
            "research_question": "q",
            "synthesized_findings": [
                {"claim": "C", "supporting_source_ids": ["pubmed#2"], "confidence": 0.9}],
            "retrieved_evidence": [
                {"evidence_id": "pubmed#1", "source_id": "pubmed",
                 "source_tier": "peer_reviewed", "title": "A", "url": "https://x/1"},
                {"evidence_id": "pubmed#2", "source_id": "pubmed",
                 "source_tier": "peer_reviewed", "title": "B", "url": "https://x/2"}],
            "identified_gaps": [],
        })["formatted_output"]
        # Without the handle a reader cannot tell which of the connector's
        # documents a claim rests on — the whole point of the change.
        assert "- [pubmed#2] [peer_reviewed] B — https://x/2" in rendered
        assert "pubmed#1" not in rendered

    def test_the_prompt_lets_the_model_map_ids_to_documents(self):
        # The allowed list alone is not enough: if the Evidence block omits
        # evidence_id the model cannot tell which document each id refers to.
        stub = _StubBaseLLM()
        BaseLLMSynthesisAdapter(stub).synthesize(
            "q",
            [{"evidence_id": "pubmed#1", "source_id": "pubmed", "title": "Trial"},
             {"evidence_id": "pubmed#2", "source_id": "pubmed", "title": "Unrelated"}],
            "peer_reviewed_only",
        )
        evidence_block = stub.calls[0][0]["content"].split("Evidence: ", 1)[1]
        assert '"evidence_id": "pubmed#1"' in evidence_block
        assert '"evidence_id": "pubmed#2"' in evidence_block
