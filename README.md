# CMN-C2-249 — Deep Research Synthesis & Evidence Report Agent

> **Category**: Cat 2 (domain workflow)
> **Industry**: CMN

## Overview

This agent turns a research question into a structured evidence report. It plans the search,
optionally pauses for a person to confirm or revise the plan, retrieves material from a
configured catalog of sources, synthesizes findings across them, and returns a report in which
every claim is bound to the specific documents that support it.

It suits work where the provenance of an answer matters as much as the answer: a researcher
sweeping several approved sources for one question, an analyst who needs each claim attributable
and scored, or a reviewer who needs to see what could *not* be evidenced as clearly as what
could.

The division of labour between the language model and the deterministic layer is the point of
the design. The model writes prose and cites reference ids; it does not author the provenance:

- **Evidence ids are issued by the retrieval step, not by the model.** Every
  `supporting_source_id` the model returns is resolved against the evidence actually retrieved;
  an id that does not exist is dropped rather than displayed as a citation.
- **Source identity and tier are assigned, not accepted.** The catalog entry and the connector
  actually called are authoritative, so a connector cannot present itself as a
  higher-tier source than it is.
- **Confidence is validated.** A score outside `0..1`, or a malformed synthesis payload, is
  rejected rather than rendered.
- **An empty result still says why.** When nothing supportable is found, that is recorded as a
  gap, not returned as a clean, empty report.
- **Credentials are blocked in nested payloads too.** The report is returned as nested
  structures, so the template scans them recursively with the framework's own detector and fails
  closed on anything it cannot traverse — a value whose type the walker does not understand is
  treated as unproven, not as safe.

It does not browse arbitrary endpoints, and it does not guarantee that its findings are complete,
accurate, or suitable for professional decisions.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Integration

Provide:

- an AgentCore runtime and a secret provider;
- a language model — either injected as a framework `BaseLLM` through `config["llm"]`, or left to
  the built-in per-invocation builder, which resolves `LLM_API_KEY` from `ctx.secrets` and
  constructs the provider named by `llm_provider`. Only key-based providers can be built that way
  (`anthropic`, `openai`); anything else injects its own client;
- **a real corpus or connector.** The shipped `config/research_corpus.sample.json` is a
  smoke-test fixture, not a data source. A deployment injects a connector-backed research service
  through `config["research_service"]`, or replaces that path with a governed corpus;
- caller authentication in front of the entry adapter.

Runtime parameters read from `config/config.yaml` (defaults in parentheses):

| Key | Meaning |
|---|---|
| `max_retry` (`3`) | Retries when retrieval yields nothing |
| `timeout_seconds` (`30`) | Execution timeout |
| `research_corpus_path` | Sample corpus used when no research service is injected |
| `research_corpus_top_k` (`3`) | Documents taken per source from that corpus |
| `llm_provider` (`anthropic`) | Provider built per invocation when `config["llm"]` is absent |
| `llm_model` / `llm_temperature` / `llm_max_tokens` (unset) | Passed to that provider client. Left unset, the client's own defaults apply — pin the model for a deployment, since a wheel upgrade can otherwise move it |
| `memory_enabled` (`true`), `hitl.enabled` (`true`), `hitl.max_hitl` (`8`), `hitl_enabled` (`true`) | The plan gate. A checkpointer is required for pause/resume, so these stay on unless the plan gate is deliberately removed |
| `configurable.allowed_sources` | The source catalog — `internal_kb`, `arxiv`, `pubmed`, `news` by default, each with a type and a tier |

A source outside the catalog cannot be requested at run time. The template does not fetch
arbitrary URLs, does not maintain its own index, and does not rank or de-duplicate across
retrieval runs.

Secrets, declared in `config/agent.yaml` and resolved through the platform's secret provider:
`LLM_API_KEY`, `ARXIV_API_KEY`, `PUBMED_API_KEY`. None of them is read from the process
environment by the agent, and none is written to state.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent is designed to run on the platform, which supplies the entry adapter, the secret provider and — in a real deployment — the durable checkpointer the plan gate resumes from. Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Supplied by the platform's own package registry at build time. It is not published on public PyPI, so `pip install -e .` alone does not fetch it. The `anthropic` extra is required for the default provider. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run on AGENTIC STAR, and the agent does not fall back to a degraded
mode. The standalone server in `src/api/server.py` does start without the platform — it builds
the graph, attaches an **in-memory** checkpointer and installs the secret provider — but two
things then differ from a real deployment. Secrets resolve only if the surrounding provider can
supply them, so an un-injected synthesis call fails when `LLM_API_KEY` is unavailable; and the
in-memory checkpointer means a paused plan gate does **not** survive a restart, so a suspended
run is lost. The test suite runs without a platform connection, using fakes for both.

## Input

`POST /invoke` takes a JSON body with `input` (the research question), `session_id` and
`input_context`. `input_context.source_spec` may pin `sources`, `quality_bar` and `depth`;
setting `input_context.plan_confirmed` to `true` skips the human plan gate.

Callers must be authenticated. The standalone entry adapter starts every request at `ANONYMOUS`
unless upstream middleware stamps a verified trust level; when `INVOKE_AUTH_TOKEN` is set on the
server environment, a matching `Authorization: Bearer` header promotes the caller to
`VERIFIED_EXTERNAL`. A caller that is neither middleware-verified nor token-authenticated is
rejected with **401** — there is no anonymous path into the graph.

```bash
curl -X POST http://127.0.0.1:8000/invoke \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${INVOKE_AUTH_TOKEN}" \
  -d '{
    "input": "What is X?",
    "session_id": "research-demo-001",
    "input_context": {
      "plan_confirmed": true,
      "source_spec": {
        "sources": ["arxiv"],
        "quality_bar": "peer_reviewed_only",
        "depth": "standard"
      }
    }
  }'
```

An empty question, a question over 4,000 characters, a credential-shaped string, or a malformed
`source_spec` is rejected before retrieval begins.

### The plan gate and `/resume`

When the plan is not pre-confirmed, `/invoke` pauses at the plan gate and returns
`status: awaiting_human` together with a `thread_id` and the proposed plan. A reviewer continues
the run by posting that `thread_id` back, **echoing the `session_id` from the paused response**:

```bash
curl -X POST http://127.0.0.1:8000/resume \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${INVOKE_AUTH_TOKEN}" \
  -d '{
    "thread_id": "<thread_id from the paused /invoke>",
    "session_id": "<session_id from the paused /invoke>",
    "feedback": "Go"
  }'
```

`feedback` is `"Go"` (or `{}`) to accept the plan as proposed, or `{"research_plan": {...}}` to
accept an edited one; an edited `quality_bar` is synchronised back into the fields retrieval
actually reads, so an edit is never silently ignored. `/resume` is authenticated exactly like
`/invoke` and additionally checks ownership: the thread must exist (**404** otherwise), its
recorded `session_id` must match (**403**), and it must still be waiting (**409** for a
completed or replayed resume).

The `session_id` is the ownership key. Under a single shared `INVOKE_AUTH_TOKEN` every token
caller has the same identity, so this defends against a caller who does not know the
`session_id` — not against one who does. A deployment that needs stronger separation supplies
per-caller identity through middleware.

## Output

The standard AgentCore envelope is preserved. Alongside `output`, `status`, `trace_id`,
`correlation_id`, `session_id`, `node_history` and `error_log`, the response carries
`research_question`, `research_plan`, `plan_confirmed`, `quality_bar`, `source_whitelist`,
`retrieved_evidence`, `synthesized_findings`, `identified_gaps`, `hitl_required`, `hitl_payload`,
`evidence_report`, `formatted_output` and `error`.

```json
{
  "status": "success",
  "research_question": "What is X?",
  "plan_confirmed": true,
  "source_whitelist": ["arxiv"],
  "retrieved_evidence": [
    {
      "evidence_id": "arxiv#1",
      "source_id": "arxiv",
      "source_type": "connector",
      "source_tier": "peer_reviewed",
      "title": "arxiv result for What is X?",
      "snippet": "evidence snippet",
      "url": "https://example.test/arxiv/1",
      "published_date": "2026-01-01",
      "score": 0.9
    }
  ],
  "synthesized_findings": [
    {"claim": "Answer to: What is X?", "supporting_source_ids": ["arxiv#1"], "confidence": 0.82}
  ],
  "evidence_report": {
    "sections": [{"heading": "Findings", "claims": [
      {"claim": "Answer to: What is X?", "supporting_source_ids": ["arxiv#1"], "confidence": 0.82}
    ]}],
    "citations": [
      {"evidence_id": "arxiv#1", "source_id": "arxiv", "source_tier": "peer_reviewed",
       "title": "arxiv result for What is X?", "url": "https://example.test/arxiv/1",
       "published_date": "2026-01-01"}
    ],
    "confidence_summary": {"mean": 0.82, "min": 0.82, "max": 0.82, "n": 1},
    "gaps": []
  },
  "identified_gaps": []
}
```

Values above are illustrative; real output depends on the question and the corpus supplied.

## Security and Limitations

Report text and retrieved snippets are scanned for credential patterns on the way out, including
inside the nested report payload — the run ends as an error rather than returning a masked
document, because the node that first produced the value owns that decision and a downstream mask
would never be reached. Provider exception text never reaches state: only the exception type name
is recorded, since a rejection message can quote the request, or the key used to sign it, back.

Known limits:

- **A citation proves attribution, not correctness.** Binding a claim to a document shows where
  it came from; it does not establish that the document supports it or that the document is
  right.
- **Retrieval is best-effort.** A source that fails transiently is recorded as a gap and the
  report is returned from what was retrieved — a thinner report, not an error.
- **`internal_kb` is in the default catalog.** A deployment that should not consult internal
  material must narrow `configurable.allowed_sources`.
- **The shipped corpus is a fixture.** Findings produced against
  `config/research_corpus.sample.json` demonstrate the pipeline; they are not research results.
- **A paused run is only as durable as the checkpointer.** The standalone server uses an
  in-memory saver; a restart discards suspended plan gates.

Output is machine-generated and assistive. It does not substitute for expert review, and any
decision that matters should be checked against the primary sources by a person.

## Release

- Version: 1.0.0
- Category: fixed multi-step domain workflow (Cat 2), with an inner workflow subgraph
- Generation mode: `llm`
- HITL: enabled — a plan gate that pauses via `interrupt()` and resumes through `/resume`
- Persistence: a checkpointer is required; the standalone server attaches an in-memory one
- External API clients: deployment-injected; the default provider builder covers key-based
  providers only

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection, using a fake language model and a fake research service
through the same `config["llm"]` and `config["research_service"]` seams production uses — there
is no test-only wiring, so a passing test exercises the production path.

## Documentation

`docs/02_design.md` carries the node flow, the state schema, the HITL plan-gate contract and the
security design. `docs/03_test_spec.md` records the test specification and the verification
results, including a live-model run and the known open items.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and test specifications
```

See `docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
