"""AgentCore Platform v1.0 — recursive S-3 credential scan for nested results.

Why this exists (verified against the installed `agenticstar-agentcore==1.0.0`):
`FunctionNode._security_gate_output()` iterates `result.items()` and does
`if not isinstance(value, str): continue` — it scans **top-level strings only**
and never descends into a list or dict
(`framework/nodes/function_node.py`). Later wheels recurse into
`dict`/`list`/`tuple`; this template does not depend on which behaviour the
installed wheel has.

This template returns its whole payload as nested structures —
`retrieved_evidence`, `synthesized_findings`, `identified_gaps`,
`evidence_report` — and surfaces them to the caller through
`merge_output()` -> `get_output()`. Under the 1.0.0 gate a credential in
retrieved source text, or in LLM-authored prose, reaches the caller unscanned:
the run is blocked on the flattened `formatted_output` string, but the nested
copies are already in state and are returned.

The detector itself is NOT re-implemented here — `detect_credentials` from the
substrate is reused. This module only supplies
the traversal the pinned gate is missing, with the same fail-closed BLOCK
posture the framework applies to top-level strings (S-2 masks, S-3 blocks).

Design rules, each of which exists because its opposite is a hole:

* **Fail closed on anything not traversable.** A value whose type this walker
  does not understand is *unproven*, not *safe*. A `set` is the concrete case:
  it is not JSON/msgpack-serialisable (so it already violates the State contract
  of the State contract), the walker cannot index it stably, and FastAPI's encoder
  happily renders it into the response as a list. Silently skipping it would let
  a credential straight through.
* **Scan dict KEYS as well as values.** Retrieval hits come from external
  connectors, which control both halves of any dict they return.
* **Never put a key's text in the error message.** The raised message is copied
  verbatim into `error_log` and the S-4 audit event by the framework, so echoing
  a credential-bearing key would re-open the very hole this closes. Dict entries
  are located by ordinal position instead.
* **Bound the walk by identity and by total nodes.** Depth alone does not stop a
  shared-reference graph from being re-walked exponentially.
"""

from __future__ import annotations

from typing import Any

from shared.security.credential_detector import detect_credentials

# Report payloads are shallow; an unbounded walk over caller-influenced data is
# its own denial-of-service surface. Exceeding either bound is an ERROR, not a
# pass — an unscanned subtree is exactly what this module exists to prevent.
_MAX_DEPTH = 32
_MAX_NODES = 20_000

# Leaf types that carry no text and therefore cannot carry a credential.
_SCALARS = (bool, int, float, type(None))


class NestedScanError(RuntimeError):
    """A nested result value could not be proven credential-free."""


def _scan(value: Any, path: str, depth: int, seen: set[int], budget: list[int]) -> None:
    """Depth-first walk. Raises NestedScanError on the first problem."""
    budget[0] -= 1
    if budget[0] < 0:
        raise NestedScanError(
            f"S-3 output gate: result is too large to scan safely (at {path}). Output blocked."
        )
    if depth > _MAX_DEPTH:
        raise NestedScanError(
            f"S-3 output gate: result nests deeper than {_MAX_DEPTH} levels (at {path}); "
            "the remainder cannot be scanned. Output blocked."
        )

    if isinstance(value, str):
        findings = detect_credentials(value)
        if findings:
            raise NestedScanError(
                f"S-3 output gate: credential pattern '{findings[0]['type']}' detected in "
                f"nested result[{path}]. Output blocked."
            )
        return
    if isinstance(value, _SCALARS):
        return

    marker = id(value)
    if marker in seen:
        # Already proven clean on this walk; re-walking a shared reference is
        # what turns a diamond-shaped payload into exponential work.
        return
    seen.add(marker)

    if isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            # Ordinal, never the key text — see the module docstring.
            entry = f"{path}" + "{" + f"{index}" + "}"
            _scan(key, f"{entry}<key>", depth + 1, seen, budget)
            _scan(item, entry, depth + 1, seen, budget)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan(item, f"{path}[{index}]", depth + 1, seen, budget)
        return

    raise NestedScanError(
        f"S-3 output gate: result[{path}] is a {type(value).__name__}, which is neither "
        "JSON-serialisable nor traversable, so it cannot be proven credential-free. "
        "Output blocked."
    )


def assert_no_nested_credentials(result: dict, node: str) -> dict:
    """S-3 extension: block a result whose NESTED values carry a credential.

    Top-level strings are deliberately skipped — the `@final` framework gate has
    already scanned those before this hook runs, so re-scanning them would only
    duplicate work and produce a competing error message.
    """
    seen: set[int] = set()
    budget = [_MAX_NODES]
    for key, value in result.items():
        if isinstance(value, str):
            continue
        try:
            _scan(value, key, 0, seen, budget)
        except NestedScanError as exc:
            raise NestedScanError(f"[{node}] {exc}") from None
    return result
