from __future__ import annotations

from typing import Any


def to_legacy_result(*, improved_result: dict[str, Any]) -> dict[str, Any]:
    """Attach legacy top-level fields (`baseline`, `cases`, `tool_results`) to improved payloads."""
    if not isinstance(improved_result, dict):
        raise ValueError("Invalid improved benchmark result. Expected a dictionary payload.")

    raw_source_results = improved_result.get("source_results", {})
    if not isinstance(raw_source_results, dict):
        raise ValueError(
            "Invalid improved benchmark result: 'source_results' must be a dictionary keyed by source."
        )
    source_results = dict(raw_source_results)

    raw_retrieval_methods = improved_result.get("retrieval_methods", [])
    if not isinstance(raw_retrieval_methods, list):
        raise ValueError(
            "Invalid improved benchmark result: 'retrieval_methods' must be a list of method ids."
        )
    retrieval_methods = list(raw_retrieval_methods)

    primary_source = "auto" if "auto" in source_results else next(iter(source_results), None)
    primary_method = retrieval_methods[0] if retrieval_methods else None
    primary_method_result = (
        source_results.get(primary_source, {}).get("methods", {}).get(primary_method, {})
        if primary_source and primary_method
        else {}
    )

    legacy_payload = dict(improved_result)
    legacy_payload["baseline"] = dict(primary_method_result.get("baseline", {}))
    legacy_payload["cases"] = list(primary_method_result.get("cases", []))
    legacy_payload["tool_results"] = dict(primary_method_result.get("tool_results", {}))
    return legacy_payload
