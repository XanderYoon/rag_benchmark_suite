from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_report_model(*, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build a stable report model from one benchmark snapshot."""
    if not isinstance(snapshot, dict):
        raise ValueError("Invalid snapshot: expected a dictionary benchmark snapshot.")

    signature = dict(snapshot.get("ui_run_signature", {}))
    run_config = {
        "embedded_chunks_path": snapshot.get("embedded_chunks_path"),
        "corpus": signature.get("corpus"),
        "embedding_model": signature.get("embedding_model"),
        "retrieval_model": snapshot.get("retrieval_model") or signature.get("retrieval_model"),
        "evaluation_model": snapshot.get("evaluation_model") or signature.get("evaluation_model"),
        "tools": signature.get("tools") or sorted(dict(snapshot.get("tool_results", {})).keys()),
        "max_cases": signature.get("max_cases"),
        "top_k": signature.get("top_k"),
        "retrieval_methods": list(snapshot.get("retrieval_methods", [])),
    }

    timing = dict(snapshot.get("timing", {}))
    source_baselines = _extract_source_baselines(snapshot=snapshot)
    tool_summaries = _extract_tool_summaries(snapshot=snapshot)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_config": run_config,
        "timing": timing,
        "source_baselines": source_baselines,
        "tool_summaries": tool_summaries,
    }


def _extract_source_baselines(*, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract baseline blocks separated by source and retrieval method."""
    source_results = _coerce_mapping(
        value=snapshot.get("source_results", {}),
        field_name="snapshot.source_results",
    )
    baseline_rows: list[dict[str, Any]] = []

    for source_name, source_bucket in source_results.items():
        source_bucket_dict = _coerce_mapping(
            value=source_bucket,
            field_name=f"snapshot.source_results[{source_name!r}]",
        )
        methods = _coerce_mapping(
            value=source_bucket_dict.get("methods", {}),
            field_name=f"snapshot.source_results[{source_name!r}].methods",
        )
        for method_name, method_result in methods.items():
            method_result_dict = _coerce_mapping(
                value=method_result,
                field_name=f"snapshot.source_results[{source_name!r}].methods[{method_name!r}]",
            )
            baseline_rows.append(
                {
                    "source": str(source_name),
                    "retrieval_method": str(method_name),
                    "baseline": _coerce_mapping(
                        value=method_result_dict.get("baseline", {}),
                        field_name=(
                            "snapshot.source_results"
                            f"[{source_name!r}].methods[{method_name!r}].baseline"
                        ),
                    ),
                }
            )

    if baseline_rows:
        return baseline_rows

    legacy_baseline = dict(snapshot.get("baseline", {}))
    if legacy_baseline:
        return [
            {
                "source": "primary",
                "retrieval_method": "faiss",
                "baseline": legacy_baseline,
            }
        ]
    return []


def _extract_tool_summaries(*, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool summaries separated by source and retrieval method."""
    source_results = _coerce_mapping(
        value=snapshot.get("source_results", {}),
        field_name="snapshot.source_results",
    )
    summary_rows: list[dict[str, Any]] = []

    for source_name, source_bucket in source_results.items():
        source_bucket_dict = _coerce_mapping(
            value=source_bucket,
            field_name=f"snapshot.source_results[{source_name!r}]",
        )
        methods = _coerce_mapping(
            value=source_bucket_dict.get("methods", {}),
            field_name=f"snapshot.source_results[{source_name!r}].methods",
        )
        for method_name, method_result in methods.items():
            method_result_dict = _coerce_mapping(
                value=method_result,
                field_name=f"snapshot.source_results[{source_name!r}].methods[{method_name!r}]",
            )
            tool_results = _coerce_mapping(
                value=method_result_dict.get("tool_results", {}),
                field_name=(
                    f"snapshot.source_results[{source_name!r}]"
                    f".methods[{method_name!r}].tool_results"
                ),
            )
            for tool_name, tool_result in tool_results.items():
                normalized_result = _coerce_mapping(
                    value=tool_result,
                    field_name=(
                        "snapshot.source_results"
                        f"[{source_name!r}].methods[{method_name!r}].tool_results[{tool_name!r}]"
                    ),
                )
                summary_rows.append(
                    {
                        "source": str(source_name),
                        "retrieval_method": str(method_name),
                        "tool": str(tool_name),
                        "status": str(normalized_result.get("status", "unknown")),
                        "summary": _numeric_summary(
                            summary=_coerce_mapping(
                                value=normalized_result.get("summary", {}),
                                field_name=(
                                    "snapshot.source_results"
                                    f"[{source_name!r}].methods[{method_name!r}]"
                                    f".tool_results[{tool_name!r}].summary"
                                ),
                            )
                        ),
                    }
                )

    if summary_rows:
        return summary_rows

    legacy_tool_results = _coerce_mapping(
        value=snapshot.get("tool_results", {}),
        field_name="snapshot.tool_results",
    )
    for tool_name, tool_result in legacy_tool_results.items():
        normalized_result = _coerce_mapping(
            value=tool_result,
            field_name=f"snapshot.tool_results[{tool_name!r}]",
        )
        summary_rows.append(
            {
                "source": "primary",
                "retrieval_method": "faiss",
                "tool": str(tool_name),
                "status": str(normalized_result.get("status", "unknown")),
                "summary": _numeric_summary(
                    summary=_coerce_mapping(
                        value=normalized_result.get("summary", {}),
                        field_name=f"snapshot.tool_results[{tool_name!r}].summary",
                    )
                ),
            }
        )
    return summary_rows


def _numeric_summary(*, summary: dict[str, Any]) -> dict[str, float]:
    """Keep only numeric metrics so report rendering remains stable."""
    normalized: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            normalized[str(key)] = float(value)
    return normalized


def _coerce_mapping(*, value: Any, field_name: str) -> dict[str, Any]:
    """Validate dictionary-like fields at report boundaries with actionable errors."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid field '{field_name}'. Expected a dictionary, got {type(value).__name__}.")
    return dict(value)
