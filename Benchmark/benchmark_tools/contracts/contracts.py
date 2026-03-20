from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_BENCHMARK_TOOLS = ("ragas",)
DEFAULT_EVALUATION_MODEL = "gpt-4o-mini"


def validate_probe_selection_policy(
    *,
    include_auto: bool,
    auto_probe_count: int | None,
    include_verified: bool,
) -> dict[str, Any]:
    """Validate source-toggle policy for benchmark probe selection."""
    if not include_auto and not include_verified:
        raise ValueError("At least one probe source must be enabled.")

    normalized_auto_count = None
    if include_auto:
        if auto_probe_count is None:
            raise ValueError("auto_probe_count is required when include_auto is enabled.")
        try:
            normalized_auto_count = int(auto_probe_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid auto_probe_count '{auto_probe_count}'. Expected a positive integer."
            ) from exc
        if normalized_auto_count <= 0:
            raise ValueError(
                f"auto_probe_count must be positive when include_auto is enabled, got {normalized_auto_count}."
            )

    return {
        "include_auto_probes": bool(include_auto),
        "auto_probe_count": normalized_auto_count,
        "include_verified_probes": bool(include_verified),
    }


def validate_run_request(*, request: dict) -> dict[str, Any]:
    """Validate and normalize benchmark run request arguments."""
    embedded_chunks_path = request.get("embedded_chunks_path")
    if not str(embedded_chunks_path).strip():
        raise ValueError("embedded_chunks_path is required.")

    retrieval_model = str(request.get("retrieval_model", "")).strip()
    if not retrieval_model:
        raise ValueError("retrieval_model is required.")

    evaluation_model_raw = request.get("evaluation_model")
    evaluation_model = (
        str(evaluation_model_raw).strip() if evaluation_model_raw is not None else ""
    )
    if not evaluation_model:
        evaluation_model = DEFAULT_EVALUATION_MODEL

    try:
        max_cases = int(request.get("max_cases", 24))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid max_cases '{request.get('max_cases')}'. Expected a positive integer.") from exc
    if max_cases <= 0:
        raise ValueError(f"max_cases must be positive, got {max_cases}.")

    try:
        top_k = int(request.get("top_k", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid top_k '{request.get('top_k')}'. Expected a positive integer.") from exc
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    tools = request.get("tools")
    if tools is None:
        normalized_tools = list(SUPPORTED_BENCHMARK_TOOLS)
    else:
        if not isinstance(tools, list):
            raise ValueError("tools must be a list of benchmark tool names.")
        normalized_tools: list[str] = []
        seen_tools: set[str] = set()
        for raw_tool in tools:
            tool_name = str(raw_tool).strip().lower()
            if not tool_name or tool_name in seen_tools:
                continue
            normalized_tools.append(tool_name)
            seen_tools.add(tool_name)
        invalid = sorted(set(normalized_tools) - set(SUPPORTED_BENCHMARK_TOOLS))
        if invalid:
            raise ValueError(
                f"Unsupported benchmark tools {invalid}. Supported: {list(SUPPORTED_BENCHMARK_TOOLS)}"
            )

    policy = validate_probe_selection_policy(
        include_auto=bool(request.get("include_auto", True)),
        auto_probe_count=request.get("auto_probe_count", max_cases),
        include_verified=bool(request.get("include_verified", False)),
    )

    verified_questions_path = Path(str(request.get("verified_questions_path", "data/verified_questions.json")))

    return {
        "embedded_chunks_path": Path(str(embedded_chunks_path)),
        "retrieval_model": retrieval_model,
        "evaluation_model": evaluation_model,
        "max_cases": max_cases,
        "top_k": top_k,
        "tools": normalized_tools,
        "probe_selection_policy": policy,
        "verified_questions_path": verified_questions_path,
    }
