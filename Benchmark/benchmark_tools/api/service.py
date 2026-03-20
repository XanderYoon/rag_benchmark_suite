from __future__ import annotations

from pathlib import Path
from typing import Any

from Benchmark.benchmark_tools.api.compat import to_legacy_result
from Benchmark.benchmark_tools.contracts.contracts import validate_run_request
from Benchmark.benchmark_tools.observability import estimate_runtime, load_recent_telemetry
from Benchmark.benchmark_tools.runtime import run_benchmark


def run_improved_benchmarks(*, request: dict) -> dict[str, Any]:
    """Run benchmarks and return the improved source-separated contract."""
    return run_benchmark(request=request)


def estimate_benchmark_runtime(*, request: dict) -> dict[str, float]:
    """Estimate benchmark runtime from request settings and recent telemetry."""
    normalized_request = validate_run_request(request=request)
    retrieval_methods = _normalize_retrieval_methods(raw_methods=request.get("retrieval_methods"))

    telemetry_output_dir = _resolve_telemetry_output_dir(raw_dir=request.get("telemetry_output_dir"))
    telemetry_history = load_recent_telemetry(output_dir=telemetry_output_dir, limit=50)

    estimated_verified_cases = request.get("estimated_verified_cases")
    if estimated_verified_cases is None and normalized_request["probe_selection_policy"]["include_verified_probes"]:
        estimated_verified_cases = normalized_request["max_cases"]

    return estimate_runtime(
        request={
            "include_auto": bool(normalized_request["probe_selection_policy"]["include_auto_probes"]),
            "include_verified": bool(normalized_request["probe_selection_policy"]["include_verified_probes"]),
            "auto_probe_count": normalized_request["probe_selection_policy"]["auto_probe_count"],
            "estimated_verified_cases": estimated_verified_cases,
            "retrieval_methods": retrieval_methods,
            "tools": list(normalized_request["tools"]),
        },
        history=telemetry_history,
    )


def run_retrieval_benchmarks(
    *,
    embedded_chunks_path: str | Path,
    retrieval_model: str,
    evaluation_model: str | None = None,
    max_cases: int = 24,
    top_k: int = 5,
    tools: list[str] | None = None,
    include_auto_probes: bool = True,
    include_verified_probes: bool = False,
    verified_questions_path: str | Path = Path("data/verified_questions.json"),
    retrieval_methods: list[str] | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run benchmarks and return payload compatible with legacy UI consumers."""
    improved_result = run_improved_benchmarks(
        request={
            "embedded_chunks_path": embedded_chunks_path,
            "retrieval_model": retrieval_model,
            "evaluation_model": evaluation_model,
            "max_cases": max_cases,
            "top_k": top_k,
            "tools": tools,
            "include_auto": include_auto_probes,
            "auto_probe_count": max_cases,
            "include_verified": include_verified_probes,
            "verified_questions_path": verified_questions_path,
            "retrieval_methods": retrieval_methods or ["faiss"],
            "max_workers": max_workers,
        }
    )
    return to_legacy_result(improved_result=improved_result)


def _normalize_retrieval_methods(*, raw_methods: Any) -> list[str]:
    """Normalize retrieval method ids and preserve first-seen order."""
    if raw_methods is None:
        return ["faiss"]
    if not isinstance(raw_methods, list):
        raise ValueError("Invalid retrieval_methods value: expected a list of method ids.")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_method in raw_methods:
        method = str(raw_method).strip().lower()
        if not method or method in seen:
            continue
        normalized.append(method)
        seen.add(method)

    if not normalized:
        raise ValueError("At least one retrieval method must be provided.")
    return normalized


def _resolve_telemetry_output_dir(*, raw_dir: Any) -> Path:
    """Resolve benchmark telemetry output directory with a stable project default."""
    if raw_dir is None:
        return Path("data/benchmark_runs/telemetry").resolve()
    return Path(str(raw_dir)).expanduser().resolve()
