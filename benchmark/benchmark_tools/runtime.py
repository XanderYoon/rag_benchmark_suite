from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmark.benchmark_tools.adapters import failed_tool_result, run_ragas_benchmark
from benchmark.benchmark_tools.artifacts import load_chunk_artifacts, summarize_retrieval_results
from benchmark.benchmark_tools.contracts.contracts import validate_run_request
from benchmark.benchmark_tools.models import BenchmarkProbe, RetrievalCaseResult
from benchmark.benchmark_tools.observability import (
    append_run_telemetry,
    capture_stage_duration,
    estimate_runtime,
    load_recent_telemetry,
)
from benchmark.benchmark_tools.probe_sources import ProbeContext, build_probe_buckets
from benchmark.benchmark_tools.retrieval_runners.registry import get_runner


def run_benchmark(*, request: dict) -> dict[str, Any]:
    """Run one benchmark configuration sequentially and return source-scoped results."""
    started = time.perf_counter()
    normalized_request = validate_run_request(request=request)
    retrieval_methods = _normalize_retrieval_methods(raw_methods=request.get("retrieval_methods"))
    retrieval_method = retrieval_methods[0]
    telemetry_output_dir = _resolve_telemetry_output_dir(raw_dir=request.get("telemetry_output_dir"))
    telemetry_history = load_recent_telemetry(output_dir=telemetry_output_dir, limit=50)
    runtime_estimate = estimate_runtime(
        request={
            "include_auto": bool(normalized_request["probe_selection_policy"]["include_auto_probes"]),
            "include_verified": bool(normalized_request["probe_selection_policy"]["include_verified_probes"]),
            "auto_probe_count": normalized_request["probe_selection_policy"]["auto_probe_count"],
            "retrieval_methods": [retrieval_method],
            "tools": list(normalized_request["tools"]),
        },
        history=telemetry_history,
    )
    run_id = f"benchmark_{uuid4().hex}"
    stage_timings: dict[str, float] = {}

    artifacts_manifest, stage_timings["load_artifacts"] = capture_stage_duration(
        stage_name="load_artifacts",
        fn=lambda: load_chunk_artifacts(normalized_request["embedded_chunks_path"]),
    )
    artifacts, manifest = artifacts_manifest
    chunk_lookup = {artifact.chunk_id: artifact.text for artifact in artifacts}
    probe_context = ProbeContext(
        artifacts=artifacts,
        chunk_lookup=chunk_lookup,
        verified_path=_resolve_verified_path(raw_path=normalized_request.get("verified_questions_path")),
        policy=dict(normalized_request["probe_selection_policy"]),
    )

    probe_buckets, stage_timings["build_probe_buckets"] = capture_stage_duration(
        stage_name="build_probe_buckets",
        fn=lambda: build_probe_buckets(
            policy=dict(normalized_request["probe_selection_policy"]),
            context=probe_context,
        ),
    )
    outputs, stage_timings["run_selected_benchmarks"] = capture_stage_duration(
        stage_name="run_selected_benchmarks",
        fn=lambda: _run_selected_benchmarks(
            probe_buckets=probe_buckets,
            retrieval_method=retrieval_method,
            request=normalized_request,
        ),
    )
    source_results = _assemble_source_results(outputs=outputs)
    total_duration = time.perf_counter() - started
    probe_source_breakdown = {
        "auto_cases": len(probe_buckets.get("auto", [])),
        "verified_cases": len(probe_buckets.get("verified", [])),
        "total_cases": sum(len(probes) for probes in probe_buckets.values()),
    }
    timing_payload = {
        "actual_total_seconds": max(total_duration, 0.0),
        "estimate_seconds": dict(runtime_estimate),
        "stages_seconds": dict(stage_timings),
    }

    payload = {
        "embedded_chunks_path": str(Path(str(normalized_request["embedded_chunks_path"]))),
        "retrieval_model": str(normalized_request["retrieval_model"]),
        "evaluation_model": normalized_request["evaluation_model"],
        "index_manifest": manifest,
        "probe_selection_policy": dict(normalized_request["probe_selection_policy"]),
        "probe_source_breakdown": probe_source_breakdown,
        "retrieval_methods": [retrieval_method],
        "source_results": source_results,
        "jobs": [_serialize_job_result(output=output) for output in outputs],
        "timing": timing_payload,
    }

    telemetry_payload = {
        "timing": dict(timing_payload),
        "probe_source_breakdown": dict(probe_source_breakdown),
        "request_summary": {
            "job_count": len(outputs),
            "total_cases": probe_source_breakdown["total_cases"],
            "tools_count": max(len(normalized_request["tools"]), 1),
            "retrieval_methods": [retrieval_method],
            "include_auto": bool(normalized_request["probe_selection_policy"]["include_auto_probes"]),
            "include_verified": bool(normalized_request["probe_selection_policy"]["include_verified_probes"]),
        },
        "status_summary": {
            "total_jobs": len(outputs),
            "failed_jobs": sum(
                1 for output in outputs if str(output.get("status", "")).strip().lower() != "completed"
            ),
        },
    }
    try:
        telemetry_file = append_run_telemetry(
            run_id=run_id,
            telemetry=telemetry_payload,
            output_dir=telemetry_output_dir,
        )
        payload["timing"]["telemetry_file"] = str(telemetry_file)
        payload["timing"]["run_id"] = run_id
    except Exception as exc:
        payload["timing"]["telemetry_error"] = f"{type(exc).__name__}: {exc}"
        payload["timing"]["run_id"] = run_id

    return payload


def _run_selected_benchmarks(
    *,
    probe_buckets: dict[str, list[BenchmarkProbe]],
    retrieval_method: str,
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run enabled probe sources sequentially for one selected retrieval method."""
    outputs: list[dict[str, Any]] = []
    ordered_sources = [source for source in ("auto", "verified") if source in probe_buckets]
    for source in probe_buckets:
        if source not in ordered_sources:
            ordered_sources.append(source)

    for order_index, source in enumerate(ordered_sources):
        probes = list(probe_buckets.get(source, []))
        if not probes:
            continue
        outputs.append(
            _run_benchmark_slice(
                source=source,
                retrieval_method=retrieval_method,
                probes=probes,
                request=request,
                order_index=order_index,
            )
        )
    if not outputs:
        raise ValueError("Benchmark run produced no executable source selections.")
    return outputs


def _run_benchmark_slice(
    *,
    source: str,
    retrieval_method: str,
    probes: list[BenchmarkProbe],
    request: dict[str, Any],
    order_index: int,
) -> dict[str, Any]:
    """Execute one source selection sequentially and normalize failures."""
    started = time.perf_counter()
    job_id = f"{source}:{retrieval_method}"
    try:
        runner = get_runner(
            method_id=retrieval_method,
            config={
                "embedded_chunks_path": request["embedded_chunks_path"],
                "retrieval_model": str(request["retrieval_model"]),
                "top_k": int(request["top_k"]),
            },
        )
        retrieval_results = runner.benchmark(probes=probes, top_k=int(request["top_k"]))
        tool_results = _run_tool_benchmarks(
            retrieval_results=retrieval_results,
            tools=list(request["tools"]),
            request=request,
            source=source,
            retrieval_method=retrieval_method,
        )
        return {
            "job_id": job_id,
            "order_index": order_index,
            "source": source,
            "retrieval_method": retrieval_method,
            "status": "completed",
            "duration_seconds": max(time.perf_counter() - started, 0.0),
            "baseline": summarize_retrieval_results(retrieval_results),
            "cases": [_serialize_case_result(case_result=case) for case in retrieval_results],
            "tool_results": tool_results,
            "error": None,
        }
    except Exception as exc:
        return {
            "job_id": job_id,
            "order_index": order_index,
            "source": source,
            "retrieval_method": retrieval_method,
            "status": "failed",
            "duration_seconds": max(time.perf_counter() - started, 0.0),
            "baseline": {},
            "cases": [],
            "tool_results": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _assemble_source_results(*, outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Assemble source-scoped results for the selected sequential benchmark run."""
    source_results: dict[str, dict[str, Any]] = {}
    for output in sorted(outputs, key=lambda item: int(item["order_index"])):
        source_bucket = source_results.setdefault(
            str(output["source"]),
            {"status": "completed", "methods": {}, "errors": [], "total_duration_seconds": 0.0},
        )
        source_bucket["total_duration_seconds"] = float(source_bucket["total_duration_seconds"]) + float(
            output["duration_seconds"]
        )
        source_bucket["methods"][str(output["retrieval_method"])] = {
            "status": output["status"],
            "baseline": dict(output["baseline"]),
            "cases": list(output["cases"]),
            "tool_results": dict(output["tool_results"]),
            "duration_seconds": float(output["duration_seconds"]),
            "error": output["error"],
        }
        if str(output["status"]).strip().lower() != "completed":
            source_bucket["status"] = "partial_failed"
            source_bucket["errors"].append(
                {
                    "job_id": output["job_id"],
                    "retrieval_method": output["retrieval_method"],
                    "error": output["error"] or "Unknown benchmark failure.",
                }
            )
    return source_results


def _run_tool_benchmarks(
    *,
    retrieval_results: list[RetrievalCaseResult],
    tools: list[str],
    request: dict[str, Any],
    source: str,
    retrieval_method: str,
) -> dict[str, dict[str, Any]]:
    """Run evaluator tool adapters sequentially in the selected tool order."""
    if not tools:
        return {}
    completed_by_tool: dict[str, dict[str, Any]] = {}
    for tool_name in tools:
        try:
            completed_by_tool[tool_name] = _run_single_tool(
                tool_name=tool_name,
                retrieval_results=retrieval_results,
                request=request,
                source=source,
                retrieval_method=retrieval_method,
            )
        except Exception as exc:
            completed_by_tool[tool_name] = failed_tool_result(tool_name=tool_name, exc=exc)
    return completed_by_tool


def _run_single_tool(
    *,
    tool_name: str,
    retrieval_results: list[RetrievalCaseResult],
    request: dict[str, Any],
    source: str,
    retrieval_method: str,
) -> dict[str, Any]:
    """Dispatch one benchmark tool adapter and normalize failures."""
    if tool_name == "ragas":
        return run_ragas_benchmark(
            results=retrieval_results,
            retrieval_model=str(request["retrieval_model"]),
            evaluation_model=request["evaluation_model"],
        )
    return failed_tool_result(
        tool_name=tool_name,
        exc=ValueError(f"Unsupported benchmark tool '{tool_name}'. Supported tools are ragas."),
    )


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


def _resolve_verified_path(*, raw_path: Any) -> Path:
    """Resolve verified question source path with a stable project default."""
    if raw_path is None:
        return Path("data/verified_questions.json").resolve()
    return Path(str(raw_path)).expanduser().resolve()


def _resolve_telemetry_output_dir(*, raw_dir: Any) -> Path:
    """Resolve benchmark telemetry output directory with a stable project default."""
    if raw_dir is None:
        return Path("data/benchmark_runs/telemetry").resolve()
    return Path(str(raw_dir)).expanduser().resolve()


def _serialize_case_result(*, case_result: RetrievalCaseResult) -> dict[str, Any]:
    """Serialize one retrieval case result into a JSON-safe payload."""
    return {
        "case_id": case_result.case_id,
        "query": case_result.query,
        "expected_chunk_id": case_result.expected_chunk_id,
        "expected_answer": case_result.expected_answer,
        "reference_contexts": list(case_result.reference_contexts),
        "top_hit_chunk_id": case_result.top_hit_chunk_id,
        "hit_at_1": case_result.hit_at_1,
        "hit_at_3": case_result.hit_at_3,
        "reciprocal_rank": case_result.reciprocal_rank,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "rank": chunk.rank,
                "text": chunk.text,
            }
            for chunk in case_result.retrieved_chunks
        ],
    }


def _serialize_job_result(*, output: dict[str, Any]) -> dict[str, Any]:
    """Serialize one sequential benchmark slice into a stable contract."""
    return {
        "job_id": output.get("job_id"),
        "order_index": output.get("order_index"),
        "source": output.get("source"),
        "retrieval_method": output.get("retrieval_method"),
        "status": output.get("status"),
        "duration_seconds": output.get("duration_seconds"),
        "baseline": dict(output.get("baseline", {})),
        "cases": list(output.get("cases", [])),
        "tool_results": dict(output.get("tool_results", {})),
        "error": output.get("error"),
    }
