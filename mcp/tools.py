from __future__ import annotations

from typing import Any

from benchmark.benchmark_tools.api.compat import to_legacy_result
from benchmark.benchmark_tools.api.service import run_improved_benchmarks
from benchmark.config import DEFAULT_CONFIG
from benchmark.domain.models import Chunk
from benchmark.services.retrieval_service import RetrievalService
from mcp.contracts import (
    BenchmarkPrimarySummary,
    BenchmarkTimingSummary,
    EvidenceItem,
    KnowledgeBaseSummary,
    RetrieveEvidenceRequest,
    RetrieveEvidenceResult,
    RunRetrievalBenchmarkRequest,
    RunRetrievalBenchmarkResult,
)
from RAG.services.knowledge_base_service import load_knowledge_base


def retrieve_evidence(*, request: RetrieveEvidenceRequest) -> RetrieveEvidenceResult:
    """Retrieve top evidence chunks from a validated knowledge base."""
    loaded_knowledge_base = load_knowledge_base(knowledge_base_dir=request.knowledge_base_path)
    retrieval_method = request.retrieval_method or str(loaded_knowledge_base["method_id"])
    retrieval_model = request.retrieval_model or str(loaded_knowledge_base.get("embedding_model", "")).strip()
    if not retrieval_model:
        retrieval_model = DEFAULT_CONFIG.embedding_model

    retrieval_provider = request.retrieval_provider or str(
        loaded_knowledge_base.get("embedding_provider", "")
    ).strip().lower()
    if not retrieval_provider:
        retrieval_provider = RetrievalService._provider_for_embedding_model(retrieval_model)

    retrieval_service = RetrievalService(DEFAULT_CONFIG)
    candidates = retrieval_service.retrieve_top_artifact(
        request.query,
        retrieval_method=retrieval_method,
        limit=request.top_k,
        retrieval_model=retrieval_model,
        retrieval_provider=retrieval_provider,
        artifact_output_dir=loaded_knowledge_base["knowledge_base_dir"],
    )
    chunks_by_id = retrieval_service.load_chunks_for_candidates(candidates)
    evidence = [
        _build_evidence_item(candidate=candidate, chunk=chunks_by_id[candidate.chunk_id])
        for candidate in candidates
        if candidate.chunk_id in chunks_by_id
    ]

    warnings = [str(item) for item in loaded_knowledge_base.get("warnings", [])]
    if retrieval_service.retrieval_error:
        warnings.append(str(retrieval_service.retrieval_error))

    return RetrieveEvidenceResult(
        query=request.query,
        knowledge_base=KnowledgeBaseSummary(
            knowledge_base_path=str(loaded_knowledge_base["knowledge_base_dir"]),
            method_id=str(loaded_knowledge_base["method_id"]),
            chunk_count=int(loaded_knowledge_base["chunk_count"]),
            embedding_provider=str(loaded_knowledge_base.get("embedding_provider", "")).strip(),
            embedding_model=str(loaded_knowledge_base.get("embedding_model", "")).strip(),
            warnings=[str(item) for item in loaded_knowledge_base.get("warnings", [])],
        ),
        retrieval_method=retrieval_method,
        retrieval_model=retrieval_model,
        retrieval_provider=retrieval_provider,
        evidence=evidence,
        warnings=warnings,
    )


def run_retrieval_benchmark(*, request: RunRetrievalBenchmarkRequest) -> RunRetrievalBenchmarkResult:
    """Execute one retrieval benchmark run and return the normalized result."""
    benchmark_request = {
        "embedded_chunks_path": request.embedded_chunks_path,
        "retrieval_model": request.retrieval_model,
        "evaluation_model": request.evaluation_model,
        "max_cases": request.max_cases,
        "top_k": request.top_k,
        "tools": list(request.tools),
        "include_auto": request.include_auto_probes,
        "auto_probe_count": request.max_cases,
        "include_verified": request.include_verified_probes,
        "verified_questions_path": request.verified_questions_path,
        "retrieval_methods": list(request.retrieval_methods),
        "telemetry_output_dir": request.telemetry_output_dir,
    }
    improved_result = run_improved_benchmarks(request=benchmark_request)
    legacy_result = to_legacy_result(improved_result=improved_result)
    timing_payload = dict(improved_result.get("timing", {}))
    estimate_payload = timing_payload.get("estimate_seconds", {})

    return RunRetrievalBenchmarkResult(
        retrieval_methods=[str(item) for item in improved_result.get("retrieval_methods", [])],
        probe_source_breakdown={
            str(key): int(value)
            for key, value in dict(improved_result.get("probe_source_breakdown", {})).items()
        },
        primary_summary=BenchmarkPrimarySummary(
            baseline=dict(legacy_result.get("baseline", {})),
            tool_results=dict(legacy_result.get("tool_results", {})),
            case_count=len(list(legacy_result.get("cases", []))),
        ),
        timing=BenchmarkTimingSummary(
            run_id=_optional_str(timing_payload.get("run_id")),
            actual_total_seconds=_optional_float(timing_payload.get("actual_total_seconds")),
            expected_seconds=_optional_float(dict(estimate_payload).get("expected_seconds")),
            telemetry_file=_optional_str(timing_payload.get("telemetry_file")),
            telemetry_error=_optional_str(timing_payload.get("telemetry_error")),
        ),
        source_results=dict(improved_result.get("source_results", {})),
        jobs=list(improved_result.get("jobs", [])),
    )


def build_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON-schema tool metadata for MCP `tools/list` responses."""
    return [
        {
            "name": "retrieve_evidence",
            "description": (
                "Retrieve the highest-ranked evidence chunks from a validated knowledge base "
                "for a scientific or benchmark query."
            ),
            "inputSchema": RetrieveEvidenceRequest.model_json_schema(),
        },
        {
            "name": "run_retrieval_benchmark",
            "description": (
                "Run one retrieval benchmark configuration and return source-separated metrics, "
                "tool results, and timing metadata."
            ),
            "inputSchema": RunRetrievalBenchmarkRequest.model_json_schema(),
        },
    ]


def call_tool(*, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate one tool request, execute it, and return a serializable payload."""
    if tool_name == "retrieve_evidence":
        result = retrieve_evidence(request=RetrieveEvidenceRequest.model_validate(arguments))
        return result.model_dump(mode="json")
    if tool_name == "run_retrieval_benchmark":
        result = run_retrieval_benchmark(request=RunRetrievalBenchmarkRequest.model_validate(arguments))
        return result.model_dump(mode="json")
    raise ValueError(f"Unknown MCP tool '{tool_name}'. Supported: ['retrieve_evidence', 'run_retrieval_benchmark']")


def _build_evidence_item(*, candidate: Any, chunk: Chunk) -> EvidenceItem:
    """Convert one retrieval candidate and chunk into the MCP output shape."""
    return EvidenceItem(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        rank=int(candidate.rank),
        score=float(candidate.score),
        text=chunk.text,
        citation_label=f"{chunk.paper_id} | chunk {chunk.index}",
    )


def _optional_str(value: Any) -> str | None:
    """Return a normalized optional string for API payloads."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_float(value: Any) -> float | None:
    """Return a normalized optional float for API payloads."""
    if value is None:
        return None
    return float(value)
