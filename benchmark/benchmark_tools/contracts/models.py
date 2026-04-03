from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.benchmark_tools.models import RetrievalCaseResult


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Store the normalized benchmark run result payload."""

    embedded_chunks_path: str
    retrieval_model: str
    evaluation_model: str | None
    index_manifest: dict[str, Any]
    baseline: dict[str, float | int]
    cases: list[RetrievalCaseResult]
    tool_results: dict[str, dict[str, Any]]
    probe_selection_policy: dict[str, Any]
    probe_source_breakdown: dict[str, int] = field(default_factory=dict)


def serialize_run_result(*, result: BenchmarkRunResult) -> dict[str, Any]:
    """Serialize a benchmark run model into a JSON-safe dictionary."""
    return {
        "embedded_chunks_path": result.embedded_chunks_path,
        "retrieval_model": result.retrieval_model,
        "evaluation_model": result.evaluation_model,
        "index_manifest": result.index_manifest,
        "baseline": result.baseline,
        "cases": [_serialize_case_result(case) for case in result.cases],
        "tool_results": result.tool_results,
        "probe_selection_policy": dict(result.probe_selection_policy),
        "probe_source_breakdown": dict(result.probe_source_breakdown),
    }


def _serialize_case_result(case: RetrievalCaseResult) -> dict[str, Any]:
    """Serialize one retrieval benchmark case for UI consumption."""
    return {
        "case_id": case.case_id,
        "query": case.query,
        "expected_chunk_id": case.expected_chunk_id,
        "expected_answer": case.expected_answer,
        "top_hit_chunk_id": case.top_hit_chunk_id,
        "hit_at_1": case.hit_at_1,
        "hit_at_3": case.hit_at_3,
        "reciprocal_rank": case.reciprocal_rank,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "rank": chunk.rank,
                "text": chunk.text,
            }
            for chunk in case.retrieved_chunks
        ],
    }
