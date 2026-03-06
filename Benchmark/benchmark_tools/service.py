from __future__ import annotations

from pathlib import Path
from typing import Any

from Benchmark.benchmark_tools.adapters import (
    run_deepeval_benchmark,
    run_langsmith_benchmark,
    run_ragas_benchmark,
)
from Benchmark.benchmark_tools.artifacts import (
    FaissRetriever,
    build_benchmark_probes,
    load_chunk_artifacts,
    summarize_retrieval_results,
)


SUPPORTED_BENCHMARK_TOOLS = ("ragas", "deepeval", "langsmith")


def run_retrieval_benchmarks(
    *,
    embedded_chunks_path: str | Path,
    retrieval_model: str,
    evaluation_model: str | None = None,
    max_cases: int = 24,
    top_k: int = 5,
    tools: list[str] | None = None,
) -> dict[str, Any]:
    """Benchmark a stored FAISS chunk index with optional third-party evaluators.

    Args:
        embedded_chunks_path: Path to the FAISS index directory or one of its files.
        retrieval_model: OpenAI embedding model used to embed probe queries.
        evaluation_model: Optional OpenAI chat model for evaluator SDKs that support it.
        max_cases: Maximum number of probe cases to derive from stored chunks.
        top_k: Number of retrieved chunks to retain per probe.
        tools: Optional subset of benchmark tools to run.

    Returns:
        Dictionary with stable keys for baseline retrieval metrics, per-case data,
        and per-tool benchmark results.

    Raises:
        FileNotFoundError: When required FAISS artifacts are missing.
        RuntimeError: When retrieval dependencies or API configuration are unavailable.
        ValueError: When arguments are invalid.
    """
    selected_tools = _normalize_tools(tools)
    artifacts, manifest = load_chunk_artifacts(embedded_chunks_path)
    probes = build_benchmark_probes(artifacts, max_cases=max_cases)

    retriever = FaissRetriever(
        embedded_chunks_path=embedded_chunks_path,
        retrieval_model=retrieval_model,
        top_k=top_k,
    )
    retrieval_results = retriever.benchmark(probes)

    tool_results: dict[str, dict[str, Any]] = {}
    for tool_name in selected_tools:
        if tool_name == "ragas":
            tool_results[tool_name] = run_ragas_benchmark(
                results=retrieval_results,
                retrieval_model=retrieval_model,
                evaluation_model=evaluation_model,
            )
        elif tool_name == "deepeval":
            tool_results[tool_name] = run_deepeval_benchmark(
                results=retrieval_results,
                evaluation_model=evaluation_model,
            )
        elif tool_name == "langsmith":
            tool_results[tool_name] = run_langsmith_benchmark(
                results=retrieval_results,
                experiment_name=f"retrieval-benchmark-{Path(embedded_chunks_path).stem}",
            )

    return {
        "embedded_chunks_path": str(Path(embedded_chunks_path)),
        "retrieval_model": retrieval_model,
        "evaluation_model": evaluation_model,
        "index_manifest": manifest,
        "baseline": summarize_retrieval_results(retrieval_results),
        "cases": [_serialize_case_result(result) for result in retrieval_results],
        "tool_results": tool_results,
    }


def _normalize_tools(tools: list[str] | None) -> list[str]:
    """Validate and normalize the requested benchmark tool list."""
    if tools is None:
        return list(SUPPORTED_BENCHMARK_TOOLS)

    normalized = [tool.strip().lower() for tool in tools if tool and tool.strip()]
    invalid = sorted(set(normalized) - set(SUPPORTED_BENCHMARK_TOOLS))
    if invalid:
        raise ValueError(
            f"Unsupported benchmark tools {invalid}. Supported: {list(SUPPORTED_BENCHMARK_TOOLS)}"
        )
    return normalized


def _serialize_case_result(result: Any) -> dict[str, Any]:
    """Serialize one retrieval benchmark case for UI consumption."""
    return {
        "case_id": result.case_id,
        "query": result.query,
        "expected_chunk_id": result.expected_chunk_id,
        "expected_answer": result.expected_answer,
        "top_hit_chunk_id": result.top_hit_chunk_id,
        "hit_at_1": result.hit_at_1,
        "hit_at_3": result.hit_at_3,
        "reciprocal_rank": result.reciprocal_rank,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "rank": chunk.rank,
                "text": chunk.text,
            }
            for chunk in result.retrieved_chunks
        ],
    }

