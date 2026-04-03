from __future__ import annotations

from typing import Any

from RAG.retrieval.index_runtime import GraphArtifactRetriever
from benchmark.benchmark_tools.models import BenchmarkProbe, RetrievalCaseResult, RetrievedChunk
from benchmark.benchmark_tools.retrieval_runners.base import RetrievalRunner, validate_top_k


def benchmark_graphrag(
    *,
    probes: list[BenchmarkProbe],
    config: dict[str, Any],
) -> list[RetrievalCaseResult]:
    """Run GraphRAG retrieval benchmarking and return normalized case results."""
    embedded_chunks_path = config.get("embedded_chunks_path")
    retrieval_model = str(config.get("retrieval_model", "")).strip()
    top_k = validate_top_k(top_k=int(config.get("top_k", 5)))

    if embedded_chunks_path is None:
        raise ValueError("GraphRAG retrieval config is missing 'embedded_chunks_path'.")
    if not retrieval_model:
        raise ValueError("GraphRAG retrieval config is missing 'retrieval_model'.")

    retriever = GraphArtifactRetriever(
        embedded_chunks_path=str(embedded_chunks_path),
        retrieval_model=retrieval_model,
        top_k=top_k,
        ollama_base_url=str(config.get("ollama_base_url", "")),
    )
    results: list[RetrievalCaseResult] = []
    for probe in probes:
        retrieved_chunks = retriever.retrieve(query=probe.query, limit=top_k)
        results.append(
            RetrievalCaseResult(
                case_id=probe.case_id,
                query=probe.query,
                expected_chunk_id=probe.expected_chunk_id,
                expected_answer=probe.expected_answer,
                reference_contexts=probe.reference_contexts,
                retrieved_chunks=[
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=chunk.score,
                        rank=chunk.rank,
                    )
                    for chunk in retrieved_chunks
                ],
            )
        )
    return results


class GraphRagRetrievalRunner(RetrievalRunner):
    """GraphRAG retrieval adapter with normalized benchmark contract."""

    def __init__(self, *, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def benchmark(self, *, probes: list[BenchmarkProbe], top_k: int) -> list[RetrievalCaseResult]:
        merged_config = dict(self._config)
        merged_config["top_k"] = validate_top_k(top_k=top_k)
        return benchmark_graphrag(probes=probes, config=merged_config)

    def capabilities(self) -> dict[str, Any]:
        return {
            "method_id": "graphrag",
            "supports_prebuilt_index": True,
            "requires_index_build": False,
            "supports_chunk_scores": True,
            "supports_chunk_ranks": True,
            "required_config_fields": ["embedded_chunks_path", "retrieval_model"],
            "optional_config_fields": ["top_k", "ollama_base_url"],
            "status": "implemented",
        }
