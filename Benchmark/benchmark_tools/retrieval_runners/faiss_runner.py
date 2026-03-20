from __future__ import annotations

from pathlib import Path
from typing import Any

from Benchmark.benchmark_tools.artifacts import FaissRetriever
from Benchmark.benchmark_tools.models import BenchmarkProbe, RetrievalCaseResult
from Benchmark.benchmark_tools.retrieval_runners.base import RetrievalRunner, validate_top_k


def benchmark_faiss(
    *,
    probes: list[BenchmarkProbe],
    config: dict[str, Any],
) -> list[RetrievalCaseResult]:
    """Run FAISS retrieval benchmarking with normalized retrieval case results."""
    embedded_chunks_path = config.get("embedded_chunks_path")
    retrieval_model = str(config.get("retrieval_model", "")).strip()
    top_k = validate_top_k(top_k=int(config.get("top_k", 5)))

    if embedded_chunks_path is None:
        raise ValueError("FAISS retrieval config is missing 'embedded_chunks_path'.")
    if not retrieval_model:
        raise ValueError("FAISS retrieval config is missing 'retrieval_model'.")

    retriever = FaissRetriever(
        embedded_chunks_path=Path(str(embedded_chunks_path)),
        retrieval_model=retrieval_model,
        top_k=top_k,
    )
    return retriever.benchmark(probes)


class FaissRetrievalRunner(RetrievalRunner):
    """Run benchmark retrieval against stored FAISS index artifacts."""

    def __init__(self, *, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def benchmark(self, *, probes: list[BenchmarkProbe], top_k: int) -> list[RetrievalCaseResult]:
        merged_config = dict(self._config)
        merged_config["top_k"] = validate_top_k(top_k=top_k)
        return benchmark_faiss(probes=probes, config=merged_config)

    def capabilities(self) -> dict[str, Any]:
        return {
            "method_id": "faiss",
            "supports_prebuilt_index": True,
            "requires_index_build": False,
            "supports_chunk_scores": True,
            "supports_chunk_ranks": True,
            "required_config_fields": ["embedded_chunks_path", "retrieval_model"],
            "optional_config_fields": ["top_k"],
        }
