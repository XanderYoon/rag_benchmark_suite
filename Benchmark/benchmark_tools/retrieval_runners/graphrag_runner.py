from __future__ import annotations

from typing import Any

from Benchmark.benchmark_tools.models import BenchmarkProbe, RetrievalCaseResult
from Benchmark.benchmark_tools.retrieval_runners.base import RetrievalRunner, validate_top_k


def benchmark_graphrag(
    *,
    probes: list[BenchmarkProbe],
    config: dict[str, Any],
) -> list[RetrievalCaseResult]:
    """Run GraphRAG retrieval benchmarking and return normalized case results."""
    _ = probes
    validate_top_k(top_k=int(config.get("top_k", 5)))
    raise NotImplementedError(
        "GraphRAG retrieval runner is not available yet. "
        "Install GraphRAG dependencies and implement graph adapter wiring."
    )


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
            "supports_prebuilt_index": False,
            "requires_index_build": True,
            "supports_chunk_scores": True,
            "supports_chunk_ranks": True,
            "required_config_fields": [],
            "optional_config_fields": ["top_k"],
            "status": "planned",
        }
