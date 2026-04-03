from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmark.benchmark_tools.models import BenchmarkProbe, RetrievalCaseResult


class RetrievalRunner(ABC):
    """Define the retrieval benchmarking contract for all retrieval methods."""

    @abstractmethod
    def benchmark(self, *, probes: list[BenchmarkProbe], top_k: int) -> list[RetrievalCaseResult]:
        """Run retrieval for probes and return normalized case results."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return retrieval-method capability metadata for UI/service layers."""


def validate_top_k(*, top_k: int) -> int:
    """Validate top-k for retrieval runners and return the normalized value."""
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"Invalid top_k value '{top_k}'. Expected a positive integer.")
    return top_k
