from benchmark.benchmark_tools.api.compat import to_legacy_result
from benchmark.benchmark_tools.api.service import (
    estimate_benchmark_runtime,
    run_improved_benchmarks,
    run_retrieval_benchmarks,
)

__all__ = [
    "estimate_benchmark_runtime",
    "run_improved_benchmarks",
    "run_retrieval_benchmarks",
    "to_legacy_result",
]
