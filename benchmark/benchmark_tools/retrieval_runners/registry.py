from __future__ import annotations

from typing import Callable

from benchmark.benchmark_tools.retrieval_runners.base import RetrievalRunner
from benchmark.benchmark_tools.retrieval_runners.faiss_runner import FaissRetrievalRunner
from benchmark.benchmark_tools.retrieval_runners.graphrag_runner import GraphRagRetrievalRunner
from benchmark.benchmark_tools.retrieval_runners.lightrag_runner import LightRagRetrievalRunner


RunnerFactory = Callable[[dict], RetrievalRunner]


def get_runner(*, method_id: str, config: dict) -> RetrievalRunner:
    """Create a retrieval runner for the requested method identifier."""
    normalized_method_id = str(method_id).strip().lower()
    registry: dict[str, RunnerFactory] = {
        "faiss": lambda resolved_config: FaissRetrievalRunner(config=resolved_config),
        "graphrag": lambda resolved_config: GraphRagRetrievalRunner(config=resolved_config),
        "lightrag": lambda resolved_config: LightRagRetrievalRunner(config=resolved_config),
    }
    if normalized_method_id not in registry:
        raise ValueError(
            f"Unsupported retrieval method '{method_id}'. Supported: {sorted(registry)}"
        )
    return registry[normalized_method_id](dict(config))
