import pytest

from Benchmark.benchmark_tools.retrieval_runners.faiss_runner import FaissRetrievalRunner
from Benchmark.benchmark_tools.retrieval_runners.graphrag_runner import GraphRagRetrievalRunner
from Benchmark.benchmark_tools.retrieval_runners.lightrag_runner import LightRagRetrievalRunner
from Benchmark.benchmark_tools.retrieval_runners.registry import get_runner


def test_get_runner_returns_faiss_runner() -> None:
    runner = get_runner(
        method_id="faiss",
        config={"embedded_chunks_path": "data/faiss_rag_index", "retrieval_model": "text-embedding-3-small"},
    )
    assert isinstance(runner, FaissRetrievalRunner)


def test_get_runner_returns_graph_runner() -> None:
    runner = get_runner(method_id="graphrag", config={})
    assert isinstance(runner, GraphRagRetrievalRunner)


def test_get_runner_returns_light_runner() -> None:
    runner = get_runner(method_id="lightrag", config={})
    assert isinstance(runner, LightRagRetrievalRunner)


def test_get_runner_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        get_runner(method_id="unknown", config={})


def test_runner_capabilities_expose_required_metadata() -> None:
    faiss_caps = FaissRetrievalRunner(config={}).capabilities()
    graph_caps = GraphRagRetrievalRunner(config={}).capabilities()
    light_caps = LightRagRetrievalRunner(config={}).capabilities()

    assert faiss_caps["method_id"] == "faiss"
    assert graph_caps["method_id"] == "graphrag"
    assert light_caps["method_id"] == "lightrag"
    assert "required_config_fields" in faiss_caps
    assert "supports_chunk_scores" in graph_caps
    assert "supports_chunk_ranks" in light_caps
