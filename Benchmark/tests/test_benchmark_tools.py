from __future__ import annotations

import sys
from types import ModuleType

import pytest

from Benchmark.benchmark_tools.artifacts import build_benchmark_probes, summarize_retrieval_results
from Benchmark.benchmark_tools.adapters import (
    _build_langchain_chat_model,
    _build_langchain_embeddings,
    _extract_ragas_case_scores,
    failed_tool_result,
)
from Benchmark.benchmark_tools.contracts.contracts import (
    SUPPORTED_BENCHMARK_TOOLS,
    validate_probe_selection_policy,
    validate_run_request,
)
from Benchmark.benchmark_tools.contracts.models import BenchmarkRunResult, serialize_run_result
from Benchmark.benchmark_tools.models import ChunkArtifact, RetrievalCaseResult, RetrievedChunk


def test_build_benchmark_probes_returns_bounded_cases() -> None:
    artifacts = [
        ChunkArtifact(
            faiss_id=index,
            paper_id="paper-a",
            chunk_id=f"paper-a_chunk_{index:04d}",
            file_path=__file__,
            text=f"Chunk text number {index}. This is benchmark content for retrieval testing.",
        )
        for index in range(10)
    ]

    probes = build_benchmark_probes(artifacts, max_cases=4)

    assert len(probes) == 4
    assert probes[0].expected_chunk_id == "paper-a_chunk_0000"
    assert probes[-1].expected_chunk_id == "paper-a_chunk_0006"
    assert probes[0].query.startswith("Retrieve the chunk")


def test_summarize_retrieval_results_calculates_hit_rates() -> None:
    results = [
        RetrievalCaseResult(
            case_id="probe_001",
            query="question one",
            expected_chunk_id="chunk_a",
            expected_answer="answer a",
            reference_contexts=["context a"],
            retrieved_chunks=[
                RetrievedChunk(chunk_id="chunk_a", text="context a", score=0.9, rank=1),
                RetrievedChunk(chunk_id="chunk_b", text="context b", score=0.8, rank=2),
            ],
        ),
        RetrievalCaseResult(
            case_id="probe_002",
            query="question two",
            expected_chunk_id="chunk_c",
            expected_answer="answer c",
            reference_contexts=["context c"],
            retrieved_chunks=[
                RetrievedChunk(chunk_id="chunk_b", text="context b", score=0.7, rank=1),
                RetrievedChunk(chunk_id="chunk_c", text="context c", score=0.6, rank=2),
            ],
        ),
    ]

    summary = summarize_retrieval_results(results)

    assert summary["num_cases"] == 2
    assert summary["hit_at_1"] == 0.5
    assert summary["hit_at_3"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["average_top_score"] == 0.8


def test_validate_run_request_applies_defaults_and_normalizes_tools() -> None:
    normalized = validate_run_request(
        request={
            "embedded_chunks_path": "data/faiss_rag_index",
            "retrieval_model": " text-embedding-3-small ",
            "tools": [" RAGAS ", "ragas"],
        }
    )

    assert str(normalized["embedded_chunks_path"]).endswith("data/faiss_rag_index")
    assert normalized["retrieval_model"] == "text-embedding-3-small"
    assert normalized["evaluation_model"] == "gpt-4o-mini"
    assert normalized["max_cases"] == 24
    assert normalized["top_k"] == 5
    assert normalized["tools"] == ["ragas"]
    assert normalized["probe_selection_policy"]["include_auto_probes"] is True
    assert normalized["probe_selection_policy"]["auto_probe_count"] == 24
    assert normalized["probe_selection_policy"]["include_verified_probes"] is False


def test_validate_run_request_rejects_unsupported_tools() -> None:
    try:
        validate_run_request(
            request={
                "embedded_chunks_path": "data/faiss_rag_index",
                "retrieval_model": "text-embedding-3-small",
                "tools": ["ragas", "unsupported_tool"],
            }
        )
    except ValueError as exc:
        assert "Unsupported benchmark tools" in str(exc)
        assert "supported" in str(exc).lower()
        for tool_name in SUPPORTED_BENCHMARK_TOOLS:
            assert tool_name in str(exc)
        return

    raise AssertionError("Expected validate_run_request to reject unsupported tools.")


def test_validate_probe_selection_policy_requires_one_enabled_source() -> None:
    try:
        validate_probe_selection_policy(
            include_auto=False,
            auto_probe_count=None,
            include_verified=False,
        )
    except ValueError as exc:
        assert "at least one source" in str(exc).lower()
        return

    raise AssertionError("Expected validate_probe_selection_policy to reject disabled sources.")


def test_serialize_run_result_returns_stable_keys() -> None:
    case_result = RetrievalCaseResult(
        case_id="probe_001",
        query="question one",
        expected_chunk_id="chunk_a",
        expected_answer="answer a",
        reference_contexts=["context a"],
        retrieved_chunks=[
            RetrievedChunk(chunk_id="chunk_a", text="context a", score=0.9, rank=1),
            RetrievedChunk(chunk_id="chunk_b", text="context b", score=0.8, rank=2),
        ],
    )
    result = BenchmarkRunResult(
        embedded_chunks_path="data/faiss_rag_index",
        retrieval_model="text-embedding-3-small",
        evaluation_model="gpt-4o-mini",
        index_manifest={"metric": "cosine"},
        baseline={"num_cases": 1, "hit_at_1": 1.0},
        cases=[case_result],
        tool_results={"ragas": {"status": "completed", "summary": {}, "details": {}}},
        probe_selection_policy={"include_auto": True, "auto_probe_count": 1, "include_verified": False},
    )

    payload = serialize_run_result(result=result)

    assert payload["embedded_chunks_path"] == "data/faiss_rag_index"
    assert payload["retrieval_model"] == "text-embedding-3-small"
    assert payload["evaluation_model"] == "gpt-4o-mini"
    assert payload["index_manifest"]["metric"] == "cosine"
    assert payload["baseline"]["num_cases"] == 1
    assert payload["cases"][0]["case_id"] == "probe_001"
    assert payload["cases"][0]["top_hit_chunk_id"] == "chunk_a"
    assert payload["probe_selection_policy"]["include_auto"] is True


def test_extract_ragas_case_scores_returns_metric_rows() -> None:
    class FakeFrame:
        def __init__(self) -> None:
            self.columns = [
                "context_precision",
                "faithfulness",
                "answer_relevancy",
            ]
            self.index = [0, 1]
            self.iloc = [
                {
                    "context_precision": 0.8,
                    "faithfulness": 0.7,
                    "answer_relevancy": 0.9,
                },
                {
                    "context_precision": 0.6,
                    "faithfulness": 0.5,
                    "answer_relevancy": 0.4,
                },
            ]

    class FakeScoreResult:
        def to_pandas(self) -> FakeFrame:
            return FakeFrame()

    results = [
        RetrievalCaseResult(
            case_id="probe_001",
            query="question one",
            expected_chunk_id="chunk_a",
            expected_answer="answer a",
            reference_contexts=["context a"],
            retrieved_chunks=[
                RetrievedChunk(chunk_id="chunk_a", text="context a", score=0.9, rank=1),
            ],
        ),
        RetrievalCaseResult(
            case_id="probe_002",
            query="question two",
            expected_chunk_id="chunk_b",
            expected_answer="answer b",
            reference_contexts=["context b"],
            retrieved_chunks=[
                RetrievedChunk(chunk_id="chunk_b", text="context b", score=0.8, rank=1),
            ],
        ),
    ]

    case_scores = _extract_ragas_case_scores(score_result=FakeScoreResult(), results=results)

    assert case_scores == [
        {
            "case_id": "probe_001",
            "context_precision": 0.8,
            "faithfulness": 0.7,
            "answer_relevancy": 0.9,
        },
        {
            "case_id": "probe_002",
            "context_precision": 0.6,
            "faithfulness": 0.5,
            "answer_relevancy": 0.4,
        },
    ]


def test_build_langchain_embeddings_uses_openai_for_openai_embedding_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("langchain_openai")

    class FakeOpenAIEmbeddings:
        def __init__(self, *, model: str) -> None:
            self.model = model

    fake_module.OpenAIEmbeddings = FakeOpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    embeddings = _build_langchain_embeddings("text-embedding-3-small")

    assert isinstance(embeddings, FakeOpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-small"


def test_build_langchain_embeddings_uses_ollama_for_local_embedding_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("langchain_ollama")

    class FakeOllamaEmbeddings:
        def __init__(self, *, model: str, base_url: str) -> None:
            self.model = model
            self.base_url = base_url

    fake_module.OllamaEmbeddings = FakeOllamaEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    embeddings = _build_langchain_embeddings("nomic-embed-text")

    assert isinstance(embeddings, FakeOllamaEmbeddings)
    assert embeddings.model == "nomic-embed-text"
    assert embeddings.base_url == "http://localhost:11434"


def test_build_langchain_chat_model_uses_ollama_for_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("langchain_ollama")

    class FakeChatOllama:
        def __init__(
            self,
            *,
            model: str,
            base_url: str,
            temperature: int,
            reasoning: bool | None = None,
        ) -> None:
            self.model = model
            self.base_url = base_url
            self.temperature = temperature
            self.reasoning = reasoning

    fake_module.ChatOllama = FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = _build_langchain_chat_model("qwen3:8b")

    assert isinstance(llm, FakeChatOllama)
    assert llm.model == "qwen3:8b"
    assert llm.base_url == "http://localhost:11434"
    assert llm.temperature == 0
    assert llm.reasoning is False


def test_failed_tool_result_includes_structured_debug_fields() -> None:
    try:
        raise TimeoutError("evaluation timed out")
    except TimeoutError as exc:
        payload = failed_tool_result("ragas", exc, details={"debug_log_file": "debug.jsonl"})

    assert payload["status"] == "failed"
    assert payload["details"]["error"] == "TimeoutError: evaluation timed out"
    assert payload["details"]["error_type"] == "TimeoutError"
    assert payload["details"]["error_message"] == "evaluation timed out"
    assert "TimeoutError: evaluation timed out" in payload["details"]["traceback"]
