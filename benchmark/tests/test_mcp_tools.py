from __future__ import annotations

import pytest

from rag_benchmark_mcp.contracts import RetrieveEvidenceRequest, RunRetrievalBenchmarkRequest
from rag_benchmark_mcp.tools import build_tool_definitions, retrieve_evidence, run_retrieval_benchmark


def test_retrieve_evidence_returns_typed_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_benchmark_mcp.tools.load_knowledge_base",
        lambda **_: {
            "knowledge_base_dir": "/tmp/kb",
            "method_id": "faiss",
            "chunk_count": 12,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "warnings": ["Optional build metadata is missing; core retrieval artifacts are still valid."],
        },
    )

    class FakeCandidate:
        chunk_id = "paper_a_chunk_0001"
        score = 0.88
        rank = 1

    class FakeChunk:
        chunk_id = "paper_a_chunk_0001"
        paper_id = "paper_a"
        text = "Measured conductivity increased after annealing."
        index = 1

    monkeypatch.setattr(
        "rag_benchmark_mcp.tools.RetrievalService.retrieve_top_artifact",
        lambda self, *args, **kwargs: [FakeCandidate()],
    )
    monkeypatch.setattr(
        "rag_benchmark_mcp.tools.RetrievalService.load_chunks_for_candidates",
        lambda self, candidates: {"paper_a_chunk_0001": FakeChunk()},
    )

    result = retrieve_evidence(
        request=RetrieveEvidenceRequest(
            query="What evidence links annealing to conductivity?",
            knowledge_base_path="/tmp/kb",
            top_k=3,
        )
    )

    assert result.retrieval_method == "faiss"
    assert result.retrieval_provider == "openai"
    assert result.evidence[0].citation_label == "paper_a | chunk 1"
    assert "Optional build metadata is missing" in result.warnings[0]


def test_run_retrieval_benchmark_returns_compact_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_benchmark_mcp.tools.run_improved_benchmarks",
        lambda **_: {
            "retrieval_methods": ["faiss"],
            "probe_source_breakdown": {"auto_cases": 3, "verified_cases": 0, "total_cases": 3},
            "source_results": {
                "auto": {
                    "methods": {
                        "faiss": {
                            "baseline": {"num_cases": 3, "hit_at_1": 0.66},
                            "cases": [{"case_id": "case-1"}],
                            "tool_results": {"ragas": {"status": "completed"}},
                        }
                    }
                }
            },
            "jobs": [{"job_id": "auto:faiss", "status": "completed"}],
            "timing": {
                "run_id": "benchmark_123",
                "actual_total_seconds": 1.5,
                "estimate_seconds": {"expected_seconds": 2.0},
                "telemetry_file": "/tmp/telemetry.json",
            },
        },
    )

    result = run_retrieval_benchmark(
        request=RunRetrievalBenchmarkRequest(
            embedded_chunks_path="data/faiss_rag_index",
            retrieval_model="text-embedding-3-small",
        )
    )

    assert result.timing.run_id == "benchmark_123"
    assert result.primary_summary.baseline["num_cases"] == 3
    assert result.primary_summary.case_count == 1


def test_build_tool_definitions_exposes_expected_tools() -> None:
    definitions = build_tool_definitions()

    assert [definition["name"] for definition in definitions] == [
        "retrieve_evidence",
        "run_retrieval_benchmark",
    ]
    assert "properties" in definitions[0]["inputSchema"]


def test_build_http_app_exposes_healthcheck() -> None:
    pytest.importorskip("mcp")
    testclient = pytest.importorskip("starlette.testclient")

    from rag_benchmark_mcp.server import build_http_app

    with testclient.TestClient(build_http_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["server"] == "rag-benchmark-mcp"
