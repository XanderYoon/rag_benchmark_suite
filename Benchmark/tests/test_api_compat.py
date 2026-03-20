from __future__ import annotations

from pathlib import Path

from Benchmark.benchmark_tools.api.compat import to_legacy_result
from Benchmark.benchmark_tools.api.service import (
    estimate_benchmark_runtime,
    run_improved_benchmarks,
    run_retrieval_benchmarks,
)


def test_to_legacy_result_uses_auto_source_and_first_retrieval_method() -> None:
    improved_result = {
        "source_results": {
            "verified": {
                "methods": {
                    "faiss": {
                        "baseline": {"num_cases": 2},
                        "cases": [{"case_id": "verified-case"}],
                        "tool_results": {"ragas": {"status": "completed"}},
                    }
                }
            },
            "auto": {
                "methods": {
                    "faiss": {
                        "baseline": {"num_cases": 4},
                        "cases": [{"case_id": "auto-case"}],
                        "tool_results": {"ragas": {"status": "completed"}},
                    }
                }
            },
        },
        "retrieval_methods": ["faiss", "lightrag"],
    }

    payload = to_legacy_result(improved_result=improved_result)

    assert payload["baseline"]["num_cases"] == 4
    assert payload["cases"][0]["case_id"] == "auto-case"
    assert "ragas" in payload["tool_results"]


def test_to_legacy_result_falls_back_to_first_available_source() -> None:
    improved_result = {
        "source_results": {
            "verified": {
                "methods": {
                    "faiss": {
                        "baseline": {"num_cases": 1},
                        "cases": [{"case_id": "verified-only"}],
                        "tool_results": {},
                    }
                }
            }
        },
        "retrieval_methods": ["faiss"],
    }

    payload = to_legacy_result(improved_result=improved_result)

    assert payload["baseline"]["num_cases"] == 1
    assert payload["cases"][0]["case_id"] == "verified-only"


def test_to_legacy_result_rejects_invalid_improved_shape() -> None:
    try:
        to_legacy_result(improved_result={"source_results": [], "retrieval_methods": ["faiss"]})  # type: ignore[arg-type]
    except ValueError as exc:
        assert "source_results" in str(exc)
        assert "dictionary" in str(exc).lower()
        return

    raise AssertionError("Expected ValueError for invalid improved result shape.")


def test_run_improved_benchmarks_delegates_to_orchestrator(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_benchmark(*, request: dict) -> dict:
        captured["request"] = request
        return {"source_results": {}, "retrieval_methods": ["faiss"]}

    monkeypatch.setattr("Benchmark.benchmark_tools.api.service.run_benchmark", fake_run_benchmark)
    payload = run_improved_benchmarks(request={"retrieval_model": "text-embedding-3-small"})

    assert payload["retrieval_methods"] == ["faiss"]
    assert captured["request"] == {"retrieval_model": "text-embedding-3-small"}


def test_run_retrieval_benchmarks_returns_legacy_shape(monkeypatch) -> None:
    def fake_run_benchmark(*, request: dict) -> dict:
        return {
            "source_results": {
                "auto": {
                    "methods": {
                        "faiss": {
                            "baseline": {"num_cases": 3},
                            "cases": [{"case_id": "auto-1"}],
                            "tool_results": {"ragas": {"status": "completed"}},
                        }
                    }
                }
            },
            "retrieval_methods": ["faiss"],
            "timing": {"actual_total_seconds": 0.1},
        }

    monkeypatch.setattr("Benchmark.benchmark_tools.api.service.run_benchmark", fake_run_benchmark)
    payload = run_retrieval_benchmarks(
        embedded_chunks_path="data/faiss_rag_index",
        retrieval_model="text-embedding-3-small",
        tools=["ragas"],
        include_auto_probes=True,
    )

    assert payload["baseline"]["num_cases"] == 3
    assert payload["cases"][0]["case_id"] == "auto-1"
    assert "ragas" in payload["tool_results"]
    assert "source_results" in payload


def test_estimate_benchmark_runtime_uses_validated_request_and_history(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.api.service.load_recent_telemetry",
        lambda **_: [{"timing": {"actual_total_seconds": 5.0}, "request_summary": {"total_cases": 10, "job_count": 1}}],
    )

    payload = estimate_benchmark_runtime(
        request={
            "embedded_chunks_path": "data/faiss_rag_index",
            "retrieval_model": "text-embedding-3-small",
            "tools": ["ragas"],
            "include_auto": True,
            "auto_probe_count": 10,
            "include_verified": False,
            "retrieval_methods": ["faiss"],
            "telemetry_output_dir": tmp_path,
        }
    )

    assert set(payload.keys()) == {"low_seconds", "expected_seconds", "high_seconds"}
    assert payload["expected_seconds"] > 0
