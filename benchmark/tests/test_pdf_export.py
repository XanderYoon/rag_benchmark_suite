from __future__ import annotations

import pytest

from benchmark.benchmark_tools.reporting.pdf_report import build_pdf_report
from benchmark.benchmark_tools.reporting.report_view_model import build_report_model


def _sample_snapshot() -> dict:
    return {
        "embedded_chunks_path": "data/benchmark_runs/retrieval_indexes/example",
        "retrieval_model": "text-embedding-3-small",
        "evaluation_model": "gpt-4o-mini",
        "timing": {"actual_total_seconds": 1.23},
        "ui_run_signature": {
            "corpus": "data/rag_corpus_chunked",
            "embedding_model": "text-embedding-3-small",
            "retrieval_model": "text-embedding-3-small",
            "evaluation_model": "gpt-4o-mini",
            "max_cases": 24,
            "top_k": 5,
            "tools": ["ragas"],
        },
        "source_results": {
            "auto": {
                "methods": {
                    "faiss": {
                        "baseline": {"num_cases": 24, "hit_at_1": 0.5, "hit_at_3": 0.75, "mrr": 0.62},
                        "tool_results": {
                            "ragas": {
                                "status": "completed",
                                "summary": {"context_precision": 0.81, "faithfulness": 0.76},
                            }
                        },
                    }
                }
            }
        },
    }


def test_build_report_model_flattens_snapshot_sections() -> None:
    report_model = build_report_model(snapshot=_sample_snapshot())

    assert report_model["run_config"]["retrieval_model"] == "text-embedding-3-small"
    assert report_model["timing"]["actual_total_seconds"] == 1.23
    assert report_model["source_baselines"][0]["source"] == "auto"
    assert report_model["source_baselines"][0]["retrieval_method"] == "faiss"
    assert report_model["tool_summaries"][0]["tool"] == "ragas"
    assert report_model["tool_summaries"][0]["summary"]["context_precision"] == 0.81


def test_build_pdf_report_returns_pdf_bytes_with_sections() -> None:
    report_model = build_report_model(snapshot=_sample_snapshot())

    pdf_bytes = build_pdf_report(report_model=report_model)

    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"Run Configuration" in pdf_bytes
    assert b"Timing" in pdf_bytes
    assert b"Baselines by Source/Method" in pdf_bytes
    assert b"Tool Summaries" in pdf_bytes


def test_build_report_model_rejects_non_dict_snapshot() -> None:
    with pytest.raises(ValueError, match="expected a dictionary"):
        build_report_model(snapshot="invalid")  # type: ignore[arg-type]


def test_build_pdf_report_rejects_non_dict_report_model() -> None:
    with pytest.raises(ValueError, match="expected a dictionary"):
        build_pdf_report(report_model="invalid")  # type: ignore[arg-type]


def test_build_report_model_falls_back_to_legacy_baseline_and_tool_results() -> None:
    snapshot = {
        "embedded_chunks_path": "data/faiss_rag_index",
        "retrieval_model": "text-embedding-3-small",
        "baseline": {"num_cases": 2, "hit_at_1": 0.5},
        "tool_results": {
            "ragas": {
                "status": "completed",
                "summary": {"context_precision": 0.7},
            }
        },
    }

    report_model = build_report_model(snapshot=snapshot)

    assert report_model["source_baselines"][0]["source"] == "primary"
    assert report_model["source_baselines"][0]["baseline"]["num_cases"] == 2
    assert report_model["tool_summaries"][0]["tool"] == "ragas"
    assert report_model["tool_summaries"][0]["summary"]["context_precision"] == 0.7


def test_build_report_model_rejects_invalid_nested_source_results_shape() -> None:
    snapshot = {
        "retrieval_model": "text-embedding-3-small",
        "source_results": {"auto": {"methods": []}},
    }

    with pytest.raises(ValueError, match="snapshot.source_results\\['auto'\\]\\.methods"):
        build_report_model(snapshot=snapshot)
