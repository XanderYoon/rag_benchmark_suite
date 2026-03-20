from __future__ import annotations

from pathlib import Path

import pytest

from Benchmark.benchmark_tools.models import BenchmarkProbe
from Benchmark.benchmark_tools.observability.telemetry import (
    append_debug_event,
    append_run_telemetry,
    load_recent_telemetry,
)
from Benchmark.benchmark_tools.observability.timing import (
    capture_stage_duration,
    estimate_runtime,
)
from Benchmark.benchmark_tools.runtime import run_benchmark


def test_estimate_runtime_handles_empty_history() -> None:
    estimate = estimate_runtime(
        request={
            "include_auto": True,
            "include_verified": False,
            "auto_probe_count": 20,
            "retrieval_methods": ["faiss"],
            "tools": ["ragas"],
        },
        history=[],
    )

    assert estimate["expected_seconds"] == pytest.approx(2.4, abs=1e-9)
    assert estimate["low_seconds"] == pytest.approx(1.68, abs=1e-9)
    assert estimate["high_seconds"] == pytest.approx(3.6, abs=1e-9)


def test_estimate_runtime_uses_history_median() -> None:
    estimate = estimate_runtime(
        request={
            "include_auto": True,
            "include_verified": True,
            "auto_probe_count": 10,
            "retrieval_methods": ["faiss", "lightrag"],
            "tools": ["ragas"],
        },
        history=[
            {
                "timing": {"actual_total_seconds": 12.0},
                "request_summary": {"total_cases": 10, "job_count": 2, "tools_count": 1},
            },
            {
                "timing": {"actual_total_seconds": 8.0},
                "request_summary": {"total_cases": 8, "job_count": 2, "tools_count": 1},
            },
        ],
    )

    assert estimate["expected_seconds"] == pytest.approx(33.0, abs=1e-9)
    assert estimate["low_seconds"] == pytest.approx(23.1, abs=1e-9)
    assert estimate["high_seconds"] == pytest.approx(49.5, abs=1e-9)


def test_capture_stage_duration_returns_result_and_duration() -> None:
    result, duration = capture_stage_duration(stage_name="sample", fn=lambda: "ok")

    assert result == "ok"
    assert duration >= 0.0


def test_capture_stage_duration_wraps_stage_failure() -> None:
    with pytest.raises(RuntimeError, match="stage 'failing_stage'"):
        capture_stage_duration(
            stage_name="failing_stage",
            fn=lambda: (_ for _ in ()).throw(ValueError("boom")),
        )


def test_telemetry_append_and_load_recent_rows(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    append_run_telemetry(
        run_id="run_001",
        telemetry={"timing": {"actual_total_seconds": 1.0}},
        output_dir=telemetry_dir,
    )
    append_run_telemetry(
        run_id="run_002",
        telemetry={"timing": {"actual_total_seconds": 2.0}},
        output_dir=telemetry_dir,
    )
    append_run_telemetry(
        run_id="run_003",
        telemetry={"timing": {"actual_total_seconds": 3.0}},
        output_dir=telemetry_dir,
    )

    rows = load_recent_telemetry(output_dir=telemetry_dir, limit=2)

    assert len(rows) == 2
    assert rows[0]["run_id"] == "run_002"
    assert rows[1]["run_id"] == "run_003"


def test_append_run_telemetry_rejects_blank_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id is required"):
        append_run_telemetry(
            run_id=" ",
            telemetry={"timing": {"actual_total_seconds": 1.0}},
            output_dir=tmp_path,
        )


def test_load_recent_telemetry_rejects_non_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        load_recent_telemetry(output_dir=tmp_path, limit=0)


def test_load_recent_telemetry_skips_invalid_json_lines(tmp_path: Path) -> None:
    telemetry_file = tmp_path / "benchmark_telemetry.jsonl"
    telemetry_file.write_text('{"run_id":"run_001"}\nnot-json\n{"run_id":"run_002"}\n', encoding="utf-8")

    rows = load_recent_telemetry(output_dir=tmp_path, limit=10)

    assert [row["run_id"] for row in rows] == ["run_001", "run_002"]


def test_append_debug_event_writes_jsonl_row(tmp_path: Path) -> None:
    debug_file = append_debug_event(
        event_type="ragas_failed",
        payload={"evaluation_model": "qwen3:8b", "error_type": "TimeoutError"},
        output_dir=tmp_path,
    )

    assert debug_file.exists()
    rows = debug_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert '"event_type": "ragas_failed"' in rows[0]
    assert '"evaluation_model": "qwen3:8b"' in rows[0]


def test_run_benchmark_adds_timing_estimate_and_writes_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = BenchmarkProbe(
        case_id="probe_001",
        query="query",
        expected_chunk_id="chunk_001",
        expected_answer="answer",
        reference_contexts=["context"],
    )
    cases = [
        {
            "case_id": "probe_001",
            "query": "query",
            "expected_chunk_id": "chunk_001",
            "expected_answer": "answer",
            "reference_contexts": ["context"],
            "top_hit_chunk_id": "chunk_001",
            "hit_at_1": True,
            "hit_at_3": True,
            "reciprocal_rank": 1.0,
            "retrieved_chunks": [],
        }
    ]
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime.load_chunk_artifacts",
        lambda _: ([], {"metric": "cosine"}),
    )
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime.build_probe_buckets",
        lambda **_: {"auto": [probe]},
    )
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime._run_selected_benchmarks",
        lambda **_: [
            {
                "job_id": "auto:faiss",
                "order_index": 0,
                "source": "auto",
                "retrieval_method": "faiss",
                "status": "completed",
                "duration_seconds": 0.1,
                "baseline": {"num_cases": 1, "hit_at_1": 1.0},
                "cases": cases,
                "tool_results": {"ragas": {"status": "completed"}},
                "error": None,
            }
        ],
    )
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime.load_recent_telemetry",
        lambda **_: [],
    )

    payload = run_benchmark(
        request={
            "embedded_chunks_path": "data/faiss_rag_index",
            "retrieval_model": "text-embedding-3-small",
            "tools": ["ragas"],
            "include_auto": True,
            "auto_probe_count": 1,
            "include_verified": False,
            "retrieval_methods": ["faiss"],
            "max_workers": 1,
            "telemetry_output_dir": tmp_path,
        }
    )

    timing = payload["timing"]
    assert timing["actual_total_seconds"] >= 0.0
    assert set(timing["estimate_seconds"]) == {"low_seconds", "expected_seconds", "high_seconds"}
    assert set(timing["stages_seconds"]) == {
        "load_artifacts",
        "build_probe_buckets",
        "run_selected_benchmarks",
    }
    assert "run_id" in timing
    assert Path(timing["telemetry_file"]).exists()

    assert "source_results" in payload
    assert payload["source_results"]["auto"]["methods"]["faiss"]["baseline"]["num_cases"] == 1
    assert payload["source_results"]["auto"]["methods"]["faiss"]["cases"][0]["case_id"] == "probe_001"


def test_run_benchmark_sets_telemetry_error_when_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = BenchmarkProbe(
        case_id="probe_001",
        query="query",
        expected_chunk_id="chunk_001",
        expected_answer="answer",
        reference_contexts=["context"],
    )
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime.load_chunk_artifacts",
        lambda _: ([], {"metric": "cosine"}),
    )
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime.build_probe_buckets",
        lambda **_: {"auto": [probe]},
    )
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime._run_selected_benchmarks",
        lambda **_: [
            {
                "job_id": "auto:faiss",
                "order_index": 0,
                "source": "auto",
                "retrieval_method": "faiss",
                "status": "completed",
                "duration_seconds": 0.1,
                "baseline": {"num_cases": 1, "hit_at_1": 1.0},
                "cases": [],
                "tool_results": {},
                "error": None,
            }
        ],
    )
    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime.load_recent_telemetry",
        lambda **_: [],
    )

    def fail_append(**_: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(
        "Benchmark.benchmark_tools.runtime.append_run_telemetry",
        fail_append,
    )

    payload = run_benchmark(
        request={
            "embedded_chunks_path": "data/faiss_rag_index",
            "retrieval_model": "text-embedding-3-small",
            "tools": [],
            "include_auto": True,
            "auto_probe_count": 1,
            "include_verified": False,
            "retrieval_methods": ["faiss"],
            "max_workers": 1,
            "telemetry_output_dir": tmp_path,
        }
    )

    assert "telemetry_error" in payload["timing"]
    assert payload["timing"]["telemetry_error"].startswith("OSError:")
    assert "run_id" in payload["timing"]
