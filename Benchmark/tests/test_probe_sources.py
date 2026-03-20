from __future__ import annotations

import json
from pathlib import Path

import pytest

from Benchmark.benchmark_tools.models import ChunkArtifact
from Benchmark.benchmark_tools.probe_sources.auto_source import load_auto_probes
from Benchmark.benchmark_tools.probe_sources.base import ProbeContext
from Benchmark.benchmark_tools.probe_sources.composer import build_probe_buckets
from Benchmark.benchmark_tools.probe_sources.verified_source import (
    count_verified_questions,
    load_verified_probes,
)


def _build_artifacts(count: int) -> list[ChunkArtifact]:
    return [
        ChunkArtifact(
            faiss_id=index,
            paper_id="paper-a",
            chunk_id=f"paper-a_chunk_{index:04d}",
            file_path=Path(__file__),
            text=f"Chunk text number {index}. This is benchmark content for retrieval testing.",
        )
        for index in range(count)
    ]


def test_load_auto_probes_returns_bounded_cases() -> None:
    probes = load_auto_probes(artifacts=_build_artifacts(10), max_cases=4)

    assert len(probes) == 4
    assert probes[0].expected_chunk_id == "paper-a_chunk_0000"
    assert probes[-1].expected_chunk_id == "paper-a_chunk_0006"
    assert probes[0].query.startswith("Retrieve the chunk")


def test_load_verified_probes_maps_rows_to_expected_probe_shape(tmp_path: Path) -> None:
    verified_path = tmp_path / "verified_questions.json"
    verified_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_000001",
                    "question_text": "What is the main claim?",
                    "ground_truth": "The main claim is X.",
                    "golden_chunk_ids": ["paper-a_chunk_0000", "paper-a_chunk_0001"],
                }
            ]
        ),
        encoding="utf-8",
    )

    probes = load_verified_probes(
        verified_path=verified_path,
        chunk_lookup={
            "paper-a_chunk_0000": "Chunk zero text.",
            "paper-a_chunk_0001": "Chunk one text.",
        },
    )

    assert len(probes) == 1
    assert probes[0].case_id == "verified_probe_q_000001"
    assert probes[0].query == "What is the main claim?"
    assert probes[0].expected_chunk_id == "paper-a_chunk_0000"
    assert probes[0].expected_answer == "The main claim is X."
    assert probes[0].reference_contexts == ["Chunk zero text.", "Chunk one text."]


def test_load_verified_probes_raises_for_unknown_chunk_id(tmp_path: Path) -> None:
    verified_path = tmp_path / "verified_questions.json"
    verified_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_000002",
                    "question_text": "Question text",
                    "ground_truth": "Answer text",
                    "golden_chunk_ids": ["paper-a_chunk_9999"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown chunk id"):
        load_verified_probes(verified_path=verified_path, chunk_lookup={})


def test_load_verified_probes_skips_negative_null_rows(tmp_path: Path) -> None:
    verified_path = tmp_path / "verified_questions.json"
    verified_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_000001",
                    "question_text": "Who is the best superhero?",
                    "ground_truth": "Subjective question.",
                    "golden_chunk_ids": [],
                    "difficulty": "Negative / Null",
                },
                {
                    "question_id": "q_000002",
                    "question_text": "What is the main claim?",
                    "ground_truth": "The main claim is X.",
                    "golden_chunk_ids": ["paper-a_chunk_0000"],
                },
            ]
        ),
        encoding="utf-8",
    )

    probes = load_verified_probes(
        verified_path=verified_path,
        chunk_lookup={
            "paper-a_chunk_0000": "Chunk zero text.",
        },
    )

    assert len(probes) == 1
    assert probes[0].case_id == "verified_probe_q_000002"


def test_count_verified_questions_returns_row_count(tmp_path: Path) -> None:
    verified_path = tmp_path / "verified_questions.json"
    verified_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_000001",
                    "question_text": "Question one",
                    "ground_truth": "Answer one",
                    "golden_chunk_ids": ["paper-a_chunk_0000"],
                },
                {
                    "question_id": "q_000002",
                    "question_text": "Question two",
                    "ground_truth": "Answer two",
                    "golden_chunk_ids": ["paper-a_chunk_0001"],
                },
            ]
        ),
        encoding="utf-8",
    )

    assert count_verified_questions(verified_path=verified_path) == 2


def test_build_probe_buckets_returns_source_separated_lists(tmp_path: Path) -> None:
    artifacts = _build_artifacts(6)
    verified_path = tmp_path / "verified_questions.json"
    verified_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_000003",
                    "question_text": "Verified question",
                    "ground_truth": "Verified answer",
                    "golden_chunk_ids": ["paper-a_chunk_0002"],
                }
            ]
        ),
        encoding="utf-8",
    )
    context = ProbeContext(
        artifacts=artifacts,
        chunk_lookup={artifact.chunk_id: artifact.text for artifact in artifacts},
        verified_path=verified_path,
        policy={},
    )

    buckets = build_probe_buckets(
        policy={
            "include_auto_probes": True,
            "auto_probe_count": 3,
            "include_verified_probes": True,
        },
        context=context,
    )

    assert set(buckets.keys()) == {"auto", "verified"}
    assert len(buckets["auto"]) == 3
    assert len(buckets["verified"]) == 1
