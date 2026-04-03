from __future__ import annotations

import json
from pathlib import Path

from benchmark.benchmark_tools.models import BenchmarkProbe
from benchmark.benchmark_tools.probe_sources.base import ProbeContext, ProbeSource


def count_verified_questions(*, verified_path: Path) -> int:
    """Return the number of verified-question rows available on disk."""
    return len(_read_verified_rows(verified_path=verified_path))


def load_verified_probes(*, verified_path: Path, chunk_lookup: dict[str, str]) -> list[BenchmarkProbe]:
    """Load probes from verified questions with strict row validation."""
    rows = _read_verified_rows(verified_path=verified_path)
    probes: list[BenchmarkProbe] = []
    for index, row in enumerate(rows, start=1):
        if _is_negative_null_row(row=row):
            continue

        question_text = str(row.get("question_text", "")).strip()
        if not question_text:
            raise ValueError(
                f"Verified question row {index} is missing a non-empty 'question_text' field."
            )

        ground_truth = str(row.get("ground_truth", "")).strip()
        if not ground_truth:
            raise ValueError(
                f"Verified question row {index} is missing a non-empty 'ground_truth' field."
            )

        golden_chunk_ids = row.get("golden_chunk_ids")
        if not isinstance(golden_chunk_ids, list) or not golden_chunk_ids:
            raise ValueError(
                f"Verified question row {index} must include a non-empty 'golden_chunk_ids' list."
            )

        resolved_chunk_ids: list[str] = []
        reference_contexts: list[str] = []
        for chunk_id in golden_chunk_ids:
            chunk_id_value = str(chunk_id).strip()
            if not chunk_id_value:
                raise ValueError(
                    f"Verified question row {index} includes an empty chunk id in 'golden_chunk_ids'."
                )
            if chunk_id_value not in chunk_lookup:
                raise ValueError(
                    f"Verified question row {index} references unknown chunk id '{chunk_id_value}'."
                )
            resolved_chunk_ids.append(chunk_id_value)
            reference_contexts.append(chunk_lookup[chunk_id_value])

        question_id = str(row.get("question_id", "")).strip()
        case_suffix = question_id or f"{index:03d}"
        probes.append(
            BenchmarkProbe(
                case_id=f"verified_probe_{case_suffix}",
                query=question_text,
                expected_chunk_id=resolved_chunk_ids[0],
                expected_answer=ground_truth,
                reference_contexts=reference_contexts,
            )
        )
    return probes


def _is_negative_null_row(*, row: dict) -> bool:
    """Return True when a verified row is intentionally ungrounded and should be skipped."""
    difficulty = str(row.get("difficulty", "")).strip().lower()
    golden_chunk_ids = row.get("golden_chunk_ids")
    has_golden_ids = isinstance(golden_chunk_ids, list) and len(golden_chunk_ids) > 0
    return "negative / null" in difficulty and not has_golden_ids


class VerifiedQuestionProbeSource(ProbeSource):
    """Load probes from the persisted verified questions dataset."""

    def load_probes(self, *, context: ProbeContext) -> list[BenchmarkProbe]:
        return load_verified_probes(
            verified_path=context.verified_path,
            chunk_lookup=context.chunk_lookup,
        )


def _read_verified_rows(*, verified_path: Path) -> list[dict]:
    """Read and validate verified question rows from disk."""
    if not verified_path.exists():
        raise FileNotFoundError(f"Verified question dataset not found at {verified_path}.")

    raw = verified_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Verified question dataset is empty at {verified_path}.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse verified question dataset at {verified_path}.") from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"Verified question dataset must be a list of rows at {verified_path}."
        )
    if not payload:
        raise ValueError(f"No verified questions found in {verified_path}.")
    return payload
