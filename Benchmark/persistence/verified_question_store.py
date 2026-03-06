from __future__ import annotations

import json
import re
from pathlib import Path

from Benchmark.domain.models import BenchmarkRecord


class VerifiedQuestionStore:
    def __init__(self, output_path: Path = Path("data/verified_questions.json")) -> None:
        self.output_path = output_path

    def append_verified(
        self,
        record: BenchmarkRecord,
        notes: str,
        ground_truth: str,
        difficulty_label: str | None = None,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._read_all()
        source_paper_ids = [paper_id for paper_id in record.source_paper_ids if str(paper_id).strip()]
        if not source_paper_ids and str(record.paper_id).strip():
            source_paper_ids = [str(record.paper_id).strip()]
        rows.append(
            {
                "question_id": record.question_id,
                "question_text": record.question_text,
                "source_paper_ids": source_paper_ids,
                "ground_truth": ground_truth,
                "golden_chunk_ids": list(record.gold_chunk_ids),
                "top_k_chunk_ids": list(record.top_k_chunk_ids),
                "difficulty": difficulty_label or record.difficulty_final.value,
                "date_created": record.created_at,
                "notes": notes,
            }
        )
        self.output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _read_all(self) -> list[dict]:
        if not self.output_path.exists():
            return []
        raw = self.output_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            sanitized = re.sub(r",(\s*[\]}])", r"\1", raw)
            data = json.loads(sanitized)
        if isinstance(data, list):
            return data
        return []
