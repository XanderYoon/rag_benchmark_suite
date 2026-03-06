from __future__ import annotations

import json
import re
from pathlib import Path

from Benchmark.domain.models import BenchmarkRecord


class UnverifiedQuestionStore:
    def __init__(self, output_path: Path = Path("data/unverified_questions.json")) -> None:
        self.output_path = output_path

    @staticmethod
    def paper_ids_for_row(row: dict) -> list[str]:
        """Return normalized source paper ids from a stored unverified row."""
        paper_ids = row.get("paper_ids")
        if isinstance(paper_ids, list):
            return [str(paper_id) for paper_id in paper_ids if str(paper_id).strip()]

        legacy_paper_id = str(row.get("paper_id", "")).strip()
        if legacy_paper_id:
            return [legacy_paper_id]
        return []

    def append_accepted(self, record: BenchmarkRecord) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.read_all()
        if any(str(row.get("question_id")) == record.question_id for row in rows):
            return
        paper_ids = [paper_id for paper_id in record.source_paper_ids if str(paper_id).strip()]
        if not paper_ids and str(record.paper_id).strip():
            paper_ids = [str(record.paper_id).strip()]
        rows.append(
            {
                "question_id": record.question_id,
                "question_text": record.question_text,
                "paper_ids": paper_ids,
                "default_difficulty": str(
                    record.audit.get("difficulty_profile", record.target_difficulty.value)
                ),
            }
        )
        self.output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def remove_question(self, question_id: str) -> bool:
        rows = self.read_all()
        filtered = [row for row in rows if str(row.get("question_id")) != question_id]
        if len(filtered) == len(rows):
            return False
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
        return True

    def read_all(self) -> list[dict]:
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
