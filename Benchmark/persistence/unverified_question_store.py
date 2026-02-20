from __future__ import annotations

import json
from pathlib import Path

from Benchmark.domain.models import BenchmarkRecord


class UnverifiedQuestionStore:
    def __init__(self, output_path: Path = Path("data/unverified_questions.json")) -> None:
        self.output_path = output_path

    def append_accepted(self, record: BenchmarkRecord) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.read_all()
        if any(str(row.get("question_id")) == record.question_id for row in rows):
            return
        rows.append(
            {
                "question_id": record.question_id,
                "question_text": record.question_text,
                "paper_id": record.paper_id,
                "default_difficulty": record.target_difficulty.value,
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
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
