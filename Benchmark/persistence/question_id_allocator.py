from __future__ import annotations

import json
import re
from pathlib import Path


class QuestionIdAllocator:
    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = data_dir
        self.counter_path = data_dir / "question_id_counter.txt"
        self.verified_path = data_dir / "verified_questions.json"
        self.unverified_path = data_dir / "unverified_questions.json"
        self._current_id: int | None = None

    def next_id(self) -> str:
        if self._current_id is None:
            self._current_id = self._load_starting_id()
        self._current_id += 1
        self._persist_counter(self._current_id)
        return f"q_{self._current_id:06d}"

    def _load_starting_id(self) -> int:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.counter_path.exists():
            raw = self.counter_path.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                return int(raw)
        max_existing = max(
            self._max_id_from_json(self.verified_path),
            self._max_id_from_json(self.unverified_path),
        )
        self._persist_counter(max_existing)
        return max_existing

    def _persist_counter(self, value: int) -> None:
        self.counter_path.write_text(str(value), encoding="utf-8")

    def _max_id_from_json(self, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0
        if not isinstance(data, list):
            return 0

        max_id = 0
        for row in data:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("question_id", ""))
            value = self._parse_id(question_id)
            if value > max_id:
                max_id = value
        return max_id

    @staticmethod
    def _parse_id(question_id: str) -> int:
        match = re.search(r"(?:q_)?(\d+)$", question_id)
        if not match:
            return 0
        return int(match.group(1))
