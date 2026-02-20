from __future__ import annotations

from Benchmark.config import AppConfig
from Benchmark.domain.enums import QuestionStatus
from Benchmark.domain.models import BenchmarkRecord


class BenchmarkStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.run_dir = config.benchmark_runs_dir
        self.draft_path = self.run_dir / "benchmark_draft.jsonl"
        self.verified_path = self.run_dir / "benchmark_verified.jsonl"
        self.settings_path = self.run_dir / "settings.json"
        self._draft_rows: list[dict] = []
        self._verified_rows: list[dict] = []

    def append_draft(self, record: BenchmarkRecord) -> None:
        self._draft_rows.append(record.to_dict())

    def append_verified(self, record: BenchmarkRecord) -> None:
        if record.status != QuestionStatus.VERIFIED:
            return
        self._verified_rows.append(record.to_dict())
