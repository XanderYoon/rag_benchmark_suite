from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from Benchmark.domain.enums import DifficultyLabel, QuestionStatus


UTC_NOW = lambda: datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Paper:
    paper_id: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    index: int


@dataclass
class EvidenceCandidate:
    chunk_id: str
    score: float
    rank: int


@dataclass
class Question:
    question_id: str
    paper_id: str
    question_text: str
    status: QuestionStatus = QuestionStatus.DRAFT

    @classmethod
    def create(cls, paper_id: str, question_text: str, question_id: str | None = None) -> "Question":
        return cls(
            question_id=question_id or str(uuid4()),
            paper_id=paper_id,
            question_text=question_text,
        )


@dataclass
class BenchmarkRecord:
    question_id: str
    paper_id: str
    question_text: str
    status: QuestionStatus = QuestionStatus.DRAFT
    target_difficulty: DifficultyLabel = DifficultyLabel.SINGLE_HOP
    difficulty_auto: DifficultyLabel = DifficultyLabel.SINGLE_HOP
    difficulty_final: DifficultyLabel = DifficultyLabel.SINGLE_HOP
    difficulty_notes: str | None = None
    retrieval_candidates: list[EvidenceCandidate] = field(default_factory=list)
    candidate_gold_chunk_ids: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=UTC_NOW)
    updated_at: str = field(default_factory=UTC_NOW)
    audit: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = UTC_NOW()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["target_difficulty"] = self.target_difficulty.value
        payload["difficulty_auto"] = self.difficulty_auto.value
        payload["difficulty_final"] = self.difficulty_final.value
        return payload

    @classmethod
    def from_question(cls, question: Question) -> "BenchmarkRecord":
        return cls(
            question_id=question.question_id,
            paper_id=question.paper_id,
            question_text=question.question_text,
        )
