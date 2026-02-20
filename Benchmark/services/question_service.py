from __future__ import annotations

from Benchmark.config import AppConfig
from Benchmark.domain.enums import DifficultyLabel
from Benchmark.domain.models import BenchmarkRecord, Chunk, Question
from Benchmark.generation.difficulty_classifier import DifficultyClassifier
from Benchmark.generation.evidence_proposer import EvidenceProposer
from Benchmark.generation.question_generator import QuestionGenerator
from Benchmark.persistence.question_id_allocator import QuestionIdAllocator
from Benchmark.services.retrieval_service import RetrievalService


class QuestionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.generator = QuestionGenerator()
        self.retrieval = RetrievalService(config)
        self.evidence = EvidenceProposer()
        self.difficulty = DifficultyClassifier()
        self.id_allocator = QuestionIdAllocator()

    def generate_records(self, paper_id: str, chunks: list[Chunk]) -> list[BenchmarkRecord]:
        records: list[BenchmarkRecord] = []
        target_difficulties = self._target_difficulties()
        for i, target in enumerate(target_difficulties):
            q = self.generator.generate_one(
                paper_id=paper_id,
                chunks=chunks,
                target_difficulty=target,
                slot_index=i,
            )
            question = Question.create(
                paper_id=paper_id,
                question_text=q,
                question_id=self.id_allocator.next_id(),
            )
            record = BenchmarkRecord.from_question(question)
            record.target_difficulty = target
            record.retrieval_candidates = self.retrieval.retrieve_generous(record.question_text, chunks)
            record.candidate_gold_chunk_ids = self.evidence.propose(record.retrieval_candidates)
            record.difficulty_auto = target
            record.difficulty_final = record.difficulty_auto
            records.append(record)
        return records

    def regenerate_record(
        self,
        paper_id: str,
        chunks: list[Chunk],
        target_difficulty: DifficultyLabel,
        slot_index: int,
        feedback: str,
        avoid_questions: list[str],
    ) -> BenchmarkRecord:
        q = self.generator.generate_one(
            paper_id=paper_id,
            chunks=chunks,
            target_difficulty=target_difficulty,
            slot_index=slot_index,
            feedback=feedback,
            avoid_questions=avoid_questions,
        )
        question = Question.create(
            paper_id=paper_id,
            question_text=q,
            question_id=self.id_allocator.next_id(),
        )
        record = BenchmarkRecord.from_question(question)
        record.target_difficulty = target_difficulty
        record.retrieval_candidates = self.retrieval.retrieve_generous(record.question_text, chunks)
        max_candidates = 4 if target_difficulty == DifficultyLabel.MULTI_HOP else 3
        record.candidate_gold_chunk_ids = self.evidence.propose(record.retrieval_candidates, max_candidates=max_candidates)
        record.difficulty_auto = target_difficulty
        record.difficulty_final = target_difficulty
        return record

    def _target_difficulties(self) -> list[DifficultyLabel]:
        # Requirement: exactly 2 single-hop and 1 multi-hop by default.
        return [
            DifficultyLabel.SINGLE_HOP,
            DifficultyLabel.SINGLE_HOP,
            DifficultyLabel.MULTI_HOP,
        ]
