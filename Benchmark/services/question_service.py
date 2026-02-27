from __future__ import annotations

from Benchmark.config import AppConfig
from Benchmark.domain.difficulty_profiles import DIFFICULTY_PROFILES as DEFAULT_DIFFICULTY_PROFILES
from Benchmark.domain.enums import DifficultyLabel
from Benchmark.domain.models import BenchmarkRecord, Chunk, Question
from Benchmark.generation.difficulty_classifier import DifficultyClassifier
from Benchmark.generation.evidence_proposer import EvidenceProposer
from Benchmark.generation.question_generator import QuestionGenerator
from Benchmark.persistence.question_id_allocator import QuestionIdAllocator
from Benchmark.services.retrieval_service import RetrievalService


class QuestionService:
    DIFFICULTY_PROFILES: list[tuple[str, str, DifficultyLabel]] = DEFAULT_DIFFICULTY_PROFILES

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.generator = QuestionGenerator(model=config.question_model)
        self.retrieval = RetrievalService(config)
        self.evidence = EvidenceProposer()
        self.difficulty = DifficultyClassifier()
        self.id_allocator = QuestionIdAllocator()

    def generate_records(self, paper_id: str, chunks: list[Chunk]) -> list[BenchmarkRecord]:
        """Generate benchmark records for all configured difficulty profiles."""
        records: list[BenchmarkRecord] = []
        for i, (profile_label, reference_type, target) in enumerate(self.DIFFICULTY_PROFILES):
            question_text = self.generator.generate_profile_question(
                paper_id=paper_id,
                chunks=chunks,
                reference_type=reference_type,
                hop_type=target,
                slot_index=i,
                avoid_questions=[record.question_text for record in records],
            )
            question = Question.create(
                paper_id=paper_id,
                question_text=question_text,
                question_id=self.id_allocator.next_id(),
            )
            record = BenchmarkRecord.from_question(question)
            record.target_difficulty = target
            record.audit["difficulty_profile"] = profile_label
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
        """Regenerate a single benchmark record for the selected slot."""
        profile_label, reference_type, inferred_target = self.profile_for_slot(slot_index)
        question_text = self.generator.generate_profile_question(
            paper_id=paper_id,
            chunks=chunks,
            reference_type=reference_type,
            hop_type=inferred_target,
            slot_index=slot_index,
            feedback=feedback,
            avoid_questions=avoid_questions,
        )
        question = Question.create(
            paper_id=paper_id,
            question_text=question_text,
            question_id=self.id_allocator.next_id(),
        )
        record = BenchmarkRecord.from_question(question)
        record.target_difficulty = inferred_target
        record.audit["difficulty_profile"] = profile_label
        record.retrieval_candidates = self.retrieval.retrieve_generous(record.question_text, chunks)
        max_candidates = 4 if inferred_target == DifficultyLabel.MULTI_HOP else 3
        record.candidate_gold_chunk_ids = self.evidence.propose(
            record.retrieval_candidates,
            max_candidates=max_candidates,
        )
        record.difficulty_auto = inferred_target
        record.difficulty_final = inferred_target
        return record

    def profile_for_slot(self, slot_index: int) -> tuple[str, str, DifficultyLabel]:
        """Return the configured profile tuple for a UI slot index."""
        if slot_index < 0:
            slot_index = 0
        return self.DIFFICULTY_PROFILES[slot_index % len(self.DIFFICULTY_PROFILES)]
