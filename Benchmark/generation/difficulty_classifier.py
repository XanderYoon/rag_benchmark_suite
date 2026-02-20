from __future__ import annotations

from Benchmark.domain.enums import DifficultyLabel


class DifficultyClassifier:
    def classify(self, question_text: str, candidate_gold_chunk_ids: list[str]) -> DifficultyLabel:
        q = question_text.lower()
        if "compare" in q or "difference" in q:
            return DifficultyLabel.COMPARISON
        if "not" in q or "except" in q:
            return DifficultyLabel.NEGATIVE
        if q.startswith("what is") or "define" in q:
            return DifficultyLabel.DEFINITION
        if len(candidate_gold_chunk_ids) >= 2:
            return DifficultyLabel.MULTI_HOP
        return DifficultyLabel.SINGLE_HOP
