from __future__ import annotations

import re

from Benchmark.domain.enums import DifficultyLabel
from Benchmark.domain.models import Chunk


class QuestionGenerator:
    def generate_one(
        self,
        paper_id: str,
        chunks: list[Chunk],
        target_difficulty: DifficultyLabel,
        slot_index: int,
        feedback: str = "",
        avoid_questions: list[str] | None = None,
    ) -> str:
        if not chunks:
            return f"What key claim is made in {paper_id}?"

        avoid_questions = avoid_questions or []
        feedback_text = feedback.strip()

        offset = slot_index
        if feedback_text:
            # Use feedback signal to move away from the original seed window.
            offset += sum(ord(c) for c in feedback_text) % max(len(chunks), 1)

        seed_a = self._snippet(chunks[offset % len(chunks)].text)
        seed_b = self._snippet(chunks[(offset + max(len(chunks) // 2, 1)) % len(chunks)].text)

        if target_difficulty == DifficultyLabel.MULTI_HOP:
            prompt = self._build_multi_hop(paper_id, seed_a, seed_b, feedback_text)
        else:
            prompt = self._build_single_hop(paper_id, seed_a, feedback_text)

        # Retry with alternate slices until we get a question sufficiently distinct
        # from the previously rejected candidates.
        if self._is_too_similar(prompt, avoid_questions):
            for step in range(1, min(len(chunks), 8)):
                alt_a = self._snippet(chunks[(offset + step) % len(chunks)].text)
                alt_b = self._snippet(chunks[(offset + step + max(len(chunks) // 2, 1)) % len(chunks)].text)
                if target_difficulty == DifficultyLabel.MULTI_HOP:
                    alt_prompt = self._build_multi_hop(paper_id, alt_a, alt_b, feedback_text)
                else:
                    alt_prompt = self._build_single_hop(paper_id, alt_a, feedback_text)
                if not self._is_too_similar(alt_prompt, avoid_questions):
                    prompt = alt_prompt
                    break

        return prompt

    @staticmethod
    def _snippet(text: str, width: int = 220) -> str:
        return " ".join(text.strip().split())[:width]

    @staticmethod
    def _build_single_hop(paper_id: str, seed: str, feedback: str) -> str:
        if feedback:
            return (
                f"In {paper_id}, generate a question about this passage with a different angle: '{seed}'. "
                f"Address this reviewer guidance: {feedback}."
            )
        return f"What does {paper_id} state about: {seed}?"

    @staticmethod
    def _build_multi_hop(paper_id: str, seed_a: str, seed_b: str, feedback: str) -> str:
        if feedback:
            return (
                f"In {paper_id}, generate a multi-hop question linking '{seed_a}' and '{seed_b}' from a different angle. "
                f"Address this reviewer guidance: {feedback}."
            )
        return f"In {paper_id}, how do these two parts connect: '{seed_a}' and '{seed_b}'?"

    @staticmethod
    def _is_too_similar(candidate: str, previous_questions: list[str]) -> bool:
        candidate_tokens = QuestionGenerator._token_set(candidate)
        if not candidate_tokens:
            return False
        for previous in previous_questions:
            prev_tokens = QuestionGenerator._token_set(previous)
            if not prev_tokens:
                continue
            overlap = len(candidate_tokens & prev_tokens) / len(candidate_tokens | prev_tokens)
            if overlap >= 0.75:
                return True
        return False

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
