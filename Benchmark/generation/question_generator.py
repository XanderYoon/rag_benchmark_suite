from __future__ import annotations

import os
import re
from typing import Any

from Benchmark.domain.enums import DifficultyLabel
from Benchmark.domain.models import Chunk


class QuestionGenerator:
    COMMON_PROFILE_INSTRUCTION = (
        "You are a RAG (Retrieval-Augmented Generation) tester. "
        "Your task is to generate questions based strictly on the content of a provided document in order to "
        "evaluate retrieval performance. Do not generate questions about content found in the acknowledgements "
        "or references sections."
    )

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._openai_client: Any | None = None

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

    def generate_profile_question(
        self,
        paper_id: str,
        chunks: list[Chunk],
        reference_type: str,
        hop_type: DifficultyLabel,
        slot_index: int,
        feedback: str = "",
        avoid_questions: list[str] | None = None,
    ) -> str:
        if not chunks:
            if hop_type == DifficultyLabel.MULTI_HOP:
                return "How do two key ideas in this work connect to support the main claim?"
            return "What key claim is made, and what evidence supports it?"

        avoid_questions = avoid_questions or []
        feedback_text = feedback.strip()
        offset = slot_index + (sum(ord(c) for c in feedback_text) % max(len(chunks), 1) if feedback_text else 0)
        seed_a = self._snippet(chunks[offset % len(chunks)].text)
        seed_b = self._snippet(chunks[(offset + max(len(chunks) // 2, 1)) % len(chunks)].text)

        near = self._snippet(chunks[(offset + 1) % len(chunks)].text)
        prompt = self._build_profile_prompt(
            paper_id=paper_id,
            reference_type=reference_type,
            hop_type=hop_type,
            seed_a=seed_a,
            seed_b=seed_b,
            near=near,
            feedback_text=feedback_text,
            avoid_questions=avoid_questions,
        )
        question = self._generate_with_llm(prompt)
        if question and not self._is_too_similar(question, avoid_questions):
            return question

        # Retry with alternate evidence windows to encourage a different output.
        for step in range(1, min(len(chunks), 8)):
            alt_a = self._snippet(chunks[(offset + step) % len(chunks)].text)
            alt_b = self._snippet(chunks[(offset + step + max(len(chunks) // 2, 1)) % len(chunks)].text)
            alt_near = self._snippet(chunks[(offset + step + 1) % len(chunks)].text)
            alt_prompt = self._build_profile_prompt(
                paper_id=paper_id,
                reference_type=reference_type,
                hop_type=hop_type,
                seed_a=alt_a,
                seed_b=alt_b,
                near=alt_near,
                feedback_text=feedback_text,
                avoid_questions=avoid_questions,
            )
            alt_question = self._generate_with_llm(alt_prompt)
            if alt_question and not self._is_too_similar(alt_question, avoid_questions):
                return alt_question

        # Deterministic fallback when model call is unavailable or low quality.
        return self._fallback_profile_question(
            paper_id=paper_id,
            reference_type=reference_type,
            hop_type=hop_type,
            seed_a=seed_a,
            seed_b=seed_b,
        )

    def _build_profile_prompt(
        self,
        paper_id: str,
        reference_type: str,
        hop_type: DifficultyLabel,
        seed_a: str,
        seed_b: str,
        near: str,
        feedback_text: str,
        avoid_questions: list[str],
    ) -> str:
        base = self.COMMON_PROFILE_INSTRUCTION
        if reference_type == "single_single":
            prompt = (
                f"{base} "
                "Create a query grounded in one document only. "
                "The question should be answerable by a "
                "Single-hop Retrieval-Augmented Generation (RAG) system. The question should require retrieving "
                "information from a single, self-contained passage within the document, without cross-referencing "
                f"multiple sections. Use this document title and passage as evidence context: '{paper_id}' and '{seed_a}'."
            )
        elif reference_type == "single_multi":
            prompt = (
                f"{base} "
                "Create a query grounded in one document only. "
                "The question should be answerable by a "
                "Multi-hop Retrieval-Augmented Generation (RAG) system. The question should require retrieving "
                "and combining information from multiple sections or passages within the document. "
                f"Use this document title and these evidence contexts: '{paper_id}', '{seed_a}', and '{seed_b}'."
            )
        elif reference_type == "multiple":
            prompt = (
                f"{base} "
                "Create a query that is best answered by retrieving evidence from multiple different documents in the corpus. "
                "The question should not depend on a single local passage. "
                f"Use these evidence contexts as anchors: '{seed_a}' and '{seed_b}'."
            )
        elif reference_type == "comparison":
            prompt = (
                f"{base} "
                "Create a comparison question that asks the system to distinguish, contrast, or weigh two related ideas. "
                "The question should require comparing separate pieces of evidence rather than repeating one fact. "
                f"Use these evidence contexts: '{seed_a}' and '{seed_b}'."
            )
        elif reference_type == "negative":
            prompt = (
                f"{base} "
                "Create a question whose correct answer is negative, absent, null, or explicitly indicates that a claim "
                "is unsupported by the provided evidence. The question should still be grounded in the document. "
                f"Use this evidence context and nearby text to define the boundary of what is not stated: '{seed_a}' and '{near}'."
            )
        else:
            prompt = (
                f"{base} "
                "Create a grounded question about the document content. "
                f"Use these evidence contexts: '{seed_a}' and '{seed_b}'."
            )

        prompt = (
            f"{prompt} Output exactly one question only, with no explanation, no preface, and no quotation marks."
        )
        if avoid_questions:
            avoid_list = "\n".join(f"- {q}" for q in avoid_questions[-5:])
            prompt = (
                f"{prompt} Avoid duplicating or closely paraphrasing any of these existing questions:\n{avoid_list}"
            )
        if feedback_text:
            prompt = (
                f"{prompt} Regenerate with a clearly different angle and satisfy this feedback: {feedback_text}."
            )
        return prompt

    def _generate_with_llm(self, prompt: str) -> str | None:
        client = self._get_openai_client()
        if client is None:
            return None
        try:
            resp = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You generate benchmark questions for retrieval evaluation. "
                            "Return one question only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.output_text or "").strip()
            if not text:
                return None
            line = text.splitlines()[0].strip().strip('"').strip("'")
            return line if line else None
        except Exception:
            return None

    def _get_openai_client(self) -> Any | None:
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import OpenAI
        except ImportError:
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _fallback_profile_question(
        self,
        paper_id: str,
        reference_type: str,
        hop_type: DifficultyLabel,
        seed_a: str,
        seed_b: str,
    ) -> str:
        if reference_type == "single_single":
            return f"In {paper_id}, what is stated about {seed_a}?"
        if reference_type == "single_multi":
            return f"In {paper_id}, how do these two points connect: {seed_a} and {seed_b}?"
        if reference_type == "multiple":
            return f"What conclusion requires combining evidence from multiple documents related to: {seed_a} and {seed_b}?"
        if reference_type == "comparison":
            return f"How do these two points compare: {seed_a} versus {seed_b}?"
        if reference_type == "negative":
            return f"What claim is not supported by the evidence around: {seed_a}?"
        return f"What grounded question can be answered using: {seed_a} and {seed_b}?"

    def generate_section_question(
        self,
        chunks: list[Chunk],
        section_hint: str,
        slot_index: int,
        avoid_questions: list[str] | None = None,
    ) -> str:
        if not chunks:
            if section_hint == "methodology":
                return "What key methodological choice supports the main approach?"
            if section_hint == "results":
                return "What result best demonstrates performance gains?"
            return "What conclusion follows from the reported findings?"

        avoid_questions = avoid_questions or []
        seed = self._section_seed(chunks, section_hint, slot_index)

        if section_hint == "methodology":
            prompt = f"Which methodological decision is most important here, and why: '{seed}'?"
        elif section_hint == "results":
            prompt = f"What is the most important result reported here, and how is it supported: '{seed}'?"
        else:
            prompt = f"What conclusion is justified by this evidence: '{seed}'?"

        if self._is_too_similar(prompt, avoid_questions):
            for step in range(1, min(len(chunks), 8)):
                alt_seed = self._section_seed(chunks, section_hint, slot_index + step)
                if section_hint == "methodology":
                    alt_prompt = f"How does this methodological setup influence the outcome: '{alt_seed}'?"
                elif section_hint == "results":
                    alt_prompt = f"How should this result be interpreted in context: '{alt_seed}'?"
                else:
                    alt_prompt = f"What key takeaway is supported by this passage: '{alt_seed}'?"
                if not self._is_too_similar(alt_prompt, avoid_questions):
                    prompt = alt_prompt
                    break

        return prompt

    def _section_seed(self, chunks: list[Chunk], section_hint: str, slot_index: int) -> str:
        keywords = {
            "methodology": ["method", "methodology", "approach", "experiment", "setup"],
            "results": ["result", "performance", "evaluation", "metric", "table", "accuracy"],
            "conclusion": ["conclusion", "summary", "future", "limitation", "discussion"],
        }
        targets = keywords.get(section_hint, [])
        for step in range(len(chunks)):
            chunk = chunks[(slot_index + step) % len(chunks)]
            lowered = chunk.text.lower()
            if any(token in lowered for token in targets):
                return self._snippet(chunk.text)
        return self._snippet(chunks[slot_index % len(chunks)].text)
