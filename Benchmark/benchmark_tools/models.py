from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkArtifact:
    """Represent one chunk row stored alongside the FAISS index."""

    faiss_id: int
    paper_id: str
    chunk_id: str
    file_path: Path
    text: str


@dataclass(frozen=True)
class BenchmarkProbe:
    """Represent one deterministic retrieval probe."""

    case_id: str
    query: str
    expected_chunk_id: str
    expected_answer: str
    reference_contexts: list[str]


@dataclass(frozen=True)
class RetrievedChunk:
    """Represent one retrieved chunk with hydrated text."""

    chunk_id: str
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Capture retrieval outputs for one probe case."""

    case_id: str
    query: str
    expected_chunk_id: str
    expected_answer: str
    reference_contexts: list[str]
    retrieved_chunks: list[RetrievedChunk]

    @property
    def top_hit_chunk_id(self) -> str | None:
        """Return the first retrieved chunk identifier when present."""
        if not self.retrieved_chunks:
            return None
        return self.retrieved_chunks[0].chunk_id

    @property
    def actual_answer(self) -> str:
        """Use the top retrieved chunk text as the benchmark answer surrogate."""
        if not self.retrieved_chunks:
            return ""
        return self.retrieved_chunks[0].text

    @property
    def hit_at_1(self) -> bool:
        """Return whether the expected chunk was ranked first."""
        return self.top_hit_chunk_id == self.expected_chunk_id

    @property
    def hit_at_3(self) -> bool:
        """Return whether the expected chunk was retrieved in the top three."""
        return any(
            chunk.chunk_id == self.expected_chunk_id and chunk.rank <= 3
            for chunk in self.retrieved_chunks
        )

    @property
    def reciprocal_rank(self) -> float:
        """Return the reciprocal rank for the expected chunk."""
        for chunk in self.retrieved_chunks:
            if chunk.chunk_id == self.expected_chunk_id:
                return 1.0 / float(chunk.rank)
        return 0.0

