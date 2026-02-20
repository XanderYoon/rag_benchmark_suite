from __future__ import annotations

from Benchmark.domain.models import Chunk


class Chunker:
    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def chunk_text(self, paper_id: str, text: str) -> list[Chunk]:
        tokens = text.split()
        if not tokens:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size_tokens - self.chunk_overlap_tokens

        idx = 0
        chunk_index = 0
        while idx < len(tokens):
            segment = tokens[idx : idx + self.chunk_size_tokens]
            chunk_id = f"{paper_id}_chunk_{chunk_index:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=" ".join(segment),
                    index=chunk_index,
                )
            )
            if idx + self.chunk_size_tokens >= len(tokens):
                break
            idx += step
            chunk_index += 1
        return chunks
