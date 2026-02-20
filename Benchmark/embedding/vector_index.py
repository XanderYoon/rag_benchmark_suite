from __future__ import annotations

from abc import ABC, abstractmethod

from Benchmark.domain.models import Chunk
from Benchmark.embedding.embedder import Embedder, TokenVector


class VectorIndex(ABC):
    @abstractmethod
    def add(self, chunk: Chunk) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError


class InMemoryVectorIndex(VectorIndex):
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._rows: list[tuple[Chunk, TokenVector]] = []

    def add(self, chunk: Chunk) -> None:
        self._rows.append((chunk, self.embedder.embed(chunk.text)))

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed(query)
        scored = [(chunk, self._cosine(q, vec)) for chunk, vec in self._rows]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _cosine(a: TokenVector, b: TokenVector) -> float:
        if len(a) > len(b):
            a, b = b, a
        return float(sum(v * b.get(k, 0.0) for k, v in a.items()))
