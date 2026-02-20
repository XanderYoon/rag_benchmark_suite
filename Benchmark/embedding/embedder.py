from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter


TokenVector = dict[str, float]


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> TokenVector:
        raise NotImplementedError


class SimpleTextEmbedder(Embedder):
    def embed(self, text: str) -> TokenVector:
        words = re.findall(r"[A-Za-z0-9_]+", text.lower())
        counts = Counter(words)
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, text: str) -> TokenVector:  # pragma: no cover
        raise NotImplementedError(
            "OpenAIEmbedder is a placeholder. Wire provider client if remote embedding is needed."
        )
