from __future__ import annotations

from Benchmark.domain.models import EvidenceCandidate


class EvidenceProposer:
    def propose(self, retrieval_candidates: list[EvidenceCandidate], max_candidates: int = 3) -> list[str]:
        return [c.chunk_id for c in sorted(retrieval_candidates, key=lambda x: x.score, reverse=True)[:max_candidates]]
