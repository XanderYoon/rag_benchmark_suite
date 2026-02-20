from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from Benchmark.config import AppConfig
from Benchmark.domain.models import Chunk, EvidenceCandidate
from Benchmark.embedding.embedder import SimpleTextEmbedder
from Benchmark.embedding.vector_index import InMemoryVectorIndex


class RetrievalService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._faiss_index: Any | None = None
        self._faiss: Any | None = None
        self._np: Any | None = None
        self._openai_client: Any | None = None
        self._faiss_rows_by_id: dict[int, dict[str, Any]] | None = None
        self._faiss_rows_by_chunk_id: dict[str, dict[str, Any]] | None = None
        self._faiss_metric: str = "cosine"
        self._faiss_embedding_model: str = config.embedding_model
        self._chunk_text_cache: dict[str, str] = {}
        self.faiss_error: str | None = None

    def retrieve_generous(self, question: str, chunks: list[Chunk]) -> list[EvidenceCandidate]:
        index = InMemoryVectorIndex(SimpleTextEmbedder())
        for chunk in chunks:
            index.add(chunk)

        scored = index.search(question, limit=max(len(chunks), self.config.retrieval_cap))
        by_id = {chunk.chunk_id: (chunk, score) for chunk, score in scored}

        top_hits = scored[: self.config.retrieval_top_k]
        selected: dict[str, float] = {}

        for chunk, score in top_hits:
            selected[chunk.chunk_id] = score
            for neighbor_idx in (chunk.index - 1, chunk.index + 1):
                if neighbor_idx < 0:
                    continue
                neighbor_id = f"{chunk.paper_id}_chunk_{neighbor_idx:04d}"
                if neighbor_id in by_id:
                    selected[neighbor_id] = by_id[neighbor_id][1]

        for chunk, score in scored:
            if score >= self.config.retrieval_threshold:
                selected[chunk.chunk_id] = score
            if len(selected) >= self.config.retrieval_cap:
                break

        ranked = sorted(selected.items(), key=lambda x: x[1], reverse=True)[: self.config.retrieval_cap]
        return [EvidenceCandidate(chunk_id=cid, score=float(score), rank=i + 1) for i, (cid, score) in enumerate(ranked)]

    def retrieve_top_faiss(self, question: str, limit: int = 20) -> list[EvidenceCandidate]:
        if limit <= 0:
            return []
        if not self._ensure_faiss_ready():
            return []

        assert self._np is not None
        assert self._faiss is not None
        assert self._faiss_index is not None
        assert self._openai_client is not None
        assert self._faiss_rows_by_id is not None

        response = self._openai_client.embeddings.create(
            model=self._faiss_embedding_model,
            input=question,
        )
        vector = self._np.array(response.data[0].embedding, dtype=self._np.float32).reshape(1, -1)
        if self._faiss_metric == "cosine":
            self._faiss.normalize_L2(vector)

        scores, ids = self._faiss_index.search(vector, limit)
        ranked: list[EvidenceCandidate] = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if int(faiss_id) < 0:
                continue
            row = self._faiss_rows_by_id.get(int(faiss_id))
            if not row:
                continue
            ranked.append(
                EvidenceCandidate(
                    chunk_id=str(row["chunk_id"]),
                    score=float(score),
                    rank=len(ranked) + 1,
                )
            )
        return ranked

    def load_chunks_for_candidates(self, candidates: list[EvidenceCandidate]) -> dict[str, Chunk]:
        if not self._ensure_faiss_ready():
            return {}
        assert self._faiss_rows_by_chunk_id is not None

        chunks_by_id: dict[str, Chunk] = {}
        for cand in candidates:
            row = self._faiss_rows_by_chunk_id.get(cand.chunk_id)
            if not row:
                continue
            chunk_path = Path(str(row["file_path"]))
            if not chunk_path.exists():
                continue

            if cand.chunk_id not in self._chunk_text_cache:
                self._chunk_text_cache[cand.chunk_id] = chunk_path.read_text(encoding="utf-8", errors="replace")

            chunk_index = 0
            try:
                chunk_index = int(cand.chunk_id.rsplit("_", 1)[-1])
            except (IndexError, ValueError):
                pass

            chunks_by_id[cand.chunk_id] = Chunk(
                chunk_id=cand.chunk_id,
                paper_id=str(row.get("paper_id", "")),
                text=self._chunk_text_cache[cand.chunk_id],
                index=chunk_index,
            )
        return chunks_by_id

    def _ensure_faiss_ready(self) -> bool:
        if self._faiss_index is not None and self._faiss_rows_by_id is not None:
            return True
        if self.faiss_error:
            return False

        try:
            import faiss
            import numpy as np
            from openai import OpenAI
        except ImportError as exc:
            self.faiss_error = f"Missing dependency for FAISS retrieval: {exc}"
            return False

        output_dir = Path("data/faiss_rag_index")
        index_path = output_dir / "chunks.faiss"
        metadata_path = output_dir / "chunks_metadata.jsonl"
        manifest_path = output_dir / "index_manifest.json"
        if not index_path.exists() or not metadata_path.exists():
            self.faiss_error = (
                f"Missing FAISS artifacts. Expected {index_path} and {metadata_path}."
            )
            return False

        rows_by_id: dict[int, dict[str, Any]] = {}
        rows_by_chunk_id: dict[str, dict[str, Any]] = {}
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                faiss_id = int(row["faiss_id"])
                rows_by_id[faiss_id] = row
                rows_by_chunk_id[str(row["chunk_id"])] = row

        self._faiss_metric = "cosine"
        self._faiss_embedding_model = self.config.embedding_model
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._faiss_metric = str(manifest.get("metric", self._faiss_metric))
            self._faiss_embedding_model = str(
                manifest.get("embedding_model", self._faiss_embedding_model)
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.faiss_error = "OPENAI_API_KEY is not set."
            return False

        self._faiss = faiss
        self._np = np
        self._openai_client = OpenAI(api_key=api_key)
        self._faiss_index = faiss.read_index(str(index_path))
        self._faiss_rows_by_id = rows_by_id
        self._faiss_rows_by_chunk_id = rows_by_chunk_id
        return True
