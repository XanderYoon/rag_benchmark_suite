from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

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
        self._faiss_output_dir: Path | None = None
        self._faiss_metric: str = "cosine"
        self._faiss_embedding_provider: str = "openai"
        self._faiss_embedding_model: str = config.embedding_model
        self._chunk_text_cache: dict[str, str] = {}
        self.faiss_error: str | None = None

    def _can_retry_from_cached_error(self) -> bool:
        """Return True when a previously cached FAISS error can be retried."""
        if self.faiss_error == "OPENAI_API_KEY is not set.":
            return bool(os.getenv("OPENAI_API_KEY", "").strip())
        if self.faiss_error == "Ollama server is not reachable.":
            try:
                response = requests.get(f"{self.config.ollama_base_url.rstrip('/')}/api/tags", timeout=2)
                return response.status_code == 200
            except Exception:
                return False
        return False

    def retrieve_generous(self, question: str, chunks: list[Chunk]) -> list[EvidenceCandidate]:
        """Retrieve generous local candidates from in-memory chunk embeddings."""
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

        ranked = sorted(selected.items(), key=lambda item: item[1], reverse=True)[: self.config.retrieval_cap]
        return [
            EvidenceCandidate(chunk_id=chunk_id, score=float(score), rank=index_ + 1)
            for index_, (chunk_id, score) in enumerate(ranked)
        ]

    @staticmethod
    def _provider_for_embedding_model(model: str) -> str:
        """Infer embedding provider from model naming convention."""
        normalized = str(model).strip().lower()
        if normalized.startswith("text-embedding-"):
            return "openai"
        return "ollama"

    def retrieve_top_faiss(
        self,
        question: str,
        limit: int = 20,
        retrieval_model: str | None = None,
        retrieval_provider: str | None = None,
        faiss_output_dir: str | Path | None = None,
    ) -> list[EvidenceCandidate]:
        """Retrieve top FAISS candidates for the question if the index is available."""
        if limit <= 0:
            return []
        if not self._ensure_faiss_ready(faiss_output_dir=faiss_output_dir):
            return []

        assert self._np is not None
        assert self._faiss is not None
        assert self._faiss_index is not None
        assert self._faiss_rows_by_id is not None

        query_model = str(retrieval_model or self._faiss_embedding_model).strip()
        if not query_model:
            self.faiss_error = "No retrieval embedding model is configured."
            return []
        query_provider = str(retrieval_provider or self._provider_for_embedding_model(query_model)).strip().lower()

        try:
            vector = self._query_embedding_vector(
                question=question,
                provider=query_provider,
                model=query_model,
            )
            if self._faiss_metric == "cosine":
                self._faiss.normalize_L2(vector)
            distances, indices = self._faiss_index.search(vector, limit)
        except Exception as exc:
            self.faiss_error = f"FAISS retrieval failed: {exc}"
            return []

        candidates: list[EvidenceCandidate] = []
        self._chunk_text_cache = {}
        for rank, faiss_id in enumerate(indices[0], start=1):
            if int(faiss_id) < 0:
                continue
            row = self._faiss_rows_by_id.get(int(faiss_id))
            if row is None:
                continue

            distance = float(distances[0][rank - 1])
            if self._faiss_metric == "cosine":
                score = distance
            else:
                score = 1.0 / (1.0 + max(distance, 0.0))

            chunk_id = str(row["chunk_id"])
            candidates.append(EvidenceCandidate(chunk_id=chunk_id, score=score, rank=rank))

            file_path = Path(str(row["file_path"]))
            try:
                self._chunk_text_cache[chunk_id] = file_path.read_text(encoding="utf-8")
            except OSError:
                self._chunk_text_cache[chunk_id] = ""

        return candidates

    def _query_embedding_vector(self, *, question: str, provider: str, model: str) -> Any:
        """Return a normalized query embedding vector for FAISS search."""
        assert self._np is not None
        normalized_provider = provider.strip().lower()
        if normalized_provider == "openai":
            if self._openai_client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError("openai is required for OpenAI FAISS retrieval.") from exc
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    self.faiss_error = "OPENAI_API_KEY is not set."
                    raise RuntimeError(self.faiss_error)
                self._openai_client = OpenAI(api_key=api_key)
            response = self._openai_client.embeddings.create(
                model=model,
                input=[question],
            )
            return self._np.array([response.data[0].embedding], dtype=self._np.float32)

        if normalized_provider == "ollama":
            base_url = self.config.ollama_base_url.rstrip("/")
            if not base_url:
                raise RuntimeError("Ollama base URL is not configured.")
            try:
                health = requests.get(f"{base_url}/api/tags", timeout=2)
                health.raise_for_status()
            except Exception as exc:
                self.faiss_error = "Ollama server is not reachable."
                raise RuntimeError(self.faiss_error) from exc
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": [question]},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or not embeddings:
                raise RuntimeError("Ollama did not return a valid embedding payload.")
            return self._np.array([embeddings[0]], dtype=self._np.float32)

        raise RuntimeError(
            f"Unsupported retrieval embedding provider '{provider}'. Supported: ['openai', 'ollama']."
        )

    def faiss_candidates_to_chunks(self, candidates: list[EvidenceCandidate]) -> dict[str, Chunk]:
        """Hydrate FAISS candidates into chunk objects when metadata is available."""
        if not self._faiss_rows_by_chunk_id:
            return {}

        chunks_by_id: dict[str, Chunk] = {}
        for cand in candidates:
            row = self._faiss_rows_by_chunk_id.get(cand.chunk_id)
            if row is None:
                continue

            chunk_id = str(row["chunk_id"])
            if chunk_id not in self._chunk_text_cache:
                file_path = Path(str(row["file_path"]))
                try:
                    self._chunk_text_cache[chunk_id] = file_path.read_text(encoding="utf-8")
                except OSError:
                    self._chunk_text_cache[chunk_id] = ""

            try:
                chunk_index = int(chunk_id.split("_")[-1])
            except ValueError:
                chunk_index = 0

            chunks_by_id[chunk_id] = Chunk(
                chunk_id=chunk_id,
                paper_id=str(row["paper_id"]),
                text=self._chunk_text_cache[chunk_id],
                index=chunk_index,
            )
        return chunks_by_id

    def load_chunks_for_candidates(self, candidates: list[EvidenceCandidate]) -> dict[str, Chunk]:
        """Return chunk objects for retrieval candidates when backing metadata is available."""
        return self.faiss_candidates_to_chunks(candidates)

    def _ensure_faiss_ready(self, *, faiss_output_dir: str | Path | None = None) -> bool:
        target_output_dir = Path(faiss_output_dir) if faiss_output_dir is not None else Path("data/faiss_rag_index")
        if self._faiss_output_dir is not None and self._faiss_output_dir != target_output_dir:
            self._faiss_index = None
            self._faiss_rows_by_id = None
            self._faiss_rows_by_chunk_id = None
            self._chunk_text_cache = {}
            self.faiss_error = None

        if self._faiss_index is not None and self._faiss_rows_by_id is not None and self._faiss_output_dir == target_output_dir:
            return True
        if self.faiss_error:
            if self._can_retry_from_cached_error():
                self.faiss_error = None
            else:
                return False

        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            self.faiss_error = f"Missing dependency for FAISS retrieval: {exc}"
            return False

        output_dir = target_output_dir
        index_path = output_dir / "chunks.faiss"
        metadata_path = output_dir / "chunks_metadata.jsonl"
        manifest_path = output_dir / "index_manifest.json"
        if not index_path.exists() or not metadata_path.exists():
            self.faiss_error = f"Missing FAISS artifacts. Expected {index_path} and {metadata_path}."
            return False

        rows_by_id: dict[int, dict[str, Any]] = {}
        rows_by_chunk_id: dict[str, dict[str, Any]] = {}
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                faiss_id = int(row["faiss_id"])
                rows_by_id[faiss_id] = row
                rows_by_chunk_id[str(row["chunk_id"])] = row

        self._faiss_metric = "cosine"
        self._faiss_embedding_provider = "openai"
        self._faiss_embedding_model = self.config.embedding_model
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._faiss_metric = str(manifest.get("metric", self._faiss_metric))
            self._faiss_embedding_provider = str(
                manifest.get("embedding_provider", self._faiss_embedding_provider)
            ).strip().lower() or "openai"
            self._faiss_embedding_model = str(
                manifest.get("embedding_model", self._faiss_embedding_model)
            )

        self._faiss = faiss
        self._np = np
        self._faiss_index = faiss.read_index(str(index_path))
        self._faiss_rows_by_id = rows_by_id
        self._faiss_rows_by_chunk_id = rows_by_chunk_id
        self._faiss_output_dir = output_dir
        return True
