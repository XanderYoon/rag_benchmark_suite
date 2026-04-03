from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from benchmark.config import AppConfig
from benchmark.domain.models import Chunk, EvidenceCandidate
from RAG.embedding.embedder import SimpleTextEmbedder
from RAG.embedding.vector_index import InMemoryVectorIndex
from RAG.retrieval.index_runtime import GraphArtifactRetriever, load_index_manifest


class RetrievalService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._faiss_index: Any | None = None
        self._faiss: Any | None = None
        self._np: Any | None = None
        self._openai_client: Any | None = None
        self._artifact_rows_by_id: dict[int, dict[str, Any]] | None = None
        self._artifact_rows_by_chunk_id: dict[str, dict[str, Any]] | None = None
        self._artifact_output_dir: Path | None = None
        self._artifact_method: str | None = None
        self._artifact_metric: str = "cosine"
        self._artifact_embedding_provider: str = "openai"
        self._artifact_embedding_model: str = config.embedding_model
        self._chunk_text_cache: dict[str, str] = {}
        self.retrieval_error: str | None = None

    @property
    def faiss_error(self) -> str | None:
        """Preserve the legacy FAISS error accessor for existing callers."""
        return self.retrieval_error

    @faiss_error.setter
    def faiss_error(self, value: str | None) -> None:
        self.retrieval_error = value

    def _can_retry_from_cached_error(self) -> bool:
        """Return True when a previously cached retrieval error can be retried."""
        if self.retrieval_error == "OPENAI_API_KEY is not set.":
            return bool(os.getenv("OPENAI_API_KEY", "").strip())
        if self.retrieval_error == "Ollama server is not reachable.":
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

    def retrieve_top_artifact(
        self,
        question: str,
        *,
        retrieval_method: str,
        limit: int = 20,
        retrieval_model: str | None = None,
        retrieval_provider: str | None = None,
        artifact_output_dir: str | Path | None = None,
    ) -> list[EvidenceCandidate]:
        """Retrieve top candidates from the selected stored retrieval artifacts."""
        normalized_method = str(retrieval_method).strip().lower()
        if normalized_method in {"", "none"} or limit <= 0:
            return []

        if normalized_method == "faiss":
            return self.retrieve_top_faiss(
                question,
                limit=limit,
                retrieval_model=retrieval_model,
                retrieval_provider=retrieval_provider,
                faiss_output_dir=artifact_output_dir,
            )

        return self._retrieve_top_graph_artifact(
            question,
            retrieval_method=normalized_method,
            limit=limit,
            retrieval_model=retrieval_model,
            artifact_output_dir=artifact_output_dir,
        )

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
        assert self._artifact_rows_by_id is not None

        query_model = str(retrieval_model or self._artifact_embedding_model).strip()
        if not query_model:
            self.retrieval_error = "No retrieval embedding model is configured."
            return []
        query_provider = str(retrieval_provider or self._provider_for_embedding_model(query_model)).strip().lower()

        try:
            vector = self._query_embedding_vector(question=question, provider=query_provider, model=query_model)
            if self._artifact_metric == "cosine":
                self._faiss.normalize_L2(vector)
            distances, indices = self._faiss_index.search(vector, limit)
        except Exception as exc:
            self.retrieval_error = f"FAISS retrieval failed: {exc}"
            return []

        candidates: list[EvidenceCandidate] = []
        self._chunk_text_cache = {}
        for rank, faiss_id in enumerate(indices[0], start=1):
            if int(faiss_id) < 0:
                continue
            row = self._artifact_rows_by_id.get(int(faiss_id))
            if row is None:
                continue

            distance = float(distances[0][rank - 1])
            if self._artifact_metric == "cosine":
                score = distance
            else:
                score = 1.0 / (1.0 + max(distance, 0.0))

            chunk_id = str(row["chunk_id"])
            candidates.append(EvidenceCandidate(chunk_id=chunk_id, score=score, rank=rank))
            self._chunk_text_cache[chunk_id] = self._read_chunk_text(Path(str(row["file_path"])))

        return candidates

    def _retrieve_top_graph_artifact(
        self,
        question: str,
        *,
        retrieval_method: str,
        limit: int,
        retrieval_model: str | None,
        artifact_output_dir: str | Path | None,
    ) -> list[EvidenceCandidate]:
        """Retrieve top candidates from LightRAG or GraphRAG artifacts."""
        target_output_dir = Path(artifact_output_dir) if artifact_output_dir is not None else Path(f"data/{retrieval_method}_index")
        if not self._ensure_artifact_metadata_ready(output_dir=target_output_dir, expected_method=retrieval_method):
            return []

        query_model = str(retrieval_model or self._artifact_embedding_model).strip()
        if not query_model:
            self.retrieval_error = "No retrieval embedding model is configured."
            return []

        try:
            retriever = GraphArtifactRetriever(
                embedded_chunks_path=target_output_dir,
                retrieval_model=query_model,
                top_k=limit,
                ollama_base_url=self.config.ollama_base_url,
            )
            retrieved_chunks = retriever.retrieve(query=question, limit=limit)
        except Exception as exc:
            self.retrieval_error = f"{retrieval_method} retrieval failed: {exc}"
            return []

        candidates: list[EvidenceCandidate] = []
        self._chunk_text_cache = {}
        for chunk in retrieved_chunks:
            candidates.append(EvidenceCandidate(chunk_id=chunk.chunk_id, score=float(chunk.score), rank=int(chunk.rank)))
            self._chunk_text_cache[chunk.chunk_id] = chunk.text
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
                    self.retrieval_error = "OPENAI_API_KEY is not set."
                    raise RuntimeError(self.retrieval_error)
                self._openai_client = OpenAI(api_key=api_key)
            response = self._openai_client.embeddings.create(model=model, input=[question])
            return self._np.array([response.data[0].embedding], dtype=self._np.float32)

        if normalized_provider == "ollama":
            base_url = self.config.ollama_base_url.rstrip("/")
            if not base_url:
                raise RuntimeError("Ollama base URL is not configured.")
            try:
                health = requests.get(f"{base_url}/api/tags", timeout=2)
                health.raise_for_status()
            except Exception as exc:
                self.retrieval_error = "Ollama server is not reachable."
                raise RuntimeError(self.retrieval_error) from exc
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

    def candidates_to_chunks(self, candidates: list[EvidenceCandidate]) -> dict[str, Chunk]:
        """Hydrate retrieval candidates into chunk objects when metadata is available."""
        if not self._artifact_rows_by_chunk_id:
            return {}

        chunks_by_id: dict[str, Chunk] = {}
        for cand in candidates:
            row = self._artifact_rows_by_chunk_id.get(cand.chunk_id)
            if row is None:
                continue

            chunk_id = str(row["chunk_id"])
            if chunk_id not in self._chunk_text_cache:
                self._chunk_text_cache[chunk_id] = self._read_chunk_text(Path(str(row["file_path"])))

            chunks_by_id[chunk_id] = Chunk(
                chunk_id=chunk_id,
                paper_id=str(row["paper_id"]),
                text=self._chunk_text_cache[chunk_id],
                index=self._chunk_index_from_id(chunk_id),
            )
        return chunks_by_id

    def faiss_candidates_to_chunks(self, candidates: list[EvidenceCandidate]) -> dict[str, Chunk]:
        """Preserve the legacy FAISS chunk hydration alias."""
        return self.candidates_to_chunks(candidates)

    def load_chunks_for_candidates(self, candidates: list[EvidenceCandidate]) -> dict[str, Chunk]:
        """Return chunk objects for retrieval candidates when backing metadata is available."""
        return self.candidates_to_chunks(candidates)

    def _ensure_faiss_ready(self, *, faiss_output_dir: str | Path | None = None) -> bool:
        target_output_dir = Path(faiss_output_dir) if faiss_output_dir is not None else Path("data/faiss_rag_index")
        if not self._ensure_artifact_metadata_ready(output_dir=target_output_dir, expected_method="faiss"):
            return False

        if self._faiss_index is not None and self._artifact_output_dir == target_output_dir:
            return True

        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            self.retrieval_error = f"Missing dependency for FAISS retrieval: {exc}"
            return False

        index_path = target_output_dir / "chunks.faiss"
        if not index_path.exists():
            self.retrieval_error = f"Missing FAISS artifact. Expected {index_path}."
            return False

        self._faiss = faiss
        self._np = np
        self._faiss_index = faiss.read_index(str(index_path))
        return True

    def _ensure_artifact_metadata_ready(self, *, output_dir: Path, expected_method: str | None = None) -> bool:
        """Load retrieval artifact metadata into cache for hydration and config."""
        target_output_dir = output_dir.expanduser()
        if self._artifact_output_dir is not None and self._artifact_output_dir != target_output_dir:
            self._reset_cached_artifacts()

        if (
            self._artifact_rows_by_chunk_id is not None
            and self._artifact_output_dir == target_output_dir
            and (expected_method is None or self._artifact_method == expected_method)
        ):
            return True
        if self.retrieval_error:
            if self._can_retry_from_cached_error():
                self.retrieval_error = None
            elif self._artifact_output_dir == target_output_dir:
                return False

        metadata_path = target_output_dir / "chunks_metadata.jsonl"
        manifest_path = target_output_dir / "index_manifest.json"
        if not metadata_path.exists() or not manifest_path.exists():
            self.retrieval_error = f"Missing retrieval artifacts. Expected {metadata_path} and {manifest_path}."
            return False

        try:
            manifest = load_index_manifest(embedded_chunks_path=target_output_dir)
        except Exception as exc:
            self.retrieval_error = f"Failed to load retrieval manifest at {manifest_path}: {exc}"
            return False

        method_id = str(manifest.get("method_id", "faiss")).strip().lower() or "faiss"
        if expected_method is not None and method_id != expected_method:
            self.retrieval_error = (
                f"Selected retrieval artifacts are for method '{method_id}', expected '{expected_method}'."
            )
            return False

        rows_by_id: dict[int, dict[str, Any]] = {}
        rows_by_chunk_id: dict[str, dict[str, Any]] = {}
        try:
            with metadata_path.open("r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "faiss_id" in row:
                        rows_by_id[int(row["faiss_id"])] = row
                    rows_by_chunk_id[str(row["chunk_id"])] = row
        except Exception as exc:
            self.retrieval_error = f"Failed to load retrieval metadata at {metadata_path}: {exc}"
            return False

        self._artifact_rows_by_id = rows_by_id
        self._artifact_rows_by_chunk_id = rows_by_chunk_id
        self._artifact_output_dir = target_output_dir
        self._artifact_method = method_id
        self._artifact_metric = str(manifest.get("metric", "cosine"))
        self._artifact_embedding_provider = str(
            manifest.get("embedding_provider", self._provider_for_embedding_model(self.config.embedding_model))
        ).strip().lower() or "openai"
        self._artifact_embedding_model = str(manifest.get("embedding_model", self.config.embedding_model))
        self.retrieval_error = None
        return True

    def _reset_cached_artifacts(self) -> None:
        """Clear cached retrieval artifacts when switching output directories."""
        self._faiss_index = None
        self._artifact_rows_by_id = None
        self._artifact_rows_by_chunk_id = None
        self._artifact_output_dir = None
        self._artifact_method = None
        self._chunk_text_cache = {}
        self.retrieval_error = None

    @staticmethod
    def _read_chunk_text(path: Path) -> str:
        """Read chunk text from disk with a permissive fallback."""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    @staticmethod
    def _chunk_index_from_id(chunk_id: str) -> int:
        """Extract the numeric chunk index from a chunk identifier."""
        try:
            return int(chunk_id.split("_")[-1])
        except ValueError:
            return 0
