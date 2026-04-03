from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class RetrievedArtifactChunk:
    """Represent one retrieved chunk hydrated from retrieval artifacts."""

    chunk_id: str
    text: str
    score: float
    rank: int


def load_index_manifest(*, embedded_chunks_path: str | Path) -> dict[str, Any]:
    """Load one retrieval index manifest from an artifact directory."""
    base_dir = _resolve_artifact_base_dir(embedded_chunks_path)
    manifest_path = base_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing index manifest at {manifest_path}.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class GraphArtifactRetriever:
    """Run retrieval against local LightRAG or GraphRAG artifact directories."""

    def __init__(
        self,
        *,
        embedded_chunks_path: str | Path,
        retrieval_model: str,
        top_k: int = 5,
        ollama_base_url: str | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.top_k = top_k
        self.retrieval_model = retrieval_model
        self.ollama_base_url = str(ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip()
        self.base_dir = _resolve_artifact_base_dir(embedded_chunks_path)
        self.manifest = load_index_manifest(embedded_chunks_path=self.base_dir)
        self.method_id = str(self.manifest.get("method_id", "")).strip().lower()
        if self.method_id not in {"lightrag", "graphrag"}:
            raise ValueError(
                f"GraphArtifactRetriever requires a lightrag or graphrag manifest, got '{self.method_id}'."
            )

        self._np: Any | None = None
        self._openai_client: Any | None = None
        self._rows_by_chunk_id: dict[str, dict[str, Any]] = {}
        self._chunk_ids_in_order: list[str] = []
        self._embeddings: Any | None = None
        self._graph_edges: dict[str, list[dict[str, Any]]] = {}
        self._ready = False

    def retrieve(self, *, query: str, limit: int | None = None) -> list[RetrievedArtifactChunk]:
        """Retrieve the top chunks for one query using graph-aware scoring."""
        self._ensure_ready()
        assert self._np is not None
        assert self._embeddings is not None

        query_limit = self.top_k if limit is None else limit
        if query_limit <= 0:
            return []

        query_vector = self._embed_query(query=query)
        base_scores = self._score_by_similarity(query_vector=query_vector)
        if self.method_id == "lightrag":
            final_scores = self._score_lightrag(base_scores=base_scores)
        else:
            final_scores = self._score_graphrag(base_scores=base_scores)

        ranked_chunk_ids = sorted(final_scores, key=lambda chunk_id: (-final_scores[chunk_id], chunk_id))[:query_limit]
        results: list[RetrievedArtifactChunk] = []
        for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
            row = self._rows_by_chunk_id.get(chunk_id)
            if row is None:
                continue
            results.append(
                RetrievedArtifactChunk(
                    chunk_id=chunk_id,
                    text=str(row["text"]),
                    score=float(final_scores[chunk_id]),
                    rank=rank,
                )
            )
        return results

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy is required to run graph-based retrieval.") from exc

        embeddings_path = self.base_dir / "chunk_embeddings.npy"
        metadata_path = self.base_dir / str(self.manifest.get("metadata_file", "chunks_metadata.jsonl"))
        graph_path = self.base_dir / str(self.manifest.get("graph_file", "graph_edges.json"))
        missing = [str(path) for path in (embeddings_path, metadata_path, graph_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing graph retrieval artifacts: {', '.join(missing)}")

        self._np = np
        self._embeddings = np.load(str(embeddings_path))
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                chunk_id = str(row["chunk_id"])
                file_path = Path(str(row["file_path"]))
                self._chunk_ids_in_order.append(chunk_id)
                self._rows_by_chunk_id[chunk_id] = {
                    **row,
                    "text": file_path.read_text(encoding="utf-8", errors="replace"),
                }
        self._graph_edges = json.loads(graph_path.read_text(encoding="utf-8"))
        self._ready = True

    def _score_by_similarity(self, *, query_vector: Any) -> dict[str, float]:
        """Score chunks by dense vector similarity."""
        assert self._embeddings is not None
        metric = str(self.manifest.get("metric", "cosine")).strip().lower()
        if metric == "l2":
            distances = ((self._embeddings - query_vector[0]) ** 2).sum(axis=1)
            similarities = 1.0 / (1.0 + distances)
        else:
            similarities = self._embeddings @ query_vector[0]
        return {
            chunk_id: float(similarities[index])
            for index, chunk_id in enumerate(self._chunk_ids_in_order)
        }

    def _score_lightrag(self, *, base_scores: dict[str, float]) -> dict[str, float]:
        """Blend dense similarity with one-hop graph expansion."""
        seed_count = min(max(self.top_k * 2, 4), len(base_scores))
        seed_chunk_ids = sorted(base_scores, key=base_scores.get, reverse=True)[:seed_count]
        propagated: dict[str, float] = {chunk_id: 0.0 for chunk_id in base_scores}

        for chunk_id in seed_chunk_ids:
            seed_score = max(base_scores[chunk_id], 0.0)
            for edge in self._graph_edges.get(chunk_id, []):
                target_chunk_id = str(edge.get("target_chunk_id", "")).strip()
                if target_chunk_id not in propagated:
                    continue
                propagated[target_chunk_id] = max(
                    propagated[target_chunk_id],
                    seed_score * float(edge.get("weight", 0.0)),
                )

        return {
            chunk_id: float((0.78 * base_scores[chunk_id]) + (0.22 * propagated.get(chunk_id, 0.0)))
            for chunk_id in base_scores
        }

    def _score_graphrag(self, *, base_scores: dict[str, float]) -> dict[str, float]:
        """Diffuse dense similarity across the chunk graph for graph retrieval."""
        propagated = {chunk_id: max(score, 0.0) for chunk_id, score in base_scores.items()}
        current = dict(propagated)

        for _ in range(2):
            next_scores = {chunk_id: 0.35 * propagated.get(chunk_id, 0.0) for chunk_id in base_scores}
            for chunk_id, score in current.items():
                if score <= 0:
                    continue
                for edge in self._graph_edges.get(chunk_id, []):
                    target_chunk_id = str(edge.get("target_chunk_id", "")).strip()
                    if target_chunk_id not in next_scores:
                        continue
                    next_scores[target_chunk_id] += score * float(edge.get("weight", 0.0)) * 0.25
            current = next_scores

        return {
            chunk_id: float((0.6 * base_scores[chunk_id]) + (0.4 * current.get(chunk_id, 0.0)))
            for chunk_id in base_scores
        }

    def _embed_query(self, *, query: str) -> Any:
        """Embed one query with the manifest-configured provider."""
        assert self._np is not None
        provider = str(self.manifest.get("embedding_provider", _infer_provider(self.retrieval_model))).strip().lower()
        if provider == "openai":
            if self._openai_client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError("openai is required for OpenAI graph retrieval.") from exc
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is required to run graph retrieval.")
                self._openai_client = OpenAI(api_key=api_key)
            response = self._openai_client.embeddings.create(model=self.retrieval_model, input=[query])
            vector = self._np.array([response.data[0].embedding], dtype=self._np.float32)
        elif provider == "ollama":
            base_url = self.ollama_base_url.rstrip("/")
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": self.retrieval_model, "input": [query]},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or not embeddings:
                raise RuntimeError("Ollama did not return embeddings for graph retrieval.")
            vector = self._np.array([embeddings[0]], dtype=self._np.float32)
        else:
            raise RuntimeError(
                f"Unsupported retrieval embedding provider '{provider}'. Supported: ['openai', 'ollama']."
            )

        if str(self.manifest.get("metric", "cosine")) == "cosine":
            norm = self._np.linalg.norm(vector, axis=1, keepdims=True)
            norm[norm == 0.0] = 1.0
            vector = vector / norm
        return vector


def _resolve_artifact_base_dir(embedded_chunks_path: str | Path) -> Path:
    """Resolve one retrieval artifact directory from a directory or manifest file."""
    raw_path = Path(embedded_chunks_path).expanduser()
    resolved = raw_path.resolve() if raw_path.exists() else raw_path
    if resolved.is_dir():
        return resolved
    if resolved.name == "index_manifest.json":
        return resolved.parent
    raise FileNotFoundError(
        "Unsupported embedded_chunks_path. Expected an index directory or index_manifest.json: "
        f"{raw_path}"
    )


def _infer_provider(model: str) -> str:
    """Infer embedding provider from a model identifier."""
    normalized = str(model).strip().lower()
    if normalized.startswith("text-embedding-"):
        return "openai"
    return "ollama"
