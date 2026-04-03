from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from benchmark.benchmark_tools.models import (
    BenchmarkProbe,
    ChunkArtifact,
    RetrievalCaseResult,
    RetrievedChunk,
)
from benchmark.benchmark_tools.probe_sources.auto_source import load_auto_probes
from RAG.retrieval.index_runtime import load_index_manifest


def resolve_chunk_artifact_paths(embedded_chunks_path: str | Path) -> dict[str, Path]:
    """Resolve chunk metadata and manifest paths from a retrieval artifact directory."""
    raw_path = Path(embedded_chunks_path).expanduser()
    target_path = raw_path.resolve() if raw_path.exists() else raw_path

    if target_path.is_dir():
        base_dir = target_path
    elif target_path.name in {"chunks.faiss", "chunks_metadata.jsonl", "index_manifest.json"}:
        base_dir = target_path.parent
    else:
        raise FileNotFoundError(
            "Unsupported embedded_chunks_path. Expected an index directory or one of "
            f"'chunks.faiss', 'chunks_metadata.jsonl', or 'index_manifest.json': {raw_path}"
        )

    paths = {"metadata_path": base_dir / "chunks_metadata.jsonl", "manifest_path": base_dir / "index_manifest.json"}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing benchmark artifacts: "
            f"{', '.join(missing)}"
        )
    return paths


def load_chunk_artifacts(embedded_chunks_path: str | Path) -> tuple[list[ChunkArtifact], dict[str, Any]]:
    """Load chunk metadata rows and index manifest for benchmarking."""
    paths = resolve_chunk_artifact_paths(embedded_chunks_path)
    manifest = load_index_manifest(embedded_chunks_path=paths["manifest_path"])
    artifacts: list[ChunkArtifact] = []

    with paths["metadata_path"].open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_path = Path(str(row["file_path"]))
            artifacts.append(
                ChunkArtifact(
                    faiss_id=int(row["faiss_id"]),
                    paper_id=str(row["paper_id"]),
                    chunk_id=str(row["chunk_id"]),
                    file_path=chunk_path,
                    text=chunk_path.read_text(encoding="utf-8", errors="replace"),
                )
            )

    if not artifacts:
        raise RuntimeError(
            f"No chunk metadata rows found in {paths['metadata_path']}."
        )
    return artifacts, manifest


def build_benchmark_probes(
    artifacts: list[ChunkArtifact],
    *,
    max_cases: int = 24,
) -> list[BenchmarkProbe]:
    """Create evenly spaced retrieval probes from chunk text."""
    return load_auto_probes(artifacts=artifacts, max_cases=max_cases)


def summarize_text(text: str, *, limit: int) -> str:
    """Collapse whitespace and return a bounded text snippet."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: max(limit - 3, 0)].rstrip()}..."


class FaissRetriever:
    """Load a FAISS index and run retrieval queries against it."""

    def __init__(
        self,
        *,
        embedded_chunks_path: str | Path,
        retrieval_model: str,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.top_k = top_k
        self.retrieval_model = retrieval_model
        self.paths = resolve_faiss_artifact_paths(embedded_chunks_path)
        self._faiss_index: Any | None = None
        self._faiss: Any | None = None
        self._np: Any | None = None
        self._openai_client: Any | None = None
        self._rows_by_id: dict[int, ChunkArtifact] = {}
        self._metric = "cosine"
        self._embedding_provider = "openai"

    def benchmark(self, probes: list[BenchmarkProbe]) -> list[RetrievalCaseResult]:
        """Run retrieval for each benchmark probe."""
        self._ensure_ready()
        results: list[RetrievalCaseResult] = []
        for probe in probes:
            results.append(self._run_case(probe))
        return results

    def _run_case(self, probe: BenchmarkProbe) -> RetrievalCaseResult:
        assert self._np is not None
        assert self._faiss is not None
        assert self._faiss_index is not None

        vector = self._embed_query(probe.query)
        if self._metric == "cosine":
            self._faiss.normalize_L2(vector)
        distances, indices = self._faiss_index.search(vector, self.top_k)

        retrieved_chunks: list[RetrievedChunk] = []
        for rank, faiss_id in enumerate(indices[0], start=1):
            if int(faiss_id) < 0:
                continue
            row = self._rows_by_id.get(int(faiss_id))
            if row is None:
                continue

            raw_distance = float(distances[0][rank - 1])
            score = raw_distance if self._metric == "cosine" else 1.0 / (1.0 + max(raw_distance, 0.0))
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    score=score,
                    rank=rank,
                )
            )

        return RetrievalCaseResult(
            case_id=probe.case_id,
            query=probe.query,
            expected_chunk_id=probe.expected_chunk_id,
            expected_answer=probe.expected_answer,
            reference_contexts=probe.reference_contexts,
            retrieved_chunks=retrieved_chunks,
        )

    def _ensure_ready(self) -> None:
        if self._faiss_index is not None:
            return

        try:
            import faiss
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "FAISS benchmarking requires optional dependencies. Install with: pip install faiss-cpu numpy"
            ) from exc

        manifest = json.loads(self.paths["manifest_path"].read_text(encoding="utf-8"))
        self._metric = str(manifest.get("metric", "cosine"))
        self._embedding_provider = str(
            manifest.get("embedding_provider", self._infer_provider(self.retrieval_model))
        ).strip().lower()

        with self.paths["metadata_path"].open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                chunk_path = Path(str(row["file_path"]))
                self._rows_by_id[int(row["faiss_id"])] = ChunkArtifact(
                    faiss_id=int(row["faiss_id"]),
                    paper_id=str(row["paper_id"]),
                    chunk_id=str(row["chunk_id"]),
                    file_path=chunk_path,
                    text=chunk_path.read_text(encoding="utf-8", errors="replace"),
                )

        self._faiss = faiss
        self._np = np
        self._faiss_index = faiss.read_index(str(self.paths["index_path"]))

    @staticmethod
    def _infer_provider(model: str) -> str:
        normalized = str(model).strip().lower()
        if normalized.startswith("text-embedding-"):
            return "openai"
        return "ollama"

    def _embed_query(self, query: str) -> Any:
        assert self._np is not None
        provider = self._embedding_provider or self._infer_provider(self.retrieval_model)
        if provider == "openai":
            if self._openai_client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError("openai is required for OpenAI retrieval embeddings.") from exc
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is required to run retrieval benchmarking.")
                self._openai_client = OpenAI(api_key=api_key)
            response = self._openai_client.embeddings.create(
                model=self.retrieval_model,
                input=[query],
            )
            return self._np.array([response.data[0].embedding], dtype=self._np.float32)

        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": self.retrieval_model, "input": [query]},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or not embeddings:
                raise RuntimeError("Ollama did not return embeddings for benchmark retrieval.")
            return self._np.array([embeddings[0]], dtype=self._np.float32)

        raise RuntimeError(f"Unsupported embedding provider '{provider}'.")


def resolve_faiss_artifact_paths(embedded_chunks_path: str | Path) -> dict[str, Path]:
    """Resolve FAISS artifact paths from an index directory or one artifact file."""
    paths = resolve_chunk_artifact_paths(embedded_chunks_path)
    index_path = paths["manifest_path"].parent / "chunks.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS benchmark artifact: {index_path}")
    return {
        "index_path": index_path,
        "metadata_path": paths["metadata_path"],
        "manifest_path": paths["manifest_path"],
    }


def summarize_retrieval_results(results: list[RetrievalCaseResult]) -> dict[str, float | int]:
    """Summarize retrieval metrics into stable numeric fields."""
    total_cases = len(results)
    if total_cases == 0:
        return {
            "num_cases": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "mrr": 0.0,
            "average_top_score": 0.0,
        }

    top_scores = [result.retrieved_chunks[0].score for result in results if result.retrieved_chunks]
    return {
        "num_cases": total_cases,
        "hit_at_1": sum(1 for result in results if result.hit_at_1) / total_cases,
        "hit_at_3": sum(1 for result in results if result.hit_at_3) / total_cases,
        "mrr": sum(result.reciprocal_rank for result in results) / total_cases,
        "average_top_score": (sum(top_scores) / len(top_scores)) if top_scores else 0.0,
    }
