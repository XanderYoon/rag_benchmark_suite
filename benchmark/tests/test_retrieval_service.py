from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.config import DEFAULT_CONFIG
from benchmark.services.retrieval_service import RetrievalService
from RAG.retrieval.index_runtime import GraphArtifactRetriever


def test_can_retry_from_cached_error_when_api_key_now_exists(monkeypatch) -> None:
    service = RetrievalService(DEFAULT_CONFIG)
    service.faiss_error = "OPENAI_API_KEY is not set."

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    assert service._can_retry_from_cached_error() is True


def test_cannot_retry_from_cached_error_without_api_key(monkeypatch) -> None:
    service = RetrievalService(DEFAULT_CONFIG)
    service.faiss_error = "OPENAI_API_KEY is not set."

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert service._can_retry_from_cached_error() is False


def _write_graph_artifacts(tmp_path: Path, *, method_id: str) -> Path:
    output_dir = tmp_path / method_id
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / "paper_a_chunk_0000.txt"
    chunk_path.write_text("alpha context", encoding="utf-8")

    with (output_dir / "chunks_metadata.jsonl").open("w", encoding="utf-8") as file_obj:
        file_obj.write(
            json.dumps(
                {
                    "faiss_id": 0,
                    "paper_id": "paper_a",
                    "chunk_id": "paper_a_chunk_0000",
                    "file_path": str(chunk_path),
                }
            )
            + "\n"
        )

    np.save(output_dir / "chunk_embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    (output_dir / "graph_edges.json").write_text(
        json.dumps({"paper_a_chunk_0000": []}),
        encoding="utf-8",
    )
    (output_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "method_id": method_id,
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "metric": "cosine",
                "dimension": 2,
                "num_vectors": 1,
                "embeddings_file": "chunk_embeddings.npy",
                "graph_file": "graph_edges.json",
                "metadata_file": "chunks_metadata.jsonl",
            }
        ),
        encoding="utf-8",
    )
    return output_dir


def test_retrieve_top_artifact_returns_graph_candidates(monkeypatch, tmp_path: Path) -> None:
    service = RetrievalService(DEFAULT_CONFIG)
    output_dir = _write_graph_artifacts(tmp_path, method_id="lightrag")
    monkeypatch.setattr(
        GraphArtifactRetriever,
        "_embed_query",
        lambda self, query: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    candidates = service.retrieve_top_artifact(
        "alpha question",
        retrieval_method="lightrag",
        limit=3,
        retrieval_model="text-embedding-3-small",
        artifact_output_dir=output_dir,
    )
    chunks = service.load_chunks_for_candidates(candidates)

    assert [candidate.chunk_id for candidate in candidates] == ["paper_a_chunk_0000"]
    assert chunks["paper_a_chunk_0000"].text == "alpha context"


def test_retrieve_top_artifact_rejects_wrong_method_manifest(tmp_path: Path) -> None:
    service = RetrievalService(DEFAULT_CONFIG)
    output_dir = _write_graph_artifacts(tmp_path, method_id="graphrag")

    candidates = service.retrieve_top_artifact(
        "alpha question",
        retrieval_method="lightrag",
        limit=3,
        retrieval_model="text-embedding-3-small",
        artifact_output_dir=output_dir,
    )

    assert candidates == []
    assert service.retrieval_error == "Selected retrieval artifacts are for method 'graphrag', expected 'lightrag'."
