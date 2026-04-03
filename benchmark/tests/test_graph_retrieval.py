from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from RAG.retrieval.index_runtime import GraphArtifactRetriever
from benchmark.benchmark_tools.artifacts import load_chunk_artifacts
from benchmark.benchmark_tools.models import BenchmarkProbe
from benchmark.benchmark_tools.retrieval_runners.graphrag_runner import benchmark_graphrag
from benchmark.benchmark_tools.retrieval_runners.lightrag_runner import benchmark_lightrag


def _write_graph_artifacts(tmp_path: Path, *, method_id: str) -> Path:
    output_dir = tmp_path / method_id
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_a = output_dir / "paper_a_chunk_0000.txt"
    chunk_b = output_dir / "paper_a_chunk_0001.txt"
    chunk_c = output_dir / "paper_a_chunk_0002.txt"
    chunk_a.write_text("alpha context", encoding="utf-8")
    chunk_b.write_text("beta context", encoding="utf-8")
    chunk_c.write_text("gamma context", encoding="utf-8")

    rows = [
        {"faiss_id": 0, "paper_id": "paper_a", "chunk_id": "paper_a_chunk_0000", "file_path": str(chunk_a)},
        {"faiss_id": 1, "paper_id": "paper_a", "chunk_id": "paper_a_chunk_0001", "file_path": str(chunk_b)},
        {"faiss_id": 2, "paper_id": "paper_a", "chunk_id": "paper_a_chunk_0002", "file_path": str(chunk_c)},
    ]
    with (output_dir / "chunks_metadata.jsonl").open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row) + "\n")

    np.save(
        output_dir / "chunk_embeddings.npy",
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    (output_dir / "graph_edges.json").write_text(
        json.dumps(
            {
                "paper_a_chunk_0000": [{"target_chunk_id": "paper_a_chunk_0001", "weight": 1.0}],
                "paper_a_chunk_0001": [{"target_chunk_id": "paper_a_chunk_0000", "weight": 1.0}],
                "paper_a_chunk_0002": [],
            }
        ),
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
                "num_vectors": 3,
                "embeddings_file": "chunk_embeddings.npy",
                "graph_file": "graph_edges.json",
                "metadata_file": "chunks_metadata.jsonl",
            }
        ),
        encoding="utf-8",
    )
    return output_dir


def test_load_chunk_artifacts_supports_graph_indexes(tmp_path: Path) -> None:
    output_dir = _write_graph_artifacts(tmp_path, method_id="lightrag")

    artifacts, manifest = load_chunk_artifacts(output_dir)

    assert len(artifacts) == 3
    assert manifest["method_id"] == "lightrag"
    assert artifacts[0].text == "alpha context"


def test_graph_artifact_retriever_returns_ranked_chunks(monkeypatch, tmp_path: Path) -> None:
    output_dir = _write_graph_artifacts(tmp_path, method_id="graphrag")
    retriever = GraphArtifactRetriever(
        embedded_chunks_path=output_dir,
        retrieval_model="text-embedding-3-small",
        top_k=2,
    )
    monkeypatch.setattr(
        GraphArtifactRetriever,
        "_embed_query",
        lambda self, query: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    results = retriever.retrieve(query="alpha", limit=2)

    assert [chunk.chunk_id for chunk in results] == ["paper_a_chunk_0000", "paper_a_chunk_0001"]
    assert results[0].score >= results[1].score


def test_graph_retrieval_runners_execute_with_local_artifacts(monkeypatch, tmp_path: Path) -> None:
    light_dir = _write_graph_artifacts(tmp_path, method_id="lightrag")
    graph_dir = _write_graph_artifacts(tmp_path, method_id="graphrag")
    monkeypatch.setattr(
        GraphArtifactRetriever,
        "_embed_query",
        lambda self, query: np.array([[1.0, 0.0]], dtype=np.float32),
    )
    probes = [
        BenchmarkProbe(
            case_id="case_1",
            query="alpha question",
            expected_chunk_id="paper_a_chunk_0000",
            expected_answer="alpha answer",
            reference_contexts=["alpha context"],
        )
    ]

    light_results = benchmark_lightrag(
        probes=probes,
        config={"embedded_chunks_path": light_dir, "retrieval_model": "text-embedding-3-small", "top_k": 2},
    )
    graph_results = benchmark_graphrag(
        probes=probes,
        config={"embedded_chunks_path": graph_dir, "retrieval_model": "text-embedding-3-small", "top_k": 2},
    )

    assert light_results[0].top_hit_chunk_id == "paper_a_chunk_0000"
    assert graph_results[0].top_hit_chunk_id == "paper_a_chunk_0000"
