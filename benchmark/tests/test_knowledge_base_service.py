from __future__ import annotations

import json
from pathlib import Path

from RAG.services.knowledge_base_service import (
    APPEND_METADATA_FILE,
    append_to_knowledge_base,
    load_knowledge_base,
    validate_knowledge_base,
)


def _write_faiss_knowledge_base(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "chunks.faiss").write_bytes(b"faiss-index")
    (base_dir / "chunks_metadata.jsonl").write_text(
        json.dumps({"chunk_id": "paper_chunk_0001", "file_path": str(base_dir / "chunk.txt")}) + "\n",
        encoding="utf-8",
    )
    (base_dir / "chunk.txt").write_text("chunk body", encoding="utf-8")
    (base_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "method_id": "faiss",
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "index_file": "chunks.faiss",
                "metadata_file": "chunks_metadata.jsonl",
            }
        ),
        encoding="utf-8",
    )


def _write_graph_knowledge_base(base_dir: Path, *, method_id: str) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "chunk_embeddings.npy").write_bytes(b"fake-npy")
    (base_dir / "graph_edges.json").write_text(json.dumps({"paper_chunk_0001": []}), encoding="utf-8")
    (base_dir / "chunks_metadata.jsonl").write_text(
        json.dumps({"chunk_id": "paper_chunk_0001", "file_path": str(base_dir / "chunk.txt")}) + "\n",
        encoding="utf-8",
    )
    (base_dir / "chunk.txt").write_text("chunk body", encoding="utf-8")
    (base_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "method_id": method_id,
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
                "metadata_file": "chunks_metadata.jsonl",
                "graph_file": "graph_edges.json",
                "embedding_matrix_file": "chunk_embeddings.npy",
            }
        ),
        encoding="utf-8",
    )


def test_validate_knowledge_base_accepts_faiss_directory(tmp_path: Path) -> None:
    kb_dir = tmp_path / "faiss_kb"
    _write_faiss_knowledge_base(kb_dir)

    payload = validate_knowledge_base(knowledge_base_dir=kb_dir)

    assert payload["method_id"] == "faiss"
    assert payload["chunk_count"] == 1
    assert payload["artifact_paths"]["index_path"].endswith("chunks.faiss")
    assert payload["warnings"]


def test_validate_knowledge_base_requires_method_specific_graph_artifacts(tmp_path: Path) -> None:
    kb_dir = tmp_path / "graph_kb"
    _write_graph_knowledge_base(kb_dir, method_id="graphrag")
    (kb_dir / "graph_edges.json").unlink()

    try:
        validate_knowledge_base(knowledge_base_dir=kb_dir)
    except FileNotFoundError as exc:
        assert "graph_edges.json" in str(exc)
        return

    raise AssertionError("Expected FileNotFoundError for missing graph artifact.")


def test_load_knowledge_base_returns_session_safe_payload(tmp_path: Path) -> None:
    kb_dir = tmp_path / "load_kb"
    _write_faiss_knowledge_base(kb_dir)

    payload = load_knowledge_base(knowledge_base_dir=kb_dir)

    assert payload["status"] == "loaded"
    assert payload["knowledge_base_dir"] == str(kb_dir.resolve())
    assert "loaded_at" in payload


def test_append_to_knowledge_base_records_uploads_and_directories(tmp_path: Path) -> None:
    kb_dir = tmp_path / "append_kb"
    _write_faiss_knowledge_base(kb_dir)
    source_dir = tmp_path / "source_docs"
    source_dir.mkdir()
    (source_dir / "notes.txt").write_text("append me", encoding="utf-8")

    payload = append_to_knowledge_base(
        knowledge_base_dir=kb_dir,
        uploaded_files=[("paper.pdf", b"%PDF-1.4")],
        source_directories=[source_dir],
    )

    metadata_path = kb_dir / APPEND_METADATA_FILE
    metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]

    assert payload["knowledge_base"]["status"] == "loaded"
    assert len(payload["appended_items"]) == 2
    assert len(metadata_rows) == 2
    assert {row["item_type"] for row in metadata_rows} == {"upload", "directory"}
