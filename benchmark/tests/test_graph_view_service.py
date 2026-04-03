from __future__ import annotations

import json
from pathlib import Path

from RAG.services.graph_view_service import build_graphviz_dot, build_interactive_graph_payload, load_graph_view_data


def _write_graph_kb(base_dir: Path, *, method_id: str = "lightrag") -> dict:
    base_dir.mkdir(parents=True, exist_ok=True)
    chunk_a = base_dir / "paper_a_chunk_0000.txt"
    chunk_b = base_dir / "paper_a_chunk_0001.txt"
    chunk_a.write_text("alpha context", encoding="utf-8")
    chunk_b.write_text("beta context", encoding="utf-8")
    (base_dir / "graph_edges.json").write_text(
        json.dumps(
            {
                "paper_a_chunk_0000": [{"target_chunk_id": "paper_a_chunk_0001", "weight": 1.0}],
                "paper_a_chunk_0001": [{"target_chunk_id": "paper_a_chunk_0000", "weight": 0.8}],
            }
        ),
        encoding="utf-8",
    )
    (base_dir / "chunks_metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "faiss_id": 0,
                        "paper_id": "paper_a",
                        "chunk_id": "paper_a_chunk_0000",
                        "file_path": str(chunk_a),
                    }
                ),
                json.dumps(
                    {
                        "faiss_id": 1,
                        "paper_id": "paper_a",
                        "chunk_id": "paper_a_chunk_0001",
                        "file_path": str(chunk_b),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (base_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "method_id": method_id,
                "graph_file": "graph_edges.json",
                "metadata_file": "chunks_metadata.jsonl",
            }
        ),
        encoding="utf-8",
    )
    return {"knowledge_base_dir": str(base_dir), "method_id": method_id}


def test_load_graph_view_data_reads_graph_capable_knowledge_base(tmp_path: Path) -> None:
    kb_payload = _write_graph_kb(tmp_path / "graph_kb")

    payload = load_graph_view_data(knowledge_base=kb_payload)

    assert payload["method_id"] == "lightrag"
    assert payload["node_count"] == 2
    assert payload["edge_count"] == 2
    assert payload["paper_ids"] == ["paper_a"]


def test_load_graph_view_data_rejects_faiss_knowledge_base(tmp_path: Path) -> None:
    kb_payload = _write_graph_kb(tmp_path / "faiss_kb", method_id="faiss")

    try:
        load_graph_view_data(knowledge_base=kb_payload)
    except ValueError as exc:
        assert "LightRAG or GraphRAG" in str(exc)
        return

    raise AssertionError("Expected ValueError for non-graph knowledge base.")


def test_build_graphviz_dot_filters_visible_nodes() -> None:
    graph_payload = {
        "nodes": [
            {"chunk_id": "paper_a_chunk_0000", "paper_id": "paper_a", "degree": 2},
            {"chunk_id": "paper_b_chunk_0000", "paper_id": "paper_b", "degree": 1},
        ],
        "edges": [
            {"source_chunk_id": "paper_a_chunk_0000", "target_chunk_id": "paper_b_chunk_0000", "weight": 0.5}
        ],
    }

    dot_graph, visible_nodes, visible_edges = build_graphviz_dot(
        graph_payload=graph_payload,
        selected_paper_id="paper_a",
        max_nodes=10,
    )

    assert "paper_a_chunk_0000" in dot_graph
    assert "paper_b_chunk_0000" not in dot_graph
    assert len(visible_nodes) == 1
    assert visible_edges == []


def test_build_interactive_graph_payload_assigns_positions() -> None:
    graph_payload = {
        "nodes": [
            {"chunk_id": "paper_a_chunk_0000", "paper_id": "paper_a", "degree": 2, "preview_text": "alpha"},
            {"chunk_id": "paper_a_chunk_0001", "paper_id": "paper_a", "degree": 1, "preview_text": "beta"},
        ],
        "edges": [
            {"source_chunk_id": "paper_a_chunk_0000", "target_chunk_id": "paper_a_chunk_0001", "weight": 1.0}
        ],
    }

    payload = build_interactive_graph_payload(
        graph_payload=graph_payload,
        selected_paper_id="paper_a",
        max_nodes=10,
    )

    assert len(payload["nodes"]) == 2
    assert payload["nodes"][0]["x"] > 0
    assert payload["nodes"][0]["y"] > 0
    assert payload["edges"][0]["source_chunk_id"] == "paper_a_chunk_0000"
