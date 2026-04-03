from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_graph_view_data(*, knowledge_base: dict[str, Any]) -> dict[str, Any]:
    """Load graph visualization data from a loaded LightRAG or GraphRAG KB.

    Args:
        knowledge_base: Session payload for the loaded knowledge base.

    Returns:
        Dictionary with stable keys for graph nodes, edges, and paper filters.

    Raises:
        ValueError: When the loaded KB is missing graph-capable metadata.
        FileNotFoundError: When required graph files are missing.
    """

    method_id = str(knowledge_base.get("method_id", "")).strip().lower()
    if method_id not in {"lightrag", "graphrag"}:
        raise ValueError(
            "The loaded knowledge base does not contain graph artifacts. Load a LightRAG or GraphRAG KB first."
        )

    base_dir = Path(str(knowledge_base.get("knowledge_base_dir", "")).strip()).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {base_dir}")

    manifest_path = base_dir / "index_manifest.json"
    graph_path = base_dir / "graph_edges.json"
    metadata_path = base_dir / "chunks_metadata.jsonl"
    missing = [str(path) for path in (manifest_path, graph_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing graph visualization artifacts: {', '.join(missing)}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph_edges = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(graph_edges, dict):
        raise ValueError(f"Invalid graph file at {graph_path}. Expected a JSON object.")

    nodes_by_chunk_id: dict[str, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse graph metadata at {metadata_path}:{line_number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid graph metadata row at {metadata_path}:{line_number}. Expected an object.")
            chunk_id = str(row.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            paper_id = str(row.get("paper_id", "")).strip()
            file_path = Path(str(row.get("file_path", "")).strip())
            preview_text = ""
            if file_path.exists():
                preview_text = file_path.read_text(encoding="utf-8", errors="replace")[:220]
            nodes_by_chunk_id[chunk_id] = {
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "file_path": str(file_path),
                "preview_text": preview_text,
                "degree": 0,
            }

    edges: list[dict[str, Any]] = []
    for source_chunk_id, raw_edges in graph_edges.items():
        source_id = str(source_chunk_id).strip()
        source_node = nodes_by_chunk_id.setdefault(
            source_id,
            {
                "chunk_id": source_id,
                "paper_id": "",
                "file_path": "",
                "preview_text": "",
                "degree": 0,
            },
        )
        if not isinstance(raw_edges, list):
            continue
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            target_chunk_id = str(raw_edge.get("target_chunk_id", "")).strip()
            if not target_chunk_id:
                continue
            target_node = nodes_by_chunk_id.setdefault(
                target_chunk_id,
                {
                    "chunk_id": target_chunk_id,
                    "paper_id": "",
                    "file_path": "",
                    "preview_text": "",
                    "degree": 0,
                },
            )
            weight = float(raw_edge.get("weight", 0.0))
            source_node["degree"] = int(source_node.get("degree", 0)) + 1
            target_node["degree"] = int(target_node.get("degree", 0)) + 1
            edges.append(
                {
                    "source_chunk_id": source_id,
                    "target_chunk_id": target_chunk_id,
                    "weight": weight,
                }
            )

    nodes = sorted(
        nodes_by_chunk_id.values(),
        key=lambda item: (-int(item.get("degree", 0)), str(item.get("paper_id", "")), str(item.get("chunk_id", ""))),
    )
    papers = sorted({str(node.get("paper_id", "")).strip() for node in nodes if str(node.get("paper_id", "")).strip()})
    return {
        "method_id": method_id,
        "manifest": manifest,
        "nodes": nodes,
        "edges": edges,
        "paper_ids": papers,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def build_graphviz_dot(
    *,
    graph_payload: dict[str, Any],
    selected_paper_id: str | None,
    max_nodes: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a Graphviz DOT graph for a filtered chunk graph view."""

    all_nodes = list(graph_payload.get("nodes", []))
    all_edges = list(graph_payload.get("edges", []))
    paper_filter = str(selected_paper_id or "").strip()

    filtered_nodes = [
        node for node in all_nodes if not paper_filter or str(node.get("paper_id", "")).strip() == paper_filter
    ]
    limited_nodes = filtered_nodes[: max(1, int(max_nodes))]
    allowed_chunk_ids = {str(node.get("chunk_id", "")).strip() for node in limited_nodes}
    filtered_edges = [
        edge
        for edge in all_edges
        if str(edge.get("source_chunk_id", "")).strip() in allowed_chunk_ids
        and str(edge.get("target_chunk_id", "")).strip() in allowed_chunk_ids
    ]

    lines = [
        "digraph knowledge_graph {",
        '  graph [overlap=false, splines=true, rankdir=LR];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f0f9ff", color="#7dd3fc"];',
        '  edge [color="#94a3b8"];',
    ]
    for node in limited_nodes:
        chunk_id = str(node.get("chunk_id", "")).strip()
        paper_id = str(node.get("paper_id", "")).strip()
        degree = int(node.get("degree", 0))
        label = _escape_dot_label(f"{paper_id}\\n{chunk_id}\\ndegree={degree}")
        lines.append(f'  "{chunk_id}" [label="{label}"];')
    for edge in filtered_edges:
        source_chunk_id = str(edge.get("source_chunk_id", "")).strip()
        target_chunk_id = str(edge.get("target_chunk_id", "")).strip()
        weight = float(edge.get("weight", 0.0))
        pen_width = max(1.0, min(4.0, 1.0 + weight))
        lines.append(
            f'  "{source_chunk_id}" -> "{target_chunk_id}" '
            f'[label="{weight:.2f}", penwidth={pen_width:.2f}];'
        )
    lines.append("}")
    return "\n".join(lines), limited_nodes, filtered_edges


def build_interactive_graph_payload(
    *,
    graph_payload: dict[str, Any],
    selected_paper_id: str | None,
    max_nodes: int,
) -> dict[str, Any]:
    """Build a filtered graph payload with deterministic node positions."""
    all_nodes = list(graph_payload.get("nodes", []))
    all_edges = list(graph_payload.get("edges", []))
    paper_filter = str(selected_paper_id or "").strip()

    filtered_nodes = [
        node for node in all_nodes if not paper_filter or str(node.get("paper_id", "")).strip() == paper_filter
    ]
    limited_nodes = filtered_nodes[: max(1, int(max_nodes))]
    laid_out_nodes = _layout_nodes(nodes=limited_nodes)
    allowed_chunk_ids = {str(node.get("chunk_id", "")).strip() for node in laid_out_nodes}
    filtered_edges = [
        edge
        for edge in all_edges
        if str(edge.get("source_chunk_id", "")).strip() in allowed_chunk_ids
        and str(edge.get("target_chunk_id", "")).strip() in allowed_chunk_ids
    ]
    return {
        "nodes": laid_out_nodes,
        "edges": filtered_edges,
    }


def _escape_dot_label(value: str) -> str:
    """Escape Graphviz label text."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _layout_nodes(*, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign deterministic seed positions for an interactive graph canvas."""
    if not nodes:
        return []
    laid_out: list[dict[str, Any]] = []
    count = len(nodes)
    center_x = 520.0
    center_y = 320.0
    base_radius = max(140.0, min(320.0, 70.0 + (count * 14.0)))
    ordered_nodes = sorted(nodes, key=lambda item: str(item.get("chunk_id", "")))
    for index, raw_node in enumerate(ordered_nodes):
        node = dict(raw_node)
        angle = (2.0 * 3.141592653589793 * index) / max(count, 1)
        orbit = base_radius + ((index % 3) * 28.0)
        chunk_id = str(node.get("chunk_id", "")).strip()
        node["x"] = center_x + (orbit * __import__("math").cos(angle))
        node["y"] = center_y + (orbit * __import__("math").sin(angle))
        node["radius"] = max(18, min(34, 16 + (2 * int(node.get("degree", 0)))))
        node["label"] = chunk_id
        laid_out.append(node)
    return laid_out
