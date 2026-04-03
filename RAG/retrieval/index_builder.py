from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from RAG.embedding.build_faiss_rag_index import (
    _import_dependencies,
    build_index,
    chunk_rows_from_files,
    discover_chunks,
    embed_texts_ollama,
    embed_texts_openai,
    resolve_repo_path,
    write_outputs,
)


ProgressCallback = Callable[[float, str], None]
SUPPORTED_RETRIEVAL_METHODS = ("faiss", "lightrag", "graphrag")
BUILD_METADATA_FILE = "build_metadata.json"


def build_retrieval_index(
    *,
    method_id: str,
    chunks_root: Path,
    output_dir: Path,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    metric: str,
    overwrite: bool,
    ollama_base_url: str = "http://localhost:11434",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str | int]:
    """Build retrieval artifacts for the selected retrieval method."""
    normalized_method = str(method_id).strip().lower()
    if normalized_method not in SUPPORTED_RETRIEVAL_METHODS:
        raise ValueError(
            f"Unsupported retrieval method '{method_id}'. Supported: {list(SUPPORTED_RETRIEVAL_METHODS)}"
        )

    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()

    if normalized_method == "faiss":
        return _build_faiss_artifacts(
            chunks_root=chunks_root,
            output_dir=output_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            batch_size=batch_size,
            metric=metric,
            overwrite=overwrite,
            ollama_base_url=ollama_base_url,
            started_at=started_at,
            started_perf=started_perf,
            progress_callback=progress_callback,
        )

    return _build_graph_artifacts(
        method_id=normalized_method,
        chunks_root=chunks_root,
        output_dir=output_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        metric=metric,
        overwrite=overwrite,
        ollama_base_url=ollama_base_url,
        started_at=started_at,
        started_perf=started_perf,
        progress_callback=progress_callback,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for retrieval artifact generation."""
    parser = argparse.ArgumentParser(description="Build retrieval artifacts for FAISS, LightRAG, or GraphRAG.")
    parser.add_argument("--method-id", choices=list(SUPPORTED_RETRIEVAL_METHODS), default="faiss")
    parser.add_argument("--chunks-root", type=Path, default=Path("data/rag_corpus_chunked"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    return parser.parse_args(argv)


def _build_faiss_artifacts(
    *,
    chunks_root: Path,
    output_dir: Path,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    metric: str,
    overwrite: bool,
    ollama_base_url: str,
    started_at: datetime,
    started_perf: float,
    progress_callback: ProgressCallback | None,
) -> dict[str, str | int]:
    """Build FAISS artifacts using the existing FAISS implementation."""
    if progress_callback is not None:
        progress_callback(0.02, "Loading FAISS build dependencies...")
    np, faiss = _import_dependencies()
    if progress_callback is not None:
        progress_callback(0.05, "Discovering chunk files...")
    files = discover_chunks(chunks_root)
    rows, texts = chunk_rows_from_files(files)
    corpus_metadata = _build_corpus_metadata(files=files, texts=texts, chunks_root=chunks_root)
    embeddings = _embed_chunk_texts(
        texts=texts,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        ollama_base_url=ollama_base_url,
        np=np,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(0.82, "Building FAISS index...")
    index = build_index(embeddings=embeddings, metric=metric, faiss=faiss)
    if progress_callback is not None:
        progress_callback(0.92, "Writing index artifacts...")
    write_outputs(
        output_dir=output_dir,
        index=index,
        rows=rows,
        provider=str(embedding_provider).strip().lower(),
        model=embedding_model,
        metric=metric,
        dimension=int(embeddings.shape[1]),
        num_vectors=int(embeddings.shape[0]),
        faiss=faiss,
        overwrite=overwrite,
    )
    resolved_output_dir = resolve_repo_path(output_dir)
    metadata_path = _write_build_metadata(
        output_dir=resolved_output_dir,
        overwrite=overwrite,
        metadata=_build_build_metadata(
            method_id="faiss",
            chunks_root=chunks_root,
            output_dir=resolved_output_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            batch_size=batch_size,
            metric=metric,
            overwrite=overwrite,
            corpus_metadata=corpus_metadata,
            started_at=started_at,
            started_perf=started_perf,
            finished_at=datetime.now(timezone.utc),
            artifact_summary={
                "num_chunks": int(embeddings.shape[0]),
                "dimension": int(embeddings.shape[1]),
                "index_path": str(resolved_output_dir / "chunks.faiss"),
                "chunk_metadata_path": str(resolved_output_dir / "chunks_metadata.jsonl"),
                "manifest_path": str(resolved_output_dir / "index_manifest.json"),
            },
        ),
    )
    return {
        "method_id": "faiss",
        "num_chunks": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "embedding_provider": str(embedding_provider).strip().lower(),
        "embedding_model": embedding_model,
        "output_dir": str(resolved_output_dir),
        "index_path": str(resolved_output_dir / "chunks.faiss"),
        "metadata_path": str(resolved_output_dir / "chunks_metadata.jsonl"),
        "manifest_path": str(resolved_output_dir / "index_manifest.json"),
        "build_metadata_path": str(metadata_path),
    }


def _build_graph_artifacts(
    *,
    method_id: str,
    chunks_root: Path,
    output_dir: Path,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    metric: str,
    overwrite: bool,
    ollama_base_url: str,
    started_at: datetime,
    started_perf: float,
    progress_callback: ProgressCallback | None,
) -> dict[str, str | int]:
    """Build local graph-assisted retrieval artifacts for LightRAG or GraphRAG."""
    if progress_callback is not None:
        progress_callback(0.02, f"Loading dependencies for {method_id}...")
    np = _import_numpy_dependency()
    if progress_callback is not None:
        progress_callback(0.05, "Discovering chunk files...")
    files = discover_chunks(chunks_root)
    rows, texts = chunk_rows_from_files(files)
    corpus_metadata = _build_corpus_metadata(files=files, texts=texts, chunks_root=chunks_root)
    embeddings = _embed_chunk_texts(
        texts=texts,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        ollama_base_url=ollama_base_url,
        np=np,
        progress_callback=progress_callback,
    )

    if metric == "cosine":
        normalized_embeddings = embeddings.copy()
        row_norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        normalized_embeddings = normalized_embeddings / row_norms
    else:
        normalized_embeddings = embeddings

    if progress_callback is not None:
        progress_callback(0.82, f"Building {method_id} graph relationships...")
    graph_edges = _build_graph_edges(
        method_id=method_id,
        rows=rows,
        embeddings=normalized_embeddings,
        np=np,
    )
    if progress_callback is not None:
        progress_callback(0.92, f"Writing {method_id} artifacts...")
    resolved_output_dir = resolve_repo_path(output_dir)
    _write_graph_outputs(
        output_dir=resolved_output_dir,
        rows=rows,
        embeddings=normalized_embeddings,
        graph_edges=graph_edges,
        provider=str(embedding_provider).strip().lower(),
        model=embedding_model,
        metric=metric,
        overwrite=overwrite,
        method_id=method_id,
        np=np,
    )
    metadata_path = _write_build_metadata(
        output_dir=resolved_output_dir,
        overwrite=overwrite,
        metadata=_build_build_metadata(
            method_id=method_id,
            chunks_root=chunks_root,
            output_dir=resolved_output_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            batch_size=batch_size,
            metric=metric,
            overwrite=overwrite,
            corpus_metadata=corpus_metadata,
            started_at=started_at,
            started_perf=started_perf,
            finished_at=datetime.now(timezone.utc),
            artifact_summary={
                "num_chunks": int(normalized_embeddings.shape[0]),
                "dimension": int(normalized_embeddings.shape[1]),
                "embedding_matrix_path": str(resolved_output_dir / "chunk_embeddings.npy"),
                "graph_path": str(resolved_output_dir / "graph_edges.json"),
                "chunk_metadata_path": str(resolved_output_dir / "chunks_metadata.jsonl"),
                "manifest_path": str(resolved_output_dir / "index_manifest.json"),
            },
        ),
    )
    if progress_callback is not None:
        progress_callback(1.0, f"{method_id} index build complete.")
    return {
        "method_id": method_id,
        "num_chunks": int(normalized_embeddings.shape[0]),
        "dimension": int(normalized_embeddings.shape[1]),
        "embedding_provider": str(embedding_provider).strip().lower(),
        "embedding_model": embedding_model,
        "output_dir": str(resolved_output_dir),
        "embedding_matrix_path": str(resolved_output_dir / "chunk_embeddings.npy"),
        "graph_path": str(resolved_output_dir / "graph_edges.json"),
        "metadata_path": str(resolved_output_dir / "chunks_metadata.jsonl"),
        "manifest_path": str(resolved_output_dir / "index_manifest.json"),
        "build_metadata_path": str(metadata_path),
    }


def _import_numpy_dependency() -> object:
    """Load numpy for graph-artifact generation."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required. Install with: pip install numpy") from exc
    return np


def _embed_chunk_texts(
    *,
    texts: list[str],
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    ollama_base_url: str,
    np: object,
    progress_callback: ProgressCallback | None,
) -> object:
    """Embed chunk text using the selected provider."""
    normalized_provider = str(embedding_provider).strip().lower()
    if normalized_provider == "openai":
        return embed_texts_openai(
            texts=texts,
            model=embedding_model,
            batch_size=batch_size,
            np=np,
            progress_callback=progress_callback,
        )
    if normalized_provider == "ollama":
        return embed_texts_ollama(
            texts=texts,
            model=embedding_model,
            batch_size=batch_size,
            np=np,
            ollama_base_url=ollama_base_url,
            progress_callback=progress_callback,
        )
    raise RuntimeError(
        f"Unsupported embedding provider '{embedding_provider}'. Supported: ['openai', 'ollama']"
    )


def _build_graph_edges(*, method_id: str, rows: list[object], embeddings: object, np: object) -> dict[str, list[dict[str, float | str]]]:
    """Build graph edges used by the local graph-assisted retrievers."""
    rows_by_paper: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        rows_by_paper[str(row.paper_id)].append((index, row))

    graph_edges: dict[str, list[dict[str, float | str]]] = {str(row.chunk_id): [] for row in rows}
    similar_neighbors = 2 if method_id == "lightrag" else 4

    for paper_rows in rows_by_paper.values():
        ordered_rows = sorted(paper_rows, key=lambda item: str(item[1].chunk_id))
        local_indexes = [index for index, _ in ordered_rows]
        paper_embeddings = embeddings[local_indexes]
        similarity_matrix = paper_embeddings @ paper_embeddings.T

        for local_index, (_, row) in enumerate(ordered_rows):
            chunk_id = str(row.chunk_id)
            edges: dict[str, float] = {}

            for neighbor_offset in (-1, 1):
                neighbor_local_index = local_index + neighbor_offset
                if 0 <= neighbor_local_index < len(ordered_rows):
                    neighbor_chunk_id = str(ordered_rows[neighbor_local_index][1].chunk_id)
                    edges[neighbor_chunk_id] = max(edges.get(neighbor_chunk_id, 0.0), 1.0)

            candidate_indexes = np.argsort(-similarity_matrix[local_index]).tolist()
            added = 0
            for candidate_local_index in candidate_indexes:
                if candidate_local_index == local_index:
                    continue
                weight = float(similarity_matrix[local_index][candidate_local_index])
                if weight <= 0:
                    continue
                neighbor_chunk_id = str(ordered_rows[candidate_local_index][1].chunk_id)
                edges[neighbor_chunk_id] = max(edges.get(neighbor_chunk_id, 0.0), weight)
                added += 1
                if added >= similar_neighbors:
                    break

            graph_edges[chunk_id] = [
                {"target_chunk_id": neighbor_chunk_id, "weight": round(weight, 6)}
                for neighbor_chunk_id, weight in sorted(edges.items(), key=lambda item: (-item[1], item[0]))
            ]

    return graph_edges


def _write_graph_outputs(
    *,
    output_dir: Path,
    rows: list[object],
    embeddings: object,
    graph_edges: dict[str, list[dict[str, float | str]]],
    provider: str,
    model: str,
    metric: str,
    overwrite: bool,
    method_id: str,
    np: object,
) -> None:
    """Persist graph-assisted retrieval artifacts and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "chunk_embeddings.npy"
    graph_path = output_dir / "graph_edges.json"
    metadata_path = output_dir / "chunks_metadata.jsonl"
    manifest_path = output_dir / "index_manifest.json"
    existing = [path for path in (embeddings_path, graph_path, metadata_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"Output files already exist ({joined}). Use overwrite to replace them.")

    np.save(str(embeddings_path), embeddings)
    graph_path.write_text(json.dumps(graph_edges, indent=2), encoding="utf-8")
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row.__dict__) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method_id": method_id,
        "embedding_provider": provider,
        "embedding_model": model,
        "metric": metric,
        "dimension": int(embeddings.shape[1]),
        "num_vectors": int(embeddings.shape[0]),
        "embeddings_file": embeddings_path.name,
        "graph_file": graph_path.name,
        "metadata_file": metadata_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _build_corpus_metadata(*, files: list[Path], texts: list[str], chunks_root: Path) -> dict[str, object]:
    """Build stable corpus metadata for one retrieval artifact build."""
    total_characters = sum(len(text) for text in texts)
    total_tokens = sum(len(text.split()) for text in texts)
    paper_ids = sorted({path.parent.name for path in files})
    per_paper_chunk_counts: dict[str, int] = defaultdict(int)
    for path in files:
        per_paper_chunk_counts[path.parent.name] += 1

    return {
        "chunks_root": str(resolve_repo_path(chunks_root)),
        "paper_count": len(paper_ids),
        "paper_ids": paper_ids,
        "chunk_count": len(files),
        "total_characters": total_characters,
        "total_tokens_estimate": total_tokens,
        "average_tokens_per_chunk_estimate": (total_tokens / len(files)) if files else 0.0,
        "average_characters_per_chunk": (total_characters / len(files)) if files else 0.0,
        "per_paper_chunk_counts": dict(sorted(per_paper_chunk_counts.items())),
        "source_files": [str(path.resolve()) for path in files],
    }


def _build_build_metadata(
    *,
    method_id: str,
    chunks_root: Path,
    output_dir: Path,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    metric: str,
    overwrite: bool,
    corpus_metadata: dict[str, object],
    started_at: datetime,
    started_perf: float,
    finished_at: datetime,
    artifact_summary: dict[str, object],
) -> dict[str, object]:
    """Build one rich metadata payload for a retrieval artifact build."""
    duration_seconds = max(time.perf_counter() - started_perf, 0.0)
    total_tokens_estimate = int(corpus_metadata.get("total_tokens_estimate", 0))

    return {
        "schema_version": 1,
        "build_started_at": started_at.isoformat(),
        "build_finished_at": finished_at.isoformat(),
        "build_duration_seconds": duration_seconds,
        "retrieval_framework": method_id,
        "embedding_provider": str(embedding_provider).strip().lower(),
        "embedding_model": embedding_model,
        "batch_size": int(batch_size),
        "metric": metric,
        "overwrite": bool(overwrite),
        "output_dir": str(output_dir),
        "chunks_root": str(resolve_repo_path(chunks_root)),
        "estimated_embedding_input_tokens": total_tokens_estimate,
        "corpus": corpus_metadata,
        "artifacts": artifact_summary,
    }


def _write_build_metadata(*, output_dir: Path, overwrite: bool, metadata: dict[str, object]) -> Path:
    """Write build metadata JSON alongside retrieval artifacts."""
    metadata_path = output_dir / BUILD_METADATA_FILE
    if metadata_path.exists() and not overwrite:
        raise RuntimeError(f"Output file already exists ({metadata_path}). Use overwrite to replace it.")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main(argv: Sequence[str] | None = None) -> None:
    """Build retrieval artifacts from chunk text files."""
    args = parse_args(argv)
    build_retrieval_index(
        method_id=args.method_id,
        chunks_root=args.chunks_root,
        output_dir=args.output_dir,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        metric=args.metric,
        overwrite=args.overwrite,
        ollama_base_url=args.ollama_base_url,
    )


if __name__ == "__main__":
    main()
