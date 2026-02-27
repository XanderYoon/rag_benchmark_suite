from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_ROOT = PROJECT_ROOT / "data" / "rag_corpus_chunked"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "faiss_rag_index"
ProgressCallback = Callable[[float, str], None]


def _import_dependencies() -> tuple[object, object, object]:
    """Load optional runtime dependencies for FAISS index building."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required. Install with: pip install numpy") from exc

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("faiss is required. Install with: pip install faiss-cpu") from exc

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai is required. Install with: pip install openai") from exc

    return np, faiss, OpenAI


@dataclass(frozen=True)
class ChunkRow:
    faiss_id: int
    paper_id: str
    chunk_id: str
    file_path: str


def resolve_repo_path(path: Path) -> Path:
    """Resolve a possibly relative path against the repository root."""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for FAISS index generation."""
    parser = argparse.ArgumentParser(
        description="Build and store a FAISS index over chunk files for RAG retrieval."
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=DEFAULT_CHUNKS_ROOT,
        help="Root folder containing per-paper chunk .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where FAISS index and metadata are written.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks per embedding request.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="Similarity metric used in FAISS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args(argv)


def discover_chunks(chunks_root: Path) -> list[Path]:
    """Validate chunk storage and return all chunk files."""
    chunks_root = resolve_repo_path(chunks_root)
    if not chunks_root.exists():
        raise FileNotFoundError(f"Chunks root not found: {chunks_root}")

    files = sorted(chunks_root.glob("*/*_chunk_*.txt"))
    if not files:
        raise RuntimeError(f"No chunk files found under: {chunks_root}")
    return files


def chunk_rows_from_files(files: list[Path]) -> tuple[list[ChunkRow], list[str]]:
    """Build chunk metadata rows and load chunk text payloads."""
    rows: list[ChunkRow] = []
    texts: list[str] = []

    for i, path in enumerate(files):
        paper_id = path.parent.name
        rows.append(
            ChunkRow(
                faiss_id=i,
                paper_id=paper_id,
                chunk_id=path.stem,
                file_path=str(path.resolve()),
            )
        )
        texts.append(path.read_text(encoding="utf-8", errors="replace"))

    return rows, texts


def batched(values: list[str], batch_size: int) -> list[list[str]]:
    """Split values into fixed-size batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def embed_texts_openai(
    texts: list[str],
    model: str,
    batch_size: int,
    np: object,
    openai_client_cls: object,
    progress_callback: ProgressCallback | None = None,
) -> object:
    """Embed chunk text with OpenAI and return a dense matrix."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required in the environment.")
    if not texts:
        raise RuntimeError("No chunk texts were provided for embedding.")

    client = openai_client_cls(api_key=api_key)
    vectors = []
    batches = batched(texts, batch_size)

    print(f"Embedding {len(texts)} chunks in {len(batches)} batches...")
    if progress_callback is not None:
        progress_callback(0.1, f"Embedding {len(texts)} chunks in {len(batches)} batches...")
    for idx, batch in enumerate(batches, start=1):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
        if idx % 10 == 0 or idx == len(batches):
            print(f"  completed {idx}/{len(batches)} batches")
        if progress_callback is not None:
            progress_callback(
                0.1 + (0.65 * idx / len(batches)),
                f"Embedded batch {idx}/{len(batches)}",
            )

    matrix = np.array(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(f"Unexpected embedding matrix shape: {matrix.shape}")
    return matrix


def build_index(embeddings: object, metric: str, faiss: object) -> object:
    """Create and populate the FAISS index."""
    dimension = int(embeddings.shape[1])

    if metric == "cosine":
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    return index


def write_outputs(
    output_dir: Path,
    index: object,
    rows: list[ChunkRow],
    model: str,
    metric: str,
    dimension: int,
    num_vectors: int,
    faiss: object,
    overwrite: bool,
) -> None:
    """Persist the FAISS index, metadata rows, and build manifest."""
    output_dir = resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "chunks.faiss"
    metadata_path = output_dir / "chunks_metadata.jsonl"
    manifest_path = output_dir / "index_manifest.json"

    if not overwrite:
        existing = [p for p in [index_path, metadata_path, manifest_path] if p.exists()]
        if existing:
            joined = ", ".join(str(p) for p in existing)
            raise RuntimeError(
                f"Output files already exist ({joined}). Use --overwrite to replace them."
            )

    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model,
        "metric": metric,
        "dimension": dimension,
        "num_vectors": num_vectors,
        "index_file": index_path.name,
        "metadata_file": metadata_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Index build complete")
    print(f"  index:    {index_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  manifest: {manifest_path}")


def build_faiss_index(
    *,
    chunks_root: Path,
    output_dir: Path,
    embedding_model: str,
    batch_size: int,
    metric: str,
    overwrite: bool,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str | int]:
    """Build a FAISS index over stored chunks and return output metadata."""
    if progress_callback is not None:
        progress_callback(0.02, "Loading FAISS build dependencies...")
    np, faiss, openai_client_cls = _import_dependencies()

    if progress_callback is not None:
        progress_callback(0.05, "Discovering chunk files...")
    files = discover_chunks(chunks_root)
    rows, texts = chunk_rows_from_files(files)

    embeddings = embed_texts_openai(
        texts=texts,
        model=embedding_model,
        batch_size=batch_size,
        np=np,
        openai_client_cls=openai_client_cls,
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
        model=embedding_model,
        metric=metric,
        dimension=int(embeddings.shape[1]),
        num_vectors=int(embeddings.shape[0]),
        faiss=faiss,
        overwrite=overwrite,
    )

    resolved_output_dir = resolve_repo_path(output_dir)
    if progress_callback is not None:
        progress_callback(1.0, "FAISS index build complete.")
    return {
        "num_chunks": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "output_dir": str(resolved_output_dir),
        "index_path": str(resolved_output_dir / "chunks.faiss"),
        "metadata_path": str(resolved_output_dir / "chunks_metadata.jsonl"),
        "manifest_path": str(resolved_output_dir / "index_manifest.json"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Build a FAISS index over stored chunk files."""
    args = parse_args(argv)
    build_faiss_index(
        chunks_root=args.chunks_root,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        metric=args.metric,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
