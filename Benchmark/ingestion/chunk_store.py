from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from Benchmark.domain.models import Chunk


class ChunkStore:
    def __init__(self, chunk_root: Path) -> None:
        self.chunk_root = chunk_root
        self.chunk_root.mkdir(parents=True, exist_ok=True)

    def write_chunks(self, paper_id: str, chunks: list[Chunk]) -> list[Path]:
        paper_dir = self.chunk_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for chunk in chunks:
            path = paper_dir / f"{chunk.chunk_id}.txt"
            path.write_text(chunk.text, encoding="utf-8", errors="replace")
            paths.append(path)
        return paths

    def read_chunks(self, paper_id: str) -> list[Chunk]:
        paper_dir = self.chunk_root / paper_id
        files = sorted(paper_dir.glob(f"{paper_id}_chunk_*.txt"))
        chunks: list[Chunk] = []
        for file in files:
            index = int(file.stem.split("_")[-1])
            chunks.append(
                Chunk(
                    chunk_id=file.stem,
                    paper_id=paper_id,
                    text=file.read_text(encoding="utf-8"),
                    index=index,
                )
            )
        return chunks

    def update_manifest(self, manifest_path: Path, paper_id: str, source_path: Path, chunk_count: int) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))

        digest = sha256(source_path.read_bytes()).hexdigest()
        existing[paper_id] = {
            "paper_id": paper_id,
            "source_path": str(source_path),
            "sha256": digest,
            "chunk_count": chunk_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
