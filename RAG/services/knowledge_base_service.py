from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INDEX_MANIFEST_FILE = "index_manifest.json"
BUILD_METADATA_FILE = "build_metadata.json"
APPEND_METADATA_FILE = "append_metadata.jsonl"
APPEND_STAGING_DIR = "append_staging"
SUPPORTED_METHODS = {"faiss", "lightrag", "graphrag"}


def validate_knowledge_base(*, knowledge_base_dir: str | Path) -> dict[str, Any]:
    """Validate and inspect an existing knowledge base on disk.

    Args:
        knowledge_base_dir: Directory expected to contain retrieval artifacts.

    Returns:
        Dictionary with stable keys describing the validated knowledge base.

    Raises:
        FileNotFoundError: When the directory or required artifacts are missing.
        ValueError: When metadata files are malformed or unsupported.
    """

    base_dir = _resolve_knowledge_base_dir(knowledge_base_dir)
    manifest_path = base_dir / INDEX_MANIFEST_FILE
    manifest = _load_json_file(manifest_path=manifest_path, error_label="index manifest")
    method_id = _resolve_method_id(base_dir=base_dir, manifest=manifest)
    required_artifacts = _resolve_required_artifacts(base_dir=base_dir, manifest=manifest, method_id=method_id)

    missing_paths = [path for path in required_artifacts.values() if not path.exists()]
    if missing_paths:
        joined = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Knowledge base is missing required artifacts: {joined}")

    metadata_path = required_artifacts["metadata_path"]
    chunk_count = _count_metadata_rows(metadata_path=metadata_path)
    if chunk_count <= 0:
        raise ValueError(f"Knowledge base metadata file is empty: {metadata_path}")

    build_metadata_path = base_dir / BUILD_METADATA_FILE
    build_metadata = _load_optional_json_file(build_metadata_path)

    return {
        "knowledge_base_dir": str(base_dir),
        "method_id": method_id,
        "manifest_path": str(manifest_path),
        "build_metadata_path": str(build_metadata_path) if build_metadata_path.exists() else "",
        "append_metadata_path": str(base_dir / APPEND_METADATA_FILE),
        "artifact_paths": {key: str(path) for key, path in required_artifacts.items()},
        "chunk_count": chunk_count,
        "embedding_provider": str(manifest.get("embedding_provider", "")).strip(),
        "embedding_model": str(
            manifest.get("embedding_model", build_metadata.get("embedding_model", ""))
        ).strip(),
        "build_metadata_present": build_metadata_path.exists(),
        "warnings": _build_warnings(build_metadata_present=build_metadata_path.exists()),
    }


def load_knowledge_base(*, knowledge_base_dir: str | Path) -> dict[str, Any]:
    """Validate one knowledge base and return a session-safe load payload."""

    validation_result = validate_knowledge_base(knowledge_base_dir=knowledge_base_dir)
    return {
        **validation_result,
        "status": "loaded",
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }


def append_to_knowledge_base(
    *,
    knowledge_base_dir: str | Path,
    uploaded_files: list[tuple[str, bytes]] | None = None,
    source_directories: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Append uploaded files or source directories to a validated knowledge base.

    Args:
        knowledge_base_dir: Existing knowledge base directory to extend.
        uploaded_files: Uploaded files as `(filename, bytes)` tuples.
        source_directories: Existing directories whose contents should be copied in.

    Returns:
        Dictionary containing append records and the loaded knowledge base payload.

    Raises:
        ValueError: When no append inputs are provided.
        FileNotFoundError: When a selected append directory is missing.
    """

    upload_payloads = uploaded_files or []
    directory_payloads = source_directories or []
    if not upload_payloads and not directory_payloads:
        raise ValueError("Provide at least one uploaded file or source directory to append.")

    validation_result = validate_knowledge_base(knowledge_base_dir=knowledge_base_dir)
    base_dir = Path(validation_result["knowledge_base_dir"])
    append_root = base_dir / APPEND_STAGING_DIR
    append_root.mkdir(parents=True, exist_ok=True)

    append_operation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    append_records: list[dict[str, Any]] = []

    for filename, payload in upload_payloads:
        append_records.append(
            _append_uploaded_file(
                append_root=append_root,
                append_operation_id=append_operation_id,
                filename=filename,
                payload=payload,
            )
        )

    for source_directory in directory_payloads:
        append_records.append(
            _append_source_directory(
                append_root=append_root,
                append_operation_id=append_operation_id,
                source_directory=source_directory,
            )
        )

    append_metadata_path = base_dir / APPEND_METADATA_FILE
    _append_metadata_records(
        append_metadata_path=append_metadata_path,
        knowledge_base_dir=base_dir,
        append_operation_id=append_operation_id,
        append_records=append_records,
    )

    return {
        "append_operation_id": append_operation_id,
        "append_metadata_path": str(append_metadata_path),
        "appended_items": append_records,
        "knowledge_base": load_knowledge_base(knowledge_base_dir=base_dir),
    }


def _resolve_knowledge_base_dir(knowledge_base_dir: str | Path) -> Path:
    """Resolve and validate a knowledge base directory path."""
    candidate = Path(knowledge_base_dir).expanduser()
    resolved = candidate.resolve() if candidate.exists() else candidate
    if not resolved.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {candidate}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Knowledge base path must be a directory: {resolved}")
    return resolved


def _load_json_file(*, manifest_path: Path, error_label: str) -> dict[str, Any]:
    """Load one required JSON file with contextual errors."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {error_label} at {manifest_path}.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {error_label} at {manifest_path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid {error_label} at {manifest_path}. Expected a JSON object.")
    return payload


def _load_optional_json_file(path: Path) -> dict[str, Any]:
    """Load one optional JSON file when present."""
    if not path.exists():
        return {}
    return _load_json_file(manifest_path=path, error_label="build metadata")


def _resolve_method_id(*, base_dir: Path, manifest: dict[str, Any]) -> str:
    """Resolve the retrieval method from manifest data or local artifacts."""
    declared_method = str(manifest.get("method_id", "")).strip().lower()
    if declared_method in SUPPORTED_METHODS:
        return declared_method
    if (base_dir / "chunks.faiss").exists():
        return "faiss"
    if (base_dir / "graph_edges.json").exists() and (base_dir / "chunk_embeddings.npy").exists():
        graph_hint = str(manifest.get("retrieval_framework", "")).strip().lower()
        if graph_hint in {"lightrag", "graphrag"}:
            return graph_hint
        return "lightrag"
    raise ValueError(
        "Could not determine knowledge base retrieval method. Expected one of "
        f"{sorted(SUPPORTED_METHODS)} in the manifest or matching artifact files in {base_dir}."
    )


def _resolve_required_artifacts(*, base_dir: Path, manifest: dict[str, Any], method_id: str) -> dict[str, Path]:
    """Return required artifact paths for one supported knowledge base."""
    metadata_name = str(manifest.get("metadata_file", "chunks_metadata.jsonl")).strip() or "chunks_metadata.jsonl"
    artifact_paths: dict[str, Path] = {
        "manifest_path": base_dir / INDEX_MANIFEST_FILE,
        "metadata_path": base_dir / metadata_name,
    }
    if method_id == "faiss":
        index_name = str(manifest.get("index_file", "chunks.faiss")).strip() or "chunks.faiss"
        artifact_paths["index_path"] = base_dir / index_name
        return artifact_paths

    graph_name = str(manifest.get("graph_file", "graph_edges.json")).strip() or "graph_edges.json"
    embedding_name = (
        str(manifest.get("embeddings_file", manifest.get("embedding_matrix_file", "chunk_embeddings.npy"))).strip()
        or "chunk_embeddings.npy"
    )
    artifact_paths["graph_path"] = base_dir / graph_name
    artifact_paths["embedding_matrix_path"] = base_dir / embedding_name
    return artifact_paths


def _count_metadata_rows(*, metadata_path: Path) -> int:
    """Count and lightly validate JSONL metadata rows."""
    count = 0
    with metadata_path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse chunk metadata at {metadata_path}:{line_number}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid chunk metadata row at {metadata_path}:{line_number}. Expected an object.")
            count += 1
    return count


def _build_warnings(*, build_metadata_present: bool) -> list[str]:
    """Return non-fatal validation warnings."""
    warnings: list[str] = []
    if not build_metadata_present:
        warnings.append("Optional build metadata is missing; core retrieval artifacts are still valid.")
    return warnings


def _append_uploaded_file(
    *,
    append_root: Path,
    append_operation_id: str,
    filename: str,
    payload: bytes,
) -> dict[str, Any]:
    """Write one uploaded file into append staging and return append metadata."""
    safe_name = _safe_entry_name(filename)
    if not safe_name:
        raise ValueError("Uploaded file name is required.")
    target_dir = append_root / append_operation_id / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = _deduplicated_target_path(target_dir=target_dir, name=safe_name)
    target_path.write_bytes(payload)
    return {
        "item_type": "upload",
        "source_name": filename,
        "stored_path": str(target_path),
        "file_count": 1,
        "byte_count": len(payload),
    }


def _append_source_directory(
    *,
    append_root: Path,
    append_operation_id: str,
    source_directory: str | Path,
) -> dict[str, Any]:
    """Copy one source directory into append staging and return append metadata."""
    source_path = Path(source_directory).expanduser()
    resolved_source = source_path.resolve() if source_path.exists() else source_path
    if not resolved_source.exists():
        raise FileNotFoundError(f"Append source directory not found: {source_path}")
    if not resolved_source.is_dir():
        raise FileNotFoundError(f"Append source path must be a directory: {resolved_source}")

    target_parent = append_root / append_operation_id / "directories"
    target_parent.mkdir(parents=True, exist_ok=True)
    target_path = _deduplicated_directory_path(
        target_parent=target_parent,
        name=_safe_entry_name(resolved_source.name) or "directory",
    )
    shutil.copytree(resolved_source, target_path)
    file_count = sum(1 for path in target_path.rglob("*") if path.is_file())
    return {
        "item_type": "directory",
        "source_name": str(resolved_source),
        "stored_path": str(target_path),
        "file_count": file_count,
        "byte_count": sum(path.stat().st_size for path in target_path.rglob("*") if path.is_file()),
    }


def _append_metadata_records(
    *,
    append_metadata_path: Path,
    knowledge_base_dir: Path,
    append_operation_id: str,
    append_records: list[dict[str, Any]],
) -> None:
    """Append one metadata entry per staged append item."""
    append_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    written_at = datetime.now(timezone.utc).isoformat()
    with append_metadata_path.open("a", encoding="utf-8") as file_obj:
        for record in append_records:
            payload = {
                "append_operation_id": append_operation_id,
                "knowledge_base_dir": str(knowledge_base_dir),
                "written_at": written_at,
                **record,
            }
            file_obj.write(json.dumps(payload) + "\n")


def _safe_entry_name(name: str) -> str:
    """Return a filesystem-safe leaf name."""
    return Path(name).name.strip().replace("/", "_").replace("\\", "_")


def _deduplicated_target_path(*, target_dir: Path, name: str) -> Path:
    """Return a non-conflicting file path inside one target directory."""
    candidate = target_dir / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        alternate = target_dir / f"{stem}_{counter}{suffix}"
        if not alternate.exists():
            return alternate
        counter += 1


def _deduplicated_directory_path(*, target_parent: Path, name: str) -> Path:
    """Return a non-conflicting directory path inside one parent directory."""
    candidate = target_parent / name
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        alternate = target_parent / f"{name}_{counter}"
        if not alternate.exists():
            return alternate
        counter += 1
