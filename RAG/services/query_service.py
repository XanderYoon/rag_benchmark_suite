from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

from benchmark.config import DEFAULT_CONFIG
from benchmark.domain.models import Chunk
from benchmark.services.retrieval_service import RetrievalService
from RAG.ingestion.chunker import Chunker
from RAG.llm.provider_client import generate_llm_text


SUPPORTED_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".log",
}


def run_query(
    *,
    question: str,
    knowledge_base: dict[str, Any],
    retrieval_provider: str,
    retrieval_model: str,
    generation_provider: str,
    generation_model: str,
    diversity: float,
    creativity: float,
    uploaded_files: list[tuple[str, bytes]] | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> dict[str, Any]:
    """Execute one query against a loaded KB and optional temporary uploads.

    Args:
        question: User query text.
        knowledge_base: Loaded KB session payload.
        retrieval_provider: Provider used for retrieval embeddings.
        retrieval_model: Embedding model used for retrieval.
        generation_provider: Provider used for final answer generation.
        generation_model: Chat model used for final answer generation.
        diversity: Retrieval diversity control in ``[0, 1]``.
        creativity: Generation creativity control in ``[0, 1]``.
        uploaded_files: Optional `(filename, bytes)` tuples for query-only context.
        ollama_base_url: Optional Ollama endpoint override.
        openai_api_key: Optional OpenAI API key override.

    Returns:
        Dictionary with stable keys for answer text, sources, warnings, and context.

    Raises:
        ValueError: When the question or KB payload is invalid.
    """

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Query text is required.")

    kb_dir = str(knowledge_base.get("knowledge_base_dir", "")).strip()
    kb_method = str(knowledge_base.get("method_id", "")).strip().lower()
    if not kb_dir or not kb_method:
        raise ValueError("Loaded knowledge base payload is missing required directory or method fields.")

    config = DEFAULT_CONFIG
    if ollama_base_url:
        config = replace(DEFAULT_CONFIG, ollama_base_url=ollama_base_url)
    retrieval_service = RetrievalService(config)
    kb_limit = _kb_retrieval_limit(diversity=diversity)
    kb_candidates = retrieval_service.retrieve_top_artifact(
        normalized_question,
        retrieval_method=kb_method,
        limit=kb_limit,
        retrieval_model=retrieval_model,
        retrieval_provider=retrieval_provider,
        artifact_output_dir=kb_dir,
    )
    kb_chunks = retrieval_service.load_chunks_for_candidates(kb_candidates)
    kb_context_entries = _build_kb_context_entries(candidates=kb_candidates, chunks_by_id=kb_chunks)

    upload_chunks: list[Chunk] = []
    upload_warnings: list[str] = []
    for filename, payload in uploaded_files or []:
        parsed_chunks, warnings = parse_query_upload(
            filename=filename,
            payload=payload,
            chunk_size_tokens=DEFAULT_CONFIG.chunk_size_tokens,
            chunk_overlap_tokens=DEFAULT_CONFIG.chunk_overlap_tokens,
        )
        upload_chunks.extend(parsed_chunks)
        upload_warnings.extend(warnings)

    upload_context_entries = _build_upload_context_entries(
        question=normalized_question,
        upload_chunks=upload_chunks,
        diversity=diversity,
    )

    context_text = _build_context_text(
        question=normalized_question,
        kb_context_entries=kb_context_entries,
        upload_context_entries=upload_context_entries,
    )
    generation_warning = ""
    answer_text = generate_llm_text(
        provider=generation_provider,
        model=generation_model,
        system_prompt=_query_system_prompt(),
        user_prompt=_query_user_prompt(question=normalized_question, context_text=context_text),
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
        temperature=_creativity_to_temperature(creativity),
        top_p=_diversity_to_top_p(diversity),
    )
    if not answer_text:
        generation_warning = (
            "Model generation is unavailable with the current provider settings. Retrieved context is shown below."
        )
        answer_text = _fallback_answer_text(
            question=normalized_question,
            kb_context_entries=kb_context_entries,
            upload_context_entries=upload_context_entries,
        )

    retrieval_error = retrieval_service.retrieval_error or ""
    warnings = [warning for warning in [retrieval_error, generation_warning, *upload_warnings] if warning]
    return {
        "question": normalized_question,
        "answer_text": answer_text,
        "context_text": context_text,
        "warnings": warnings,
        "knowledge_base_sources": kb_context_entries,
        "upload_sources": upload_context_entries,
        "retrieval_provider": retrieval_provider,
        "retrieval_model": retrieval_model,
        "generation_provider": generation_provider,
        "generation_model": generation_model,
        "diversity": float(diversity),
        "creativity": float(creativity),
    }


def parse_query_upload(
    *,
    filename: str,
    payload: bytes,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> tuple[list[Chunk], list[str]]:
    """Parse one uploaded file into temporary query chunks."""
    suffix = Path(filename).suffix.lower()
    extracted_text = ""
    warnings: list[str] = []
    if suffix == ".pdf":
        extracted_text = _extract_pdf_text(filename=filename, payload=payload)
    elif suffix in SUPPORTED_TEXT_SUFFIXES:
        extracted_text = _decode_text_payload(filename=filename, payload=payload)
    else:
        warnings.append(f"Skipped unsupported query upload '{filename}'. Supported: PDF and text-like files.")
        return [], warnings

    normalized_text = extracted_text.strip()
    if not normalized_text:
        warnings.append(f"No usable text was extracted from uploaded file '{filename}'.")
        return [], warnings

    paper_id = _upload_paper_id(filename)
    chunker = Chunker(chunk_size_tokens, chunk_overlap_tokens)
    return chunker.chunk_text(paper_id=paper_id, text=normalized_text), warnings


def _extract_pdf_text(*, filename: str, payload: bytes) -> str:
    """Extract text from one PDF payload."""
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required for PDF query uploads.") from exc
    try:
        reader = PdfReader(BytesIO(payload))
    except Exception as exc:
        raise RuntimeError(f"Failed to read uploaded PDF '{filename}'.") from exc
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _decode_text_payload(*, filename: str, payload: bytes) -> str:
    """Decode one uploaded text payload with UTF-8 fallback handling."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return payload.decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"Failed to decode uploaded text file '{filename}'.") from exc


def _upload_paper_id(filename: str) -> str:
    """Return a stable synthetic paper id for one query upload."""
    stem = Path(filename).stem.strip().lower().replace(" ", "_")
    return f"uploaded_{stem or 'file'}"


def _kb_retrieval_limit(*, diversity: float) -> int:
    """Map diversity control to a bounded KB retrieval limit."""
    return min(20, max(6, int(round(6 + (10 * float(diversity))))))


def _upload_retrieval_limit(*, diversity: float) -> int:
    """Map diversity control to a bounded upload retrieval limit."""
    return min(8, max(2, int(round(2 + (4 * float(diversity))))))


def _build_kb_context_entries(
    *,
    candidates: list[Any],
    chunks_by_id: dict[str, Chunk],
) -> list[dict[str, Any]]:
    """Build ordered KB context entries from retrieved chunks."""
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        chunk = chunks_by_id.get(candidate.chunk_id)
        if chunk is None:
            continue
        entries.append(
            {
                "label": f"{chunk.paper_id} | chunk {chunk.index}",
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "rank": int(candidate.rank),
                "score": float(candidate.score),
                "text": chunk.text,
                "source_type": "knowledge_base",
            }
        )
    return entries


def _build_upload_context_entries(
    *,
    question: str,
    upload_chunks: list[Chunk],
    diversity: float,
) -> list[dict[str, Any]]:
    """Retrieve top temporary-upload chunks for one query."""
    if not upload_chunks:
        return []

    retrieval_service = RetrievalService(DEFAULT_CONFIG)
    candidates = retrieval_service.retrieve_generous(question, upload_chunks)
    candidates = candidates[: _upload_retrieval_limit(diversity=diversity)]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in upload_chunks}

    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        chunk = chunks_by_id.get(candidate.chunk_id)
        if chunk is None:
            continue
        entries.append(
            {
                "label": f"{chunk.paper_id} | chunk {chunk.index}",
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "rank": int(candidate.rank),
                "score": float(candidate.score),
                "text": chunk.text,
                "source_type": "upload",
            }
        )
    return entries


def _build_context_text(
    *,
    question: str,
    kb_context_entries: list[dict[str, Any]],
    upload_context_entries: list[dict[str, Any]],
) -> str:
    """Build the merged context passed to the generator."""
    lines = [f"User query: {question}", "", "Retrieved context:"]
    for entry in kb_context_entries:
        lines.append(f"[{entry['label']}]: {entry['text']}")
    for entry in upload_context_entries:
        lines.append(f"[{entry['label']} | uploaded]: {entry['text']}")
    if len(lines) == 3:
        lines.append("No retrieval context was found.")
    return "\n\n".join(lines)


def _query_system_prompt() -> str:
    """Return the grounded-answer system prompt."""
    return (
        "Answer the user using only the provided retrieval context when possible. "
        "Be explicit when the context is insufficient. "
        "Reference source labels from the context in your answer when they support a claim."
    )


def _query_user_prompt(*, question: str, context_text: str) -> str:
    """Build the final user prompt for answer generation."""
    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Write a grounded answer. If evidence is limited, say so."
    )


def _creativity_to_temperature(creativity: float) -> float:
    """Map the UI creativity control to model temperature."""
    return round(max(0.0, min(1.0, float(creativity))), 2)


def _diversity_to_top_p(diversity: float) -> float:
    """Map the UI diversity control to sampling top-p."""
    return round(max(0.1, min(1.0, 0.4 + (0.6 * float(diversity)))), 2)


def _fallback_answer_text(
    *,
    question: str,
    kb_context_entries: list[dict[str, Any]],
    upload_context_entries: list[dict[str, Any]],
) -> str:
    """Return a deterministic fallback answer when generation is unavailable."""
    summary = {
        "question": question,
        "knowledge_base_sources": [entry["label"] for entry in kb_context_entries],
        "upload_sources": [entry["label"] for entry in upload_context_entries],
    }
    return (
        "No model-generated answer is available with the current settings.\n\n"
        "Retrieved evidence summary:\n"
        f"{json.dumps(summary, indent=2)}"
    )
