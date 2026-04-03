from __future__ import annotations

from RAG.services.query_service import (
    _build_context_text,
    _creativity_to_temperature,
    _diversity_to_top_p,
    parse_query_upload,
)


def test_parse_query_upload_chunks_text_payload() -> None:
    chunks, warnings = parse_query_upload(
        filename="notes.txt",
        payload=b"alpha beta gamma delta epsilon",
        chunk_size_tokens=3,
        chunk_overlap_tokens=1,
    )

    assert not warnings
    assert len(chunks) >= 2
    assert chunks[0].paper_id == "uploaded_notes"


def test_parse_query_upload_rejects_unsupported_suffix() -> None:
    chunks, warnings = parse_query_upload(
        filename="image.png",
        payload=b"png",
        chunk_size_tokens=3,
        chunk_overlap_tokens=1,
    )

    assert chunks == []
    assert warnings


def test_build_context_text_includes_knowledge_base_and_upload_entries() -> None:
    context = _build_context_text(
        question="What changed?",
        kb_context_entries=[{"label": "paper_a | chunk 0", "text": "kb text"}],
        upload_context_entries=[{"label": "uploaded_notes | chunk 0", "text": "upload text"}],
    )

    assert "paper_a | chunk 0" in context
    assert "uploaded_notes | chunk 0 | uploaded" in context


def test_sampling_controls_are_bounded() -> None:
    assert _creativity_to_temperature(1.5) == 1.0
    assert _creativity_to_temperature(-1.0) == 0.0
    assert _diversity_to_top_p(0.0) == 0.4
    assert _diversity_to_top_p(1.0) == 1.0
