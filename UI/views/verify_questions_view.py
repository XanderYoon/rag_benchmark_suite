from __future__ import annotations

import os

import streamlit as st

from Benchmark.domain.difficulty_profiles import difficulty_from_profile_label
from Benchmark.domain.enums import DifficultyLabel, QuestionStatus
from Benchmark.domain.models import BenchmarkRecord, Chunk, EvidenceCandidate
from Benchmark.persistence.unverified_question_store import UnverifiedQuestionStore
from Benchmark.persistence.verified_question_store import VerifiedQuestionStore
from UI.components.difficulty_editor import render_difficulty_editor
from UI.components.evidence_picker import render_evidence_picker
from UI.state.session_state import (
    get_current_paper_index,
    get_pipeline,
    get_verifier,
    set_current_paper_index,
)


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Verify Questions")
    else:
        st.subheader("Verify Questions")


def _generate_ground_truth(question: str, selected_chunk_ids: list[str], chunks_by_id: dict[str, Chunk]) -> str:
    context_parts = [
        chunks_by_id[cid].text.strip()
        for cid in selected_chunk_ids
        if cid in chunks_by_id and chunks_by_id[cid].text.strip()
    ]
    if not context_parts:
        return ""

    context = "\n\n".join(context_parts)[:12000]
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None  # type: ignore[assignment]

    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is not None and api_key:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Write a concise ground-truth answer strictly from provided chunks. "
                            "Do not add facts not in context."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question:\n{question}\n\nContext chunks:\n{context}",
                    },
                ],
            )
            text = (resp.output_text or "").strip()
            if text:
                return text
        except Exception:
            pass

    fallback = " ".join(context_parts)[:900].strip()
    if len(fallback) == 900:
        fallback += "..."
    return fallback


def _difficulty_from_profile(profile_label: str) -> DifficultyLabel:
    return difficulty_from_profile_label(profile_label)


def _paper_ids_for_row(row: dict) -> list[str]:
    """Return normalized paper ids for an unverified row."""
    return UnverifiedQuestionStore.paper_ids_for_row(row)


def _paper_scope_key(paper_ids: list[str]) -> str:
    """Build a stable scope key for a paper-id list."""
    return "|".join(paper_ids)


def _paper_scope_label(paper_ids: list[str]) -> str:
    """Return a user-facing label for a paper-id list."""
    if not paper_ids:
        return "No selected papers"
    if len(paper_ids) == 1:
        return paper_ids[0]
    return ", ".join(paper_ids)


def _render_top_k_picker(record: BenchmarkRecord, candidates: list[EvidenceCandidate]) -> list[str]:
    """Select up to five retrieval candidates for ranked top-k evaluation."""
    candidate_ids = [cand.chunk_id for cand in candidates]
    default_ids = [chunk_id for chunk_id in (record.top_k_chunk_ids or candidate_ids[:5]) if chunk_id in candidate_ids]
    default_ids = default_ids[:5]
    return st.multiselect(
        "Top-k retrieval chunks (choose up to 5, then rank below on verify)",
        options=candidate_ids,
        default=default_ids,
        max_selections=5,
        key=f"verify_top_k_{record.question_id}",
    )


def _build_display_candidates(
    record: BenchmarkRecord,
    chunks_by_id: dict[str, Chunk],
    *,
    limit_to_top_k: bool,
    top_k: int,
) -> list[EvidenceCandidate]:
    """Return the chunk candidates to display for verification."""
    ranked_candidates = list(record.retrieval_candidates)
    if limit_to_top_k:
        return ranked_candidates[:top_k]

    display_candidates: list[EvidenceCandidate] = list(ranked_candidates)
    seen_chunk_ids = {candidate.chunk_id for candidate in display_candidates}
    next_rank = len(display_candidates) + 1
    for chunk_id, chunk in chunks_by_id.items():
        if chunk_id in seen_chunk_ids:
            continue
        display_candidates.append(
            EvidenceCandidate(
                chunk_id=chunk_id,
                score=0.0,
                rank=next_rank,
            )
        )
        next_rank += 1
    return display_candidates


def _candidate_matches_filters(
    candidate: EvidenceCandidate,
    chunks_by_id: dict[str, Chunk],
    selected_docs: list[str],
    search_text: str,
) -> bool:
    """Return True when a retrieval candidate satisfies the active chunk filters."""
    chunk = chunks_by_id.get(candidate.chunk_id)
    row_paper_ids = [chunk.paper_id] if chunk else []
    if selected_docs:
        if not set(selected_docs).intersection(row_paper_ids):
            return False

    query = search_text.strip().lower()
    if query:
        haystacks = [
            candidate.chunk_id,
            " ".join(row_paper_ids),
            chunk.text if chunk else "",
        ]
        if not any(query in haystack.lower() for haystack in haystacks):
            return False

    return True


def render(show_title: bool = True) -> None:
    _show_title(show_title)
    st.markdown(
        """
        <style>
        div[data-testid="stTextArea"] textarea {
            font-size: 0.88rem !important;
            line-height: 1.35 !important;
        }
        div[data-testid="stDialog"] div[role="dialog"] {
            width: min(94vw, 1200px) !important;
        }
        div[data-testid="stDialog"] .sortable-item {
            color: #1f77ff !important;
            font-weight: 600 !important;
        }
        div[data-testid="stDialog"] div[data-testid="stExpander"] summary p {
            color: #1f77ff !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()
    verifier = get_verifier()
    verified_store = VerifiedQuestionStore()
    unverified_store = UnverifiedQuestionStore()
    rows = unverified_store.read_all()

    if not rows:
        st.info("No unverified questions found. Accept questions on Question Generation first.")
        return

    scope_keys: list[str] = []
    scope_paper_ids: dict[str, list[str]] = {}
    for row in rows:
        paper_ids = _paper_ids_for_row(row)
        scope_key = _paper_scope_key(paper_ids)
        if scope_key in scope_paper_ids:
            continue
        scope_keys.append(scope_key)
        scope_paper_ids[scope_key] = paper_ids

    current_index = min(get_current_paper_index(), len(scope_keys) - 1)
    set_current_paper_index(current_index)
    scope_key = scope_keys[current_index]
    selected_paper_ids = scope_paper_ids[scope_key]
    paper_scope_label = _paper_scope_label(selected_paper_ids)
    paper_rows = [row for row in rows if _paper_scope_key(_paper_ids_for_row(row)) == scope_key]

    if not paper_rows:
        st.info("No unverified questions found for this paper.")
        return

    st.subheader(f"Papers: {paper_scope_label}")
    chunks_by_id: dict[str, Chunk] = {}
    for selected_paper_id in selected_paper_ids:
        for chunk in pipeline.load_chunks(selected_paper_id):
            chunks_by_id[chunk.chunk_id] = chunk

    faiss_top_k = int(
        st.number_input(
            "Top-k",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
            key="verify_faiss_top_k",
        )
    )
    limit_to_top_k = st.toggle(
        "Restrict displayed chunks to top-k",
        value=True,
        key=f"verify_limit_top_k_{scope_key or 'none'}",
    )

    verify_idx_key = f"verify_question_index_{scope_key or 'none'}"
    if verify_idx_key not in st.session_state:
        st.session_state[verify_idx_key] = 0
    verify_idx = min(int(st.session_state[verify_idx_key]), len(paper_rows) - 1)
    row = paper_rows[verify_idx]

    qid = str(row.get("question_id", ""))
    question_text = str(row.get("question_text", ""))
    default_diff = str(row.get("default_difficulty", DifficultyLabel.SINGLE_HOP.value))
    target_difficulty = _difficulty_from_profile(default_diff)

    record = BenchmarkRecord(
        question_id=qid,
        paper_id=selected_paper_ids[0] if selected_paper_ids else "",
        question_text=question_text,
        source_paper_ids=list(selected_paper_ids),
        status=QuestionStatus.DRAFT,
        target_difficulty=target_difficulty,
        difficulty_auto=target_difficulty,
        difficulty_final=target_difficulty,
    )

    st.markdown(f"### Question {verify_idx + 1} of {len(paper_rows)}")
    record.question_text = st.text_area("Question", value=record.question_text, key=f"verify_q_{record.question_id}")

    faiss_candidates = pipeline.question_service.retrieval.retrieve_top_faiss(record.question_text, limit=faiss_top_k)
    if faiss_candidates:
        record.retrieval_candidates = faiss_candidates
    else:
        faiss_error = pipeline.question_service.retrieval.faiss_error
        if faiss_error:
            st.warning(f"FAISS retrieval unavailable: {faiss_error}")

    faiss_chunks_by_id = pipeline.question_service.retrieval.load_chunks_for_candidates(record.retrieval_candidates)
    chunks_by_id.update(faiss_chunks_by_id)
    display_candidates = _build_display_candidates(
        record,
        chunks_by_id,
        limit_to_top_k=limit_to_top_k,
        top_k=faiss_top_k,
    )

    available_chunk_docs = sorted(
        {
            chunk.paper_id
            for cand in display_candidates
            for chunk in [chunks_by_id.get(cand.chunk_id)]
            if chunk is not None and str(chunk.paper_id).strip()
        }
    )
    filter_col, search_col = st.columns(2)
    filter_docs = filter_col.multiselect(
        "Filter chunk documents TODO",
        options=available_chunk_docs,
        key=f"verify_filter_documents_{record.question_id}",
    )
    search_text = search_col.text_input(
        "Search chunks TODO",
        value="",
        key=f"verify_filter_search_{record.question_id}",
        placeholder="Search within chunk text, chunk id, or document id",
    )
    filtered_candidates = [
        cand
        for cand in display_candidates
        if _candidate_matches_filters(cand, chunks_by_id, filter_docs, search_text)
    ]

    if not filtered_candidates:
        st.info("No chunks match the active filters.")

    record.gold_chunk_ids = render_evidence_picker(
        record=record,
        chunks_by_id=chunks_by_id,
        key_prefix=f"verify_{scope_key or 'none'}_{verify_idx}",
        candidates=filtered_candidates,
    )
    record.top_k_chunk_ids = _render_top_k_picker(record, filtered_candidates)
    selected_difficulty_profile = render_difficulty_editor(
        record,
        key_prefix=f"verify_{scope_key or 'none'}_{verify_idx}",
        default_label=default_diff,
    )
    record.difficulty_final = _difficulty_from_profile(selected_difficulty_profile)
    record.audit["difficulty_profile"] = selected_difficulty_profile

    st.subheader("Ground Truth")
    ground_truth_key = f"verify_ground_truth_{record.question_id}"
    ground_truth_pending_key = f"{ground_truth_key}__pending"
    if ground_truth_pending_key in st.session_state:
        st.session_state[ground_truth_key] = str(st.session_state.pop(ground_truth_pending_key))
    if ground_truth_key not in st.session_state:
        st.session_state[ground_truth_key] = ""
    record_ground_truth = st.text_area(
        "Ground truth answer",
        value=st.session_state.get(ground_truth_key, ""),
        key=ground_truth_key,
        height=140,
        placeholder="Write or generate the reference answer here.",
    )
    if st.button("Generate answer", key=f"gen_answer_{record.question_id}", disabled=not bool(record.gold_chunk_ids)):
        generated = _generate_ground_truth(record.question_text, record.gold_chunk_ids, chunks_by_id)
        if generated:
            st.session_state[ground_truth_pending_key] = generated
            st.rerun()
        else:
            st.error("Could not generate answer. Make sure selected chunks have text.")

    notes = st.text_input(
        "Notes",
        value="",
        key=f"verify_notes_{record.question_id}",
    )

    c1, c2, c3 = st.columns(3)
    if c1.button("Verify", key=f"verify_btn_{record.question_id}"):
        if not record.gold_chunk_ids:
            st.error("Select at least one chunk before verifying.")
        elif not record.top_k_chunk_ids:
            st.error("Select at least one top-k retrieval chunk before verifying.")
        else:
            final = BenchmarkRecord(
                question_id=record.question_id,
                paper_id=record.paper_id,
                question_text=record.question_text,
                source_paper_ids=list(record.source_paper_ids),
                status=QuestionStatus.DRAFT,
                target_difficulty=record.target_difficulty,
                difficulty_auto=record.target_difficulty,
                difficulty_final=record.difficulty_final,
            )
            final.gold_chunk_ids = list(record.gold_chunk_ids)
            final.top_k_chunk_ids = list(record.top_k_chunk_ids)
            final.audit["difficulty_profile"] = selected_difficulty_profile
            verifier.verify(final, verified_by="streamlit_user", notes=notes)
            pipeline.audit_log.append("question_verified", final.to_dict())
            verified_store.append_verified(
                final,
                notes=notes,
                ground_truth=record_ground_truth,
                difficulty_label=selected_difficulty_profile,
            )
            unverified_store.remove_question(final.question_id)
            st.success("Question saved to data/verified_questions.json")
            st.rerun()

    if c2.button("Needs revision", key=f"revise_btn_{record.question_id}"):
        verifier.needs_revision(record, verified_by="streamlit_user", notes=notes)
        pipeline.audit_log.append("question_needs_revision", record.to_dict())
        st.rerun()

    if c3.button("Reject", key=f"reject_btn_{record.question_id}"):
        verifier.reject(record, verified_by="streamlit_user", notes=notes)
        pipeline.audit_log.append("question_rejected", record.to_dict())
        unverified_store.remove_question(record.question_id)
        st.rerun()
