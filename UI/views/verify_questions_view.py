from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from benchmark.config import AppConfig
from benchmark.domain.difficulty_profiles import difficulty_from_profile_label
from benchmark.domain.enums import DifficultyLabel, QuestionStatus
from benchmark.domain.models import BenchmarkRecord, Chunk, EvidenceCandidate
from RAG.llm import generate_llm_text
from benchmark.persistence.unverified_question_store import UnverifiedQuestionStore
from benchmark.persistence.verified_question_store import VerifiedQuestionStore
from UI.components.difficulty_editor import render_difficulty_editor
from UI.components.evidence_picker import render_evidence_picker
from UI.state.session_state import (
    get_current_paper_index,
    get_pipeline,
    get_verifier,
    set_current_paper_index,
)

OPENAI_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
OLLAMA_EMBEDDING_MODELS = ["nomic-embed-text"]
RETRIEVAL_METHODS = {
    "none": "None",
    "faiss": "FAISS",
    "lightrag": "LightRAG",
    "graphrag": "GraphRAG",
}


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Verify Probes")
    else:
        st.subheader("Verify Probes")


def _generate_ground_truth(
    question: str,
    selected_chunk_ids: list[str],
    chunks_by_id: dict[str, Chunk],
    config: AppConfig,
) -> str:
    context_parts = [
        chunks_by_id[cid].text.strip()
        for cid in selected_chunk_ids
        if cid in chunks_by_id and chunks_by_id[cid].text.strip()
    ]
    if not context_parts:
        return ""

    context = "\n\n".join(context_parts)[:12000]
    answer = generate_llm_text(
        provider=config.llm_provider,
        model=config.question_model,
        system_prompt=(
            "Write a concise ground-truth answer strictly from provided chunks. "
            "Do not add facts not in context."
        ),
        user_prompt=f"Question:\n{question}\n\nContext chunks:\n{context}",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        ollama_base_url=config.ollama_base_url,
    )
    if answer:
        return answer

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


def _resolve_retrieval_model() -> tuple[str, str]:
    """Return selected retrieval `(provider, model)` from enabled providers."""
    enabled_providers = list(st.session_state.get("llm_providers", ["openai"]))
    options: list[tuple[str, str, str]] = []
    if "openai" in enabled_providers:
        for model in OPENAI_EMBEDDING_MODELS:
            options.append((f"openai::{model}", "openai", model))
    if "ollama" in enabled_providers:
        for model in OLLAMA_EMBEDDING_MODELS:
            options.append((f"ollama::{model}", "ollama", model))
    if not options:
        options.append(("openai::text-embedding-3-small", "openai", "text-embedding-3-small"))

    option_lookup = {key: (provider, model) for key, provider, model in options}
    default_provider = str(st.session_state.get("verify_retrieval_provider", "")).strip().lower()
    default_model = str(st.session_state.get("verify_retrieval_model", "")).strip()
    default_key = f"{default_provider}::{default_model}" if default_provider and default_model else options[0][0]
    if default_key not in option_lookup:
        default_key = options[0][0]

    selected_key = st.selectbox(
        "Retrieval model",
        options=[key for key, _, _ in options],
        index=[key for key, _, _ in options].index(default_key),
        format_func=lambda key: f"{option_lookup[key][1]} ({option_lookup[key][0]})",
        key="verify_retrieval_model_select",
    )
    provider, model = option_lookup[selected_key]
    st.session_state["verify_retrieval_provider"] = provider
    st.session_state["verify_retrieval_model"] = model
    return provider, model


def _discover_retrieval_directories(method_id: str) -> list[Path]:
    """Return artifact directories available for the selected retrieval method."""
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    candidates: set[Path] = set()
    if data_root.exists():
        for manifest_path in data_root.rglob("index_manifest.json"):
            parent = manifest_path.parent
            if not (parent / "chunks_metadata.jsonl").exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            manifest_method = str(manifest.get("method_id", "faiss")).strip().lower() or "faiss"
            if manifest_method == method_id:
                candidates.add(parent.resolve())
    default_dir = (data_root / f"{method_id}_index").resolve()
    if (default_dir / "index_manifest.json").exists() and (default_dir / "chunks_metadata.jsonl").exists():
        candidates.add(default_dir)
    return sorted(candidates)


def _fallback_candidates_without_retrieval(chunks_by_id: dict[str, Chunk]) -> list[EvidenceCandidate]:
    """Create deterministic non-retrieval candidates from loaded chunks."""
    ordered_chunk_ids = sorted(chunks_by_id.keys())
    return [
        EvidenceCandidate(chunk_id=chunk_id, score=0.0, rank=index + 1)
        for index, chunk_id in enumerate(ordered_chunk_ids)
    ]


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
        st.info("No unverified probes found. Accept probes on Probe Generation first.")
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
        st.info("No unverified probes found for this paper.")
        return

    st.subheader(f"Papers: {paper_scope_label}")
    chunks_by_id: dict[str, Chunk] = {}
    for selected_paper_id in selected_paper_ids:
        for chunk in pipeline.load_chunks(selected_paper_id):
            chunks_by_id[chunk.chunk_id] = chunk

    default_retrieval_method = str(st.session_state.get("verify_retrieval_method", "faiss")).strip().lower()
    if default_retrieval_method not in RETRIEVAL_METHODS:
        default_retrieval_method = "faiss"
    retrieval_method = st.selectbox(
        "Retrieval method",
        options=list(RETRIEVAL_METHODS.keys()),
        index=list(RETRIEVAL_METHODS.keys()).index(default_retrieval_method),
        format_func=lambda method_id: RETRIEVAL_METHODS[method_id],
        key="verify_retrieval_method",
    )
    retrieval_top_k = 20
    limit_to_top_k = False
    retrieval_provider = ""
    retrieval_model = ""
    selected_artifact_directory = ""
    if retrieval_method != "none":
        with st.expander("Retrieval settings", expanded=False):
            retrieval_directories = _discover_retrieval_directories(retrieval_method)
            if retrieval_directories:
                directory_options = [str(path) for path in retrieval_directories]
                saved_directory = str(
                    st.session_state.get(f"verify_{retrieval_method}_directory", directory_options[0])
                )
                if saved_directory not in directory_options:
                    saved_directory = directory_options[0]
                selected_artifact_directory = st.selectbox(
                    "Artifact directory",
                    options=directory_options,
                    index=directory_options.index(saved_directory),
                    key=f"verify_{retrieval_method}_directory_select",
                )
                st.session_state[f"verify_{retrieval_method}_directory"] = selected_artifact_directory
            else:
                st.warning(f"No {RETRIEVAL_METHODS[retrieval_method]} artifact directories found. Using chunk-only fallback.")
                retrieval_method = "none"

            if retrieval_method != "none":
                retrieval_provider, retrieval_model = _resolve_retrieval_model()
                limit_to_top_k = st.toggle(
                    "Restrict displayed chunks to top-k",
                    value=True,
                    key=f"verify_limit_top_k_{retrieval_method}_{scope_key or 'none'}",
                )
                if limit_to_top_k:
                    retrieval_top_k = int(
                        st.number_input(
                            "Top-k",
                            min_value=1,
                            max_value=200,
                            value=int(st.session_state.get("verify_retrieval_top_k", 20)),
                            step=1,
                            key="verify_retrieval_top_k",
                        )
                    )
                else:
                    retrieval_top_k = 200

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

    st.markdown(f"### Probe {verify_idx + 1} of {len(paper_rows)}")
    record.question_text = st.text_area("Probe", value=record.question_text, key=f"verify_q_{record.question_id}")

    retrieval_results_applied = False
    if retrieval_method != "none":
        retrieved_candidates = pipeline.question_service.retrieval.retrieve_top_artifact(
            record.question_text,
            retrieval_method=retrieval_method,
            limit=retrieval_top_k,
            retrieval_model=retrieval_model,
            retrieval_provider=retrieval_provider,
            artifact_output_dir=selected_artifact_directory,
        )
        if retrieved_candidates:
            record.retrieval_candidates = retrieved_candidates
            retrieval_results_applied = True
        else:
            retrieval_error = pipeline.question_service.retrieval.retrieval_error
            if retrieval_error:
                st.warning(
                    f"{RETRIEVAL_METHODS[retrieval_method]} retrieval unavailable: {retrieval_error}. "
                    "Showing chunks without retrieval."
                )
            record.retrieval_candidates = _fallback_candidates_without_retrieval(chunks_by_id)
    else:
        record.retrieval_candidates = _fallback_candidates_without_retrieval(chunks_by_id)

    retrieved_chunks_by_id = pipeline.question_service.retrieval.load_chunks_for_candidates(record.retrieval_candidates)
    chunks_by_id.update(retrieved_chunks_by_id)
    display_candidates = _build_display_candidates(
        record,
        chunks_by_id,
        limit_to_top_k=limit_to_top_k,
        top_k=retrieval_top_k,
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
        show_scores=retrieval_results_applied,
    )
    st.divider()
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
        generated = _generate_ground_truth(
            record.question_text,
            record.gold_chunk_ids,
            chunks_by_id,
            pipeline.config,
        )
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
        st.success("Probe saved to data/verified_questions.json")
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
