from __future__ import annotations

import os

import streamlit as st

from Benchmark.domain.enums import DifficultyLabel, QuestionStatus
from Benchmark.domain.models import BenchmarkRecord, Chunk
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

try:
    from streamlit_sortables import sort_items
except ImportError:  # pragma: no cover
    sort_items = None


VERIFY_PAYLOAD_KEY = "verify_order_payload"

st.title("3. Verify Questions")
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

    # Fallback: deterministic summary-ish answer when model call is unavailable.
    max_chars = 900
    fallback = " ".join(context_parts)[:max_chars].strip()
    if len(fallback) == max_chars:
        fallback += "..."
    return fallback

if not rows:
    st.info("No unverified questions found. Accept questions on page 2 first.")
    st.stop()

paper_ids = list(dict.fromkeys(str(row.get("paper_id", "")) for row in rows if row.get("paper_id")))
if not paper_ids:
    st.info("No paper ids found in data/unverified_questions.json.")
    st.stop()

current_index = min(get_current_paper_index(), len(paper_ids) - 1)
set_current_paper_index(current_index)
paper_id = paper_ids[current_index]
paper_rows = [row for row in rows if str(row.get("paper_id", "")) == paper_id]

if not paper_rows:
    st.info("No unverified questions found for this paper.")
    st.stop()

st.subheader(f"Paper: {paper_id}")
chunks = pipeline.load_chunks(paper_id)
chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

faiss_top_k = int(
    st.number_input(
        "FAISS top-k",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key="verify_faiss_top_k",
    )
)

verify_idx_key = f"verify_question_index_{paper_id}"
if verify_idx_key not in st.session_state:
    st.session_state[verify_idx_key] = 0
verify_idx = min(int(st.session_state[verify_idx_key]), len(paper_rows) - 1)
row = paper_rows[verify_idx]

qid = str(row.get("question_id", ""))
question_text = str(row.get("question_text", ""))
default_diff = str(row.get("default_difficulty", DifficultyLabel.SINGLE_HOP.value))
try:
    target_difficulty = DifficultyLabel(default_diff)
except ValueError:
    target_difficulty = DifficultyLabel.SINGLE_HOP

record = BenchmarkRecord(
    question_id=qid,
    paper_id=paper_id,
    question_text=question_text,
    status=QuestionStatus.DRAFT,
    target_difficulty=target_difficulty,
    difficulty_auto=target_difficulty,
    difficulty_final=target_difficulty,
)

st.markdown(f"### Question {verify_idx + 1} of {len(paper_rows)}")
record.question_text = st.text_area("Question", value=record.question_text, key=f"verify_q_{record.question_id}")

faiss_candidates = pipeline.question_service.retrieval.retrieve_top_faiss(
    record.question_text, limit=faiss_top_k
)
if faiss_candidates:
    record.retrieval_candidates = faiss_candidates
else:
    faiss_error = pipeline.question_service.retrieval.faiss_error
    if faiss_error:
        st.warning(f"FAISS retrieval unavailable: {faiss_error}")

faiss_chunks_by_id = pipeline.question_service.retrieval.load_chunks_for_candidates(record.retrieval_candidates)
chunks_by_id.update(faiss_chunks_by_id)

record.gold_chunk_ids = render_evidence_picker(
    record=record,
    chunks_by_id=chunks_by_id,
    key_prefix=f"verify_{paper_id}_{verify_idx}",
)
record.difficulty_final = render_difficulty_editor(record, key_prefix=f"verify_{paper_id}_{verify_idx}")

st.subheader("Ground Truth")
ground_truth_key = f"verify_ground_truth_{record.question_id}"
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
        st.session_state[ground_truth_key] = generated
        st.rerun()
    else:
        st.error("Could not generate answer. Make sure selected chunks have text.")

notes = st.text_input(
    "Notes",
    value="",
    key=f"verify_notes_{record.question_id}",
)


@st.dialog("Order Selected Chunks", width="large")
def render_verify_order_dialog() -> None:
    payload = st.session_state.get(VERIFY_PAYLOAD_KEY)
    if not payload:
        st.info("No verification payload found.")
        return

    st.write("Drag selected chunks to rank from most important (top) to least (bottom).")
    selected_chunk_ids = list(payload.get("selected_chunk_ids", []))

    if not selected_chunk_ids:
        st.warning("No chunks selected.")
        if st.button("Close", key="verify_order_close_empty"):
            st.session_state.pop(VERIFY_PAYLOAD_KEY, None)
            st.rerun()
        return

    ordered_chunk_ids = selected_chunk_ids
    if sort_items is None:
        st.warning("Install `streamlit-sortables` for drag-and-drop ordering. Using current order.")
    else:
        label_to_chunk_id: dict[str, str] = {}
        drag_labels: list[str] = []
        for idx, chunk_id in enumerate(selected_chunk_ids, start=1):
            label = f"{idx}. {chunk_id}"
            label_to_chunk_id[label] = chunk_id
            drag_labels.append(label)
        sorted_labels = sort_items(drag_labels, direction="vertical", key=f"sort_{payload['question_id']}")
        ordered_chunk_ids = [label_to_chunk_id[label] for label in sorted_labels if label in label_to_chunk_id]

    st.caption("Ordered chunks (drag list above, click dropdowns below for text)")
    chunk_texts = payload.get("selected_chunk_texts", {})
    for idx, chunk_id in enumerate(ordered_chunk_ids, start=1):
        with st.expander(f"{idx}. {chunk_id}", expanded=False):
            text = str(chunk_texts.get(chunk_id, "Chunk text unavailable."))
            st.text(text)

    c1, c2 = st.columns(2)
    if c1.button("Confirm Verify", key="confirm_verify_order"):
        final = BenchmarkRecord(
            question_id=str(payload["question_id"]),
            paper_id=str(payload["paper_id"]),
            question_text=str(payload["question_text"]),
            status=QuestionStatus.DRAFT,
            target_difficulty=DifficultyLabel(str(payload["target_difficulty"])),
            difficulty_auto=DifficultyLabel(str(payload["target_difficulty"])),
            difficulty_final=DifficultyLabel(str(payload["difficulty_final"])),
        )
        final.gold_chunk_ids = ordered_chunk_ids
        verifier.verify(final, verified_by="streamlit_user", notes=str(payload["notes"]))
        pipeline.audit_log.append("question_verified", final.to_dict())
        verified_store.append_verified(
            final,
            notes=str(payload["notes"]),
            ground_truth=str(payload.get("ground_truth", "")),
        )
        unverified_store.remove_question(final.question_id)
        st.session_state.pop(VERIFY_PAYLOAD_KEY, None)
        st.success("Question saved to data/verified_questions.json")
        st.rerun()

    if c2.button("Cancel", key="cancel_verify_order"):
        st.session_state.pop(VERIFY_PAYLOAD_KEY, None)
        st.rerun()


c1, c2, c3 = st.columns(3)
if c1.button("Verify", key=f"verify_btn_{record.question_id}"):
    if not record.gold_chunk_ids:
        st.error("Select at least one chunk before verifying.")
    else:
        st.session_state[VERIFY_PAYLOAD_KEY] = {
            "question_id": record.question_id,
            "paper_id": record.paper_id,
            "question_text": record.question_text,
            "target_difficulty": record.target_difficulty.value,
            "difficulty_final": record.difficulty_final.value,
            "notes": notes,
            "ground_truth": record_ground_truth,
            "selected_chunk_ids": list(record.gold_chunk_ids),
            "selected_chunk_texts": {
                chunk_id: (chunks_by_id.get(chunk_id).text if chunks_by_id.get(chunk_id) else "Chunk text unavailable.")
                for chunk_id in record.gold_chunk_ids
            },
        }
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

if st.session_state.get(VERIFY_PAYLOAD_KEY):
    render_verify_order_dialog()
