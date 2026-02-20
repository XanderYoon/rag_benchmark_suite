from __future__ import annotations

import streamlit as st

from Benchmark.domain.enums import QuestionStatus
from Benchmark.persistence.unverified_question_store import UnverifiedQuestionStore
from UI.components.paper_selector import render_paper_selector
from UI.state.session_state import (
    get_current_paper_index,
    get_pipeline,
    get_records_store,
    set_current_paper_index,
)


def _text_area_height(text: str) -> int:
    lines = text.splitlines() or [text]
    wrapped_lines = sum(max((len(line) // 90) + 1, 1) for line in lines)
    estimated_lines = min(max(wrapped_lines, 3), 18)
    return 28 * estimated_lines + 12


st.title("2. Question Generation")

pipeline = get_pipeline()
unverified_store = UnverifiedQuestionStore()
records_by_paper = get_records_store()
papers = pipeline.paper_service.list_papers()
all_paper_ids = [p.paper_id for p in papers]

if not all_paper_ids:
    st.info("No papers detected. Run ingestion first.")
    st.stop()

approved_papers: set[str] = set()
for paper_id, records in records_by_paper.items():
    if any(r.status == QuestionStatus.VERIFIED for r in records):
        approved_papers.add(paper_id)

paper_ids = [paper_id for paper_id in all_paper_ids if paper_id not in approved_papers]

if not paper_ids:
    st.success("All papers already have approved questions. Nothing left to generate.")
    st.stop()

current_index = render_paper_selector(paper_ids, get_current_paper_index())
set_current_paper_index(current_index)
paper_id = paper_ids[current_index]

if st.button("Generate 3 questions for current paper"):
    records = pipeline.generate_for_paper(paper_id)
    records_by_paper[paper_id] = records

records = records_by_paper.get(paper_id, [])
if not records:
    st.info("No generated questions yet for this paper.")
    st.stop()

st.write("Target mix per paper: 2 single-hop, 1 multi-hop.")

display_records = [r for r in records if not bool(r.audit.get("accepted_for_verification"))]
if not display_records:
    st.success("All generated questions for this paper were accepted.")
    st.stop()

for i, record in enumerate(display_records):
    question_key = f"qgen_{paper_id}_q_{record.question_id}"
    current_text = st.session_state.get(question_key, record.question_text)
    record.question_text = st.text_area(
        f"Question {i + 1}",
        value=current_text,
        key=question_key,
        height=_text_area_height(current_text),
    )
    record.touch()

    st.caption(f"Question {i + 1} target difficulty: {record.target_difficulty.value}")
    decline_feedback = st.text_input(
        "Decline feedback (required for regenerate)",
        value="",
        key=f"decline_feedback_{paper_id}_{record.question_id}",
        placeholder="Why should this question be replaced?",
    )

    c_decline, c_accept = st.columns(2)
    if c_decline.button("Decline and regenerate", key=f"decline_regen_{paper_id}_{record.question_id}"):
        feedback = decline_feedback.strip()
        if not feedback:
            st.error("Provide decline feedback before regenerating.")
        else:
            avoid_questions = [r.question_text for idx, r in enumerate(records) if idx != i]
            new_record = pipeline.regenerate_question(
                paper_id=paper_id,
                target_difficulty=record.target_difficulty,
                slot_index=i,
                feedback=feedback,
                avoid_questions=avoid_questions,
            )
            records[i] = new_record
            records_by_paper[paper_id] = records
            st.success(f"Replaced question {i + 1} based on your feedback.")
            st.rerun()
    if c_accept.button("Accept", key=f"accept_q_{paper_id}_{record.question_id}"):
        record.audit["accepted_for_verification"] = True
        unverified_store.append_accepted(record)
        st.success(f"Accepted Question {i + 1} and saved to data/unverified_questions.json")
        st.rerun()

    if i < len(display_records) - 1:
        st.divider()

st.write(f"Current paper: {paper_id}")
