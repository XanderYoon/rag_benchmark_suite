from __future__ import annotations

import streamlit as st

from Benchmark.domain.difficulty_profiles import (
    canonical_profile_label,
    difficulty_from_profile_label,
    difficulty_profile_labels,
)
from Benchmark.domain.models import BenchmarkRecord
from Benchmark.domain.enums import DifficultyLabel, QuestionStatus
from Benchmark.persistence.question_id_allocator import QuestionIdAllocator
from Benchmark.persistence.unverified_question_store import UnverifiedQuestionStore
from UI.components.paper_selector import render_paper_selector
from UI.state.session_state import (
    get_current_paper_index,
    get_pipeline,
    get_records_store,
    set_current_paper_index,
)


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Question Generation")
    else:
        st.subheader("Question Generation")


def _text_area_height(text: str) -> int:
    lines = text.splitlines() or [text]
    wrapped_lines = sum(max((len(line) // 90) + 1, 1) for line in lines)
    estimated_lines = min(max(wrapped_lines, 3), 18)
    return 28 * estimated_lines + 12


def _render_user_created_questions_section() -> None:
    """Render manual question entry inputs above the paper selector."""
    st.subheader("User Created Questions")
    custom_question = str(st.session_state.get("user_created_question_text", ""))
    st.text_area(
        "Question text",
        value=custom_question,
        key="user_created_question_text",
        height=_text_area_height(custom_question),
        placeholder="Write a question manually...",
    )
    difficulty_options = difficulty_profile_labels()
    default_difficulty = difficulty_options[0]
    selected_difficulty = canonical_profile_label(
        str(st.session_state.get("user_created_question_difficulty", default_difficulty))
    )
    st.session_state["user_created_question_difficulty"] = selected_difficulty
    st.selectbox(
        "Suggested difficulty",
        options=difficulty_options,
        key="user_created_question_difficulty",
    )


def _add_user_created_question(
    *,
    paper_id: str,
    unverified_store: UnverifiedQuestionStore,
    id_allocator: QuestionIdAllocator,
) -> bool:
    """Validate and append a manually authored question to the unverified queue."""
    question_text = str(st.session_state.get("user_created_question_text", "")).strip()
    if not question_text:
        st.error("Enter a question before adding it.")
        return False

    difficulty_value = canonical_profile_label(
        str(st.session_state.get("user_created_question_difficulty", difficulty_profile_labels()[0]))
    )
    difficulty = difficulty_from_profile_label(difficulty_value)

    record = BenchmarkRecord(
        question_id=id_allocator.next_id(),
        paper_id=paper_id,
        question_text=question_text,
        target_difficulty=difficulty,
        difficulty_auto=difficulty,
        difficulty_final=difficulty,
    )
    record.audit["difficulty_profile"] = difficulty_value
    record.audit["accepted_for_verification"] = True
    unverified_store.append_accepted(record)
    st.session_state["user_created_question_text"] = ""
    st.success("Added manual question to data/unverified_questions.json")
    return True


def _render_user_created_question_actions(
    *,
    paper_ids: list[str],
    current_paper_id: str | None,
    unverified_store: UnverifiedQuestionStore,
    id_allocator: QuestionIdAllocator,
) -> None:
    """Render the action row for the manual question section."""
    selected_paper_id = st.selectbox(
        "Add to paper",
        options=[None, *paper_ids],
        index=(paper_ids.index(current_paper_id) + 1) if current_paper_id in paper_ids else 0,
        format_func=lambda value: "No paper selected" if value is None else value,
        key="user_created_question_paper_id",
    )
    if st.button("Add", key="add_user_created_question"):
        if selected_paper_id is None:
            st.error("Select a paper before adding the question.")
            return
        if _add_user_created_question(
            paper_id=selected_paper_id,
            unverified_store=unverified_store,
            id_allocator=id_allocator,
        ):
            st.rerun()
    st.divider()


def render(show_title: bool = True) -> None:
    _show_title(show_title)

    pipeline = get_pipeline()
    unverified_store = UnverifiedQuestionStore()
    id_allocator = QuestionIdAllocator()
    records_by_paper = get_records_store()
    papers = pipeline.paper_service.list_papers()
    all_paper_ids = [p.paper_id for p in papers]

    if not all_paper_ids:
        st.info("No papers detected. Run ingestion first.")
        return

    approved_papers: set[str] = set()
    for paper_id, records in records_by_paper.items():
        if any(r.status == QuestionStatus.VERIFIED for r in records):
            approved_papers.add(paper_id)

    paper_ids = [paper_id for paper_id in all_paper_ids if paper_id not in approved_papers]

    if not paper_ids:
        st.success("All papers already have approved questions. Nothing left to generate.")
        return

    current_index = min(get_current_paper_index(), len(paper_ids) - 1)
    paper_id = paper_ids[current_index]

    _render_user_created_questions_section()
    _render_user_created_question_actions(
        paper_ids=all_paper_ids,
        current_paper_id=paper_id,
        unverified_store=unverified_store,
        id_allocator=id_allocator,
    )

    st.subheader("AI Generated Questions")
    current_index = render_paper_selector(paper_ids, current_index)
    set_current_paper_index(current_index)
    paper_id = paper_ids[current_index]

    if st.button("Generate 5 questions for current paper", key="generate_questions_for_paper"):
        records = pipeline.generate_for_paper(paper_id)
        records_by_paper[paper_id] = records


    records = records_by_paper.get(paper_id, [])
    if not records:
        st.info("No generated questions yet for this paper.")
        return

    st.write(
        "Target types per paper: "
        "Single document: single hop, "
        "Single document: multi hop, "
        "Multiple documents, "
        "Comparison, "
        "Negative / Null."
    )

    display_records = [r for r in records if not bool(r.audit.get("accepted_for_verification"))]
    if not display_records:
        st.success("All generated questions for this paper were accepted.")
        return

    profile_to_slot: dict[str, int] = {
        label: idx for idx, (label, _, _) in enumerate(pipeline.question_service.DIFFICULTY_PROFILES)
    }

    display_rows: list[tuple[int, object]] = [
        (idx, r) for idx, r in enumerate(records) if not bool(r.audit.get("accepted_for_verification"))
    ]

    for i, (record_idx, record) in enumerate(display_rows):
        question_key = f"qgen_{paper_id}_q_{record.question_id}"
        current_text = st.session_state.get(question_key, record.question_text)
        record.question_text = st.text_area(
            f"Question {i + 1}",
            value=current_text,
            key=question_key,
            height=_text_area_height(current_text),
        )
        record.touch()

        profile_label = str(record.audit.get("difficulty_profile", record.target_difficulty.value))
        st.caption(f"Question {i + 1} type: {profile_label}")
        decline_feedback = st.text_input(
            "Decline feedback (required for regenerate)",
            value="",
            key=f"decline_feedback_{paper_id}_{record.question_id}",
            placeholder="Why should this question be replaced?",
        )

        c_decline, c_accept, c_remove = st.columns([1.2, 1.2, 0.6])
        if c_decline.button("Decline and regenerate", key=f"decline_regen_{paper_id}_{record.question_id}"):
            feedback = decline_feedback.strip()
            if not feedback:
                st.error("Provide decline feedback before regenerating.")
            else:
                profile_label = str(record.audit.get("difficulty_profile", "")).strip().lower()
                slot_index = profile_to_slot.get(profile_label, record_idx)
                avoid_questions = [r.question_text for idx, r in enumerate(records) if idx != record_idx]
                new_record = pipeline.regenerate_question(
                    paper_id=paper_id,
                    target_difficulty=record.target_difficulty,
                    slot_index=slot_index,
                    feedback=feedback,
                    avoid_questions=avoid_questions,
                )
                records[record_idx] = new_record
                records_by_paper[paper_id] = records
                st.success(f"Replaced question {i + 1} based on your feedback.")
                st.rerun()
        if c_accept.button("Accept", key=f"accept_q_{paper_id}_{record.question_id}"):
            record.audit["accepted_for_verification"] = True
            unverified_store.append_accepted(record)
            st.success("Accepted question and saved to data/unverified_questions.json")
            st.rerun()
        if c_remove.button("X", key=f"remove_q_{paper_id}_{record.question_id}"):
            records.pop(record_idx)
            records_by_paper[paper_id] = records
            st.success(f"Removed Question {i + 1}.")
            st.rerun()

        if i < len(display_rows) - 1:
            st.divider()

    st.write(f"Current paper: {paper_id}")
