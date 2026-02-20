from __future__ import annotations

import streamlit as st

from Benchmark.domain.models import BenchmarkRecord


def render_question_editor(records: list[BenchmarkRecord], key_prefix: str) -> None:
    for i, record in enumerate(records):
        new_text = st.text_area(
            f"Question {i + 1}",
            value=record.question_text,
            key=f"{key_prefix}_q_{record.question_id}",
            height=100,
        )
        record.question_text = new_text
        record.touch()
