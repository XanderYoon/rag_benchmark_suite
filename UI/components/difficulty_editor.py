from __future__ import annotations

import streamlit as st

from Benchmark.domain.enums import DifficultyLabel
from Benchmark.domain.models import BenchmarkRecord


def render_difficulty_editor(record: BenchmarkRecord, key_prefix: str) -> DifficultyLabel:
    labels = [d.value for d in DifficultyLabel]
    idx = labels.index(record.difficulty_final.value)
    value = st.selectbox("Difficulty", labels, index=idx, key=f"{key_prefix}_diff_{record.question_id}")
    return DifficultyLabel(value)
