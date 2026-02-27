from __future__ import annotations

import streamlit as st

from Benchmark.domain.difficulty_profiles import canonical_profile_label, difficulty_profile_labels
from Benchmark.domain.models import BenchmarkRecord


DIFFICULTY_OPTIONS = difficulty_profile_labels()


def render_difficulty_editor(record: BenchmarkRecord, key_prefix: str, default_label: str | None = None) -> str:
    initial = canonical_profile_label(default_label)
    if initial not in DIFFICULTY_OPTIONS:
        initial = canonical_profile_label(str(record.audit.get("difficulty_profile", "")))
    idx = DIFFICULTY_OPTIONS.index(initial)
    return st.selectbox("Difficulty", DIFFICULTY_OPTIONS, index=idx, key=f"{key_prefix}_diff_{record.question_id}")
