from __future__ import annotations

import streamlit as st


def render_paper_selector(paper_ids: list[str], current_index: int) -> int:
    if not paper_ids:
        st.info("No ingested papers found.")
        return 0
    idx = min(current_index, len(paper_ids) - 1)
    return st.selectbox("Paper", options=list(range(len(paper_ids))), index=idx, format_func=lambda i: paper_ids[i])
