from __future__ import annotations

import streamlit as st

from Benchmark.domain.models import Chunk


def render_chunk_preview(chunks: list[Chunk], count: int = 3) -> None:
    for chunk in chunks[:count]:
        with st.expander(f"{chunk.chunk_id}"):
            st.write(chunk.text)
