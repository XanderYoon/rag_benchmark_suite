from __future__ import annotations

from pathlib import Path

import streamlit as st

from UI.components.chunk_viewer import render_chunk_preview
from UI.state.session_state import get_pipeline


st.title("1. Ingest")

corpus_dir = st.text_input("Corpus folder", value=st.session_state.get("corpus_dir", "data/rag_corpus_pdf"))
st.session_state["corpus_dir"] = corpus_dir

pipeline = get_pipeline()

if st.button("Chunk all"):
    summary = pipeline.ingest_all()
    if not summary:
        st.warning("No PDFs found.")
    else:
        st.success(f"Chunked {len(summary)} papers")
        st.dataframe([{"paper_id": k, "chunk_count": v} for k, v in summary.items()])

pdfs = pipeline.paper_service.list_pdfs()
if pdfs:
    selected = st.selectbox("Preview paper", options=[p.stem for p in pdfs])
    chunks = pipeline.load_chunks(selected)
    if not chunks:
        st.info("No chunks yet for this paper. Run 'Chunk all' first.")
    else:
        cache = pipeline.config.text_cache_dir / f"{selected}.txt"
        if cache.exists():
            st.subheader("Extracted text sample")
            st.text(cache.read_text(encoding="utf-8")[:2000])
        st.subheader("Sample chunks")
        render_chunk_preview(chunks, count=3)
else:
    st.info(f"No PDFs in {Path(corpus_dir)}")
