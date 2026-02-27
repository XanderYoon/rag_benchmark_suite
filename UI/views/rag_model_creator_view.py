from __future__ import annotations

from pathlib import Path

import streamlit as st

from Benchmark.embedding.build_faiss_rag_index import build_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AVAILABLE_SCRIPTS = {
    "Build FAISS RAG Index": PROJECT_ROOT / "Benchmark" / "embedding" / "build_faiss_rag_index.py",
}


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("RAG Model Creator")
    else:
        st.subheader("RAG Model Creator")


def render(show_title: bool = True) -> None:
    _show_title(show_title)

    script_label = st.selectbox("Script to run", options=list(AVAILABLE_SCRIPTS.keys()))
    script_path = AVAILABLE_SCRIPTS[script_label]

    chunks_root = st.text_input("Chunks root", value="data/rag_corpus_chunked")
    output_dir = st.text_input("Output directory", value="data/faiss_rag_index")
    embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")
    batch_size = int(st.number_input("Batch size", min_value=1, max_value=512, value=64, step=1))
    metric = st.selectbox("Metric", options=["cosine", "l2"])
    overwrite = st.checkbox("Overwrite existing index files", value=False)

    cmd = [
        "python3",
        str(script_path),
        "--chunks-root",
        chunks_root,
        "--output-dir",
        output_dir,
        "--embedding-model",
        embedding_model,
        "--batch-size",
        str(batch_size),
        "--metric",
        metric,
    ]
    if overwrite:
        cmd.append("--overwrite")

    st.code(" ".join(cmd), language="bash")

    def _render_rag_progress() -> tuple[object, object]:
        """Create Streamlit progress UI for FAISS index building."""
        status_placeholder = st.empty()
        progress_bar = st.progress(0.0)
        status_placeholder.info("Starting RAG index build...")
        return progress_bar, status_placeholder

    if st.button("Run RAG script", key="run_rag_script"):
        progress_bar, status_placeholder = _render_rag_progress()

        def update_progress(progress: float, message: str) -> None:
            progress_bar.progress(min(max(progress, 0.0), 1.0))
            status_placeholder.info(message)

        try:
            result = build_faiss_index(
                chunks_root=Path(chunks_root),
                output_dir=Path(output_dir),
                embedding_model=embedding_model,
                batch_size=batch_size,
                metric=metric,
                overwrite=overwrite,
                progress_callback=update_progress,
            )
        except Exception as exc:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"RAG script failed: {exc}")
            return

        progress_bar.progress(1.0)
        status_placeholder.success("RAG index build complete.")
        st.success("RAG script completed successfully.")
        st.json(result)
