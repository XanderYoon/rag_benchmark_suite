from __future__ import annotations

from pathlib import Path

import streamlit as st

from Benchmark.embedding.build_faiss_rag_index import build_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AVAILABLE_SCRIPTS = {
    "Build FAISS RAG Index": PROJECT_ROOT / "Benchmark" / "embedding" / "build_faiss_rag_index.py",
}
OPENAI_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
OLLAMA_EMBEDDING_MODELS = ["nomic-embed-text"]


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("RAG Model Creator")
    else:
        st.subheader("RAG Model Creator")


def _provider_embedding_models(enabled_providers: list[str]) -> list[str]:
    """Return embedding model options based on enabled LLM providers."""
    models: list[str] = []
    if "openai" in enabled_providers:
        models.extend(OPENAI_EMBEDDING_MODELS)
    if "ollama" in enabled_providers:
        models.extend(OLLAMA_EMBEDDING_MODELS)
    if not models:
        models = list(OPENAI_EMBEDDING_MODELS)
    return models


def _embedding_provider_for_model(embedding_model: str) -> str:
    """Resolve embedding provider from the selected model identifier."""
    normalized = embedding_model.strip().lower()
    if normalized.startswith("text-embedding-"):
        return "openai"
    return "ollama"


def render(show_title: bool = True) -> None:
    _show_title(show_title)
    enabled_providers = list(st.session_state.get("llm_providers", ["openai"]))
    embedding_model_options = _provider_embedding_models(enabled_providers)

    script_label = st.selectbox(
        "Script to run",
        options=list(AVAILABLE_SCRIPTS.keys()),
        help="Select the indexing script to execute for building your RAG retrieval artifacts.",
    )
    script_path = AVAILABLE_SCRIPTS[script_label]

    chunks_root = st.text_input(
        "Chunks root",
        value="data/rag_corpus_chunked",
        help="Path to chunked paper text folders (one folder per paper with chunk text files).",
    )
    output_dir = st.text_input(
        "Output directory",
        value="data/faiss_rag_index",
        help="Directory where FAISS index files and metadata will be written.",
    )
    previous_embedding_model = str(st.session_state.get("rag_creator_embedding_model", embedding_model_options[0]))
    if previous_embedding_model not in embedding_model_options:
        previous_embedding_model = embedding_model_options[0]
    embedding_model = st.selectbox(
        "Embedding model",
        options=embedding_model_options,
        index=embedding_model_options.index(previous_embedding_model),
        key="rag_creator_embedding_model",
        help="Embedding models are filtered by enabled providers in Settings.",
    )
    batch_size = int(
        st.number_input(
            "Batch size",
            min_value=1,
            max_value=512,
            value=64,
            step=1,
            help="Number of chunks embedded per request batch. Larger values may be faster but use more memory.",
        )
    )
    metric = st.selectbox(
        "Metric",
        options=["cosine", "l2"],
        help="Distance metric used by FAISS for similarity search: cosine for angular similarity, l2 for Euclidean distance.",
    )
    overwrite = st.checkbox(
        "Overwrite existing index files",
        value=False,
        help="If enabled, existing files in the output directory can be replaced.",
    )
    embedding_provider = _embedding_provider_for_model(embedding_model)
    ollama_base_url = str(st.session_state.get("ollama_base_url", "http://localhost:11434")).strip()

    cmd = [
        "python3",
        str(script_path),
        "--chunks-root",
        chunks_root,
        "--output-dir",
        output_dir,
        "--embedding-provider",
        embedding_provider,
        "--embedding-model",
        embedding_model,
        "--batch-size",
        str(batch_size),
        "--metric",
        metric,
    ]
    if embedding_provider == "ollama":
        cmd.extend(["--ollama-base-url", ollama_base_url])
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
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                batch_size=batch_size,
                metric=metric,
                overwrite=overwrite,
                ollama_base_url=ollama_base_url,
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
