from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from RAG.services.query_service import run_query
from UI.state.session_state import get_loaded_knowledge_base


OPENAI_RETRIEVAL_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
OLLAMA_RETRIEVAL_MODELS = ["nomic-embed-text"]
QUERY_REQUEST_KEY = "query_model_request"
QUERY_RESULT_KEY = "query_model_result"


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Knowledge-Base")
    else:
        st.subheader("Knowledge-Base")
    st.subheader("Query Knowledge Base")


def _enabled_query_providers() -> list[str]:
    """Return enabled providers available for query-time retrieval settings."""
    raw_providers = st.session_state.get("llm_providers", ["openai"])
    providers: list[str] = []
    for raw_provider in raw_providers:
        provider = str(raw_provider).strip().lower()
        if provider in {"openai", "ollama"} and provider not in providers:
            providers.append(provider)
    return providers or ["openai"]


def _provider_retrieval_models(provider: str) -> list[str]:
    """Return retrieval model options for one selected provider."""
    if str(provider).strip().lower() == "ollama":
        return list(OLLAMA_RETRIEVAL_MODELS)
    return list(OPENAI_RETRIEVAL_MODELS)


def _default_retrieval_model(*, provider: str, loaded_knowledge_base: dict[str, Any]) -> str:
    """Pick a sensible initial retrieval model from the loaded KB or provider defaults."""
    kb_model = str(loaded_knowledge_base.get("embedding_model", "")).strip()
    provider_models = _provider_retrieval_models(provider)
    if kb_model in provider_models:
        return kb_model
    return provider_models[0]


def _uploaded_file_summaries() -> list[dict[str, Any]]:
    """Return detached metadata for per-query uploaded files."""
    uploaded_files = st.session_state.get("query_model_uploaded_files", [])
    summaries: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files:
        payload = uploaded_file.getvalue()
        summaries.append(
            {
                "name": str(uploaded_file.name),
                "size_bytes": len(payload),
                "suffix": Path(str(uploaded_file.name)).suffix.lower(),
            }
        )
    return summaries


def _persist_query_request(*, payload: dict[str, Any]) -> None:
    """Persist the pending query configuration in session state."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid query request payload. Expected a dictionary.")
    st.session_state[QUERY_REQUEST_KEY] = dict(payload)


def _persist_query_result(*, payload: dict[str, Any]) -> None:
    """Persist the latest executed query result in session state."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid query result payload. Expected a dictionary.")
    st.session_state[QUERY_RESULT_KEY] = dict(payload)


def _generation_model_for_provider(provider: str) -> str:
    """Return the answer-generation model for the selected provider."""
    normalized_provider = str(provider).strip().lower()
    if normalized_provider == "ollama":
        return str(st.session_state.get("ollama_model", "qwen3:8b")).strip() or "qwen3:8b"
    configured_model = str(st.session_state.get("question_generation_model", "")).strip()
    return configured_model or "gpt-4o-mini"


def _uploaded_file_payloads() -> list[tuple[str, bytes]]:
    """Detach uploaded files into `(name, bytes)` tuples for query execution."""
    uploaded_files = st.session_state.get("query_model_uploaded_files", [])
    return [(str(uploaded_file.name), uploaded_file.getvalue()) for uploaded_file in uploaded_files]


def _render_query_result() -> None:
    """Render the latest query result if one is present."""
    payload = st.session_state.get(QUERY_RESULT_KEY)
    if not isinstance(payload, dict):
        return
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            st.warning(str(warning))

    st.subheader("Answer")
    st.write(str(payload.get("answer_text", "")))

    kb_sources = payload.get("knowledge_base_sources", [])
    if isinstance(kb_sources, list) and kb_sources:
        st.subheader("Knowledge Base Sources")
        st.dataframe(
            [
                {
                    "label": item.get("label", ""),
                    "rank": item.get("rank", ""),
                    "score": item.get("score", ""),
                }
                for item in kb_sources
            ],
            use_container_width=True,
        )

    upload_sources = payload.get("upload_sources", [])
    if isinstance(upload_sources, list) and upload_sources:
        st.subheader("Temporary Upload Sources")
        st.dataframe(
            [
                {
                    "label": item.get("label", ""),
                    "rank": item.get("rank", ""),
                    "score": item.get("score", ""),
                }
                for item in upload_sources
            ],
            use_container_width=True,
        )

    with st.expander("Retrieved context"):
        st.text(str(payload.get("context_text", "")))


def render(show_title: bool = True) -> None:
    _show_title(show_title)
    loaded_knowledge_base = get_loaded_knowledge_base()
    if loaded_knowledge_base is None:
        st.warning("No knowledge base is loaded. Load a knowledge base before querying.")
        return

    st.info(
        "Loaded knowledge base: "
        f"{loaded_knowledge_base.get('knowledge_base_dir', '')} "
        f"({loaded_knowledge_base.get('method_id', '')})"
    )

    enabled_providers = _enabled_query_providers()
    active_provider = str(st.session_state.get("llm_provider", enabled_providers[0])).strip().lower()
    if active_provider not in enabled_providers:
        active_provider = enabled_providers[0]

    provider_index = enabled_providers.index(active_provider)
    selected_provider = st.selectbox(
        "LLM service",
        options=enabled_providers,
        index=provider_index,
        key="query_model_provider",
        help="Retrieval model options follow the currently selected service.",
    )

    retrieval_model_options = _provider_retrieval_models(selected_provider)
    default_model = _default_retrieval_model(provider=selected_provider, loaded_knowledge_base=loaded_knowledge_base)
    previous_model = str(st.session_state.get("query_model_retrieval_model", default_model)).strip()
    if previous_model not in retrieval_model_options:
        previous_model = default_model

    retrieval_model = st.selectbox(
        "Retrieval model",
        options=retrieval_model_options,
        index=retrieval_model_options.index(previous_model),
        key="query_model_retrieval_model",
        help="Models are filtered by the selected LLM service.",
    )
    generation_model = _generation_model_for_provider(selected_provider)
    st.caption(f"Answer model: {generation_model}")

    question_text = st.text_area(
        "Ask a question",
        value=str(st.session_state.get("query_model_question", "")),
        key="query_model_question",
        height=140,
        placeholder="Ask a question grounded in the loaded knowledge base...",
    )

    with st.expander("Advanced controls"):
        diversity = st.slider(
            "Diversity",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("query_model_diversity", 0.2)),
            step=0.05,
            key="query_model_diversity",
            help="Higher diversity broadens retrieval/generation behavior across candidate evidence.",
        )
        creativity = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("query_model_creativity", 0.2)),
            step=0.05,
            key="query_model_creativity",
            help="Higher creativity encourages less deterministic phrasing in later answer generation.",
        )

    st.file_uploader(
        "Upload relevant files for this query",
        accept_multiple_files=True,
        key="query_model_uploaded_files",
        help="These files are appended only for this query and are not saved back into the knowledge base.",
    )
    st.caption(
        "Per-query uploads are temporary. They are available only for this query request and are not persisted "
        "to the loaded knowledge base."
    )

    uploaded_file_summaries = _uploaded_file_summaries()
    if uploaded_file_summaries:
        st.dataframe(uploaded_file_summaries, use_container_width=True)

    if st.button("Run query", key="query_model_run"):
        normalized_question = question_text.strip()
        if not normalized_question:
            st.warning("Enter a question before running a query.")
            return

        payload = {
            "knowledge_base_dir": str(loaded_knowledge_base.get("knowledge_base_dir", "")).strip(),
            "knowledge_base_method": str(loaded_knowledge_base.get("method_id", "")).strip(),
            "provider": selected_provider,
            "retrieval_model": retrieval_model,
            "generation_model": generation_model,
            "question": normalized_question,
            "diversity": float(diversity),
            "creativity": float(creativity),
            "temporary_uploads": uploaded_file_summaries,
        }
        _persist_query_request(payload=payload)
        try:
            result = run_query(
                question=normalized_question,
                knowledge_base=loaded_knowledge_base,
                retrieval_provider=selected_provider,
                retrieval_model=retrieval_model,
                generation_provider=selected_provider,
                generation_model=generation_model,
                diversity=float(diversity),
                creativity=float(creativity),
                uploaded_files=_uploaded_file_payloads(),
                ollama_base_url=str(st.session_state.get("ollama_base_url", "http://localhost:11434")).strip(),
                openai_api_key=str(st.session_state.get("openai_api_key", "")).strip(),
            )
        except Exception as exc:
            st.error(str(exc))
            return

        _persist_query_result(payload=result)

    _render_query_result()
