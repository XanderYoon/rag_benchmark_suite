from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from RAG.services.knowledge_base_service import append_to_knowledge_base, load_knowledge_base
from UI.state.session_state import get_loaded_knowledge_base, set_loaded_knowledge_base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_BASE_DIR = "data/faiss_rag_index"


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Knowledge-Base")
    else:
        st.subheader("Knowledge-Base")
    st.subheader("Load / Append Knowledge Base")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_user_path(raw_path: str) -> Path:
    """Resolve user-entered paths relative to the repository root."""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _render_directory_browser(*, state_key: str, raw_path: str) -> None:
    """Render a minimal directory chooser that writes back to one session key."""
    candidate = _resolve_user_path(raw_path) if raw_path.strip() else PROJECT_ROOT
    browser_dir = candidate if candidate.is_dir() else candidate.parent
    if not browser_dir.exists():
        browser_dir = PROJECT_ROOT

    subdirs = sorted(path for path in browser_dir.iterdir() if path.is_dir())
    options = [".."] + [_display_path(path) for path in subdirs]
    with st.expander("Browse directories"):
        st.caption(f"Current folder: {_display_path(browser_dir)}")
        selected = st.selectbox(
            "Directories",
            options=options,
            key=f"{state_key}_browser_{_display_path(browser_dir)}",
        )
        if st.button("Use selected directory", key=f"{state_key}_use_selected_{_display_path(browser_dir)}"):
            if selected == "..":
                next_dir = browser_dir.parent
            else:
                next_dir = PROJECT_ROOT / selected if not Path(selected).is_absolute() else Path(selected)
            st.session_state[state_key] = _display_path(next_dir)
            st.rerun()


def _render_validation_summary(*, payload: dict[str, Any]) -> None:
    """Render a compact summary for one loaded knowledge base."""
    st.success("Knowledge base validation passed.")
    st.caption(f"Directory: {payload['knowledge_base_dir']}")
    st.caption(f"Method: {payload['method_id']}")
    st.caption(f"Chunks: {payload['chunk_count']}")
    embedding_model = str(payload.get("embedding_model", "")).strip()
    if embedding_model:
        st.caption(f"Embedding model: {embedding_model}")
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            st.warning(str(warning))

    artifact_rows = [
        {"artifact": key, "path": value}
        for key, value in dict(payload.get("artifact_paths", {})).items()
    ]
    if artifact_rows:
        st.dataframe(artifact_rows, use_container_width=True)


def _render_loaded_kb_summary() -> None:
    """Render the active session-loaded knowledge base, if any."""
    loaded_knowledge_base = get_loaded_knowledge_base()
    if loaded_knowledge_base is None:
        st.info("No knowledge base is loaded in this session.")
        return
    st.info(
        "Loaded knowledge base: "
        f"{loaded_knowledge_base.get('knowledge_base_dir', '')} "
        f"({loaded_knowledge_base.get('method_id', '')})"
    )


def _uploaded_file_payloads() -> list[tuple[str, bytes]]:
    """Convert Streamlit uploaded files into detached payload tuples."""
    uploaded_files = st.session_state.get("kb_append_uploads", [])
    payloads: list[tuple[str, bytes]] = []
    for uploaded_file in uploaded_files:
        payloads.append((str(uploaded_file.name), uploaded_file.getvalue()))
    return payloads


def render(show_title: bool = True) -> None:
    _show_title(show_title)
    _render_loaded_kb_summary()

    operation_mode = st.radio(
        "Knowledge base action",
        options=["load", "append"],
        format_func=lambda option: "Load existing knowledge base" if option == "load" else "Append to knowledge base",
        horizontal=True,
    )
    st.session_state["knowledge_base_action"] = operation_mode

    selected_kb_dir = st.text_input(
        "Knowledge base directory",
        value=str(st.session_state.get("knowledge_base_dir_input", DEFAULT_KNOWLEDGE_BASE_DIR)),
        key="knowledge_base_dir_input",
        help="Directory containing retrieval artifacts such as index manifest, chunk metadata, and method-specific files.",
    )
    _render_directory_browser(state_key="knowledge_base_dir_input", raw_path=selected_kb_dir)

    if operation_mode == "load":
        if st.button("Validate and load knowledge base", key="load_knowledge_base"):
            try:
                loaded_payload = load_knowledge_base(knowledge_base_dir=selected_kb_dir)
            except Exception as exc:
                st.error(str(exc))
            else:
                set_loaded_knowledge_base(knowledge_base=loaded_payload)
                _render_validation_summary(payload=loaded_payload)
    else:
        append_source_dir = st.text_input(
            "Append source directory",
            value=str(st.session_state.get("append_source_dir_input", "")),
            key="append_source_dir_input",
            help="Optional directory whose contents should be copied into append staging for this knowledge base.",
        )
        _render_directory_browser(state_key="append_source_dir_input", raw_path=append_source_dir)
        st.file_uploader(
            "Upload files to append",
            accept_multiple_files=True,
            key="kb_append_uploads",
            help="Uploaded files are copied into append staging and recorded in append metadata.",
        )

        if st.button("Append and load knowledge base", key="append_knowledge_base"):
            source_directories: list[str | Path] = []
            if append_source_dir.strip():
                source_directories.append(append_source_dir.strip())

            try:
                append_result = append_to_knowledge_base(
                    knowledge_base_dir=selected_kb_dir,
                    uploaded_files=_uploaded_file_payloads(),
                    source_directories=source_directories,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                loaded_payload = append_result["knowledge_base"]
                set_loaded_knowledge_base(knowledge_base=loaded_payload)
                st.success("Append completed and knowledge base loaded into the session.")
                st.caption(f"Append metadata: {append_result['append_metadata_path']}")
                st.dataframe(append_result["appended_items"], use_container_width=True)
                _render_validation_summary(payload=loaded_payload)
