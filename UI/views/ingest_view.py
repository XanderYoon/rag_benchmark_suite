from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from UI.components.chunk_viewer import render_chunk_preview
from UI.state.session_state import get_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Ingest")
    else:
        st.subheader("Ingest")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _render_pdf_viewer(pdf_path: Path) -> None:
    """Render an inline PDF preview for the selected paper."""
    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError:
        st.error(f"Failed to load PDF preview from {pdf_path}.")
        return

    pdf_renderer = getattr(st, "pdf", None)
    if callable(pdf_renderer):
        pdf_renderer(pdf_bytes)
    else:
        html_renderer = getattr(st, "html", None)
        if callable(html_renderer):
            pdf_data = base64.b64encode(pdf_bytes).decode("utf-8")
            html_renderer(
                f"""
                <div style="border: 1px solid #d9d9d9; border-radius: 0.5rem; overflow: hidden;">
                  <embed
                    src="data:application/pdf;base64,{pdf_data}#view=FitH"
                    type="application/pdf"
                    width="100%"
                    height="720px"
                  />
                </div>
                """
            )
        else:
            st.info("Inline PDF preview is not available in this Streamlit version.")
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=pdf_path.name,
        mime="application/pdf",
        use_container_width=True,
        key=f"download_pdf_{pdf_path.stem}",
    )


def _render_directory_browser() -> tuple[Path, str]:
    current_value = str(st.session_state.get("corpus_dir", "data/rag_corpus_pdf")).strip() or "data/rag_corpus_pdf"
    candidate = Path(current_value).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    browser_dir = candidate if candidate.is_dir() else candidate.parent
    if not browser_dir.exists():
        browser_dir = PROJECT_ROOT

    subdirs = sorted(path for path in browser_dir.iterdir() if path.is_dir())
    options = [".."] + [_display_path(path) for path in subdirs]
    with st.expander("Browse: TODO"):
        st.caption(f"Folder browser: {_display_path(browser_dir)}")
        selected = st.selectbox(
            "Browse directories",
            options=options,
            key=f"browse_corpus_dir_{_display_path(browser_dir)}",
        )
    return browser_dir, selected


def _render_ingest_progress(total_pdfs: int) -> tuple[object, object]:
    """Create Streamlit progress UI for chunk ingestion."""
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    if total_pdfs > 0:
        status_placeholder.info(f"Starting chunking for {total_pdfs} PDF(s)...")
    else:
        status_placeholder.info("Starting chunking...")
    return progress_bar, status_placeholder


def render(show_title: bool = True) -> None:
    _show_title(show_title)

    corpus_dir = st.text_input("Corpus folder", value=st.session_state.get("corpus_dir", "data/rag_corpus_pdf"))
    st.session_state["corpus_dir"] = corpus_dir
    browser_dir, selected = _render_directory_browser()

    pipeline = get_pipeline()

    c_use, c_chunk = st.columns(2)
    if c_use.button("Use selected folder", key=f"use_corpus_dir_{_display_path(browser_dir)}", use_container_width=True):
        if selected == "..":
            next_dir = browser_dir.parent
        else:
            next_dir = PROJECT_ROOT / selected if not Path(selected).is_absolute() else Path(selected)
        st.session_state["corpus_dir"] = _display_path(next_dir)
        st.rerun()

    if c_chunk.button("Chunk all", key="chunk_all_pdfs", use_container_width=True):
        total_pdfs = len(pipeline.paper_service.list_pdfs())
        progress_bar, status_placeholder = _render_ingest_progress(total_pdfs)

        def update_progress(completed: int, total: int, message: str) -> None:
            denominator = total if total > 0 else 1
            progress_bar.progress(min(completed / denominator, 1.0))
            status_placeholder.info(message)

        summary = pipeline.ingest_all(progress_callback=update_progress)
        if not summary:
            progress_bar.empty()
            status_placeholder.empty()
            st.warning("No PDFs found.")
        else:
            progress_bar.progress(1.0)
            status_placeholder.success(f"Chunking complete. Processed {len(summary)} paper(s).")
            st.success(f"Chunked {len(summary)} papers")
            st.dataframe([{"paper_id": k, "chunk_count": v} for k, v in summary.items()])

    pdfs = pipeline.paper_service.list_pdfs()
    if pdfs:
        st.subheader("Preview paper")
        selected = st.selectbox("Paper", options=[p.stem for p in pdfs], key="preview_paper")
        selected_pdf = next(pdf for pdf in pdfs if pdf.stem == selected)
        _render_pdf_viewer(selected_pdf)

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
