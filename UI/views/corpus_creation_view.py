from __future__ import annotations

from pathlib import Path

import streamlit as st

from Benchmark.ingestion.arxiv_corpus_creator import scrape_arxiv_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPIC_STATE_KEY = "corpus_topics"
TOPIC_ID_STATE_KEY = "corpus_topic_id_counter"
FOLDER_INPUT_KEY = "corpus_folder_name"


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Corpus Creation")
    else:
        st.subheader("Corpus Creation")


def _next_topic_id() -> int:
    next_id = int(st.session_state.get(TOPIC_ID_STATE_KEY, 0))
    st.session_state[TOPIC_ID_STATE_KEY] = next_id + 1
    return next_id


def _topic_row(topic: str = "", docs_per_topic: int = 10, row_id: int | None = None) -> dict[str, object]:
    return {
        "row_id": _next_topic_id() if row_id is None else row_id,
        "topic": topic,
        "docs_per_topic": docs_per_topic,
    }


def _default_docs_for_new_topic(topics: list[dict[str, object]]) -> int:
    if not topics:
        return 10
    first_docs = topics[0].get("docs_per_topic")
    if isinstance(first_docs, int) and first_docs > 0:
        return first_docs
    return 10


def _ensure_topics() -> list[dict[str, object]]:
    if TOPIC_STATE_KEY not in st.session_state:
        st.session_state[TOPIC_STATE_KEY] = [_topic_row("retrieval augmented generation benchmarking", 10)]

    raw_topics = st.session_state[TOPIC_STATE_KEY]
    normalized: list[dict[str, object]] = []
    if isinstance(raw_topics, list):
        for item in raw_topics:
            if isinstance(item, dict):
                normalized.append(
                    _topic_row(
                        str(item.get("topic", "")),
                        _default_docs_for_new_topic(normalized) if not isinstance(item.get("docs_per_topic"), int) else item["docs_per_topic"],
                        item.get("row_id") if isinstance(item.get("row_id"), int) else None,
                    )
                )
            else:
                normalized.append(_topic_row(str(item), _default_docs_for_new_topic(normalized)))

    if not normalized:
        normalized = [_topic_row("retrieval augmented generation benchmarking", 10)]

    max_row_id = max(int(topic["row_id"]) for topic in normalized)
    st.session_state[TOPIC_ID_STATE_KEY] = max(int(st.session_state.get(TOPIC_ID_STATE_KEY, 0)), max_row_id + 1)
    st.session_state[TOPIC_STATE_KEY] = normalized
    return st.session_state[TOPIC_STATE_KEY]


def _resolve_output_dir(folder_name: str) -> Path:
    path = Path(folder_name).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / "data" / path).resolve()


def _render_scrape_progress(total_requested: int) -> tuple[object, object]:
    """Create Streamlit progress UI for corpus scraping."""
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    if total_requested > 0:
        status_placeholder.info(f"Starting scrape for {total_requested} paper(s)...")
    else:
        status_placeholder.info("Starting scrape...")
    return progress_bar, status_placeholder


def render(show_title: bool = True) -> None:
    _show_title(show_title)

    topics = _ensure_topics()
    st.caption("Add one or more research topics, then scrape recent arXiv publications into a corpus folder.")
    for idx, topic_row in enumerate(list(topics)):
        row_id = int(topic_row["row_id"])
        c_text, c_count, c_remove = st.columns([6, 2, 1])
        topics[idx]["topic"] = c_text.text_input(
            f"Topic {idx + 1}",
            value=str(topic_row["topic"]),
            key=f"corpus_topic_{row_id}",
            placeholder="Enter a research topic prompt",
        )
        topics[idx]["docs_per_topic"] = int(
            c_count.number_input(
                "Docs",
                min_value=1,
                max_value=100,
                value=int(topic_row["docs_per_topic"]),
                step=1,
                key=f"corpus_topic_count_{row_id}",
            )
        )
        c_remove.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if len(topics) > 1 and c_remove.button("Remove", key=f"remove_topic_{row_id}"):
            st.session_state[TOPIC_STATE_KEY] = [row for row in topics if int(row["row_id"]) != row_id]
            st.rerun()

    if st.button("+ Add topic", key="add_corpus_topic"):
        topics.append(_topic_row("", _default_docs_for_new_topic(topics)))
        st.session_state[TOPIC_STATE_KEY] = topics
        st.rerun()
    c_folder, c_browse = st.columns([6, 1])
    folder_name = c_folder.text_input("Folder / directory name", value="rag_corpus_pdf", key=FOLDER_INPUT_KEY)
    c_browse.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
    if c_browse.button("Browse", key="browse_corpus_folder"):
        st.info("TODO")
    output_dir = _resolve_output_dir(folder_name)
    st.caption(f"Output folder: {output_dir}")

    if st.button("Scrape research publications", key="scrape_research_publications"):
        active_topics = [
            {
                "topic": str(topic["topic"]).strip(),
                "docs_per_topic": int(topic["docs_per_topic"]),
            }
            for topic in topics
            if str(topic["topic"]).strip()
        ]
        if not active_topics:
            st.error("Add at least one topic before scraping.")
            return
        if not folder_name.strip():
            st.error("Provide a folder name.")
            return

        total_requested = sum(int(topic["docs_per_topic"]) for topic in active_topics)
        progress_bar, status_placeholder = _render_scrape_progress(total_requested)

        def update_progress(completed: int, total: int, message: str) -> None:
            denominator = total if total > 0 else 1
            progress_bar.progress(min(completed / denominator, 1.0))
            status_placeholder.info(message)

        try:
            downloaded, metadata = scrape_arxiv_corpus(
                topic_rows=active_topics,
                save_dir=output_dir,
                progress_callback=update_progress,
            )
        except Exception as exc:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"Scrape failed: {exc}")
            return
        progress_bar.progress(1.0)
        status_placeholder.success(f"Scrape complete. Processed {len(metadata)} paper(s).")

        st.session_state["corpus_dir"] = str(output_dir)
        st.success(f"Downloaded {downloaded} paper(s) into {output_dir}.")
        if metadata:
            st.dataframe(
                [
                    {
                        "id": row["id"],
                        "published": row["published"],
                        "topic": row["topic_query"],
                        "title": row["title"],
                    }
                    for row in metadata
                ]
            )
