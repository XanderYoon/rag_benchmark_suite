from __future__ import annotations

import os

import requests
import streamlit as st

from UI.views.benchmarking_view import render as render_benchmarking
from UI.views.corpus_creation_view import render as render_corpus_creation
from UI.views.ingest_view import render as render_ingest
from UI.views.question_generation_view import render as render_question_generation
from UI.views.rag_model_creator_view import render as render_rag_model_creator
from UI.views.verify_questions_view import render as render_verify_questions


st.set_page_config(page_title="RAG Benchmark Builder", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .nav-active-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 6px solid #7dd3fc;
        border-radius: 12px;
        box-shadow: 0 8px 22px rgba(14, 165, 233, 0.14);
        color: #0b1220;
        font-weight: 700;
        margin: 0.15rem 0 0.4rem 0;
        padding: 0.65rem 0.85rem;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] .nav-active-card span {
        display: block;
        letter-spacing: 0.01em;
        text-align: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SECTIONS: dict[str, list[str]] = {
    "RAG Creation": ["Corpus Creation", "Ingest", "RAG Model Creator"],
    "Query Creation": ["Question Generation", "Verify Questions"],
    "Benchmarking": ["Overview"],
}


def _default_subpage(section: str) -> str:
    return SECTIONS[section][0]


def _set_navigation(section: str, subpage: str) -> None:
    st.session_state["nav_section"] = section
    st.session_state["nav_subpage"] = subpage


def _is_valid_openai_api_key(api_key: str) -> bool:
    key = api_key.strip()
    if not key or not key.startswith("sk-"):
        return False
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


if "openai_api_key_initialized" not in st.session_state:
    st.session_state["openai_api_key_initialized"] = True
    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_api_key and _is_valid_openai_api_key(env_api_key):
        st.session_state["openai_api_key"] = env_api_key
    else:
        st.session_state["openai_api_key"] = ""

session_api_key = str(st.session_state.get("openai_api_key", "")).strip()
if session_api_key:
    os.environ["OPENAI_API_KEY"] = session_api_key
else:
    os.environ.pop("OPENAI_API_KEY", None)


with st.sidebar:
    st.title("Navigation")
    if "nav_section" not in st.session_state:
        st.session_state["nav_section"] = "RAG Creation"
    if "nav_subpage" not in st.session_state:
        st.session_state["nav_subpage"] = _default_subpage(st.session_state["nav_section"])

    for section_name, pages in SECTIONS.items():
        is_active_section = st.session_state.get("nav_section") == section_name
        if section_name == "Benchmarking":
            st.markdown(f"**{section_name}**" if is_active_section else section_name)
            is_active_page = (
                st.session_state.get("nav_section") == "Benchmarking"
                and st.session_state.get("nav_subpage") == "Overview"
            )
            if is_active_page:
                st.markdown(
                    "<div class='nav-active-card'><span>Overview</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button("Overview", key="nav_benchmarking", use_container_width=True):
                    _set_navigation("Benchmarking", "Overview")
                    st.rerun()
            st.write("")
            continue

        st.markdown(f"**{section_name}**" if is_active_section else section_name)
        for page_name in pages:
            is_active_page = (
                st.session_state.get("nav_section") == section_name
                and st.session_state.get("nav_subpage") == page_name
            )
            if is_active_page:
                st.markdown(
                    f"<div class='nav-active-card'><span>{page_name}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    page_name,
                    key=f"nav_{section_name}_{page_name}",
                    use_container_width=True,
                ):
                    _set_navigation(section_name, page_name)
                    st.rerun()
        st.write("")

    section = str(st.session_state.get("nav_section", "RAG Creation"))
    subpage = str(st.session_state.get("nav_subpage", _default_subpage(section)))

    st.divider()
    with st.expander("Settings", expanded=False):
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            key="settings_openai_api_key_input",
            placeholder="sk-...",
        )
        if st.button("Set OpenAI API Key", key="set_openai_key"):
            candidate_key = api_key_input.strip()
            if not candidate_key:
                st.error("Enter an API key first.")
            elif _is_valid_openai_api_key(candidate_key):
                st.session_state["openai_api_key"] = candidate_key
                os.environ["OPENAI_API_KEY"] = candidate_key
                st.success("Valid OpenAI API key verified and set.")
                st.rerun()
            else:
                st.error("That key could not be verified with OpenAI. It was not saved.")

        if st.button("Clear key", key="clear_openai_key"):
            st.session_state["openai_api_key"] = ""
            st.session_state["settings_openai_api_key_input"] = ""
            os.environ.pop("OPENAI_API_KEY", None)
            st.success("Cleared session and environment key.")
            st.rerun()

        active_key = str(st.session_state.get("openai_api_key", "")).strip()
        if active_key:
            st.caption("OpenAI key is set for this session.")
        else:
            st.caption("No OpenAI key is set in this session.")

st.caption(f"{section} / {subpage}")

if section == "RAG Creation" and subpage == "Corpus Creation":
    render_corpus_creation(show_title=True)
elif section == "RAG Creation" and subpage == "Ingest":
    render_ingest(show_title=True)
elif section == "RAG Creation" and subpage == "RAG Model Creator":
    render_rag_model_creator(show_title=True)
elif section == "Query Creation" and subpage == "Question Generation":
    render_question_generation(show_title=True)
elif section == "Query Creation" and subpage == "Verify Questions":
    render_verify_questions(show_title=True)
else:
    render_benchmarking(show_title=False)
