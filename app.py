from __future__ import annotations

import os
import subprocess
import time
from urllib.parse import urlparse

import requests
import streamlit as st

from RAG.llm.provider_client import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, normalize_llm_provider
from UI.views.benchmarking_view import render as render_benchmarking
from UI.views.corpus_creation_view import render as render_corpus_creation
from UI.views.ingest_view import render as render_ingest
from UI.views.knowledge_graph_view import render as render_knowledge_graph
from UI.views.load_append_model_view import render as render_load_append_model
from UI.views.mcp_view import render as render_mcp
from UI.views.query_model_view import render as render_query_model
from UI.views.question_generation_view import render as render_question_generation
from UI.views.rag_model_creator_view import render as render_rag_model_creator
from UI.views.verify_questions_view import render as render_verify_questions


st.set_page_config(page_title="RAG Benchmark Builder", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] div[data-testid="stElementContainer"] {
        width: 100%;
    }
    [data-testid="stSidebar"] div[data-testid="stElementContainer"] > div {
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton {
        box-sizing: border-box;
        display: block;
        margin: 0 0 0.4rem 0;
        min-width: 100%;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > div {
        box-sizing: border-box;
        min-width: 100%;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button {
        align-items: center;
        align-self: stretch;
        border-radius: 12px;
        box-sizing: border-box;
        display: flex;
        height: 2.75rem;
        justify-content: center;
        max-width: 100%;
        margin: 0;
        min-height: 2.75rem;
        min-width: 100%;
        padding: 0 0.85rem;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button > div {
        justify-content: center;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button p {
        text-align: center;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: transparent;
        border-color: #7dd3fc;
        color: #0369a1;
        box-shadow: inset 0 0 0 1px #7dd3fc;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:disabled {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        box-shadow: inset 6px 0 0 #7dd3fc;
        color: #0b1220;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:disabled p {
        color: #0b1220;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .nav-section-heading {
        font-size: 1.45rem;
        font-weight: 500;
        line-height: 1.2;
        margin: 0 0 0.75rem 0;
    }
    [data-testid="stSidebar"] .nav-section-heading-active {
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 0 0 0.75rem 0;
    }
    [data-testid="stSidebar"] .nav-group-heading {
        font-size: 1.18rem;
        font-weight: 500;
        line-height: 1.3;
        margin: 0 0 0.55rem 0;
    }
    [data-testid="stSidebar"] .nav-subsection-heading {
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.25;
        margin: 0.1rem 0 0.4rem 0;
        padding-left: 0.1rem;
    }
    [data-testid="stSidebar"] .nav-subsection-heading-active {
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 0.1rem 0 0.4rem 0;
        padding-left: 0.1rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SECTIONS: dict[str, list[str]] = {
    "RAG Creation": ["Corpus Creation", "Ingest", "RAG Model Creator"],
    "Knowledge-Base": ["Load / Append Knowledge Base", "Query Knowledge Base", "View Knowledge Graph"],
    "Probe Creation": ["Probe Generation", "Verify Probes"],
    "Benchmarking": ["Run Benchmarks", "Compare Benchmarks"],
    "MCP": ["MCP"],
}
BENCHMARK_RUN_IN_PROGRESS_KEY = "benchmark_run_in_progress"
SUPPORTED_LLM_PROVIDERS = ("openai", "ollama")
PAGE_LABELS: dict[str, str] = {
    "Ingest": "Parse and Chunk",
}
PAGE_ALIASES: dict[str, str] = {
    "Parse and Chunk": "Ingest",
    "Question Generation": "Probe Generation",
    "Verify Questions": "Verify Probes",
}
SECTION_ALIASES: dict[str, str] = {
    "Query Creation": "Probe Creation",
    "Chat": "Knowledge-Base",
    "Model Workspace": "Knowledge-Base",
}
SECTION_GROUPS: dict[str, str] = {
    "Probe Creation": "Benchmarks",
    "Benchmarking": "Benchmarks",
}
GROUP_SECTIONS: dict[str, list[str]] = {
    "RAG Creation": ["RAG Creation"],
    "Knowledge-Base": ["Knowledge-Base"],
    "Benchmarks": ["Probe Creation", "Benchmarking"],
    "MCP": ["MCP"],
}


def _default_subpage(section: str) -> str:
    return SECTIONS[section][0]


def _set_navigation(section: str, subpage: str) -> None:
    st.session_state["nav_section"] = section
    st.session_state["nav_subpage"] = subpage


def _page_label(page_name: str) -> str:
    """Return the user-facing label for a navigation page."""
    return PAGE_LABELS.get(page_name, page_name)


def _render_section_heading(*, section_name: str, is_active: bool) -> None:
    """Render a section heading with active-state emphasis."""
    class_name = "nav-section-heading-active" if is_active else "nav-section-heading"
    st.markdown(f"<div class='{class_name}'>{section_name}</div>", unsafe_allow_html=True)


def _render_subsection_heading(*, section_name: str, is_active: bool) -> None:
    """Render a subsection heading with active-state emphasis."""
    class_name = "nav-subsection-heading-active" if is_active else "nav-subsection-heading"
    st.markdown(f"<div class='{class_name}'>{section_name}</div>", unsafe_allow_html=True)


def _normalize_subpage_name(subpage: str) -> str:
    """Map legacy or display-only page labels to internal route ids."""
    return PAGE_ALIASES.get(subpage, subpage)


def _normalize_navigation_state() -> None:
    """Keep navigation state aligned with current internal route ids."""
    raw_section = str(st.session_state.get("nav_section", "RAG Creation"))
    section = SECTION_ALIASES.get(raw_section, raw_section)
    raw_subpage = str(st.session_state.get("nav_subpage", _default_subpage(section if section in SECTIONS else "RAG Creation")))
    normalized_subpage = _normalize_subpage_name(raw_subpage)
    if section == "Knowledge-Base" and normalized_subpage == "MCP":
        section = "MCP"
    if section not in SECTIONS:
        section = "RAG Creation"
    st.session_state["nav_section"] = section

    valid_subpages = SECTIONS.get(section, [])
    if normalized_subpage not in valid_subpages:
        normalized_subpage = _default_subpage(section)
    st.session_state["nav_subpage"] = normalized_subpage


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


def _is_reachable_ollama_server(base_url: str, model: str) -> tuple[bool, str]:
    server_url = base_url.strip().rstrip("/")
    if not server_url:
        return False, "Enter an Ollama base URL."
    if not model.strip():
        return False, "Enter an Ollama model name."
    try:
        response = requests.get(f"{server_url}/api/tags", timeout=5)
        response.raise_for_status()
        body = response.json()
    except Exception:
        return False, f"Could not connect to Ollama at {server_url}."

    listed_models = {
        str(item.get("name", "")).strip()
        for item in body.get("models", [])
        if isinstance(item, dict)
    }
    if model.strip() not in listed_models:
        return (
            False,
            f"Connected to Ollama, but model '{model.strip()}' is not installed. "
            f"Run: ollama pull {model.strip()}",
        )
    return True, f"Ollama reachable and model '{model.strip()}' is available."


def _candidate_ollama_cli_paths() -> list[str]:
    """Return likely executable paths for the Ollama CLI."""
    home_dir = os.path.expanduser("~")
    return [
        str(os.getenv("OLLAMA_CLI_PATH", "")).strip(),
        os.path.join(home_dir, ".local", "bin", "ollama"),
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/opt/homebrew/bin/ollama",
    ]


def _resolve_ollama_cli_path(configured_path: str | None = None) -> str | None:
    """Resolve an executable Ollama CLI path from config and common locations."""
    candidates = [str(configured_path or "").strip(), *_candidate_ollama_cli_paths()]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _is_ollama_server_up(base_url: str) -> bool:
    """Return True when an Ollama server responds at the given URL."""
    server_url = base_url.strip().rstrip("/")
    if not server_url:
        return False
    try:
        response = requests.get(f"{server_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _normalize_provider_list(raw_providers: list[str] | tuple[str, ...] | str | None) -> list[str]:
    """Return unique supported providers preserving user order."""
    providers: list[str] = []
    if isinstance(raw_providers, str):
        candidates = [item.strip() for item in raw_providers.split(",")]
    elif isinstance(raw_providers, (list, tuple)):
        candidates = [str(item).strip() for item in raw_providers]
    else:
        candidates = []
    for candidate in candidates:
        provider = normalize_llm_provider(candidate)
        if provider in SUPPORTED_LLM_PROVIDERS and provider not in providers:
            providers.append(provider)
    return providers


def _resolve_active_provider(*, providers: list[str], preferred: str | None) -> str:
    """Pick the active provider from selected providers and preferred value."""
    if not providers:
        return "openai"
    preferred_provider = normalize_llm_provider(preferred or "")
    if preferred_provider in providers:
        return preferred_provider
    return providers[0]


def _is_local_ollama_url(base_url: str) -> bool:
    """Return True when the base URL points to localhost."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _selected_provider_ids_from_settings() -> list[str]:
    """Return enabled provider ids from sidebar widget state."""
    selected_providers: list[str] = []
    if bool(st.session_state.get("settings_provider_openai", False)):
        selected_providers.append("openai")
    if bool(st.session_state.get("settings_provider_ollama", False)):
        selected_providers.append("ollama")
    return selected_providers


def _sync_provider_settings_state() -> None:
    """Keep provider widget state and backend session state in sync."""
    selected_providers = _selected_provider_ids_from_settings()
    if not selected_providers:
        fallback_provider = normalize_llm_provider(str(st.session_state.get("llm_provider", "openai")))
        if fallback_provider not in SUPPORTED_LLM_PROVIDERS:
            fallback_provider = "openai"
        selected_providers = [fallback_provider]
        st.session_state["settings_provider_openai"] = fallback_provider == "openai"
        st.session_state["settings_provider_ollama"] = fallback_provider == "ollama"
        st.session_state["provider_selection_warning"] = "At least one provider is required."
    else:
        st.session_state["provider_selection_warning"] = ""

    st.session_state["llm_providers"] = selected_providers
    st.session_state["llm_provider"] = _resolve_active_provider(
        providers=selected_providers,
        preferred=str(st.session_state.get("llm_provider", "openai")),
    )


def _start_local_ollama_server(base_url: str) -> tuple[bool, str]:
    """Start Ollama server for local URLs and verify readiness."""
    target_url = base_url.strip() or DEFAULT_OLLAMA_BASE_URL
    if not _is_local_ollama_url(target_url):
        return False, "Can only start Ollama automatically for localhost URLs."
    if _is_ollama_server_up(target_url):
        return True, f"Ollama server is already running at {target_url}."
    ollama_cli_path = _resolve_ollama_cli_path(str(st.session_state.get("ollama_cli_path", "")).strip())
    if not ollama_cli_path:
        searched_paths = [p for p in _candidate_ollama_cli_paths() if p]
        return (
            False,
            "Could not find the Ollama CLI binary. Set 'Ollama CLI Path' in Settings "
            f"or install Ollama. Checked: {searched_paths}",
        )
    try:
        subprocess.Popen(
            [ollama_cli_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        return False, "Could not find the 'ollama' CLI on PATH."
    except Exception as exc:
        return False, f"Failed to start Ollama server: {exc}"

    for _ in range(8):
        try:
            response = requests.get(f"{target_url.rstrip('/')}/api/tags", timeout=1.5)
            if response.status_code == 200:
                return True, f"Started Ollama server at {target_url}."
        except Exception:
            pass
        time.sleep(0.5)
    return False, f"Started process but could not verify Ollama at {target_url} yet."


if "openai_api_key_initialized" not in st.session_state:
    st.session_state["openai_api_key_initialized"] = True
    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_api_key and _is_valid_openai_api_key(env_api_key):
        st.session_state["openai_api_key"] = env_api_key
    else:
        st.session_state["openai_api_key"] = ""

if "llm_settings_initialized" not in st.session_state:
    st.session_state["llm_settings_initialized"] = True
    env_provider_list = _normalize_provider_list(os.getenv("LLM_PROVIDERS", ""))
    if not env_provider_list:
        env_provider_list = _normalize_provider_list(os.getenv("LLM_PROVIDER", "openai"))
    if not env_provider_list:
        env_provider_list = ["openai"]
    st.session_state["llm_providers"] = env_provider_list
    st.session_state["llm_provider"] = env_provider_list[0]
    st.session_state["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
    st.session_state["ollama_model"] = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    st.session_state["ollama_cli_path"] = _resolve_ollama_cli_path(
        os.getenv("OLLAMA_CLI_PATH", "")
    ) or str(os.getenv("OLLAMA_CLI_PATH", "")).strip()
    st.session_state["settings_provider_openai"] = "openai" in env_provider_list
    st.session_state["settings_provider_ollama"] = "ollama" in env_provider_list
    st.session_state["provider_selection_warning"] = ""

session_api_key = str(st.session_state.get("openai_api_key", "")).strip()
if session_api_key:
    os.environ["OPENAI_API_KEY"] = session_api_key
else:
    os.environ.pop("OPENAI_API_KEY", None)
session_llm_providers = _normalize_provider_list(st.session_state.get("llm_providers", ["openai"]))
if not session_llm_providers:
    session_llm_providers = _normalize_provider_list(st.session_state.get("llm_provider", "openai"))
if not session_llm_providers:
    session_llm_providers = ["openai"]
st.session_state["llm_providers"] = session_llm_providers
session_llm_provider = _resolve_active_provider(
    providers=session_llm_providers,
    preferred=str(st.session_state.get("llm_provider", "openai")),
)
st.session_state["llm_provider"] = session_llm_provider
os.environ["LLM_PROVIDER"] = session_llm_provider
os.environ["LLM_PROVIDERS"] = ",".join(session_llm_providers)
session_ollama_base_url = str(st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)).strip()
session_ollama_model = str(st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL)).strip()
st.session_state["ollama_base_url"] = session_ollama_base_url
st.session_state["ollama_model"] = session_ollama_model
os.environ["OLLAMA_BASE_URL"] = session_ollama_base_url
os.environ["OLLAMA_MODEL"] = session_ollama_model
session_ollama_cli_path = str(st.session_state.get("ollama_cli_path", "")).strip()
if session_ollama_cli_path:
    os.environ["OLLAMA_CLI_PATH"] = session_ollama_cli_path
else:
    os.environ.pop("OLLAMA_CLI_PATH", None)


with st.sidebar:
    navigation_locked = bool(st.session_state.get(BENCHMARK_RUN_IN_PROGRESS_KEY, False))
    if "nav_section" not in st.session_state:
        st.session_state["nav_section"] = "RAG Creation"
    if "nav_subpage" not in st.session_state:
        initial_section = SECTION_ALIASES.get(
            str(st.session_state["nav_section"]),
            str(st.session_state["nav_section"]),
        )
        if initial_section not in SECTIONS:
            initial_section = "RAG Creation"
        st.session_state["nav_section"] = initial_section
        st.session_state["nav_subpage"] = _default_subpage(initial_section)
    _normalize_navigation_state()
    if navigation_locked:
        st.session_state["nav_section"] = "Benchmarking"
        st.session_state["nav_subpage"] = "Run Benchmarks"
        st.warning("Benchmark in progress: navigation is temporarily locked.")

    for group_name, group_sections in GROUP_SECTIONS.items():
        group_is_active = st.session_state.get("nav_section") in group_sections
        _render_section_heading(section_name=group_name, is_active=group_is_active)

        for section_name in group_sections:
            pages = SECTIONS[section_name]
            is_active_section = st.session_state.get("nav_section") == section_name
            if group_name == "Benchmarks":
                _render_subsection_heading(section_name=section_name, is_active=is_active_section)

            for page_name in pages:
                is_active_page = (
                    st.session_state.get("nav_section") == section_name
                    and st.session_state.get("nav_subpage") == page_name
                )
                if st.button(
                    _page_label(page_name),
                    key=f"nav_{section_name}_{page_name}",
                    disabled=navigation_locked or is_active_page,
                    type="secondary",
                ) and not is_active_page:
                    _set_navigation(section_name, page_name)
                    st.rerun()
        st.write("")

    section = str(st.session_state.get("nav_section", "RAG Creation"))
    subpage = _normalize_subpage_name(
        str(st.session_state.get("nav_subpage", _default_subpage(section)))
    )
    st.session_state["nav_subpage"] = subpage

    st.divider()
    with st.expander("Settings", expanded=False):
        selected_providers = list(st.session_state.get("llm_providers", ["openai"]))
        st.session_state["settings_provider_openai"] = "openai" in selected_providers
        st.session_state["settings_provider_ollama"] = "ollama" in selected_providers
        st.markdown("**LLM Provider**")
        provider_col_openai, provider_col_ollama = st.columns(2)
        provider_col_openai.checkbox(
            "OpenAI",
            key="settings_provider_openai",
            on_change=_sync_provider_settings_state,
        )
        provider_col_ollama.checkbox(
            "Ollama",
            key="settings_provider_ollama",
            on_change=_sync_provider_settings_state,
        )
        _sync_provider_settings_state()
        selected_providers = list(st.session_state.get("llm_providers", ["openai"]))
        provider_warning = str(st.session_state.get("provider_selection_warning", "")).strip()
        if provider_warning:
            st.warning(provider_warning)

        st.markdown("**OpenAI Settings**")
        openai_settings_enabled = "openai" in st.session_state.get("llm_providers", [])
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            key="settings_openai_api_key_input",
            placeholder="sk-...",
            disabled=not openai_settings_enabled,
        )
        openai_action_col1, openai_action_col2 = st.columns(2)
        if openai_action_col1.button("Set OpenAI Key", key="set_openai_key", disabled=not openai_settings_enabled):
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

        if openai_action_col2.button(
            "Clear OpenAI Key",
            key="clear_openai_key",
            disabled=not openai_settings_enabled,
        ):
            st.session_state["openai_api_key"] = ""
            st.session_state["settings_openai_api_key_input"] = ""
            os.environ.pop("OPENAI_API_KEY", None)
            st.success("Cleared session and environment key.")
            st.rerun()

        st.markdown("**Ollama Settings**")
        ollama_settings_enabled = "ollama" in st.session_state.get("llm_providers", [])
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
            key="settings_ollama_base_url_input",
            placeholder=DEFAULT_OLLAMA_BASE_URL,
            disabled=not ollama_settings_enabled,
        )
        ollama_model = st.text_input(
            "Ollama Model",
            value=st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL),
            key="settings_ollama_model_input",
            placeholder=DEFAULT_OLLAMA_MODEL,
            disabled=not ollama_settings_enabled,
        )
        ollama_cli_path = st.text_input(
            "Ollama CLI Path",
            value=st.session_state.get("ollama_cli_path", ""),
            key="settings_ollama_cli_path_input",
            placeholder="/usr/local/bin/ollama",
            disabled=not ollama_settings_enabled,
            help="Absolute path to the ollama executable used to auto-start the server.",
        )
        ollama_action_col1, ollama_action_col2 = st.columns(2)
        if ollama_action_col1.button(
            "Save Ollama",
            key="set_ollama_settings",
            disabled=not ollama_settings_enabled,
        ):
            st.session_state["ollama_base_url"] = ollama_base_url.strip() or DEFAULT_OLLAMA_BASE_URL
            st.session_state["ollama_model"] = ollama_model.strip() or DEFAULT_OLLAMA_MODEL
            st.session_state["ollama_cli_path"] = ollama_cli_path.strip()
            st.success("Saved Ollama settings for this session.")
            st.rerun()
        if ollama_action_col2.button(
            "Test Ollama",
            key="test_ollama_connection",
            disabled=not ollama_settings_enabled,
        ):
            ok, message = _is_reachable_ollama_server(
                base_url=str(st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)),
                model=str(st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL)),
            )
            if ok:
                st.success(message)
            else:
                st.warning(message)
        if st.button(
            "Start Ollama Server (Local)",
            key="start_ollama_server",
            disabled=not ollama_settings_enabled,
        ):
            ok, message = _start_local_ollama_server(
                str(st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL))
            )
            if ok:
                st.success(message)
            else:
                st.warning(message)

        active_key = str(st.session_state.get("openai_api_key", "")).strip()
        enabled_providers = st.session_state.get("llm_providers", ["openai"])
        st.caption(f"Enabled providers: {', '.join(enabled_providers)}")
        st.caption(f"Active provider: {st.session_state.get('llm_provider', 'openai')}")
        if "openai" in enabled_providers:
            if active_key:
                st.caption("OpenAI key is set for this session.")
            else:
                st.caption("No OpenAI key is set in this session.")
        if "ollama" in enabled_providers:
            st.caption(
                "Ollama target: "
                f"{st.session_state.get('ollama_base_url', DEFAULT_OLLAMA_BASE_URL)} / "
                f"{st.session_state.get('ollama_model', DEFAULT_OLLAMA_MODEL)}"
            )
            resolved_cli = _resolve_ollama_cli_path(str(st.session_state.get("ollama_cli_path", "")).strip())
            if resolved_cli:
                st.caption(f"Ollama CLI: {resolved_cli}")
            else:
                st.caption("Ollama CLI not found. Set 'Ollama CLI Path' to enable auto-start.")

section = str(st.session_state.get("nav_section", "RAG Creation"))
section = SECTION_ALIASES.get(section, section)
if section not in SECTIONS:
    section = "RAG Creation"
subpage = _normalize_subpage_name(
    str(st.session_state.get("nav_subpage", _default_subpage(section)))
)
if subpage not in SECTIONS[section]:
    subpage = _default_subpage(section)
    st.session_state["nav_subpage"] = subpage

st.caption(f"{section} / {_page_label(subpage)}")

if section == "RAG Creation" and subpage == "Corpus Creation":
    render_corpus_creation(show_title=True)
elif section == "RAG Creation" and subpage in {"Ingest", "Parse and Chunk"}:
    render_ingest(show_title=True)
elif section == "RAG Creation" and subpage == "RAG Model Creator":
    render_rag_model_creator(show_title=True)
elif section == "Knowledge-Base" and subpage == "Load / Append Knowledge Base":
    render_load_append_model(show_title=True)
elif section == "Knowledge-Base" and subpage == "Query Knowledge Base":
    render_query_model(show_title=True)
elif section == "MCP" and subpage == "MCP":
    render_mcp(show_title=True)
elif section == "Knowledge-Base" and subpage == "View Knowledge Graph":
    render_knowledge_graph(show_title=True)
elif section == "Probe Creation" and subpage in {"Question Generation", "Probe Generation"}:
    render_question_generation(show_title=True)
elif section == "Probe Creation" and subpage in {"Verify Questions", "Verify Probes"}:
    render_verify_questions(show_title=True)
elif section == "Benchmarking":
    render_benchmarking(show_title=False)
else:
    st.error(f"Unknown navigation target: {section} / {subpage}")
