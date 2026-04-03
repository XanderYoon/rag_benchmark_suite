from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from benchmark.config import AppConfig, DEFAULT_CONFIG
from RAG.llm import normalize_llm_provider
from benchmark.services.pipeline import PipelineService
from benchmark.verification.verifier import Verifier

BENCHMARK_SNAPSHOT_KEY = "benchmark_snapshot"
LOADED_KNOWLEDGE_BASE_KEY = "loaded_knowledge_base"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config_key(config: AppConfig) -> str:
    payload = asdict(config)
    return "|".join(f"{k}={v}" for k, v in sorted(payload.items()))


def _resolve_corpus_dir(raw_corpus_dir: str | Path) -> Path:
    """Resolve corpus directories relative to the app project root."""
    candidate = Path(raw_corpus_dir).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def get_config() -> AppConfig:
    corpus_dir = st.session_state.get("corpus_dir", str(DEFAULT_CONFIG.corpus_dir))
    llm_provider = normalize_llm_provider(st.session_state.get("llm_provider", DEFAULT_CONFIG.llm_provider))
    ollama_base_url = str(st.session_state.get("ollama_base_url", DEFAULT_CONFIG.ollama_base_url)).strip()
    ollama_model = str(st.session_state.get("ollama_model", DEFAULT_CONFIG.ollama_model)).strip()
    selected_question_provider = normalize_llm_provider(
        st.session_state.get("question_generation_provider", llm_provider)
    )
    selected_question_model = str(st.session_state.get("question_generation_model", "")).strip()
    if not selected_question_model:
        selected_question_model = DEFAULT_CONFIG.question_model
        if selected_question_provider == "ollama":
            selected_question_model = ollama_model or DEFAULT_CONFIG.ollama_model
    return AppConfig(
        chunk_size_tokens=DEFAULT_CONFIG.chunk_size_tokens,
        chunk_overlap_tokens=DEFAULT_CONFIG.chunk_overlap_tokens,
        questions_per_paper=DEFAULT_CONFIG.questions_per_paper,
        retrieval_top_k=DEFAULT_CONFIG.retrieval_top_k,
        retrieval_threshold=DEFAULT_CONFIG.retrieval_threshold,
        retrieval_cap=DEFAULT_CONFIG.retrieval_cap,
        question_model=selected_question_model,
        evidence_model=DEFAULT_CONFIG.evidence_model,
        difficulty_model=DEFAULT_CONFIG.difficulty_model,
        embedding_model=DEFAULT_CONFIG.embedding_model,
        llm_provider=selected_question_provider,
        ollama_base_url=ollama_base_url or DEFAULT_CONFIG.ollama_base_url,
        ollama_model=ollama_model or DEFAULT_CONFIG.ollama_model,
        corpus_dir=_resolve_corpus_dir(corpus_dir),
        text_cache_dir=DEFAULT_CONFIG.text_cache_dir,
        chunk_dir=DEFAULT_CONFIG.chunk_dir,
        benchmark_runs_dir=DEFAULT_CONFIG.benchmark_runs_dir,
    )


def get_pipeline() -> PipelineService:
    config = get_config()
    key = _config_key(config)
    if st.session_state.get("pipeline_config_key") != key:
        st.session_state["pipeline"] = PipelineService(config)
        st.session_state["pipeline_config_key"] = key
    return st.session_state["pipeline"]


def get_verifier() -> Verifier:
    if "verifier" not in st.session_state:
        st.session_state["verifier"] = Verifier()
    return st.session_state["verifier"]


def get_records_store() -> dict[str, list]:
    if "records_by_paper" not in st.session_state:
        st.session_state["records_by_paper"] = {}
    return st.session_state["records_by_paper"]


def get_current_paper_index() -> int:
    return int(st.session_state.get("current_paper_index", 0))


def set_current_paper_index(index: int) -> None:
    st.session_state["current_paper_index"] = max(index, 0)


def set_benchmark_snapshot(*, snapshot: dict) -> None:
    """Persist the latest benchmark snapshot for cross-page rendering."""
    if not isinstance(snapshot, dict):
        raise ValueError(
            "Invalid benchmark snapshot payload. Expected a dictionary suitable for session storage."
        )
    st.session_state[BENCHMARK_SNAPSHOT_KEY] = dict(snapshot)


def get_benchmark_snapshot() -> dict | None:
    """Return the persisted benchmark snapshot if present."""
    raw_snapshot = st.session_state.get(BENCHMARK_SNAPSHOT_KEY)
    if raw_snapshot is None:
        return None
    if not isinstance(raw_snapshot, dict):
        raise ValueError(
            f"Invalid benchmark snapshot found in session key '{BENCHMARK_SNAPSHOT_KEY}'. Expected a dictionary."
        )
    return dict(raw_snapshot)


def clear_benchmark_snapshot() -> None:
    """Remove the persisted benchmark snapshot from Streamlit session state."""
    st.session_state.pop(BENCHMARK_SNAPSHOT_KEY, None)


def set_loaded_knowledge_base(*, knowledge_base: dict) -> None:
    """Persist the active knowledge base selection for later query workflows."""
    if not isinstance(knowledge_base, dict):
        raise ValueError("Invalid knowledge base payload. Expected a dictionary suitable for session storage.")
    st.session_state[LOADED_KNOWLEDGE_BASE_KEY] = dict(knowledge_base)


def get_loaded_knowledge_base() -> dict | None:
    """Return the active knowledge base payload if one has been loaded."""
    raw_payload = st.session_state.get(LOADED_KNOWLEDGE_BASE_KEY)
    if raw_payload is None:
        return None
    if not isinstance(raw_payload, dict):
        raise ValueError(
            f"Invalid knowledge base payload found in session key '{LOADED_KNOWLEDGE_BASE_KEY}'."
        )
    return dict(raw_payload)


def clear_loaded_knowledge_base() -> None:
    """Remove the active knowledge base from session state."""
    st.session_state.pop(LOADED_KNOWLEDGE_BASE_KEY, None)
