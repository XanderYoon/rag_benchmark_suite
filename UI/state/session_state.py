from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from Benchmark.config import AppConfig, DEFAULT_CONFIG
from Benchmark.services.pipeline import PipelineService
from Benchmark.verification.verifier import Verifier


def _config_key(config: AppConfig) -> str:
    payload = asdict(config)
    return "|".join(f"{k}={v}" for k, v in sorted(payload.items()))


def get_config() -> AppConfig:
    corpus_dir = st.session_state.get("corpus_dir", str(DEFAULT_CONFIG.corpus_dir))
    return AppConfig(
        chunk_size_tokens=DEFAULT_CONFIG.chunk_size_tokens,
        chunk_overlap_tokens=DEFAULT_CONFIG.chunk_overlap_tokens,
        questions_per_paper=DEFAULT_CONFIG.questions_per_paper,
        retrieval_top_k=DEFAULT_CONFIG.retrieval_top_k,
        retrieval_threshold=DEFAULT_CONFIG.retrieval_threshold,
        retrieval_cap=DEFAULT_CONFIG.retrieval_cap,
        question_model=DEFAULT_CONFIG.question_model,
        evidence_model=DEFAULT_CONFIG.evidence_model,
        difficulty_model=DEFAULT_CONFIG.difficulty_model,
        embedding_model=DEFAULT_CONFIG.embedding_model,
        corpus_dir=Path(corpus_dir),
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
