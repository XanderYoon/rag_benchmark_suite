from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CORPUS_DIR = Path("rag_corpus_pdf") if Path("rag_corpus_pdf").exists() else Path("data/rag_corpus_pdf")


@dataclass(frozen=True)
class AppConfig:
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 60
    questions_per_paper: int = 3

    retrieval_top_k: int = 8
    retrieval_threshold: float = 0.15
    retrieval_cap: int = 25

    question_model: str = "gpt-4o-mini"
    evidence_model: str = "gpt-4o-mini"
    difficulty_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    corpus_dir: Path = DEFAULT_CORPUS_DIR
    text_cache_dir: Path = Path("data/rag_corpus_text")
    chunk_dir: Path = Path("data/rag_corpus_chunked")
    benchmark_runs_dir: Path = Path("data/benchmark_runs")


DEFAULT_CONFIG = AppConfig()
