from __future__ import annotations

from UI.views.query_model_view import (
    _default_retrieval_model,
    _provider_retrieval_models,
)


def test_provider_retrieval_models_returns_openai_models() -> None:
    assert _provider_retrieval_models("openai") == [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]


def test_provider_retrieval_models_returns_ollama_models() -> None:
    assert _provider_retrieval_models("ollama") == ["nomic-embed-text"]


def test_default_retrieval_model_prefers_loaded_knowledge_base_model() -> None:
    payload = {"embedding_model": "text-embedding-3-large"}

    assert _default_retrieval_model(provider="openai", loaded_knowledge_base=payload) == "text-embedding-3-large"


def test_default_retrieval_model_falls_back_to_provider_default() -> None:
    payload = {"embedding_model": "nomic-embed-text"}

    assert _default_retrieval_model(provider="openai", loaded_knowledge_base=payload) == "text-embedding-3-small"
