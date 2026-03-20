from __future__ import annotations

from Benchmark.config import DEFAULT_CONFIG
from Benchmark.services.retrieval_service import RetrievalService


def test_can_retry_from_cached_error_when_api_key_now_exists(monkeypatch) -> None:
    service = RetrievalService(DEFAULT_CONFIG)
    service.faiss_error = "OPENAI_API_KEY is not set."

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    assert service._can_retry_from_cached_error() is True


def test_cannot_retry_from_cached_error_without_api_key(monkeypatch) -> None:
    service = RetrievalService(DEFAULT_CONFIG)
    service.faiss_error = "OPENAI_API_KEY is not set."

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert service._can_retry_from_cached_error() is False

