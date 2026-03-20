from __future__ import annotations

import pytest

from Benchmark.llm.provider_client import generate_llm_text, normalize_llm_provider


def test_normalize_llm_provider_defaults_to_openai() -> None:
    assert normalize_llm_provider(None) == "openai"
    assert normalize_llm_provider("unknown") == "openai"
    assert normalize_llm_provider("OLLAMA") == "ollama"


def test_generate_llm_text_requires_model() -> None:
    with pytest.raises(ValueError, match="LLM model is required."):
        generate_llm_text(
            provider="openai",
            model="",
            system_prompt="sys",
            user_prompt="user",
        )


def test_generate_llm_text_ollama_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "hello from ollama"}}

    def _fake_post(*args: object, **kwargs: object) -> DummyResponse:
        return DummyResponse()

    monkeypatch.setattr("Benchmark.llm.provider_client.requests.post", _fake_post)
    text = generate_llm_text(
        provider="ollama",
        model="qwen3:8b",
        system_prompt="sys",
        user_prompt="user",
        ollama_base_url="http://localhost:11434",
    )
    assert text == "hello from ollama"


def test_generate_llm_text_openai_without_key_returns_none() -> None:
    text = generate_llm_text(
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="sys",
        user_prompt="user",
        openai_api_key="",
    )
    assert text is None
