from __future__ import annotations

import os
from typing import Any

import requests


SUPPORTED_LLM_PROVIDERS = {"openai", "ollama"}
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


def normalize_llm_provider(raw_provider: str | None) -> str:
    """Return a supported provider identifier with a safe default."""
    provider = str(raw_provider or "").strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        return "openai"
    return provider


def generate_llm_text(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
    timeout_seconds: float = 45.0,
) -> str | None:
    """Generate a text response from the selected provider."""
    normalized_provider = normalize_llm_provider(provider)
    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("LLM model is required.")

    if normalized_provider == "ollama":
        return _generate_with_ollama(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            ollama_base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            timeout_seconds=timeout_seconds,
        )

    return _generate_with_openai(
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        openai_api_key=(openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip(),
    )


def _generate_with_openai(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    openai_api_key: str,
) -> str | None:
    """Generate text with OpenAI Responses API."""
    if not openai_api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        return text or None
    except Exception:
        return None


def _generate_with_ollama(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    ollama_base_url: str,
    timeout_seconds: float,
) -> str | None:
    """Generate text with a local Ollama server."""
    base_url = ollama_base_url.rstrip("/")
    if not base_url:
        raise ValueError("Ollama base URL is required.")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
    except Exception:
        return None

    message = body.get("message")
    if isinstance(message, dict):
        content = str(message.get("content", "")).strip()
        if content:
            return content
    raw_response = str(body.get("response", "")).strip()
    return raw_response or None
