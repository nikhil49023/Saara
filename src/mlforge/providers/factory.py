from __future__ import annotations

from .mock import MockProvider
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider


def create_provider(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    provider = provider.lower()
    if provider == "mock":
        return MockProvider(model=model or "mock-local")
    if provider == "ollama":
        return OllamaProvider(model=model or "qwen", base_url=base_url or "http://localhost:11434")
    if provider in {"vllm", "openai-compatible", "openai"}:
        return OpenAICompatibleProvider(
            model=model or "local-model",
            base_url=base_url or "http://localhost:8000/v1",
            api_key=api_key or "EMPTY",
        )
    raise ValueError(f"Unsupported provider: {provider}")
