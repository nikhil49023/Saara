"""
Local LLM Provider System
Supports local inference backends only: vLLM and Ollama.

Simple API - choose a local backend or use auto fallback.

Released under the MIT License.
"""

import logging
from typing import Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported local LLM providers."""
    AUTO = "auto"
    VLLM = "vllm"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for local LLM provider."""
    provider: str = "auto"
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 300


class BaseLLMProvider(ABC):
    """Base class for local LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class LocalEngineProvider(BaseLLMProvider):
    """Provider backed by SAARA's LocalInferenceEngine (vLLM/Ollama)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from .local_inference import InferenceConfig, LocalInferenceEngine

        backend = config.provider if config.provider in {"vllm", "ollama"} else None
        self._engine = LocalInferenceEngine(
            InferenceConfig(
                backend=backend,
                model=config.model or "mistral",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                base_url=config.base_url,
                enable_auto_fallback=True,
            )
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._engine.generate(prompt=prompt, system_prompt=system_prompt)

    def is_available(self) -> bool:
        try:
            health = self._engine.health_check()
            return any(v.get("available") for v in health.values())
        except Exception:
            return False


class LLMProviderFactory:
    """Factory to create local LLM providers."""

    _providers = {
        LLMProvider.AUTO.value: LocalEngineProvider,
        LLMProvider.VLLM.value: LocalEngineProvider,
        LLMProvider.OLLAMA.value: LocalEngineProvider,
    }

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create provider from config."""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}. Use one of: {', '.join(cls.list_providers())}")
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())


class UnifiedLLM:
    """
    Unified local LLM client.

    Examples:
        >>> llm = UnifiedLLM(provider="auto", model="mistral")
        >>> response = llm.generate("Explain AI")

        >>> llm = UnifiedLLM(provider="vllm", model="mistral")
        >>> response = llm.generate("Explain AI")

        >>> llm = UnifiedLLM(provider="ollama", model="mistral")
        >>> response = llm.generate("Explain AI")
    """

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 300,
    ):
        self.config = LLMConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.provider = LLMProviderFactory.create(self.config)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt."""
        return self.provider.generate(prompt, system_prompt)

    def is_available(self) -> bool:
        """Check if provider is available and working."""
        return self.provider.is_available()

    @staticmethod
    def list_providers() -> List[str]:
        """List all supported local providers."""
        return LLMProviderFactory.list_providers()


def create_llm(
    provider: str = "auto",
    model: Optional[str] = None,
    **kwargs,
) -> UnifiedLLM:
    """Quick helper to create a local LLM client."""
    return UnifiedLLM(provider=provider, model=model, **kwargs)


def quick_generate(
    prompt: str,
    provider: str = "auto",
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """Quick one-line local generation."""
    llm = create_llm(provider=provider, model=model, **kwargs)
    return llm.generate(prompt)
