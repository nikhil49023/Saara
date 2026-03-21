"""
Unified LLM Provider System
Supports Ollama, Gemini, OpenAI, Nemotron, Claude, and custom providers.

Simple API - user provides API key, we handle the rest.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import logging
from typing import Dict, Any, Optional, List, Generator, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"           # Local Ollama
    GEMINI = "gemini"           # Google Gemini
    OPENAI = "openai"           # OpenAI GPT
    ANTHROPIC = "anthropic"     # Anthropic Claude
    NEMOTRON = "nemotron"       # NVIDIA Nemotron
    GROQ = "groq"               # Groq (fast inference)
    TOGETHER = "together"       # Together AI
    REPLICATE = "replicate"     # Replicate
    HUGGINGFACE = "huggingface" # HuggingFace Inference API
    CUSTOM = "custom"           # Custom endpoint


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str                    # Provider name (ollama, gemini, etc.)
    api_key: Optional[str] = None    # API key (not needed for Ollama)
    model: Optional[str] = None      # Model name
    base_url: Optional[str] = None   # Custom endpoint URL
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 300


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""

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


# =============================================================================
# Ollama Provider (Local)
# =============================================================================

class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import ollama
            self.client = ollama.Client(host=config.base_url or "http://localhost:11434")
            self.available = True
        except ImportError:
            logger.warning("ollama package not installed: pip install ollama")
            self.available = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.available:
            raise RuntimeError("Ollama not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.config.model or "granite4",
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        )
        return response['message']['content']

    def is_available(self) -> bool:
        if not self.available:
            return False
        try:
            self.client.list()
            return True
        except Exception:
            return False


# =============================================================================
# Gemini Provider (Google)
# =============================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            self.model = genai.GenerativeModel(config.model or "gemini-2.0-flash")
            self.available = True
        except ImportError:
            logger.warning("google-generativeai not installed: pip install google-generativeai")
            self.available = False
        except Exception as e:
            logger.warning(f"Gemini setup failed: {e}")
            self.available = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.available:
            raise RuntimeError("Gemini not available")

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
        )
        return response.text

    def is_available(self) -> bool:
        return self.available


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.api_key)
            self.available = True
        except ImportError:
            logger.warning("openai package not installed: pip install openai")
            self.available = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.available:
            raise RuntimeError("OpenAI not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model or "gpt-4",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        return self.available


# =============================================================================
# Anthropic Provider (Claude)
# =============================================================================

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key)
            self.available = True
        except ImportError:
            logger.warning("anthropic package not installed: pip install anthropic")
            self.available = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.available:
            raise RuntimeError("Anthropic not available")

        message = self.client.messages.create(
            model=self.config.model or "claude-3-5-sonnet-20241022",
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def is_available(self) -> bool:
        return self.available


# =============================================================================
# NVIDIA Nemotron Provider
# =============================================================================

class NemotronProvider(BaseLLMProvider):
    """NVIDIA Nemotron API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai  # Nemotron uses OpenAI-compatible API
            self.client = openai.OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=config.api_key
            )
            self.available = True
        except ImportError:
            logger.warning("openai package not installed: pip install openai")
            self.available = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.available:
            raise RuntimeError("Nemotron not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model or "nvidia/llama-3.1-nemotron-70b-instruct",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        return self.available


# =============================================================================
# Groq Provider (Fast Inference)
# =============================================================================

class GroqProvider(BaseLLMProvider):
    """Groq fast inference API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from groq import Groq
            self.client = Groq(api_key=config.api_key)
            self.available = True
        except ImportError:
            logger.warning("groq package not installed: pip install groq")
            self.available = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.available:
            raise RuntimeError("Groq not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model or "llama-3.1-70b-versatile",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        return self.available


# =============================================================================
# Provider Factory
# =============================================================================

class LLMProviderFactory:
    """Factory to create LLM providers."""

    _providers = {
        LLMProvider.OLLAMA.value: OllamaProvider,
        LLMProvider.GEMINI.value: GeminiProvider,
        LLMProvider.OPENAI.value: OpenAIProvider,
        LLMProvider.ANTHROPIC.value: AnthropicProvider,
        LLMProvider.NEMOTRON.value: NemotronProvider,
        LLMProvider.GROQ.value: GroqProvider,
    }

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create provider from config."""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())


# =============================================================================
# Unified LLM Client
# =============================================================================

class UnifiedLLM:
    """
    Unified LLM client - works with any provider.

    Examples:
        >>> # Ollama (local)
        >>> llm = UnifiedLLM(provider="ollama", model="granite4")
        >>> response = llm.generate("Explain AI")

        >>> # Gemini
        >>> llm = UnifiedLLM(provider="gemini", api_key="...", model="gemini-2.0-flash")
        >>> response = llm.generate("Explain AI")

        >>> # OpenAI
        >>> llm = UnifiedLLM(provider="openai", api_key="...", model="gpt-4")
        >>> response = llm.generate("Explain AI")

        >>> # Nemotron
        >>> llm = UnifiedLLM(provider="nemotron", api_key="...")
        >>> response = llm.generate("Explain AI")
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 300
    ):
        """
        Initialize unified LLM client.

        Args:
            provider: Provider name (ollama, gemini, openai, anthropic, nemotron, groq)
            api_key: API key (not needed for Ollama)
            model: Model name (provider-specific)
            base_url: Custom base URL
            temperature: Generation temperature (0-1)
            max_tokens: Max tokens to generate
            timeout: Request timeout in seconds
        """
        self.config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        self.provider = LLMProviderFactory.create(self.config)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        return self.provider.generate(prompt, system_prompt)

    def is_available(self) -> bool:
        """Check if provider is available and working."""
        return self.provider.is_available()

    @staticmethod
    def list_providers() -> List[str]:
        """List all supported providers."""
        return LLMProviderFactory.list_providers()


# =============================================================================
# Quick Helper Functions
# =============================================================================

def create_llm(
    provider: str = "ollama",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> UnifiedLLM:
    """
    Quick helper to create LLM client.

    Examples:
        >>> llm = create_llm("ollama", model="granite4")
        >>> llm = create_llm("gemini", api_key="...", model="gemini-2.0-flash")
        >>> llm = create_llm("openai", api_key="...", model="gpt-4")
        >>> llm = create_llm("nemotron", api_key="...")
    """
    return UnifiedLLM(provider=provider, api_key=api_key, model=model, **kwargs)


def quick_generate(
    prompt: str,
    provider: str = "ollama",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick one-line generation.

    Example:
        >>> response = quick_generate("Explain AI", provider="gemini", api_key="...")
    """
    llm = create_llm(provider, api_key, model, **kwargs)
    return llm.generate(prompt)
