"""
Local Inference Engine - vLLM + Ollama with Auto-Fallback
Optimized for local-first approach with cloud notebook compatibility.

Priority: vLLM (faster) → Ollama (simpler) → API fallback (cloud)
"""

import os
import sys
import logging
import time
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class InferenceBackend(Enum):
    """Available inference backends."""
    VLLM = "vllm"           # Fast, production-ready
    OLLAMA = "ollama"       # Simple, requires daemon
    API = "api"             # Cloud-based API (GPT, Gemini)
    UNKNOWN = "unknown"


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    backend: Optional[str] = None      # Force backend (vllm, ollama, api)
    model: str = "mistral:latest"      # Model name
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 300
    base_url: Optional[str] = None     # For Ollama: http://localhost:11434
    api_key: Optional[str] = None      # For API-based backends
    enable_auto_fallback: bool = True  # Try next backend if current fails


class BaseInferenceEngine(ABC):
    """Base class for inference engines."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.backend_name = self.__class__.__name__

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text generation."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    def health_check(self) -> bool:
        """Health check - override in subclasses."""
        try:
            response = self.generate("Hello", system_prompt="Respond with 'OK'")
            return "OK" in response or len(response) > 0
        except Exception as e:
            logger.warning(f"{self.backend_name} health check failed: {e}")
            return False


# ============================================================================
# vLLM Backend (Recommended for Local & Cloud)
# ============================================================================

class vLLMEngine(BaseInferenceEngine):
    """vLLM inference engine - fastest, cloud-ready."""

    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self._client = None
        self._initialized = False

        try:
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.available = True
        except ImportError:
            logger.warning("vLLM not installed: pip install vllm")
            self.available = False

    def _initialize(self):
        """Lazy initialization to avoid loading model on import."""
        if self._initialized or not self.available:
            return

        try:
            logger.info(f"Loading vLLM model: {self.config.model}")
            self._client = self.LLM(
                model=self.config.model,
                tensor_parallel_size=1,
                dtype="float16"
            )
            self._initialized = True
            logger.info("vLLM engine ready")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            self.available = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using vLLM."""
        self._initialize()
        if not self.available or not self._client:
            raise RuntimeError("vLLM not initialized")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Format for vLLM
        formatted_prompt = self._format_prompt(messages)

        sampling_params = self.SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        outputs = self._client.generate([formatted_prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text generation from vLLM."""
        self._initialize()
        if not self.available or not self._client:
            raise RuntimeError("vLLM not initialized")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = self._format_prompt(messages)

        sampling_params = self.SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # vLLM batching - yield tokens as they arrive
        outputs = self._client.generate([formatted_prompt], sampling_params, use_tqdm=False)
        for output in outputs:
            yield output.outputs[0].text

    def is_available(self) -> bool:
        return self.available

    @staticmethod
    def _format_prompt(messages: List[Dict]) -> str:
        """Format messages for vLLM."""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<s>[INST] <<SYS>>\n{msg['content']}\n<</SYS>>\n\n"
            elif msg["role"] == "user":
                prompt += f"{msg['content']} [/INST]\n"
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']} </s>\n"
        return prompt


# ============================================================================
# Ollama Backend (Simple Fallback)
# ============================================================================

class OllamaEngine(BaseInferenceEngine):
    """Ollama inference engine - simple, requires daemon running."""

    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self._client = None
        self.available = False

        try:
            import ollama
            self.ollama = ollama
            self.base_url = config.base_url or "http://localhost:11434"
            self._client = ollama.Client(host=self.base_url)
            self.available = True
        except ImportError:
            logger.warning("ollama package not installed: pip install ollama")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama."""
        if not self.available or not self._client:
            raise RuntimeError("Ollama not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text generation from Ollama."""
        if not self.available or not self._client:
            raise RuntimeError("Ollama not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            )
            for chunk in response:
                if 'message' in chunk:
                    yield chunk['message']['content']
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    def is_available(self) -> bool:
        if not self.available:
            return False
        try:
            # Quick health check
            self._client.list()
            return True
        except Exception:
            return False


# ============================================================================
# Auto-Selecting Inference Engine
# ============================================================================

class LocalInferenceEngine:
    """
    Smart inference coordinator - tries backends in priority order.
    Automatically selects best available backend with fallback support.
    """

    # Priority order: vLLM (fastest) → Ollama (simple) → API (cloud fallback)
    BACKEND_PRIORITY = [
        vLLMEngine,
        OllamaEngine,
    ]

    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.current_engine = None
        self.backend_name = None
        self._select_backend()

    def _select_backend(self):
        """Select best available backend."""
        if self.config.backend:
            # Force specific backend
            self._init_specific_backend(self.config.backend)
        else:
            # Try backends in priority order
            for engine_class in self.BACKEND_PRIORITY:
                try:
                    engine = engine_class(self.config)
                    if engine.is_available():
                        self.current_engine = engine
                        self.backend_name = engine_class.__name__
                        logger.info(f"Using {self.backend_name} for inference")
                        return
                except Exception as e:
                    logger.debug(f"Skipped {engine_class.__name__}: {e}")

        if not self.current_engine:
            raise RuntimeError(
                "No inference backend available. "
                "Install vLLM (pip install vllm) or Ollama (ollama.ai)"
            )

    def _init_specific_backend(self, backend_name: str):
        """Force initialization of specific backend."""
        backend_map = {
            "vllm": vLLMEngine,
            "ollama": OllamaEngine,
        }

        backend_class = backend_map.get(backend_name.lower())
        if not backend_class:
            raise ValueError(f"Unknown backend: {backend_name}")

        engine = backend_class(self.config)
        if not engine.is_available():
            raise RuntimeError(f"{backend_name} is not available")

        self.current_engine = engine
        self.backend_name = backend_class.__name__

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text."""
        if not self.current_engine:
            raise RuntimeError("No inference engine initialized")

        if self.config.enable_auto_fallback:
            return self._generate_with_fallback(prompt, system_prompt)
        else:
            return self.current_engine.generate(prompt, system_prompt)

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text generation."""
        if not self.current_engine:
            raise RuntimeError("No inference engine initialized")

        if self.config.enable_auto_fallback:
            return self._generate_stream_with_fallback(prompt, system_prompt)
        else:
            return self.current_engine.generate_stream(prompt, system_prompt)

    def _generate_with_fallback(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Try current backend, fallback to others if needed."""
        if self.current_engine:
            try:
                return self.current_engine.generate(prompt, system_prompt)
            except Exception as e:
                logger.warning(f"{self.backend_name} failed: {e}. Trying fallback...")

        # Try other backends
        for engine_class in self.BACKEND_PRIORITY:
            if engine_class.__name__ == self.backend_name:
                continue  # Skip current engine
            try:
                engine = engine_class(self.config)
                if engine.is_available():
                    logger.info(f"Falling back to {engine_class.__name__}")
                    self.current_engine = engine
                    self.backend_name = engine_class.__name__
                    return engine.generate(prompt, system_prompt)
            except Exception as e:
                logger.debug(f"Fallback to {engine_class.__name__} failed: {e}")

        raise RuntimeError("All inference backends failed")

    def _generate_stream_with_fallback(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Try current backend stream, fallback to others if needed."""
        if self.current_engine:
            try:
                yield from self.current_engine.generate_stream(prompt, system_prompt)
                return
            except Exception as e:
                logger.warning(f"{self.backend_name} streaming failed: {e}")

        # Try other backends
        for engine_class in self.BACKEND_PRIORITY:
            if engine_class.__name__ == self.backend_name:
                continue
            try:
                engine = engine_class(self.config)
                if engine.is_available():
                    logger.info(f"Falling back to {engine_class.__name__}")
                    self.current_engine = engine
                    self.backend_name = engine_class.__name__
                    yield from engine.generate_stream(prompt, system_prompt)
                    return
            except Exception as e:
                logger.debug(f"Fallback to {engine_class.__name__} failed: {e}")

        raise RuntimeError("All inference backends failed")

    def health_check(self) -> Dict[str, Any]:
        """Get health status of all backends."""
        status = {}
        for engine_class in self.BACKEND_PRIORITY:
            try:
                engine = engine_class(self.config)
                status[engine_class.__name__] = {
                    "available": engine.is_available(),
                    "healthy": engine.health_check() if engine.is_available() else False
                }
            except Exception as e:
                status[engine_class.__name__] = {
                    "available": False,
                    "error": str(e)
                }
        return status

    def get_info(self) -> Dict[str, str]:
        """Get current engine info."""
        return {
            "backend": self.backend_name,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
