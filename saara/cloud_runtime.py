"""
Cloud Runtime Module
Enables running SAARA CLI on cloud platforms (Google Colab, Kaggle, etc.)
Uses API-based models instead of Ollama for cloud compatibility.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)


# ============================================================================
# Cloud Environment Detection
# ============================================================================

class CloudEnvironment(Enum):
    """Supported cloud environments."""
    LOCAL = "local"
    COLAB = "google_colab"
    KAGGLE = "kaggle"
    SAGEMAKER = "aws_sagemaker"
    AZURE_ML = "azure_ml"
    PAPERSPACE = "paperspace"
    RUNPOD = "runpod"
    LAMBDA = "lambda_labs"
    UNKNOWN_CLOUD = "unknown_cloud"


def detect_environment() -> CloudEnvironment:
    """Detect the current runtime environment."""
    
    # Google Colab
    try:
        import google.colab
        return CloudEnvironment.COLAB
    except ImportError:
        pass
    
    # Kaggle
    if os.path.exists('/kaggle'):
        return CloudEnvironment.KAGGLE
    
    # AWS SageMaker
    if os.environ.get('SM_MODEL_DIR') or os.environ.get('SAGEMAKER_TRAINING_MODULE'):
        return CloudEnvironment.SAGEMAKER
    
    # Azure ML
    if os.environ.get('AZUREML_RUN_ID'):
        return CloudEnvironment.AZURE_ML
    
    # Paperspace Gradient
    if os.environ.get('PAPERSPACE_CLUSTER_ID'):
        return CloudEnvironment.PAPERSPACE
    
    # RunPod
    if os.environ.get('RUNPOD_POD_ID'):
        return CloudEnvironment.RUNPOD
    
    # Lambda Labs
    if os.path.exists('/home/ubuntu/.lambda'):
        return CloudEnvironment.LAMBDA
    
    # Check for generic cloud indicators
    if any([
        os.environ.get('CLOUD_SHELL'),
        os.environ.get('KUBERNETES_SERVICE_HOST'),
        '/content' in os.getcwd()  # Colab-like
    ]):
        return CloudEnvironment.UNKNOWN_CLOUD
    
    return CloudEnvironment.LOCAL


def is_cloud_environment() -> bool:
    """Check if running in a cloud environment."""
    return detect_environment() != CloudEnvironment.LOCAL


def get_environment_info() -> Dict[str, Any]:
    """Get detailed information about the current environment."""
    env = detect_environment()
    
    info = {
        "environment": env.value,
        "is_cloud": is_cloud_environment(),
        "python_version": sys.version,
        "cwd": os.getcwd(),
    }
    
    # GPU info
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = False
    
    return info


# ============================================================================
# Cloud API Client Base
# ============================================================================

class CloudAPIClient(ABC):
    """Abstract base class for cloud API clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate text with streaming."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API is available."""
        pass


# ============================================================================
# Google AI (Gemini) Client
# ============================================================================

class GeminiClient(CloudAPIClient):
    """Client for Google AI Gemini API - recommended for Colab."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._client = None
        
        # Try to use google-generativeai if available
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            self._use_sdk = True
        except ImportError:
            self._use_sdk = False
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 2048, **kwargs) -> str:
        """Generate text using Gemini."""
        if not self.is_available():
            raise ValueError("Gemini API key not configured")
        
        if self._use_sdk and self._client:
            try:
                response = self._client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini SDK error: {e}")
                raise
        
        # Fallback to REST API
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream generation (simplified - returns full response)."""
        yield self.generate(prompt, **kwargs)


# ============================================================================
# OpenAI Client
# ============================================================================

class OpenAIClient(CloudAPIClient):
    """Client for OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini",
                 base_url: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 2048, system_prompt: str = None, **kwargs) -> str:
        """Generate text using OpenAI."""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream generation."""
        yield self.generate(prompt, **kwargs)


# ============================================================================
# DeepSeek Client
# ============================================================================

class DeepSeekClient(CloudAPIClient):
    """Client for DeepSeek API - cost-effective alternative."""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com/v1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 2048, system_prompt: str = None, **kwargs) -> str:
        """Generate text using DeepSeek."""
        if not self.is_available():
            raise ValueError("DeepSeek API key not configured")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield self.generate(prompt, **kwargs)


# ============================================================================
# Groq Client (Fast inference)
# ============================================================================

class GroqClient(CloudAPIClient):
    """Client for Groq API - ultra-fast inference."""
    
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 2048, system_prompt: str = None, **kwargs) -> str:
        """Generate text using Groq."""
        if not self.is_available():
            raise ValueError("Groq API key not configured")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield self.generate(prompt, **kwargs)


# ============================================================================
# HuggingFace Inference Client
# ============================================================================

class HuggingFaceClient(CloudAPIClient):
    """Client for HuggingFace Inference API."""
    
    def __init__(self, api_key: str = None, 
                 model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.model = model
        self.base_url = f"https://api-inference.huggingface.co/models/{model}"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 2048, **kwargs) -> str:
        """Generate text using HuggingFace Inference."""
        if not self.is_available():
            raise ValueError("HuggingFace token not configured")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            }
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        return str(data)
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield self.generate(prompt, **kwargs)


# ============================================================================
# Unified Cloud Client
# ============================================================================

@dataclass
class CloudConfig:
    """Configuration for cloud runtime."""
    provider: str = "auto"  # auto, gemini, openai, deepseek, groq, huggingface
    api_key: str = None
    model: str = None
    temperature: float = 0.7
    max_tokens: int = 2048


class UnifiedCloudClient:
    """
    Unified client that automatically selects best available cloud provider.
    Priority: Gemini (free for Colab) > Groq (fast) > DeepSeek (cheap) > OpenAI > HuggingFace
    """
    
    PROVIDER_PRIORITY = ["gemini", "groq", "deepseek", "openai", "huggingface"]
    
    def __init__(self, config: CloudConfig = None):
        self.config = config or CloudConfig()
        self._client: Optional[CloudAPIClient] = None
        self._provider: str = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize with best available provider."""
        if self.config.provider != "auto":
            self._client = self._create_client(self.config.provider)
            self._provider = self.config.provider
            return
        
        # Auto-detect best provider
        for provider in self.PROVIDER_PRIORITY:
            client = self._create_client(provider)
            if client and client.is_available():
                self._client = client
                self._provider = provider
                logger.info(f"☁️ Cloud provider: {provider}")
                return
        
        logger.warning("No cloud API providers available")
    
    def _create_client(self, provider: str) -> Optional[CloudAPIClient]:
        """Create client for specific provider."""
        clients = {
            "gemini": lambda: GeminiClient(api_key=self.config.api_key, 
                                           model=self.config.model or "gemini-2.0-flash-exp"),
            "openai": lambda: OpenAIClient(api_key=self.config.api_key,
                                           model=self.config.model or "gpt-4o-mini"),
            "deepseek": lambda: DeepSeekClient(api_key=self.config.api_key,
                                               model=self.config.model or "deepseek-chat"),
            "groq": lambda: GroqClient(api_key=self.config.api_key,
                                       model=self.config.model or "llama-3.3-70b-versatile"),
            "huggingface": lambda: HuggingFaceClient(api_key=self.config.api_key,
                                                      model=self.config.model or "meta-llama/Llama-3.2-3B-Instruct"),
        }
        
        creator = clients.get(provider.lower())
        return creator() if creator else None
    
    @property
    def provider(self) -> str:
        """Get current provider name."""
        return self._provider or "none"
    
    def is_available(self) -> bool:
        """Check if any provider is available."""
        return self._client is not None and self._client.is_available()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using best available provider."""
        if not self._client:
            raise RuntimeError("No cloud API provider available. Set API keys.")
        
        # Merge config defaults with kwargs
        temperature = kwargs.pop('temperature', self.config.temperature)
        max_tokens = kwargs.pop('max_tokens', self.config.max_tokens)
        
        return self._client.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream generation."""
        if not self._client:
            raise RuntimeError("No cloud API provider available")
        
        yield from self._client.generate_stream(prompt, **kwargs)


# ============================================================================
# Cloud Runtime Manager
# ============================================================================

class CloudRuntime:
    """
    Manager for cloud runtime operations.
    Handles setup, configuration, and provides optimized defaults for cloud environments.
    """
    
    def __init__(self):
        self.environment = detect_environment()
        self.cloud_client: Optional[UnifiedCloudClient] = None
        self._setup_complete = False
    
    def setup(self, 
              api_key: str = None,
              provider: str = "auto",
              install_deps: bool = True) -> bool:
        """
        Setup cloud runtime environment.
        
        Args:
            api_key: API key for cloud provider
            provider: Provider to use (auto, gemini, openai, etc.)
            install_deps: Whether to install missing dependencies
        """
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[cyan]☁️ Setting up Cloud Runtime[/cyan]\n"
                f"Environment: [green]{self.environment.value}[/green]",
                title="SAARA Cloud Setup"
            ))
        
        # Install dependencies in Colab/Kaggle
        if install_deps and self.environment in [CloudEnvironment.COLAB, CloudEnvironment.KAGGLE]:
            self._install_cloud_deps()
        
        # Initialize cloud client
        config = CloudConfig(
            provider=provider,
            api_key=api_key
        )
        self.cloud_client = UnifiedCloudClient(config)
        
        if not self.cloud_client.is_available():
            if RICH_AVAILABLE:
                console.print("[yellow]⚠️ No API keys found. Set one of:[/yellow]")
                console.print("  • GOOGLE_API_KEY (Gemini - recommended)")
                console.print("  • GROQ_API_KEY (Fast inference)")
                console.print("  • DEEPSEEK_API_KEY (Cost-effective)")
                console.print("  • OPENAI_API_KEY")
            return False
        
        self._setup_complete = True
        
        if RICH_AVAILABLE:
            console.print(f"[green]✅ Cloud runtime ready![/green]")
            console.print(f"[cyan]Provider:[/cyan] {self.cloud_client.provider}")
        
        return True
    
    def _install_cloud_deps(self):
        """Install dependencies for cloud environment."""
        import subprocess
        
        packages = [
            "google-generativeai",
            "transformers",
            "datasets",
            "accelerate",
            "bitsandbytes",
            "peft",
            "trl",
            "sentencepiece",
        ]
        
        if RICH_AVAILABLE:
            with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
                task = progress.add_task("Installing dependencies...", total=len(packages))
                for pkg in packages:
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-q", pkg],
                            capture_output=True,
                            check=True
                        )
                    except subprocess.CalledProcessError:
                        pass
                    progress.advance(task)
        else:
            for pkg in packages:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], 
                             capture_output=True)
    
    def get_client(self) -> UnifiedCloudClient:
        """Get the cloud API client."""
        if not self._setup_complete:
            self.setup()
        return self.cloud_client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using cloud API."""
        return self.get_client().generate(prompt, **kwargs)
    
    def label_text(self, text: str, task: str = "qa_generation") -> str:
        """
        Use cloud API for text labeling (replacement for Ollama labeling).
        
        Args:
            text: Text to process
            task: Type of labeling task
        """
        prompts = {
            "qa_generation": f"""Generate high-quality question-answer pairs from this text.
Format as JSON array: [{{"question": "...", "answer": "..."}}]

Text:
{text}

JSON:""",

            "summarize": f"""Summarize this text concisely:

{text}

Summary:""",

            "extract_entities": f"""Extract key entities (people, places, concepts) from this text.
Format as JSON: {{"entities": [...]}}

{text}

JSON:""",

            "classify": f"""Classify the topic/domain of this text.
Categories: science, technology, medicine, law, history, literature, other

{text}

Classification:""",
        }
        
        prompt = prompts.get(task, prompts["qa_generation"])
        return self.generate(prompt)
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings for cloud training."""
        env_info = get_environment_info()
        
        # Base settings
        settings = {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "max_seq_length": 2048,
            "num_epochs": 3,
            "use_4bit": True,
            "use_gradient_checkpointing": True,
        }
        
        # Adjust based on GPU
        if env_info.get("cuda_available"):
            gpu_memory = env_info.get("gpu_memory_gb", 0)
            
            if gpu_memory >= 40:  # A100
                settings.update({
                    "batch_size": 16,
                    "max_seq_length": 4096,
                    "use_4bit": False,
                })
            elif gpu_memory >= 20:  # A10G, L4
                settings.update({
                    "batch_size": 8,
                    "max_seq_length": 2048,
                })
            elif gpu_memory >= 15:  # T4, V100
                settings.update({
                    "batch_size": 4,
                    "max_seq_length": 2048,
                })
            else:  # Limited GPU
                settings.update({
                    "batch_size": 2,
                    "max_seq_length": 1024,
                    "gradient_accumulation_steps": 8,
                })
        
        return settings
    
    def display_info(self):
        """Display cloud environment information."""
        info = get_environment_info()
        
        if RICH_AVAILABLE:
            table = Table(title="☁️ Cloud Environment Info", show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Environment", info["environment"])
            table.add_row("Is Cloud", "✅ Yes" if info["is_cloud"] else "❌ No")
            table.add_row("CUDA Available", "✅" if info.get("cuda_available") else "❌")
            
            if info.get("cuda_available"):
                table.add_row("GPU", info.get("gpu_name", "Unknown"))
                table.add_row("GPU Memory", f"{info.get('gpu_memory_gb', 0):.1f} GB")
                table.add_row("CUDA Version", info.get("cuda_version", "N/A"))
            
            if self.cloud_client and self.cloud_client.is_available():
                table.add_row("API Provider", self.cloud_client.provider)
            
            console.print(table)
        else:
            for k, v in info.items():
                print(f"{k}: {v}")


# ============================================================================
# Colab-Specific Utilities
# ============================================================================

def setup_colab(api_key: str = None, install: bool = True) -> CloudRuntime:
    """
    Quick setup for Google Colab.
    
    Usage in Colab:
        from saara.cloud_runtime import setup_colab
        runtime = setup_colab(api_key="your-gemini-key")
    """
    runtime = CloudRuntime()
    runtime.setup(api_key=api_key, provider="gemini", install_deps=install)
    return runtime


def colab_quickstart():
    """Print quickstart guide for Colab."""
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║              🌟 SAARA Cloud Quickstart Guide 🌟                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1️⃣  Install SAARA:                                              ║
║      !pip install saara-ai                                       ║
║                                                                  ║
║  2️⃣  Setup Cloud Runtime:                                        ║
║      from saara.cloud_runtime import setup_colab                 ║
║      runtime = setup_colab()                                     ║
║      # Set API key in Colab secrets or pass directly             ║
║                                                                  ║
║  3️⃣  For Training:                                               ║
║      from saara.train import LLMTrainer                          ║
║      from saara.accelerator import create_accelerator            ║
║                                                                  ║
║      accelerator = create_accelerator()                          ║
║      trainer = LLMTrainer(model_id="google/gemma-2-2b")          ║
║      trainer.train("your_data.jsonl")                            ║
║                                                                  ║
║  4️⃣  For Data Labeling (no Ollama needed):                       ║
║      response = runtime.label_text(your_text, "qa_generation")   ║
║                                                                  ║
║  💡 Recommended API (free tier): Google AI (Gemini)              ║
║     Get key: https://aistudio.google.com/apikey                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Environment
    'CloudEnvironment',
    'detect_environment',
    'is_cloud_environment',
    'get_environment_info',
    
    # Clients
    'CloudAPIClient',
    'GeminiClient',
    'OpenAIClient', 
    'DeepSeekClient',
    'GroqClient',
    'HuggingFaceClient',
    'UnifiedCloudClient',
    
    # Config & Runtime
    'CloudConfig',
    'CloudRuntime',
    
    # Utilities
    'setup_colab',
    'colab_quickstart',
]
