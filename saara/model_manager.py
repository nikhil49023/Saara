"""
Model Manager Module
Handles Ollama model installation, management, and hardware-based recommendations.
"""

import logging
import subprocess
import json
import psutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

import requests

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    display_name: str
    category: str  # vision, analyzer, embedding
    size_gb: float
    vram_required: float  # GB
    description: str
    tags: List[str] = field(default_factory=list)
    is_installed: bool = False


# Model catalog organized by category and hardware requirements
MODEL_CATALOG = {
    "vision": [
        # Lightweight (< 4GB VRAM)
        ModelInfo("moondream", "Moondream 2", "vision", 1.5, 2.0, 
                  "Fast, lightweight vision model. Best for quick OCR.", ["lightweight", "fast"]),
        ModelInfo("llava:7b", "LLaVA 7B", "vision", 4.0, 4.5,
                  "Good balance of speed and accuracy.", ["balanced"]),
        # Medium (4-8GB VRAM)
        ModelInfo("qwen2.5vl:3b", "Qwen2.5-VL 3B", "vision", 2.0, 3.5,
                  "Alibaba's vision model. Good for tables.", ["tables", "chinese"]),
        ModelInfo("qwen2.5vl:7b", "Qwen2.5-VL 7B", "vision", 4.5, 6.0,
                  "Higher accuracy vision model.", ["accurate"]),
        # Heavy (> 8GB VRAM)
        ModelInfo("llava:13b", "LLaVA 13B", "vision", 8.0, 10.0,
                  "High accuracy, slower. For detailed documents.", ["accurate", "slow"]),
        ModelInfo("qwen2.5vl:32b", "Qwen2.5-VL 32B", "vision", 20.0, 24.0,
                  "Best accuracy. Requires high-end GPU.", ["best", "heavy"]),
    ],
    "analyzer": [
        # Lightweight (< 4GB VRAM)
        ModelInfo("phi3:mini", "Phi-3 Mini", "analyzer", 2.0, 2.5,
                  "Microsoft's compact model. Fast inference.", ["lightweight", "fast"]),
        ModelInfo("gemma2:2b", "Gemma 2 2B", "analyzer", 1.5, 2.0,
                  "Google's small model. Good for simple tasks.", ["lightweight"]),
        ModelInfo("qwen2.5:3b", "Qwen 2.5 3B", "analyzer", 2.0, 3.0,
                  "Alibaba's efficient small model.", ["lightweight", "multilingual"]),
        # Medium (4-8GB VRAM)
        ModelInfo("llama3.2:3b", "Llama 3.2 3B", "analyzer", 2.0, 3.5,
                  "Meta's latest efficient model.", ["balanced"]),
        ModelInfo("granite3.1-dense:8b", "Granite 3.1 8B", "analyzer", 4.5, 6.0,
                  "IBM's enterprise model. Great for data tasks.", ["enterprise", "structured"]),
        ModelInfo("mistral:7b", "Mistral 7B", "analyzer", 4.0, 5.5,
                  "Fast and capable. Good general model.", ["balanced", "fast"]),
        ModelInfo("qwen2.5:7b", "Qwen 2.5 7B", "analyzer", 4.5, 6.0,
                  "Excellent reasoning and multilingual.", ["reasoning", "multilingual"]),
        # Heavy (> 8GB VRAM)
        ModelInfo("llama3.2:70b", "Llama 3.2 70B", "analyzer", 40.0, 48.0,
                  "Most powerful open model. Requires multiple GPUs.", ["best", "heavy"]),
        ModelInfo("qwen2.5:32b", "Qwen 2.5 32B", "analyzer", 20.0, 24.0,
                  "Excellent for complex reasoning.", ["reasoning", "heavy"]),
        ModelInfo("deepseek-coder-v2:16b", "DeepSeek Coder V2 16B", "analyzer", 10.0, 12.0,
                  "Best for code and structured data.", ["code", "technical"]),
    ],
}


class HardwareDetector:
    """Detects system hardware capabilities."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get detailed system information."""
        info = {
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "gpu_available": False,
            "gpu_name": None,
            "vram_gb": 0,
        }
        
        # Try to detect NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(',')
                    info["gpu_available"] = True
                    info["gpu_name"] = parts[0].strip()
                    info["vram_gb"] = round(float(parts[1].strip()) / 1024, 1)
        except:
            pass
        
        return info
    
    @staticmethod
    def get_recommended_tier(hardware_info: Dict[str, Any]) -> str:
        """Get recommended model tier based on hardware."""
        vram = hardware_info.get("vram_gb", 0)
        ram = hardware_info.get("ram_gb", 0)
        
        if vram >= 16:
            return "heavy"
        elif vram >= 8:
            return "medium"
        elif vram >= 4 or ram >= 32:
            return "light"
        else:
            return "minimal"
    
    @staticmethod
    def display_hardware_info(info: Dict[str, Any]):
        """Display hardware information in a nice table."""
        table = Table(title="💻 System Hardware", show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("RAM", f"{info['ram_gb']} GB (Available: {info['ram_available_gb']} GB)")
        table.add_row("CPU", f"{info['cpu_cores']} cores / {info['cpu_threads']} threads")
        
        if info["gpu_available"]:
            table.add_row("GPU", f"✓ {info['gpu_name']}")
            table.add_row("VRAM", f"{info['vram_gb']} GB")
        else:
            table.add_row("GPU", "❌ Not detected (CPU-only mode)")
        
        console.print(table)


class ModelManager:
    """Manages Ollama model installation and configuration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ollama_url = self.config.get("ollama", {}).get("base_url", "http://localhost:11434")
        
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.ok
        except:
            return False
    
    def start_ollama(self) -> bool:
        """Attempt to start Ollama."""
        try:
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            import time
            time.sleep(3)
            return self.check_ollama_running()
        except:
            return False
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed Ollama models."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.ok:
                data = response.json()
                return [m["name"].split(":")[0] for m in data.get("models", [])]
        except:
            pass
        return []
    
    def install_model(self, model_name: str, progress_callback=None) -> bool:
        """Install an Ollama model."""
        console.print(f"[cyan]Pulling model: {model_name}...[/cyan]")
        
        try:
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            for line in process.stdout:
                if progress_callback:
                    progress_callback(line.strip())
                else:
                    # Parse progress from Ollama output
                    if "pulling" in line.lower() or "%" in line:
                        console.print(f"  [dim]{line.strip()}[/dim]", end="\r")
            
            process.wait()
            console.print()  # New line after progress
            return process.returncode == 0
            
        except Exception as e:
            console.print(f"[red]Error installing model: {e}[/red]")
            return False
    
    def uninstall_model(self, model_name: str) -> bool:
        """Uninstall an Ollama model."""
        try:
            result = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def get_model_catalog(self, category: str = None, tier: str = None) -> List[ModelInfo]:
        """Get models from catalog, optionally filtered."""
        installed = self.get_installed_models()
        
        models = []
        categories = [category] if category else MODEL_CATALOG.keys()
        
        for cat in categories:
            for model in MODEL_CATALOG.get(cat, []):
                # Check if installed
                model.is_installed = any(model.name.split(":")[0] in m for m in installed)
                
                # Filter by tier
                if tier:
                    if tier == "minimal" and model.vram_required > 3:
                        continue
                    elif tier == "light" and model.vram_required > 6:
                        continue
                    elif tier == "medium" and model.vram_required > 12:
                        continue
                
                models.append(model)
        
        return models
    
    def display_models(self, category: str = None, tier: str = None):
        """Display available models in a formatted table."""
        models = self.get_model_catalog(category, tier)
        
        if not models:
            console.print("[yellow]No models found for the specified criteria.[/yellow]")
            return
        
        # Group by category
        for cat in ["vision", "analyzer"]:
            cat_models = [m for m in models if m.category == cat]
            if not cat_models:
                continue
            
            title = "👁️ Vision Models" if cat == "vision" else "🧠 Analyzer Models"
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Model", style="green", width=20)
            table.add_column("Size", width=8)
            table.add_column("VRAM", width=8)
            table.add_column("Description", width=40)
            table.add_column("Status", width=12)
            
            for i, model in enumerate(cat_models, 1):
                status = "[green]✓ Installed[/green]" if model.is_installed else "[dim]Not installed[/dim]"
                table.add_row(
                    str(i),
                    model.display_name,
                    f"{model.size_gb} GB",
                    f"{model.vram_required} GB",
                    model.description[:40] + "..." if len(model.description) > 40 else model.description,
                    status
                )
            
            console.print(table)
            console.print()


class TrainedModelManager:
    """Manages fine-tuned models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained/fine-tuned models."""
        models = []
        
        if not self.models_dir.exists():
            return models
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                adapter_path = model_dir / "final_adapter"
                if adapter_path.exists():
                    # Try to read config
                    config_path = adapter_path / "adapter_config.json"
                    base_model = "Unknown"
                    if config_path.exists():
                        try:
                            with open(config_path) as f:
                                config = json.load(f)
                                base_model = config.get("base_model_name_or_path", "Unknown")
                        except:
                            pass
                    
                    models.append({
                        "name": model_dir.name,
                        "path": str(adapter_path),
                        "base_model": base_model,
                        "size_mb": sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()) / (1024**2)
                    })
        
        return models
    
    def display_trained_models(self):
        """Display trained models in a table."""
        models = self.list_trained_models()
        
        if not models:
            console.print("[yellow]No fine-tuned models found.[/yellow]")
            console.print("[dim]Train a model with: saara train[/dim]")
            return
        
        table = Table(title="🎯 Fine-Tuned Models", show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="green")
        table.add_column("Base Model", style="yellow")
        table.add_column("Size", width=10)
        table.add_column("Path", style="dim")
        
        for i, model in enumerate(models, 1):
            table.add_row(
                str(i),
                model["name"],
                model["base_model"].split("/")[-1],
                f"{model['size_mb']:.1f} MB",
                model["path"]
            )
        
        console.print(table)
    
    def delete_trained_model(self, model_name: str) -> bool:
        """Delete a fine-tuned model."""
        import shutil
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            shutil.rmtree(model_path)
            return True
        return False
