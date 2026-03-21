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
    
    # Training requirements for different architectures (pretraining)
    # Format: (min_vram_gb, min_ram_gb, estimated_training_time_hours_per_epoch)
    PRETRAIN_REQUIREMENTS = {
        "nano": (2.0, 8, 0.5),      # ~15M params
        "micro": (4.0, 16, 2),      # ~50M params
        "mini": (6.0, 16, 5),       # ~125M params
        "small": (8.0, 32, 12),     # ~350M params
        "base": (16.0, 64, 48),     # ~1B params
        "large": (24.0, 128, 100),  # ~3B params
    }
    
    # Fine-tuning requirements (usually lower due to LoRA/QLoRA)
    FINETUNE_REQUIREMENTS = {
        "small_model": {"max_params": "3B", "min_vram": 4.0, "min_ram": 8},
        "medium_model": {"max_params": "7B", "min_vram": 8.0, "min_ram": 16},
        "large_model": {"max_params": "13B", "min_vram": 16.0, "min_ram": 32},
        "xlarge_model": {"max_params": "70B", "min_vram": 48.0, "min_ram": 64},
    }
    
    # Cloud providers for bigger models
    CLOUD_OPTIONS = [
        {
            "name": "Google Colab Pro",
            "gpu": "A100 (40GB)",
            "cost": "~$10/month",
            "good_for": "Up to 7B model fine-tuning, small pretraining",
            "url": "https://colab.research.google.com/",
        },
        {
            "name": "Google Cloud (GCP)",
            "gpu": "A100/H100 (up to 80GB)",
            "cost": "~$2-4/hour",
            "good_for": "Large-scale pretraining & fine-tuning",
            "url": "https://cloud.google.com/compute/gpus",
        },
        {
            "name": "AWS SageMaker",
            "gpu": "A100/V100 clusters",
            "cost": "Variable",
            "good_for": "Enterprise training pipelines",
            "url": "https://aws.amazon.com/sagemaker/",
        },
        {
            "name": "Lambda Labs",
            "gpu": "A100/H100 (80GB)",
            "cost": "~$1.5-2.5/hour",
            "good_for": "Cost-effective GPU rental",
            "url": "https://lambdalabs.com/",
        },
        {
            "name": "RunPod",
            "gpu": "A100/3090 (24GB+)",
            "cost": "~$0.5-2/hour",
            "good_for": "Flexible GPU instances",
            "url": "https://runpod.io/",
        },
        {
            "name": "Kaggle Notebooks",
            "gpu": "T4/P100 (16GB)",
            "cost": "Free (30h/week)",
            "good_for": "Small experiments, fine-tuning",
            "url": "https://www.kaggle.com/",
        },
    ]
    
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
    def get_pretrain_recommendations(hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get pre-training architecture recommendations based on hardware.
        
        Returns dict with:
            - recommended: list of recommended architectures
            - possible_with_optimization: architectures that might work with gradient checkpointing etc.
            - requires_cloud: architectures that need more powerful hardware
            - warnings: list of warnings/notes
        """
        vram = hardware_info.get("vram_gb", 0)
        ram = hardware_info.get("ram_gb", 0)
        has_gpu = hardware_info.get("gpu_available", False)
        
        result = {
            "recommended": [],
            "possible_with_optimization": [],
            "requires_cloud": [],
            "warnings": [],
            "hardware_tier": "cpu_only",
        }
        
        if not has_gpu:
            result["warnings"].append("⚠️ No GPU detected. Training will be VERY slow (CPU-only mode).")
            result["warnings"].append("💡 Only nano architecture is practical for CPU training.")
            result["recommended"].append("nano")
            result["possible_with_optimization"].append("micro")
            result["requires_cloud"] = ["mini", "small", "base", "large"]
            result["hardware_tier"] = "cpu_only"
            return result
        
        # Determine hardware tier
        if vram >= 24:
            result["hardware_tier"] = "high_end"
        elif vram >= 16:
            result["hardware_tier"] = "enthusiast"
        elif vram >= 8:
            result["hardware_tier"] = "mid_range"
        elif vram >= 4:
            result["hardware_tier"] = "entry_level"
        else:
            result["hardware_tier"] = "minimal"
        
        # Check each architecture
        for arch, (min_vram, min_ram, _) in HardwareDetector.PRETRAIN_REQUIREMENTS.items():
            # With gradient checkpointing, we can reduce VRAM requirement by ~40%
            optimized_vram = min_vram * 0.6
            
            if vram >= min_vram and ram >= min_ram:
                result["recommended"].append(arch)
            elif vram >= optimized_vram and ram >= min_ram * 0.8:
                result["possible_with_optimization"].append(arch)
            else:
                result["requires_cloud"].append(arch)
        
        # Add specific recommendations based on hardware
        if vram < 4:
            result["warnings"].append(f"⚠️ Your VRAM ({vram}GB) is limited. Consider using nano/micro architectures.")
        
        if ram < 16:
            result["warnings"].append(f"⚠️ Low RAM ({ram}GB) may cause issues with data loading. Consider 16GB+ RAM.")
            
        if vram >= 8 and vram < 16:
            result["warnings"].append("💡 For base (1B) models, consider using gradient checkpointing and lower batch sizes.")
            
        if not result["recommended"]:
            result["warnings"].append("❌ Your hardware cannot comfortably train any architecture. Cloud is recommended.")
            if result["possible_with_optimization"]:
                result["warnings"].append(f"💡 With optimizations, you might train: {', '.join(result['possible_with_optimization'])}")
        
        return result
    
    @staticmethod
    def get_finetune_recommendations(hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get fine-tuning recommendations based on hardware.
        Fine-tuning with LoRA/QLoRA requires less resources than pretraining.
        """
        vram = hardware_info.get("vram_gb", 0)
        ram = hardware_info.get("ram_gb", 0)
        has_gpu = hardware_info.get("gpu_available", False)
        
        result = {
            "max_model_size": "0B",
            "recommended_models": [],
            "requires_cloud": [],
            "optimizations": [],
            "warnings": [],
        }
        
        if not has_gpu:
            result["warnings"].append("⚠️ No GPU detected. Fine-tuning will be extremely slow.")
            result["warnings"].append("💡 Consider using cloud GPUs for fine-tuning.")
            result["max_model_size"] = "1B (very slow)"
            result["recommended_models"] = ["TinyLlama/TinyLlama-1.1B"]
            result["requires_cloud"] = ["3B+", "7B+", "13B+", "70B+"]
            return result
        
        # QLoRA with 4-bit quantization reduces memory significantly
        # Approximate VRAM requirements for QLoRA fine-tuning:
        # 2B model: ~4GB VRAM
        # 7B model: ~8GB VRAM  
        # 13B model: ~16GB VRAM
        # 70B model: ~48GB VRAM
        
        if vram >= 48:
            result["max_model_size"] = "70B"
            result["recommended_models"] = [
                "meta-llama/Llama-3.3-70B-Instruct",
                "Qwen/Qwen2.5-72B",
                "meta-llama/Llama-3.1-70B",
            ]
        elif vram >= 24:
            result["max_model_size"] = "34B"
            result["recommended_models"] = [
                "codellama/CodeLlama-34b",
                "Qwen/Qwen2.5-32B",
            ]
            result["requires_cloud"] = ["70B+"]
        elif vram >= 16:
            result["max_model_size"] = "13B"
            result["recommended_models"] = [
                "meta-llama/Llama-3.2-11B-Instruct",
                "Qwen/Qwen2.5-14B",
                "google/gemma-2-9b",
            ]
            result["requires_cloud"] = ["34B+", "70B+"]
        elif vram >= 8:
            result["max_model_size"] = "7B"
            result["recommended_models"] = [
                "google/gemma-2-2b",
                "meta-llama/Llama-3.2-3B",
                "Qwen/Qwen2.5-7B",
                "mistralai/Mistral-7B-v0.3",
            ]
            result["requires_cloud"] = ["13B+", "34B+", "70B+"]
            result["optimizations"].append("Use QLoRA (4-bit) for 7B models")
        elif vram >= 4:
            result["max_model_size"] = "3B"
            result["recommended_models"] = [
                "google/gemma-2-2b",
                "TinyLlama/TinyLlama-1.1B",
                "microsoft/phi-2",
            ]
            result["requires_cloud"] = ["7B+", "13B+", "34B+", "70B+"]
            result["optimizations"].append("Use QLoRA (4-bit) with gradient checkpointing")
            result["optimizations"].append("Reduce batch size to 1-2")
        else:
            result["max_model_size"] = "1B"
            result["recommended_models"] = [
                "TinyLlama/TinyLlama-1.1B",
            ]
            result["requires_cloud"] = ["2B+", "7B+", "13B+", "70B+"]
            result["warnings"].append("⚠️ Very limited VRAM. Consider cloud options.")
        
        # Add optimizations for mid-range hardware
        if 4 <= vram < 16:
            result["optimizations"].extend([
                "Enable gradient checkpointing",
                "Use bf16/fp16 mixed precision",
                "Reduce max sequence length if possible",
            ])
            
        return result
    
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
    
    @staticmethod
    def display_pretrain_recommendations(hardware_info: Dict[str, Any], show_cloud: bool = True):
        """Display pretraining recommendations with visual formatting."""
        recommendations = HardwareDetector.get_pretrain_recommendations(hardware_info)
        
        console.print(Panel(
            f"[bold cyan]🔍 Hardware Analysis for Pre-training[/bold cyan]\n\n"
            f"[green]Hardware Tier:[/green] {recommendations['hardware_tier'].replace('_', ' ').title()}\n"
            f"[green]GPU:[/green] {hardware_info.get('gpu_name', 'None')} ({hardware_info.get('vram_gb', 0)}GB VRAM)\n"
            f"[green]RAM:[/green] {hardware_info.get('ram_gb', 0)}GB",
            title="📊 System Analysis",
            border_style="cyan"
        ))
        
        # Recommended architectures
        if recommendations["recommended"]:
            rec_table = Table(title="✅ Recommended Architectures", show_header=True, header_style="bold green")
            rec_table.add_column("Architecture", style="green")
            rec_table.add_column("Parameters", style="cyan")
            rec_table.add_column("Min VRAM", style="yellow")
            rec_table.add_column("Status", style="green")
            
            for arch in recommendations["recommended"]:
                req = HardwareDetector.PRETRAIN_REQUIREMENTS.get(arch, (0, 0, 0))
                rec_table.add_row(
                    arch.capitalize(),
                    f"~{['15M', '50M', '125M', '350M', '1B', '3B'][list(HardwareDetector.PRETRAIN_REQUIREMENTS.keys()).index(arch)]}",
                    f"{req[0]}GB",
                    "✓ Ready to train"
                )
            console.print(rec_table)
        
        # Possible with optimization
        if recommendations["possible_with_optimization"]:
            opt_table = Table(title="⚠️ Possible with Optimization", show_header=True, header_style="bold yellow")
            opt_table.add_column("Architecture", style="yellow")
            opt_table.add_column("Parameters", style="cyan")
            opt_table.add_column("Optimizations Needed")
            
            for arch in recommendations["possible_with_optimization"]:
                opt_table.add_row(
                    arch.capitalize(),
                    f"~{['15M', '50M', '125M', '350M', '1B', '3B'][list(HardwareDetector.PRETRAIN_REQUIREMENTS.keys()).index(arch)]}",
                    "Gradient checkpointing, lower batch size, fp16"
                )
            console.print(opt_table)
        
        # Requires cloud
        if recommendations["requires_cloud"] and show_cloud:
            cloud_table = Table(title="☁️ Requires Cloud/Better Hardware", show_header=True, header_style="bold red")
            cloud_table.add_column("Architecture", style="red")
            cloud_table.add_column("Min VRAM Needed", style="yellow")
            cloud_table.add_column("Suggestion")
            
            for arch in recommendations["requires_cloud"]:
                req = HardwareDetector.PRETRAIN_REQUIREMENTS.get(arch, (0, 0, 0))
                cloud_table.add_row(
                    arch.capitalize(),
                    f"{req[0]}GB+",
                    "Use cloud GPU (see options below)"
                )
            console.print(cloud_table)
        
        # Warnings
        if recommendations["warnings"]:
            console.print()
            for warning in recommendations["warnings"]:
                console.print(f"  {warning}")
        
        return recommendations
    
    @staticmethod
    def display_finetune_recommendations(hardware_info: Dict[str, Any], show_cloud: bool = True):
        """Display finetuning recommendations with visual formatting."""
        recommendations = HardwareDetector.get_finetune_recommendations(hardware_info)
        
        console.print(Panel(
            f"[bold cyan]🔍 Hardware Analysis for Fine-tuning[/bold cyan]\n\n"
            f"[green]Maximum Model Size:[/green] {recommendations['max_model_size']}\n"
            f"[green]GPU:[/green] {hardware_info.get('gpu_name', 'None')} ({hardware_info.get('vram_gb', 0)}GB VRAM)\n"
            f"[green]RAM:[/green] {hardware_info.get('ram_gb', 0)}GB",
            title="📊 System Analysis",
            border_style="cyan"
        ))
        
        # Recommended models
        if recommendations["recommended_models"]:
            rec_table = Table(title="✅ Recommended Base Models", show_header=True, header_style="bold green")
            rec_table.add_column("Model", style="green")
            rec_table.add_column("Status", style="cyan")
            
            for model in recommendations["recommended_models"]:
                rec_table.add_row(model, "✓ Can fine-tune")
            console.print(rec_table)
        
        # Optimizations
        if recommendations["optimizations"]:
            console.print("\n[bold yellow]💡 Recommended Optimizations:[/bold yellow]")
            for opt in recommendations["optimizations"]:
                console.print(f"  • {opt}")
        
        # Requires cloud
        if recommendations["requires_cloud"] and show_cloud:
            console.print(f"\n[bold red]☁️ Models requiring cloud:[/bold red] {', '.join(recommendations['requires_cloud'])}")
        
        # Warnings
        if recommendations["warnings"]:
            console.print()
            for warning in recommendations["warnings"]:
                console.print(f"  {warning}")
        
        return recommendations
    
    @staticmethod
    def display_cloud_options():
        """Display available cloud GPU options."""
        console.print(Panel(
            "[bold cyan]☁️ Cloud GPU Options for Larger Models[/bold cyan]\n\n"
            "[dim]When your local hardware isn't sufficient, consider these cloud options:[/dim]",
            title="Cloud Training Options",
            border_style="blue"
        ))
        
        cloud_table = Table(show_header=True, header_style="bold magenta")
        cloud_table.add_column("Provider", style="cyan", width=18)
        cloud_table.add_column("GPU", style="green", width=18)
        cloud_table.add_column("Cost", style="yellow", width=12)
        cloud_table.add_column("Best For", width=35)
        
        for option in HardwareDetector.CLOUD_OPTIONS:
            cloud_table.add_row(
                option["name"],
                option["gpu"],
                option["cost"],
                option["good_for"]
            )
        
        console.print(cloud_table)
        
        console.print("\n[dim]💡 Tips:[/dim]")
        console.print("[dim]  • Start with Google Colab Pro for quick experiments[/dim]")
        console.print("[dim]  • Lambda Labs offers best price/performance for long training[/dim]")
        console.print("[dim]  • Kaggle is free for light fine-tuning experiments[/dim]")
    
    @staticmethod
    def run_hardware_check_wizard(mode: str = "both", interactive: bool = True) -> Dict[str, Any]:
        """
        Run interactive hardware check wizard.
        
        Args:
            mode: "pretrain", "finetune", or "both"
            interactive: If True, offer options to proceed to training
            
        Returns:
            Dict with hardware info and recommendations
        """
        from rich.prompt import Prompt, Confirm
        
        console.print(Panel(
            "[bold cyan]🔧 Hardware Check Wizard[/bold cyan]\n\n"
            "[dim]Analyzing your system to recommend optimal training configurations...[/dim]",
            title="Hardware Analysis",
            border_style="cyan"
        ))
        
        # Get hardware info
        console.print("\n[bold]📡 Detecting hardware...[/bold]")
        hardware_info = HardwareDetector.get_system_info()
        HardwareDetector.display_hardware_info(hardware_info)
        
        result = {
            "hardware": hardware_info,
            "pretrain": None,
            "finetune": None,
        }
        
        if mode in ["pretrain", "both"]:
            console.print("\n")
            result["pretrain"] = HardwareDetector.display_pretrain_recommendations(
                hardware_info, 
                show_cloud=True
            )
        
        if mode in ["finetune", "both"]:
            console.print("\n")
            result["finetune"] = HardwareDetector.display_finetune_recommendations(
                hardware_info,
                show_cloud=True
            )
        
        # Show cloud options if any architecture requires it
        needs_cloud = False
        if result["pretrain"] and result["pretrain"].get("requires_cloud"):
            needs_cloud = True
        if result["finetune"] and result["finetune"].get("requires_cloud"):
            needs_cloud = True
            
        if needs_cloud:
            console.print("\n")
            HardwareDetector.display_cloud_options()
        
        # Interactive mode - offer to proceed to training
        if interactive:
            console.print("\n")
            console.print("[bold]What would you like to do next?[/bold]\n")
            
            options = []
            if mode in ["pretrain", "both"]:
                if result["pretrain"].get("recommended") or result["pretrain"].get("possible_with_optimization"):
                    options.append(("1", "🏗️ Start Pre-training", "pretrain"))
            if mode in ["finetune", "both"]:
                if result["finetune"].get("recommended_models"):
                    options.append(("2", "🧠 Start Fine-tuning", "finetune"))
            options.append(("3", "☁️ View Cloud Options Again", "cloud"))
            options.append(("4", "✅ Exit", "exit"))
            
            for opt_id, opt_name, _ in options:
                console.print(f"  {opt_id}. {opt_name}")
            
            choice = Prompt.ask("Select option", choices=[o[0] for o in options], default="4")
            
            selected_action = next((o[2] for o in options if o[0] == choice), "exit")
            
            if selected_action == "pretrain":
                return HardwareDetector._start_pretrain_flow(hardware_info, result["pretrain"])
            elif selected_action == "finetune":
                return HardwareDetector._start_finetune_flow(hardware_info, result["finetune"])
            elif selected_action == "cloud":
                HardwareDetector.display_cloud_options()
        
        return result
    
    @staticmethod
    def _start_pretrain_flow(hardware_info: Dict[str, Any], pretrain_recs: Dict[str, Any]) -> Dict[str, Any]:
        """Start pretraining flow with architecture selection based on hardware."""
        from rich.prompt import Prompt, Confirm
        from rich.table import Table
        
        console.print("\n[bold cyan]🏗️ Pre-training Setup[/bold cyan]\n")
        
        # Combine recommended and possible with optimization
        available_archs = pretrain_recs.get("recommended", []) + pretrain_recs.get("possible_with_optimization", [])
        
        if not available_archs:
            console.print("[red]No architectures available for your hardware.[/red]")
            console.print("[dim]Consider using cloud GPU or upgrading your hardware.[/dim]")
            return {"action": "cancelled", "reason": "no_compatible_architectures"}
        
        # Show architecture selection
        arch_table = Table(title="Select Architecture", show_header=True, header_style="bold magenta")
        arch_table.add_column("#", style="cyan", width=3)
        arch_table.add_column("Architecture", style="green")
        arch_table.add_column("Parameters")
        arch_table.add_column("Status")
        
        param_map = {"nano": "~15M", "micro": "~50M", "mini": "~125M", "small": "~350M", "base": "~1B", "large": "~3B"}
        
        for i, arch in enumerate(available_archs, 1):
            status = "[green]✓ Recommended[/green]" if arch in pretrain_recs.get("recommended", []) else "[yellow]⚠ With optimization[/yellow]"
            arch_table.add_row(str(i), arch.capitalize(), param_map.get(arch, "?"), status)
        
        console.print(arch_table)
        
        choice = Prompt.ask("Select architecture", choices=[str(i) for i in range(1, len(available_archs)+1)], default="1")
        selected_arch = available_archs[int(choice) - 1]
        
        # Get model name
        model_name = Prompt.ask("Enter model name", default="my-custom-model")
        
        # Get data path
        data_path = Prompt.ask("Path to training data (text files or JSONL)")
        
        console.print(f"\n[green]✓ Selected: {selected_arch.capitalize()} ({param_map.get(selected_arch, '?')} params)[/green]")
        console.print(f"[green]✓ Model name: {model_name}[/green]")
        console.print(f"[green]✓ Data path: {data_path}[/green]")
        
        if Confirm.ask("\nStart pre-training now?", default=True):
            # Import and run pretrainer
            try:
                from saara.pretrain import PreTrainer
                
                console.print("\n[bold green]Starting pre-training...[/bold green]\n")
                
                pretrainer = PreTrainer(
                    architecture=selected_arch,
                    model_name=model_name,
                    output_dir="models"
                )
                
                result_path = pretrainer.pretrain(data_path)
                
                return {
                    "action": "pretrain_complete",
                    "architecture": selected_arch,
                    "model_name": model_name,
                    "output_path": result_path
                }
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                import traceback
                traceback.print_exc()
                return {"action": "pretrain_failed", "error": str(e)}
        else:
            console.print("[yellow]Pre-training cancelled.[/yellow]")
            return {"action": "cancelled"}
    
    @staticmethod
    def _start_finetune_flow(hardware_info: Dict[str, Any], finetune_recs: Dict[str, Any]) -> Dict[str, Any]:
        """Start finetuning flow with model selection based on hardware."""
        from rich.prompt import Prompt, Confirm
        from rich.table import Table
        
        console.print("\n[bold cyan]🧠 Fine-tuning Setup[/bold cyan]\n")
        console.print(f"[dim]Maximum recommended model size: {finetune_recs.get('max_model_size', 'Unknown')}[/dim]\n")
        
        recommended_models = finetune_recs.get("recommended_models", [])
        
        if not recommended_models:
            console.print("[red]No models available for your hardware.[/red]")
            console.print("[dim]Consider using cloud GPU or upgrading your hardware.[/dim]")
            return {"action": "cancelled", "reason": "no_compatible_models"}
        
        # Show model selection
        model_table = Table(title="Recommended Base Models", show_header=True, header_style="bold magenta")
        model_table.add_column("#", style="cyan", width=3)
        model_table.add_column("Model", style="green")
        model_table.add_column("Status", style="green")
        
        for i, model in enumerate(recommended_models, 1):
            model_table.add_row(str(i), model, "✓ Compatible")
        
        # Add "Other" option
        model_table.add_row(str(len(recommended_models) + 1), "[dim]Other (Enter HuggingFace ID)[/dim]", "[dim]Manual[/dim]")
        
        console.print(model_table)
        
        choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(recommended_models)+2)], default="1")
        choice_idx = int(choice) - 1
        
        if choice_idx < len(recommended_models):
            selected_model = recommended_models[choice_idx]
        else:
            selected_model = Prompt.ask("Enter HuggingFace Model ID")
        
        # Get dataset path
        data_path = Prompt.ask("Path to training dataset (.jsonl)", default="datasets/distilled_train.jsonl")
        
        console.print(f"\n[green]✓ Selected model: {selected_model}[/green]")
        console.print(f"[green]✓ Dataset: {data_path}[/green]")
        
        # Show optimization tips
        if finetune_recs.get("optimizations"):
            console.print("\n[bold yellow]💡 Optimization tips for your hardware:[/bold yellow]")
            for opt in finetune_recs["optimizations"][:3]:
                console.print(f"  • {opt}")
        
        if Confirm.ask("\nStart fine-tuning now?", default=True):
            # Import and run trainer
            try:
                from saara.train import LLMTrainer
                
                console.print("\n[bold green]Starting fine-tuning...[/bold green]\n")
                
                trainer = LLMTrainer(model_id=selected_model)
                trainer.train(data_path)
                
                return {
                    "action": "finetune_complete",
                    "model": selected_model,
                    "dataset": data_path
                }
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                import traceback
                traceback.print_exc()
                return {"action": "finetune_failed", "error": str(e)}
        else:
            console.print("[yellow]Fine-tuning cancelled.[/yellow]")
            return {"action": "cancelled"}


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
    """Manages fine-tuned and pre-trained models."""
    
    def __init__(self, models_dir: str = "models", datasets_dir: str = "datasets", 
                 tokenizers_dir: str = "tokenizers"):
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.tokenizers_dir = Path(tokenizers_dir)
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained/fine-tuned models."""
        models = []
        
        if not self.models_dir.exists():
            return models
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # Check for fine-tuned adapter
                adapter_path = model_dir / "final_adapter"
                is_adapter = adapter_path.exists()
                
                # Check for pre-trained model
                pretrain_config = model_dir / "config.json"
                is_pretrained = pretrain_config.exists() and not is_adapter
                
                if is_adapter or is_pretrained:
                    # Try to read config
                    config_path = adapter_path / "adapter_config.json" if is_adapter else pretrain_config
                    base_model = "Custom" if is_pretrained else "Unknown"
                    
                    if config_path.exists():
                        try:
                            with open(config_path) as f:
                                config = json.load(f)
                                if is_adapter:
                                    base_model = config.get("base_model_name_or_path", "Unknown")
                                else:
                                    base_model = f"Custom ({config.get('hidden_size', '?')}d)"
                        except:
                            pass
                    
                    # Calculate size
                    target_path = adapter_path if is_adapter else model_dir
                    try:
                        size_bytes = sum(f.stat().st_size for f in target_path.rglob("*") if f.is_file())
                        size_mb = size_bytes / (1024**2)
                    except:
                        size_mb = 0
                    
                    # Check for checkpoints
                    checkpoints = list(model_dir.glob("checkpoint-*"))
                    
                    models.append({
                        "name": model_dir.name,
                        "path": str(target_path),
                        "base_model": base_model,
                        "size_mb": size_mb,
                        "type": "adapter" if is_adapter else "pretrained",
                        "checkpoints": len(checkpoints),
                        "created": model_dir.stat().st_mtime if model_dir.exists() else 0
                    })
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x.get("created", 0), reverse=True)
        return models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            return None
        
        info = {
            "name": model_name,
            "path": str(model_path),
            "exists": True,
        }
        
        # Check for adapter
        adapter_path = model_path / "final_adapter"
        if adapter_path.exists():
            info["type"] = "fine-tuned adapter"
            info["adapter_path"] = str(adapter_path)
            
            config_path = adapter_path / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    info["base_model"] = config.get("base_model_name_or_path", "Unknown")
                    info["lora_rank"] = config.get("r", "Unknown")
                    info["lora_alpha"] = config.get("lora_alpha", "Unknown")
        else:
            # Check for pretrained model
            config_path = model_path / "config.json"
            if config_path.exists():
                info["type"] = "pre-trained model"
                with open(config_path) as f:
                    config = json.load(f)
                    info["hidden_size"] = config.get("hidden_size", "Unknown")
                    info["num_layers"] = config.get("num_hidden_layers", "Unknown")
                    info["vocab_size"] = config.get("vocab_size", "Unknown")
        
        # Calculate size
        try:
            size_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            info["size_mb"] = round(size_bytes / (1024**2), 2)
            info["size_gb"] = round(size_bytes / (1024**3), 2)
        except:
            info["size_mb"] = 0
        
        # List checkpoints
        checkpoints = sorted(model_path.glob("checkpoint-*"))
        info["checkpoints"] = [cp.name for cp in checkpoints]
        
        # List files
        info["files"] = [f.name for f in model_path.iterdir() if f.is_file()][:10]
        
        return info
    
    def display_trained_models(self):
        """Display trained models in a table."""
        models = self.list_trained_models()
        
        if not models:
            console.print("[yellow]No fine-tuned models found.[/yellow]")
            console.print("[dim]Train a model with: saara train[/dim]")
            return
        
        table = Table(title="🎯 Trained Models", show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Base Model", style="yellow")
        table.add_column("Size", width=10)
        table.add_column("Checkpoints", width=5)
        
        for i, model in enumerate(models, 1):
            model_type = "🔧 Adapter" if model["type"] == "adapter" else "🏗️ Pretrained"
            table.add_row(
                str(i),
                model["name"],
                model_type,
                model["base_model"].split("/")[-1][:20],
                f"{model['size_mb']:.1f} MB",
                str(model["checkpoints"])
            )
        
        console.print(table)
    
    def display_model_info(self, model_name: str):
        """Display detailed model information."""
        info = self.get_model_info(model_name)
        
        if not info:
            console.print(f"[red]Model not found: {model_name}[/red]")
            return
        
        console.print(Panel(
            f"[bold cyan]{info['name']}[/bold cyan]\n\n"
            f"[green]Type:[/green] {info.get('type', 'Unknown')}\n"
            f"[green]Path:[/green] {info['path']}\n"
            f"[green]Size:[/green] {info.get('size_mb', 0):.1f} MB ({info.get('size_gb', 0):.2f} GB)\n\n"
            + (f"[yellow]Base Model:[/yellow] {info.get('base_model', 'N/A')}\n" if 'base_model' in info else "")
            + (f"[yellow]LoRA Rank:[/yellow] {info.get('lora_rank', 'N/A')}\n" if 'lora_rank' in info else "")
            + (f"[yellow]Hidden Size:[/yellow] {info.get('hidden_size', 'N/A')}\n" if 'hidden_size' in info else "")
            + (f"[yellow]Layers:[/yellow] {info.get('num_layers', 'N/A')}\n" if 'num_layers' in info else "")
            + f"\n[dim]Checkpoints:[/dim] {len(info.get('checkpoints', []))}"
            + (f"\n  {', '.join(info.get('checkpoints', [])[:5])}" if info.get('checkpoints') else ""),
            title="📋 Model Details",
            border_style="cyan"
        ))
    
    def delete_trained_model(self, model_name: str, include_checkpoints: bool = True) -> bool:
        """Delete a fine-tuned model."""
        import shutil
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            shutil.rmtree(model_path)
            console.print(f"[green]✓ Deleted model: {model_name}[/green]")
            return True
        
        console.print(f"[red]Model not found: {model_name}[/red]")
        return False
    
    def delete_checkpoint(self, model_name: str, checkpoint_name: str) -> bool:
        """Delete a specific checkpoint from a model."""
        import shutil
        checkpoint_path = self.models_dir / model_name / checkpoint_name
        
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            console.print(f"[green]✓ Deleted checkpoint: {checkpoint_name}[/green]")
            return True
        return False
    
    def clear_all_models(self, confirm: bool = False) -> int:
        """Delete all trained models. Returns count of deleted models."""
        import shutil
        
        if not self.models_dir.exists():
            return 0
        
        models = self.list_trained_models()
        
        if not models:
            console.print("[yellow]No models to delete.[/yellow]")
            return 0
        
        if not confirm:
            console.print(f"[red]This will delete {len(models)} models![/red]")
            return 0
        
        count = 0
        for model in models:
            model_path = self.models_dir / model["name"]
            if model_path.exists():
                shutil.rmtree(model_path)
                count += 1
                console.print(f"  [dim]Deleted: {model['name']}[/dim]")
        
        console.print(f"[green]✓ Deleted {count} models[/green]")
        return count
    
    def clear_all_checkpoints(self, model_name: str = None) -> int:
        """Clear all checkpoints, keeping only final model."""
        import shutil
        
        if not self.models_dir.exists():
            return 0
        
        count = 0
        
        if model_name:
            # Clear checkpoints for specific model
            model_path = self.models_dir / model_name
            if model_path.exists():
                for checkpoint in model_path.glob("checkpoint-*"):
                    shutil.rmtree(checkpoint)
                    count += 1
        else:
            # Clear all checkpoints
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    for checkpoint in model_dir.glob("checkpoint-*"):
                        shutil.rmtree(checkpoint)
                        count += 1
        
        if count > 0:
            console.print(f"[green]✓ Cleared {count} checkpoints[/green]")
        return count
    
    def clear_datasets(self, confirm: bool = False) -> int:
        """Clear all generated datasets."""
        import shutil
        
        if not self.datasets_dir.exists():
            return 0
        
        files = list(self.datasets_dir.glob("*.jsonl")) + list(self.datasets_dir.glob("*.json"))
        
        if not files:
            console.print("[yellow]No datasets to delete.[/yellow]")
            return 0
        
        if not confirm:
            console.print(f"[red]This will delete {len(files)} dataset files![/red]")
            return 0
        
        count = 0
        for f in files:
            f.unlink()
            count += 1
        
        console.print(f"[green]✓ Deleted {count} dataset files[/green]")
        return count
    
    def clear_tokenizers(self, confirm: bool = False) -> int:
        """Clear all custom tokenizers."""
        import shutil
        
        if not self.tokenizers_dir.exists():
            return 0
        
        tokenizers = [d for d in self.tokenizers_dir.iterdir() if d.is_dir()]
        
        if not tokenizers:
            console.print("[yellow]No tokenizers to delete.[/yellow]")
            return 0
        
        if not confirm:
            console.print(f"[red]This will delete {len(tokenizers)} tokenizers![/red]")
            return 0
        
        count = 0
        for t in tokenizers:
            shutil.rmtree(t)
            count += 1
        
        console.print(f"[green]✓ Deleted {count} tokenizers[/green]")
        return count
    
    def reset_all(self, confirm: bool = False) -> Dict[str, int]:
        """Reset everything - delete all models, datasets, and tokenizers."""
        if not confirm:
            console.print("[red]⚠️ This will delete ALL trained models, datasets, and tokenizers![/red]")
            return {"models": 0, "datasets": 0, "tokenizers": 0}
        
        results = {
            "models": self.clear_all_models(confirm=True),
            "datasets": self.clear_datasets(confirm=True),
            "tokenizers": self.clear_tokenizers(confirm=True),
        }
        
        console.print(Panel(
            f"[green]Reset complete![/green]\n\n"
            f"Deleted models: {results['models']}\n"
            f"Deleted datasets: {results['datasets']}\n"
            f"Deleted tokenizers: {results['tokenizers']}",
            title="🔄 Factory Reset",
            border_style="green"
        ))
        
        return results
    
    def get_storage_usage(self) -> Dict[str, float]:
        """Get storage usage for models, datasets, and tokenizers."""
        usage = {}
        
        for name, path in [("models", self.models_dir), 
                           ("datasets", self.datasets_dir),
                           ("tokenizers", self.tokenizers_dir)]:
            if path.exists():
                try:
                    size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    usage[name] = round(size_bytes / (1024**3), 2)  # GB
                except:
                    usage[name] = 0
            else:
                usage[name] = 0
        
        usage["total"] = sum(usage.values())
        return usage
    
    def display_storage_usage(self):
        """Display storage usage summary."""
        usage = self.get_storage_usage()
        
        table = Table(title="💾 Storage Usage", show_header=True, header_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Size", style="green", justify="right")
        
        table.add_row("Models", f"{usage.get('models', 0):.2f} GB")
        table.add_row("Datasets", f"{usage.get('datasets', 0):.2f} GB")
        table.add_row("Tokenizers", f"{usage.get('tokenizers', 0):.2f} GB")
        table.add_row("─" * 15, "─" * 10)
        table.add_row("[bold]Total[/bold]", f"[bold]{usage.get('total', 0):.2f} GB[/bold]")
        
        console.print(table)
    
    def prepare_for_retrain(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Prepare model for retraining from scratch.
        Returns info needed to retrain, then deletes the old model.
        """
        info = self.get_model_info(model_name)
        
        if not info:
            return None
        
        # Save retrain configuration
        retrain_config = {
            "original_name": model_name,
            "base_model": info.get("base_model"),
            "type": info.get("type"),
            "lora_rank": info.get("lora_rank"),
            "lora_alpha": info.get("lora_alpha"),
        }
        
        return retrain_config

