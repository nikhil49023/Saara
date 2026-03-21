"""
Neural Network Accelerator Module
Provides GPU optimization, mixed precision training, and performance acceleration.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)


@dataclass
class AcceleratorConfig:
    """Configuration for neural network acceleration."""
    device: str = "auto"  # auto, cuda, cpu, mps
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    memory_efficient_attention: bool = True
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    benchmark: bool = True  # cudnn.benchmark
    deterministic: bool = False
    tf32: bool = True  # Allow TF32 on Ampere+


@dataclass 
class AcceleratorMetrics:
    """Real-time performance metrics."""
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    ram_used: float = 0.0
    ram_total: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    current_loss: float = 0.0
    current_lr: float = 0.0
    step: int = 0
    epoch: int = 0
    eta_seconds: float = 0.0
    throughput_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)


class NeuralAccelerator:
    """
    High-performance neural network accelerator with:
    - Automatic device selection
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Memory optimization
    - Real-time metrics tracking
    """
    
    def __init__(self, config: AcceleratorConfig = None):
        self.config = config or AcceleratorConfig()
        self.metrics = AcceleratorMetrics()
        self.scaler = None
        self.device = None
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = None
        self._callbacks: List[Callable[[AcceleratorMetrics], None]] = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize accelerator with optimal settings."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using CPU fallback.")
            self.device = "cpu"
            return
        
        # Auto-detect best device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        logger.info(f"🚀 Accelerator initialized on: {self.device.upper()}")
        
        # Configure PyTorch optimizations
        if self.device == "cuda":
            # Enable TF32 for Ampere GPUs
            if self.config.tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # CuDNN optimizations
            torch.backends.cudnn.benchmark = self.config.benchmark
            torch.backends.cudnn.deterministic = self.config.deterministic
            
            # Initialize mixed precision scaler
            if self.config.mixed_precision:
                self.scaler = GradScaler()
                logger.info("📊 Mixed precision (FP16/BF16) enabled")
        
        # Memory optimizations
        if self.device == "cuda":
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("⚡ Flash Attention / Memory-Efficient Attention enabled")
    
    def get_device(self) -> str:
        """Get the current device."""
        return self.device
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "device": self.device,
            "torch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
        }
        
        if TORCH_AVAILABLE and self.device == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
            })
        
        return info
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for accelerated training.
        
        - Moves to optimal device
        - Applies gradient checkpointing
        - Compiles model (PyTorch 2.0+)
        """
        if not TORCH_AVAILABLE:
            return model
        
        # Move to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("✅ Gradient checkpointing enabled")
        
        # Compile model for faster execution (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("🔧 Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def prepare_optimizer(self, optimizer, model: nn.Module) -> Any:
        """Configure optimizer for accelerated training."""
        # Add any optimizer wrapping here if needed
        return optimizer
    
    def prepare_dataloader(self, dataloader) -> Any:
        """Optimize dataloader for faster data loading."""
        # Return as-is, let caller configure DataLoader with our recommended settings
        return dataloader
    
    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get recommended DataLoader kwargs for optimal performance."""
        kwargs = {
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory if self.device == "cuda" else False,
            "persistent_workers": self.config.persistent_workers if self.config.num_workers > 0 else False,
        }
        if self.config.num_workers > 0:
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return kwargs
    
    def backward(self, loss, optimizer, step: int = 0):
        """
        Perform backward pass with gradient scaling and accumulation.
        """
        if not TORCH_AVAILABLE:
            return
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        if self.scaler and self.device == "cuda":
            # Mixed precision backward
            self.scaler.scale(scaled_loss).backward()
            
            # Only step optimizer on accumulation boundary
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            scaled_loss.backward()
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    
    @property
    def autocast_context(self):
        """Get autocast context for mixed precision forward pass."""
        if not TORCH_AVAILABLE or not self.config.mixed_precision:
            return torch.no_grad().__class__()  # dummy context
        
        if self.device == "cuda":
            return autocast(device_type="cuda", dtype=torch.float16)
        elif self.device == "cpu":
            return autocast(device_type="cpu", dtype=torch.bfloat16)
        return autocast(device_type="cuda", dtype=torch.float16)
    
    def update_metrics(self, loss: float = None, samples: int = 0, tokens: int = 0,
                       lr: float = None, step: int = None, epoch: int = None):
        """Update training metrics."""
        if loss is not None:
            self.metrics.current_loss = loss
            self.metrics.loss_history.append(loss)
        
        if lr is not None:
            self.metrics.current_lr = lr
        
        if step is not None:
            self.metrics.step = step
        
        if epoch is not None:
            self.metrics.epoch = epoch
        
        # Calculate throughput
        if self._start_time and samples > 0:
            elapsed = time.time() - self._start_time
            self.metrics.samples_per_second = samples / elapsed if elapsed > 0 else 0
            self.metrics.throughput_history.append(self.metrics.samples_per_second)
        
        if self._start_time and tokens > 0:
            elapsed = time.time() - self._start_time
            self.metrics.tokens_per_second = tokens / elapsed if elapsed > 0 else 0
        
        # Update GPU metrics
        if TORCH_AVAILABLE and self.device == "cuda":
            self.metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1e9
            self.metrics.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def register_callback(self, callback: Callable[[AcceleratorMetrics], None]):
        """Register a callback to be notified of metric updates."""
        self._callbacks.append(callback)
    
    def start_training(self):
        """Mark training start time."""
        self._start_time = time.time()
        self.metrics = AcceleratorMetrics()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary after completion."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        return {
            "total_time_seconds": elapsed,
            "total_steps": self.metrics.step,
            "total_epochs": self.metrics.epoch,
            "final_loss": self.metrics.current_loss,
            "avg_samples_per_second": sum(self.metrics.throughput_history) / len(self.metrics.throughput_history) if self.metrics.throughput_history else 0,
            "peak_gpu_memory_gb": self.metrics.gpu_memory_used,
            "device": self.device,
        }
    
    def display_status(self):
        """Display current accelerator status."""
        if not RICH_AVAILABLE:
            print(f"Device: {self.device}, Loss: {self.metrics.current_loss:.4f}")
            return
        
        info = self.get_device_info()
        
        table = Table(title="🚀 Neural Accelerator Status", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Device", info.get("device", "N/A").upper())
        table.add_row("PyTorch Version", info.get("torch_version", "N/A"))
        
        if self.device == "cuda":
            table.add_row("GPU", info.get("gpu_name", "N/A"))
            table.add_row("CUDA Version", info.get("cuda_version", "N/A"))
            table.add_row("GPU Memory", f"{info.get('gpu_memory_total_gb', 0):.1f} GB")
            table.add_row("Compute Capability", info.get("compute_capability", "N/A"))
        
        table.add_row("Mixed Precision", "✅ Enabled" if self.config.mixed_precision else "❌ Disabled")
        table.add_row("Gradient Accumulation", str(self.config.gradient_accumulation_steps))
        table.add_row("Gradient Checkpointing", "✅" if self.config.gradient_checkpointing else "❌")
        
        console.print(table)
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for fast inference."""
        if not TORCH_AVAILABLE:
            return model
        
        model.eval()
        model = model.to(self.device)
        
        # Enable inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            pass  # Will be used in context manager during inference
        
        # Compile for inference
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                logger.warning(f"Inference compilation failed: {e}")
        
        return model
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if TORCH_AVAILABLE and self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("🧹 GPU cache cleared")


class MemoryTracker:
    """Track memory usage during training."""
    
    def __init__(self, accelerator: NeuralAccelerator):
        self.accelerator = accelerator
        self.snapshots: List[Dict[str, Any]] = []
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if not TORCH_AVAILABLE or self.accelerator.device != "cuda":
            return
        
        snapshot = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def report(self) -> str:
        """Generate memory usage report."""
        if not self.snapshots:
            return "No memory snapshots recorded."
        
        lines = ["Memory Usage Report", "=" * 40]
        for snap in self.snapshots:
            lines.append(f"[{snap['label']}] Allocated: {snap['allocated_gb']:.2f} GB, "
                        f"Reserved: {snap['reserved_gb']:.2f} GB, "
                        f"Peak: {snap['max_allocated_gb']:.2f} GB")
        return "\n".join(lines)
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if TORCH_AVAILABLE and self.accelerator.device == "cuda":
            torch.cuda.reset_peak_memory_stats()


def create_accelerator(
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = True,
    **kwargs
) -> NeuralAccelerator:
    """
    Factory function to create an optimized accelerator.
    
    Args:
        mixed_precision: Enable FP16/BF16 training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        gradient_checkpointing: Enable gradient checkpointing for memory savings
        **kwargs: Additional AcceleratorConfig options
    
    Returns:
        Configured NeuralAccelerator instance
    """
    config = AcceleratorConfig(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs
    )
    return NeuralAccelerator(config)


# Convenience exports
__all__ = [
    'NeuralAccelerator',
    'AcceleratorConfig', 
    'AcceleratorMetrics',
    'MemoryTracker',
    'create_accelerator',
]
