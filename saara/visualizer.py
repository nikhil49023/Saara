"""
Neural Network Visualizer Module
Interactive visualization for neural network architectures, training metrics, and processes.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import os
import sys
import json
import time
import math
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    from rich.box import ROUNDED, DOUBLE, HEAVY
    from rich.align import Align
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LayerInfo:
    """Information about a neural network layer."""
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_parameters: int
    trainable: bool = True
    activation: str = ""
    
    
@dataclass
class NetworkArchitecture:
    """Complete neural network architecture representation."""
    name: str
    layers: List[LayerInfo] = field(default_factory=list)
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()


@dataclass
class TrainingSnapshot:
    """Snapshot of training state at a point in time."""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    gpu_memory_gb: float = 0.0
    throughput: float = 0.0  # samples/sec
    timestamp: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingHistory:
    """Complete training history."""
    snapshots: List[TrainingSnapshot] = field(default_factory=list)
    best_loss: float = float('inf')
    best_step: int = 0
    total_time_seconds: float = 0.0
    
    def add(self, snapshot: TrainingSnapshot):
        self.snapshots.append(snapshot)
        if snapshot.loss < self.best_loss:
            self.best_loss = snapshot.loss
            self.best_step = snapshot.step


# ============================================================================
# ASCII Art Neural Network Visualizer
# ============================================================================

class ASCIINetworkVisualizer:
    """Render neural network architecture as ASCII art."""
    
    NEURON_CHAR = "●"
    CONNECTION_CHAR = "─"
    
    def __init__(self, max_width: int = 100):
        self.max_width = max_width
    
    def visualize_layer(self, layer: LayerInfo, index: int) -> str:
        """Create ASCII representation of a single layer."""
        layer_width = min(40, self.max_width // 2)
        
        # Layer box
        name_display = layer.name[:layer_width-4] if len(layer.name) > layer_width-4 else layer.name
        type_display = layer.layer_type[:layer_width-4]
        params = f"{layer.num_parameters:,} params"
        
        lines = [
            f"┌{'─' * (layer_width-2)}┐",
            f"│ Layer {index}: {name_display}".ljust(layer_width-1) + "│",
            f"│ Type: {type_display}".ljust(layer_width-1) + "│",
            f"│ {params}".ljust(layer_width-1) + "│",
            f"│ {'🔓 Trainable' if layer.trainable else '🔒 Frozen'}".ljust(layer_width-1) + "│",
            f"└{'─' * (layer_width-2)}┘",
        ]
        
        return "\n".join(lines)
    
    def visualize_architecture(self, arch: NetworkArchitecture) -> str:
        """Create full ASCII visualization of network architecture."""
        lines = []
        
        # Header
        lines.append(f"\n{'═' * 60}")
        lines.append(f"  🧠 {arch.name}")
        lines.append(f"{'═' * 60}")
        lines.append(f"  Total Parameters: {arch.total_parameters:,}")
        lines.append(f"  Trainable: {arch.trainable_parameters:,}")
        lines.append(f"  Model Size: {arch.model_size_mb:.2f} MB")
        lines.append(f"{'─' * 60}\n")
        
        # Layers with connections
        for i, layer in enumerate(arch.layers):
            lines.append(self.visualize_layer(layer, i))
            if i < len(arch.layers) - 1:
                lines.append("        │")
                lines.append("        ▼")
        
        lines.append(f"\n{'═' * 60}")
        
        return "\n".join(lines)


# ============================================================================
# Rich Console Visualizer
# ============================================================================

class RichNetworkVisualizer:
    """Rich-powered neural network visualizer with beautiful console output."""
    
    LAYER_COLORS = {
        "Linear": "cyan",
        "Conv": "green", 
        "Attention": "magenta",
        "Embedding": "yellow",
        "LayerNorm": "blue",
        "Dropout": "dim",
        "ReLU": "red",
        "GELU": "red",
        "Softmax": "red",
    }
    
    def __init__(self):
        if not RICH_AVAILABLE:
            raise RuntimeError("Rich library required for RichNetworkVisualizer")
        self.console = Console()
    
    def _get_layer_color(self, layer_type: str) -> str:
        """Get color for layer type."""
        for key, color in self.LAYER_COLORS.items():
            if key.lower() in layer_type.lower():
                return color
        return "white"
    
    def visualize_architecture(self, arch: NetworkArchitecture):
        """Display network architecture with Rich formatting."""
        # Header panel
        header = Panel(
            f"[bold cyan]{arch.name}[/bold cyan]\n\n"
            f"[green]📊 Total Parameters:[/green] {arch.total_parameters:,}\n"
            f"[green]🔧 Trainable:[/green] {arch.trainable_parameters:,}\n"
            f"[green]💾 Model Size:[/green] {arch.model_size_mb:.2f} MB",
            title="🧠 Neural Network Architecture",
            border_style="cyan",
            box=DOUBLE
        )
        self.console.print(header)
        
        # Layers table
        table = Table(
            title="Network Layers",
            show_header=True,
            header_style="bold magenta",
            box=ROUNDED
        )
        
        table.add_column("#", style="dim", width=4)
        table.add_column("Layer Name", style="cyan", max_width=30)
        table.add_column("Type", style="green")
        table.add_column("Output Shape", style="yellow")
        table.add_column("Parameters", style="blue", justify="right")
        table.add_column("Status", style="white", justify="center")
        
        for i, layer in enumerate(arch.layers):
            color = self._get_layer_color(layer.layer_type)
            status = "[green]●[/green] Train" if layer.trainable else "[dim]○[/dim] Frozen"
            
            table.add_row(
                str(i),
                layer.name,
                f"[{color}]{layer.layer_type}[/{color}]",
                str(layer.output_shape),
                f"{layer.num_parameters:,}",
                status
            )
        
        self.console.print(table)
    
    def visualize_training_progress(self, history: TrainingHistory, current: TrainingSnapshot = None):
        """Display training progress with live metrics."""
        # Create layout
        layout = Layout()
        
        # Current metrics panel
        if current:
            metrics_text = (
                f"[bold]Step:[/bold] {current.step} | "
                f"[bold]Epoch:[/bold] {current.epoch}\n"
                f"[bold green]Loss:[/bold green] {current.loss:.6f}\n"
                f"[bold cyan]LR:[/bold cyan] {current.learning_rate:.2e}\n"
                f"[bold yellow]GPU:[/bold yellow] {current.gpu_memory_gb:.2f} GB\n"
                f"[bold magenta]Speed:[/bold magenta] {current.throughput:.1f} samples/sec"
            )
            
            current_panel = Panel(
                metrics_text,
                title="📈 Current Training State",
                border_style="green"
            )
            self.console.print(current_panel)
        
        # Loss chart (ASCII sparkline)
        if history.snapshots:
            losses = [s.loss for s in history.snapshots[-50:]]
            chart = self._create_sparkline(losses, width=50)
            
            chart_panel = Panel(
                f"[cyan]{chart}[/cyan]\n\n"
                f"[dim]Best Loss: {history.best_loss:.6f} @ Step {history.best_step}[/dim]",
                title="📉 Loss History (Last 50 Steps)",
                border_style="blue"
            )
            self.console.print(chart_panel)
    
    def _create_sparkline(self, values: List[float], width: int = 50) -> str:
        """Create ASCII sparkline chart."""
        if not values:
            return "No data"
        
        # Normalize values
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        # Sparkline characters (low to high)
        chars = "▁▂▃▄▅▆▇█"
        
        # Sample or interpolate to fit width
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values
        
        # Convert to sparkline
        sparkline = ""
        for val in sampled:
            normalized = (val - min_val) / range_val
            char_idx = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            sparkline += chars[char_idx]
        
        return sparkline
    
    def create_live_dashboard(self) -> 'TrainingDashboard':
        """Create a live training dashboard."""
        return TrainingDashboard(self.console)


# ============================================================================
# Live Training Dashboard
# ============================================================================

class TrainingDashboard:
    """Real-time training dashboard with live updates."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.history = TrainingHistory()
        self.current = None
        self.running = False
        self._live = None
        self._layout = None
        self.model_name = "Neural Network"
        self.start_time = None
        
    def _create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="chart", ratio=2)
        )
        
        return layout
    
    def _render_header(self) -> Panel:
        """Render header section."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
        
        return Panel(
            Align.center(
                f"[bold cyan]🧠 {self.model_name}[/bold cyan] | "
                f"[green]⏱️ {elapsed_str}[/green] | "
                f"[yellow]Step {self.current.step if self.current else 0}[/yellow]"
            ),
            style="on dark_blue"
        )
    
    def _render_metrics(self) -> Panel:
        """Render metrics panel."""
        if not self.current:
            return Panel("Waiting for data...", title="📊 Metrics")
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="green")
        
        table.add_row("Loss", f"{self.current.loss:.6f}")
        table.add_row("Learning Rate", f"{self.current.learning_rate:.2e}")
        table.add_row("Epoch", str(self.current.epoch))
        table.add_row("Step", str(self.current.step))
        table.add_row("GPU Memory", f"{self.current.gpu_memory_gb:.2f} GB")
        table.add_row("Throughput", f"{self.current.throughput:.1f} s/s")
        table.add_row("Best Loss", f"{self.history.best_loss:.6f}")
        
        return Panel(table, title="📊 Training Metrics", border_style="cyan")
    
    def _render_chart(self) -> Panel:
        """Render loss chart."""
        if not self.history.snapshots:
            return Panel("Collecting data...", title="📉 Loss Curve")
        
        losses = [s.loss for s in self.history.snapshots[-100:]]
        
        # Create multi-line ASCII chart
        height = 10
        width = 60
        
        min_loss = min(losses)
        max_loss = max(losses)
        range_loss = max_loss - min_loss if max_loss > min_loss else 1
        
        # Normalize and create chart
        chart_lines = []
        for row in range(height):
            threshold = max_loss - (row / height) * range_loss
            line = ""
            for i, loss in enumerate(losses[-width:]):
                if loss <= threshold:
                    line += "█"
                else:
                    line += " "
            chart_lines.append(f"[cyan]{line}[/cyan]")
        
        chart_text = "\n".join(chart_lines)
        chart_text += f"\n[dim]Min: {min_loss:.4f} | Max: {max_loss:.4f}[/dim]"
        
        return Panel(chart_text, title="📉 Loss Curve", border_style="green")
    
    def _render_footer(self) -> Panel:
        """Render footer with status."""
        status = "[green]● Training in progress...[/green]" if self.running else "[yellow]● Paused[/yellow]"
        return Panel(
            Align.center(f"{status} | Press Ctrl+C to stop"),
            style="dim"
        )
    
    def _generate_display(self) -> Layout:
        """Generate full dashboard display."""
        layout = self._create_layout()
        layout["header"].update(self._render_header())
        layout["metrics"].update(self._render_metrics())
        layout["chart"].update(self._render_chart())
        layout["footer"].update(self._render_footer())
        return layout
    
    def update(self, step: int, epoch: int, loss: float, lr: float,
               gpu_memory: float = 0.0, throughput: float = 0.0,
               extra_metrics: Dict[str, float] = None):
        """Update dashboard with new training data."""
        snapshot = TrainingSnapshot(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=lr,
            gpu_memory_gb=gpu_memory,
            throughput=throughput,
            timestamp=datetime.now().isoformat(),
            metrics=extra_metrics or {}
        )
        
        self.current = snapshot
        self.history.add(snapshot)
    
    def start(self, model_name: str = "Neural Network"):
        """Start the live dashboard."""
        self.model_name = model_name
        self.running = True
        self.start_time = time.time()
        self._live = Live(self._generate_display(), console=self.console, refresh_per_second=4)
        self._live.start()
    
    def refresh(self):
        """Refresh the dashboard display."""
        if self._live:
            self._live.update(self._generate_display())
    
    def stop(self):
        """Stop the live dashboard."""
        self.running = False
        if self._live:
            self._live.stop()
        
        # Print final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        summary = Panel(
            f"[bold green]✅ Training Complete![/bold green]\n\n"
            f"[cyan]Total Steps:[/cyan] {self.history.snapshots[-1].step if self.history.snapshots else 0}\n"
            f"[cyan]Total Time:[/cyan] {elapsed/60:.1f} minutes\n"
            f"[cyan]Final Loss:[/cyan] {self.current.loss if self.current else 0:.6f}\n"
            f"[cyan]Best Loss:[/cyan] {self.history.best_loss:.6f} @ Step {self.history.best_step}",
            title="📊 Training Summary",
            border_style="green",
            box=DOUBLE
        )
        self.console.print(summary)


# ============================================================================
# Model Architecture Analyzer
# ============================================================================

class ModelAnalyzer:
    """Analyze PyTorch models and extract architecture information."""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ModelAnalyzer")
    
    def analyze(self, model: nn.Module, input_shape: Tuple[int, ...] = None,
                model_name: str = None) -> NetworkArchitecture:
        """Analyze a PyTorch model and extract architecture."""
        
        layers = []
        total_params = 0
        trainable_params = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                # Count parameters
                num_params = sum(p.numel() for p in module.parameters(recurse=False))
                trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
                
                if num_params > 0 or isinstance(module, (nn.ReLU, nn.GELU, nn.Dropout)):
                    layer_info = LayerInfo(
                        name=name or module.__class__.__name__,
                        layer_type=module.__class__.__name__,
                        input_shape=(),
                        output_shape=(),
                        num_parameters=num_params,
                        trainable=trainable > 0
                    )
                    layers.append(layer_info)
                
                total_params += num_params
                trainable_params += trainable
        
        # Calculate model size
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return NetworkArchitecture(
            name=model_name or model.__class__.__name__,
            layers=layers,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            input_shape=input_shape or (),
            output_shape=()
        )
    
    def summary(self, model: nn.Module, input_shape: Tuple[int, ...] = None):
        """Print a summary of the model architecture."""
        arch = self.analyze(model, input_shape)
        
        if RICH_AVAILABLE:
            viz = RichNetworkVisualizer()
            viz.visualize_architecture(arch)
        else:
            ascii_viz = ASCIINetworkVisualizer()
            print(ascii_viz.visualize_architecture(arch))
        
        return arch


# ============================================================================
# Process Visualizer
# ============================================================================

class ProcessVisualizer:
    """Visualize data pipeline processes."""
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def show_pipeline(self, stages: List[Dict[str, Any]]):
        """Visualize a data pipeline with stages."""
        if not RICH_AVAILABLE:
            for i, stage in enumerate(stages):
                print(f"[{i+1}] {stage.get('name', 'Stage')} - {stage.get('status', 'pending')}")
            return
        
        # Create pipeline visualization
        panels = []
        
        for i, stage in enumerate(stages):
            name = stage.get('name', f'Stage {i+1}')
            status = stage.get('status', 'pending')
            details = stage.get('details', '')
            
            # Status icon and color
            if status == 'complete':
                icon = "✅"
                color = "green"
            elif status == 'running':
                icon = "🔄"
                color = "yellow"
            elif status == 'error':
                icon = "❌"
                color = "red"
            else:
                icon = "⏳"
                color = "dim"
            
            panel = Panel(
                f"{icon} [bold]{name}[/bold]\n[dim]{details}[/dim]",
                border_style=color,
                width=25
            )
            panels.append(panel)
        
        # Print with arrows between stages
        columns = Columns(panels, equal=True, expand=True)
        self.console.print(columns)
    
    def show_progress(self, task_name: str, total: int, description: str = "") -> Progress:
        """Create and return a progress bar for a task."""
        if not RICH_AVAILABLE:
            return None
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        
        return progress


# ============================================================================
# HTML Report Generator
# ============================================================================

class HTMLReportGenerator:
    """Generate HTML reports for training runs."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_training_report(self, arch: NetworkArchitecture,
                                  history: TrainingHistory,
                                  output_name: str = "training_report") -> str:
        """Generate an HTML training report."""
        
        # Prepare chart data
        steps = [s.step for s in history.snapshots]
        losses = [s.loss for s in history.snapshots]
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {arch.name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #00d9ff;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d9ff, #ff00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            color: #00d9ff;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-item {{
            background: rgba(0,217,255,0.1);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #00d9ff;
        }}
        .metric-label {{
            color: #888;
            margin-top: 5px;
        }}
        .chart-container {{
            height: 400px;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{ color: #00d9ff; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 {arch.name}</h1>
            <p>Training Report - Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="card">
            <h2>📊 Training Summary</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value">{history.snapshots[-1].step if history.snapshots else 0:,}</div>
                    <div class="metric-label">Total Steps</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{history.best_loss:.4f}</div>
                    <div class="metric-label">Best Loss</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{arch.total_parameters:,}</div>
                    <div class="metric-label">Parameters</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{arch.model_size_mb:.1f} MB</div>
                    <div class="metric-label">Model Size</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📉 Loss Curve</h2>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>🏗️ Model Architecture</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Layer</th>
                        <th>Type</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'<tr><td>{i}</td><td>{l.name}</td><td>{l.layer_type}</td><td>{l.num_parameters:,}</td></tr>' for i, l in enumerate(arch.layers[:20]))}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(steps[-100:])},
                datasets: [{{
                    label: 'Loss',
                    data: {json.dumps(losses[-100:])},
                    borderColor: '#00d9ff',
                    backgroundColor: 'rgba(0, 217, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                    y: {{ ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        output_path = self.output_dir / f"{output_name}.html"
        output_path.write_text(html, encoding='utf-8')
        
        return str(output_path)


# ============================================================================
# Factory Functions
# ============================================================================

def create_visualizer(use_rich: bool = True) -> Union[RichNetworkVisualizer, ASCIINetworkVisualizer]:
    """Create appropriate visualizer based on environment."""
    if use_rich and RICH_AVAILABLE:
        return RichNetworkVisualizer()
    return ASCIINetworkVisualizer()


def create_dashboard() -> TrainingDashboard:
    """Create a training dashboard."""
    return TrainingDashboard()


def analyze_model(model, name: str = None) -> NetworkArchitecture:
    """Analyze a model and return its architecture."""
    analyzer = ModelAnalyzer()
    return analyzer.analyze(model, model_name=name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'LayerInfo',
    'NetworkArchitecture', 
    'TrainingSnapshot',
    'TrainingHistory',
    'ASCIINetworkVisualizer',
    'RichNetworkVisualizer',
    'TrainingDashboard',
    'ModelAnalyzer',
    'ProcessVisualizer',
    'HTMLReportGenerator',
    'create_visualizer',
    'create_dashboard',
    'analyze_model',
]
