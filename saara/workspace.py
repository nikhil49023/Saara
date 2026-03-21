"""
Workspace Configuration Module
Manages workspace folder for storing models, datasets, and outputs.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import os
import sys
import json
import platform
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


# ============================================================================
# Workspace Configuration
# ============================================================================

@dataclass
class WorkspaceConfig:
    """Workspace configuration settings."""
    workspace_path: str
    models_dir: str = "models"
    datasets_dir: str = "datasets"
    outputs_dir: str = "outputs"
    tokenizers_dir: str = "tokenizers"
    checkpoints_dir: str = "checkpoints"
    notebooks_dir: str = "notebooks"
    logs_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "WorkspaceConfig":
        return cls(**data)
    
    def get_models_path(self) -> Path:
        return Path(self.workspace_path) / self.models_dir
    
    def get_datasets_path(self) -> Path:
        return Path(self.workspace_path) / self.datasets_dir
    
    def get_outputs_path(self) -> Path:
        return Path(self.workspace_path) / self.outputs_dir
    
    def get_tokenizers_path(self) -> Path:
        return Path(self.workspace_path) / self.tokenizers_dir
    
    def get_checkpoints_path(self) -> Path:
        return Path(self.workspace_path) / self.checkpoints_dir
    
    def get_notebooks_path(self) -> Path:
        return Path(self.workspace_path) / self.notebooks_dir
    
    def get_logs_path(self) -> Path:
        return Path(self.workspace_path) / self.logs_dir


class WorkspaceManager:
    """
    Manages the SAARA workspace - a central folder for all models, datasets, and outputs.
    Works across Windows, Linux, and macOS.
    """
    
    CONFIG_FILENAME = ".saara_workspace.json"
    
    def __init__(self):
        self._config: Optional[WorkspaceConfig] = None
        self._config_dir = self._get_config_dir()
        self._config_file = self._config_dir / self.CONFIG_FILENAME
        self._load_config()
    
    def _get_config_dir(self) -> Path:
        """Get the appropriate config directory for the current OS."""
        system = platform.system()
        
        if system == "Windows":
            # Windows: Use APPDATA or user home
            appdata = os.environ.get("APPDATA")
            if appdata:
                return Path(appdata) / "SAARA"
            return Path.home() / ".saara"
        elif system == "Darwin":
            # macOS: Use Application Support
            return Path.home() / "Library" / "Application Support" / "SAARA"
        else:
            # Linux and others: Use ~/.saara
            return Path.home() / ".saara"
    
    def _get_default_workspace(self) -> Path:
        """Get the default workspace path for the current OS."""
        system = platform.system()
        
        if system == "Windows":
            # Windows: Documents folder
            docs = Path.home() / "Documents"
            return docs / "SAARA_Workspace"
        elif system == "Darwin":
            # macOS: Documents folder
            docs = Path.home() / "Documents"
            return docs / "SAARA_Workspace"
        else:
            # Linux: Home directory
            return Path.home() / "saara_workspace"
    
    def _load_config(self):
        """Load workspace configuration from file."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    data = json.load(f)
                    self._config = WorkspaceConfig.from_dict(data)
            except Exception as e:
                self._config = None
    
    def _save_config(self):
        """Save workspace configuration to file."""
        if self._config:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, "w") as f:
                json.dump(self._config.to_dict(), f, indent=2)
    
    def is_configured(self) -> bool:
        """Check if workspace is configured."""
        return self._config is not None and Path(self._config.workspace_path).exists()
    
    def get_config(self) -> Optional[WorkspaceConfig]:
        """Get the current workspace configuration."""
        return self._config
    
    def get_workspace_path(self) -> Optional[Path]:
        """Get the workspace path."""
        if self._config:
            return Path(self._config.workspace_path)
        return None
    
    def setup_workspace(self, path: str = None, interactive: bool = True) -> bool:
        """
        Setup or change the workspace folder.
        
        Args:
            path: Optional workspace path. If None, will prompt user or use default.
            interactive: Whether to show interactive prompts.
        
        Returns:
            True if workspace was set up successfully.
        """
        if RICH_AVAILABLE and interactive:
            console.print(Panel.fit(
                "[bold cyan]📂 SAARA Workspace Setup[/bold cyan]\n\n"
                "Choose a folder to store all your models, datasets, and outputs.\n"
                "[dim]This folder will be used across all SAARA sessions.[/dim]",
                title="Workspace Configuration",
                border_style="cyan"
            ))
        
        # Determine workspace path
        if path:
            workspace_path = Path(path)
        elif interactive and RICH_AVAILABLE:
            default_path = str(self._get_default_workspace())
            
            console.print("\n[bold]Select Workspace Location:[/bold]")
            console.print(f"  1. Use default: [cyan]{default_path}[/cyan]")
            console.print("  2. Choose custom location")
            console.print("  3. Use current directory")
            
            choice = Prompt.ask("Selection", choices=["1", "2", "3"], default="1")
            
            if choice == "1":
                workspace_path = self._get_default_workspace()
            elif choice == "2":
                custom_path = Prompt.ask("Enter workspace folder path")
                workspace_path = Path(custom_path).expanduser().resolve()
            else:
                workspace_path = Path.cwd() / "saara_workspace"
        else:
            workspace_path = self._get_default_workspace()
        
        # Create workspace structure
        workspace_path = Path(workspace_path).resolve()
        
        try:
            # Create main workspace folder
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create configuration
            self._config = WorkspaceConfig(workspace_path=str(workspace_path))
            
            # Create subdirectories
            subdirs = [
                self._config.models_dir,
                self._config.datasets_dir,
                self._config.outputs_dir,
                self._config.tokenizers_dir,
                self._config.checkpoints_dir,
                self._config.notebooks_dir,
                self._config.logs_dir,
            ]
            
            for subdir in subdirs:
                (workspace_path / subdir).mkdir(exist_ok=True)
            
            # Save configuration
            self._save_config()
            
            # Create a README in the workspace
            readme_path = workspace_path / "README.md"
            if not readme_path.exists():
                readme_path.write_text(f"""# SAARA Workspace

This folder contains all your SAARA AI data:

- **models/** - Fine-tuned and pre-trained models
- **datasets/** - Training datasets (JSONL, CSV, etc.)
- **outputs/** - Generated outputs and reports
- **tokenizers/** - Custom tokenizers
- **checkpoints/** - Training checkpoints
- **notebooks/** - Generated Colab/Kaggle notebooks
- **logs/** - Training and processing logs

Created: {Path(workspace_path).stat().st_mtime}
OS: {platform.system()} {platform.release()}
""")
            
            if RICH_AVAILABLE and interactive:
                console.print(Panel(
                    f"[bold green]✅ Workspace Created![/bold green]\n\n"
                    f"[bold]Location:[/bold] [cyan]{workspace_path}[/cyan]\n\n"
                    f"[bold]Created folders:[/bold]\n"
                    f"  📁 models/      - Fine-tuned models\n"
                    f"  📁 datasets/    - Training datasets\n"
                    f"  📁 outputs/     - Generated outputs\n"
                    f"  📁 tokenizers/  - Custom tokenizers\n"
                    f"  📁 checkpoints/ - Training checkpoints\n"
                    f"  📁 notebooks/   - GPU worker notebooks\n"
                    f"  📁 logs/        - Training logs\n\n"
                    f"[dim]All SAARA operations will now use this workspace.[/dim]",
                    title="🎉 Workspace Ready",
                    border_style="green"
                ))
            
            return True
            
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ Failed to create workspace: {e}[/red]")
            return False
    
    def require_workspace(self) -> WorkspaceConfig:
        """
        Ensure workspace is configured, prompting setup if not.
        
        Returns:
            WorkspaceConfig object.
        
        Raises:
            RuntimeError if workspace cannot be configured.
        """
        if not self.is_configured():
            if RICH_AVAILABLE:
                console.print("\n[yellow]⚠ Workspace not configured.[/yellow]")
                console.print("[dim]SAARA needs a workspace folder to store models and datasets.[/dim]\n")
            
            if not self.setup_workspace(interactive=True):
                raise RuntimeError("Workspace setup failed")
        
        return self._config
    
    def display_info(self):
        """Display workspace information."""
        if not self._config:
            if RICH_AVAILABLE:
                console.print("[yellow]Workspace not configured.[/yellow]")
                console.print("[dim]Run 'saara workspace setup' to configure.[/dim]")
            return
        
        workspace_path = Path(self._config.workspace_path)
        
        if RICH_AVAILABLE:
            table = Table(title="📂 SAARA Workspace", show_header=True, header_style="bold cyan")
            table.add_column("Folder", style="cyan")
            table.add_column("Path", style="green")
            table.add_column("Size", style="yellow")
            table.add_column("Files", style="magenta")
            
            folders = [
                ("Workspace", workspace_path, ""),
                ("Models", self._config.get_models_path(), self._config.models_dir),
                ("Datasets", self._config.get_datasets_path(), self._config.datasets_dir),
                ("Outputs", self._config.get_outputs_path(), self._config.outputs_dir),
                ("Tokenizers", self._config.get_tokenizers_path(), self._config.tokenizers_dir),
                ("Checkpoints", self._config.get_checkpoints_path(), self._config.checkpoints_dir),
                ("Notebooks", self._config.get_notebooks_path(), self._config.notebooks_dir),
                ("Logs", self._config.get_logs_path(), self._config.logs_dir),
            ]
            
            for name, path, _ in folders:
                if path.exists():
                    size = self._get_folder_size(path)
                    file_count = len(list(path.glob("*")))
                    table.add_row(
                        name,
                        str(path),
                        self._format_size(size),
                        str(file_count)
                    )
                else:
                    table.add_row(name, str(path), "-", "-")
            
            console.print(table)
    
    def _get_folder_size(self, path: Path) -> int:
        """Get total size of a folder in bytes."""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except:
            pass
        return total
    
    def _format_size(self, size: int) -> str:
        """Format size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def clear_workspace(self, target: str = "all", confirm: bool = True) -> bool:
        """Clear workspace contents."""
        if not self._config:
            return False
        
        targets = {
            "models": self._config.get_models_path(),
            "datasets": self._config.get_datasets_path(),
            "outputs": self._config.get_outputs_path(),
            "tokenizers": self._config.get_tokenizers_path(),
            "checkpoints": self._config.get_checkpoints_path(),
            "notebooks": self._config.get_notebooks_path(),
            "logs": self._config.get_logs_path(),
        }
        
        if target == "all":
            paths_to_clear = list(targets.values())
        elif target in targets:
            paths_to_clear = [targets[target]]
        else:
            return False
        
        if confirm and RICH_AVAILABLE:
            if not Confirm.ask(f"Clear {target}? This cannot be undone", default=False):
                return False
        
        import shutil
        for path in paths_to_clear:
            if path.exists():
                for item in path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        
        if RICH_AVAILABLE:
            console.print(f"[green]✓ Cleared: {target}[/green]")
        
        return True


# ============================================================================
# Dataset Library
# ============================================================================

class DatasetLibrary:
    """
    Dataset library for downloading and managing datasets from Hugging Face.
    """
    
    # Popular datasets for LLM training
    RECOMMENDED_DATASETS = [
        {
            "name": "openwebtext",
            "hf_id": "Skylion007/openwebtext",
            "description": "Web text dataset for language modeling",
            "size": "~40GB",
            "category": "pretraining"
        },
        {
            "name": "wikitext",
            "hf_id": "wikitext",
            "config": "wikitext-103-raw-v1",
            "description": "Wikipedia articles for language modeling",
            "size": "~500MB",
            "category": "pretraining"
        },
        {
            "name": "alpaca",
            "hf_id": "tatsu-lab/alpaca",
            "description": "Instruction-following dataset (52K samples)",
            "size": "~25MB",
            "category": "instruction"
        },
        {
            "name": "databricks-dolly",
            "hf_id": "databricks/databricks-dolly-15k",
            "description": "Instruction-following dataset by Databricks",
            "size": "~15MB",
            "category": "instruction"
        },
        {
            "name": "oasst1",
            "hf_id": "OpenAssistant/oasst1",
            "description": "Human-generated conversations for chat models",
            "size": "~150MB",
            "category": "chat"
        },
        {
            "name": "squad",
            "hf_id": "squad",
            "description": "Stanford QA dataset for reading comprehension",
            "size": "~35MB",
            "category": "qa"
        },
        {
            "name": "gsm8k",
            "hf_id": "gsm8k",
            "config": "main",
            "description": "Grade school math word problems",
            "size": "~10MB",
            "category": "math"
        },
        {
            "name": "code_alpaca",
            "hf_id": "sahil2801/CodeAlpaca-20k",
            "description": "Code instruction dataset (20K samples)",
            "size": "~20MB",
            "category": "code"
        },
        {
            "name": "medical_meadow",
            "hf_id": "medalpaca/medical_meadow_medical_flashcards",
            "description": "Medical QA flashcards",
            "size": "~5MB",
            "category": "medical"
        },
        {
            "name": "ultrachat",
            "hf_id": "stingning/ultrachat",
            "description": "Large-scale chat dataset (1.5M conversations)",
            "size": "~5GB",
            "category": "chat"
        },
    ]
    
    def __init__(self, workspace: WorkspaceManager = None):
        self.workspace = workspace or WorkspaceManager()
        self._datasets_cache: Dict[str, Dict] = {}
    
    def get_datasets_path(self) -> Path:
        """Get the datasets folder path."""
        config = self.workspace.get_config()
        if config:
            return config.get_datasets_path()
        return Path.cwd() / "datasets"
    
    def list_recommended(self, category: str = None) -> list:
        """List recommended datasets, optionally filtered by category."""
        datasets = self.RECOMMENDED_DATASETS
        if category:
            datasets = [d for d in datasets if d.get("category") == category]
        return datasets
    
    def list_local(self) -> list:
        """List locally available datasets."""
        datasets_path = self.get_datasets_path()
        local_datasets = []
        
        if not datasets_path.exists():
            return local_datasets
        
        for item in datasets_path.iterdir():
            if item.is_dir() or item.suffix in [".jsonl", ".json", ".csv", ".parquet"]:
                size = self.workspace._get_folder_size(item) if item.is_dir() else item.stat().st_size
                local_datasets.append({
                    "name": item.stem,
                    "path": str(item),
                    "size": self.workspace._format_size(size),
                    "type": "folder" if item.is_dir() else item.suffix[1:]
                })
        
        return local_datasets
    
    def download_dataset(
        self, 
        dataset_id: str, 
        config: str = None,
        split: str = "train",
        output_format: str = "jsonl",
        max_samples: int = None,
        progress_callback = None
    ) -> Optional[Path]:
        """
        Download a dataset from Hugging Face.
        
        Args:
            dataset_id: Hugging Face dataset ID (e.g., "tatsu-lab/alpaca")
            config: Dataset configuration name
            split: Dataset split to download
            output_format: Output format (jsonl, csv, parquet)
            max_samples: Maximum number of samples (None for all)
            progress_callback: Callback for progress updates
        
        Returns:
            Path to downloaded dataset or None on failure.
        """
        try:
            from datasets import load_dataset
            import json
        except ImportError:
            if RICH_AVAILABLE:
                console.print("[red]❌ datasets library not installed. Run: pip install datasets[/red]")
            return None
        
        datasets_path = self.get_datasets_path()
        datasets_path.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename
        safe_name = dataset_id.replace("/", "_").replace("-", "_")
        output_file = datasets_path / f"{safe_name}.{output_format}"
        
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]📥 Downloading: {dataset_id}[/cyan]")
            if config:
                console.print(f"[dim]   Config: {config}[/dim]")
            console.print(f"[dim]   Split: {split}[/dim]")
        
        try:
            # Load dataset from Hugging Face
            if config:
                dataset = load_dataset(dataset_id, config, split=split)
            else:
                dataset = load_dataset(dataset_id, split=split)
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            if RICH_AVAILABLE:
                console.print(f"[dim]   Samples: {len(dataset):,}[/dim]")
            
            # Save to file
            if output_format == "jsonl":
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif output_format == "csv":
                dataset.to_csv(str(output_file))
            elif output_format == "parquet":
                dataset.to_parquet(str(output_file))
            else:
                # Default to JSON
                dataset.to_json(str(output_file))
            
            if RICH_AVAILABLE:
                file_size = self.workspace._format_size(output_file.stat().st_size)
                console.print(f"[green]✅ Saved: {output_file} ({file_size})[/green]")
            
            return output_file
            
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ Download failed: {e}[/red]")
            return None
    
    def search_huggingface(self, query: str, limit: int = 10) -> list:
        """Search for datasets on Hugging Face."""
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            results = api.list_datasets(search=query, limit=limit)
            
            return [
                {
                    "id": ds.id,
                    "downloads": getattr(ds, "downloads", 0),
                    "likes": getattr(ds, "likes", 0),
                }
                for ds in results
            ]
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[yellow]Search failed: {e}[/yellow]")
            return []
    
    def display_recommended(self, category: str = None):
        """Display recommended datasets in a table."""
        if not RICH_AVAILABLE:
            return
        
        datasets = self.list_recommended(category)
        
        table = Table(title="📚 Recommended Datasets", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Description", style="white")
        
        for i, ds in enumerate(datasets, 1):
            table.add_row(
                str(i),
                ds["name"],
                ds.get("category", "general"),
                ds.get("size", "?"),
                ds["description"][:50] + "..." if len(ds["description"]) > 50 else ds["description"]
            )
        
        console.print(table)
    
    def display_local(self):
        """Display locally available datasets."""
        if not RICH_AVAILABLE:
            return
        
        datasets = self.list_local()
        
        if not datasets:
            console.print("[yellow]No local datasets found.[/yellow]")
            console.print("[dim]Download datasets with: saara datasets download[/dim]")
            return
        
        table = Table(title="💾 Local Datasets", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Path", style="dim")
        
        for ds in datasets:
            table.add_row(
                ds["name"],
                ds["type"],
                ds["size"],
                ds["path"][:50] + "..." if len(ds["path"]) > 50 else ds["path"]
            )
        
        console.print(table)


# ============================================================================
# Cloud GPU Setup Instructions
# ============================================================================

def display_cloud_gpu_instructions():
    """Display step-by-step instructions for setting up cloud GPU."""
    if not RICH_AVAILABLE:
        print("Cloud GPU Setup Instructions")
        print("=" * 40)
        return
    
    instructions = """
[bold cyan]🌐 Cloud GPU Setup - Step by Step[/bold cyan]

[bold]Prerequisites:[/bold]
  • SAARA CLI installed on your local machine
  • Free account on Kaggle or Google Colab

[bold yellow]═══ STEP 1: Start Worker Server ═══[/bold yellow]

Run this command on your local machine:

    [cyan]saara cloud connect[/cyan]

For public access (if behind NAT/firewall):

    [cyan]saara cloud connect --ngrok[/cyan]

[dim]Note: You'll need to install pyngrok for public access:
      pip install pyngrok[/dim]

[bold yellow]═══ STEP 2: Generate Worker Notebook ═══[/bold yellow]

Generate a notebook for your preferred platform:

[bold]For Kaggle:[/bold]
    [cyan]saara cloud generate --platform kaggle --url YOUR_SERVER_URL[/cyan]

[bold]For Google Colab:[/bold]
    [cyan]saara cloud generate --platform colab --url YOUR_SERVER_URL[/cyan]

[bold yellow]═══ STEP 3: Configure Cloud Platform ═══[/bold yellow]

[bold]Kaggle Setup:[/bold]
  1. Go to kaggle.com/kernels and click "New Notebook"
  2. Click "File → Import Notebook" and upload the generated file
  3. Settings (right panel):
     • Accelerator: [green]GPU T4 x2[/green] or [green]GPU P100[/green]
     • Internet: [green]On[/green]
     • Persistence: [green]Files only[/green]
  4. Run all cells (Shift+Enter on each cell)

[bold]Google Colab Setup:[/bold]
  1. Go to colab.research.google.com
  2. File → Upload Notebook → Upload the generated file
  3. Runtime → Change runtime type:
     • Hardware accelerator: [green]T4 GPU[/green] (free) or [green]A100[/green] (Pro)
  4. Run all cells (Ctrl+F9 or Runtime → Run all)

[bold yellow]═══ STEP 4: Verify Connection ═══[/bold yellow]

On your local machine, check if worker connected:

    [cyan]saara cloud workers[/cyan]

You should see your cloud worker listed with GPU info.

[bold yellow]═══ STEP 5: Submit Training Jobs ═══[/bold yellow]

Create a training job to run on the cloud GPU:

    [cyan]saara cloud jobs create --type training --model google/gemma-2-2b[/cyan]

Monitor job progress:

    [cyan]saara cloud jobs list[/cyan]

[bold green]═══ TIPS ═══[/bold green]

  • [yellow]Kaggle[/yellow] gives ~30 hours/week of free GPU (T4/P100)
  • [yellow]Colab[/yellow] gives ~12 hours/session of free GPU (T4)
  • Use [cyan]--ngrok[/cyan] flag if your local machine can't receive connections
  • Keep the notebook running to maintain the worker connection
  • Worker auto-reconnects if connection is temporarily lost

[bold red]═══ TROUBLESHOOTING ═══[/bold red]

[yellow]"Connection refused"[/yellow]
  → Make sure saara cloud connect is running
  → Check firewall allows incoming connections on port 8765
  → Try using --ngrok for tunneling

[yellow]"Invalid token"[/yellow]
  → Regenerate token: [cyan]saara cloud token generate[/cyan]
  → Update notebook with new token

[yellow]"GPU not detected"[/yellow]
  → Enable GPU in Kaggle/Colab settings
  → Restart the runtime

[yellow]"Worker disconnected"[/yellow]
  → Kaggle/Colab sessions timeout after idle period
  → Re-run the notebook cells to reconnect
"""
    
    console.print(Panel(
        instructions,
        title="📖 Cloud GPU Setup Guide",
        border_style="cyan",
        padding=(1, 2)
    ))


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'WorkspaceConfig',
    'WorkspaceManager',
    'DatasetLibrary',
    'display_cloud_gpu_instructions',
]
