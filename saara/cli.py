"""
Command Line Interface for the Data Pipeline.

🪔 SAARA - ज्ञानस्य सारः
© 2024-2025 Kilani Sai Nikhil. All Rights Reserved.
"""


import typer
import sys
import os
import yaml
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from saara.splash import display_splash, display_animated_splash, display_goodbye, display_minimal_header

# Lazy import for DataPipeline to avoid loading heavy dependencies at startup
DataPipeline = None

def get_pipeline(config):
    """Lazy load DataPipeline."""
    global DataPipeline
    if DataPipeline is None:
        from saara.pipeline import DataPipeline as _DP
        DataPipeline = _DP
    return DataPipeline(config)

# Initialize Typer app
app = typer.Typer(
    name="saara",
    help="🧠 Saara - Autonomous Document-to-LLM Data Factory",
    add_completion=False,
    no_args_is_help=True
)
console = Console()

# --- Shared Wizards ---
# (Kept mostly as-is, just removed argparse logic)

def interactive_mode():
    """Run the interactive setup wizard."""
    # Display the beautiful animated Sanskrit splash screen with flickering flame
    display_animated_splash(duration=2.5)

    from rich.panel import Panel
    from rich.padding import Padding
    console.print(Padding(Panel("[bold]Welcome[/bold]\n[dim]Select a workflow below:[/dim]", border_style="dim", expand=False), (0, 0, 1, 20)))
    
    # Selection Mode with Table
    # Spacing handled by padding
    mode_table = Table(title="Choose Your Workflow", show_header=True, header_style="bold magenta")
    mode_table.add_column("Option", style="cyan", width=8)
    mode_table.add_column("Mode", style="green")
    mode_table.add_column("Description", style="dim")
    
    mode_table.add_row("1", "📄 Dataset Creation", "Extract data from PDFs → Generate training datasets")
    mode_table.add_row("2", "🧠 Model Training", "Fine-tune LLMs on your prepared data")
    mode_table.add_row("3", "🧪 Model Evaluation", "Test & improve trained models")
    mode_table.add_row("4", "🚀 Model Deployment", "Deploy models locally or to cloud")
    
    console.print(mode_table)
    console.print()
    
    mode_choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4"], default="1")
    
    if mode_choice == "2":
        run_training_wizard()
        return
    elif mode_choice == "3":
        run_evaluation_wizard()
        return
    elif mode_choice == "4":
        run_deployment_wizard()
        return

    # --- Comprehensive Dataset Creation Flow ---
    run_dataset_creation_wizard()


def run_dataset_creation_wizard():
    """Comprehensive dataset creation wizard with auto-detection and advanced options."""
    import requests
    
    console.print(Panel.fit(
        "[bold cyan]📄 Dataset Creation Wizard[/bold cyan]\n\n"
        "This wizard will guide you through creating high-quality training datasets from your PDFs.",
        title="Step 1: Configuration",
        border_style="cyan"
    ))
    
    # Step 1: Path Configuration
    console.print("\n[bold]📁 Step 1: Configure Paths[/bold]\n")
    
    base_dir = os.getcwd()
    raw_path = Prompt.ask(
        "Enter path to PDF files or folder",
        default=base_dir
    ).strip('"\'')
    
    raw_path_obj = Path(raw_path)
    if not raw_path_obj.exists():
        console.print(f"[red]❌ Path does not exist: {raw_path}[/red]")
        if not Confirm.ask("Continue anyway?", default=False):
            return
    else:
        # Count PDFs
        if raw_path_obj.is_dir():
            pdf_count = len(list(raw_path_obj.glob("**/*.pdf")))
            console.print(f"[green]✓ Found {pdf_count} PDF files in directory[/green]")
        else:
            console.print(f"[green]✓ Single file: {raw_path_obj.name}[/green]")
    
    output_path = Prompt.ask(
        "Enter output directory for datasets",
        default="./datasets"
    ).strip('"\'')
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓ Output directory: {output_path}[/green]")
    
    # Step 2: Auto-detect Ollama Models
    console.print("\n[bold]🔍 Step 2: Detecting Available Models[/bold]\n")
    
    available_models = []
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            models_data = response.json().get("models", [])
            available_models = [m["name"].split(":")[0] for m in models_data]
            available_models = list(set(available_models))  # Dedupe
            console.print(f"[green]✓ Ollama is running. Found {len(available_models)} models.[/green]")
        else:
            console.print("[yellow]⚠ Could not fetch Ollama models[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Ollama not running or unreachable: {e}[/red]")
        console.print("[dim]Start Ollama with: ollama serve[/dim]")
        if not Confirm.ask("Continue anyway?", default=False):
            return
    
    # Vision model selection
    console.print("\n[bold]👁️ Vision OCR Model:[/bold]")
    vision_models = {
        "1": ("moondream", "Moondream", "Fast, lightweight (~2GB VRAM)"),
        "2": ("qwen2.5vl", "Qwen2.5-VL", "High accuracy (~4GB VRAM)"),
    }
    
    v_table = Table(show_header=True, header_style="bold magenta")
    v_table.add_column("ID", style="cyan", width=4)
    v_table.add_column("Model", style="green")
    v_table.add_column("Description")
    v_table.add_column("Status", style="yellow")
    
    for key, (model_name, display_name, desc) in vision_models.items():
        status = "✓ Available" if model_name in available_models else "⚠ Not pulled"
        v_table.add_row(key, display_name, desc, status)
    
    console.print(v_table)
    v_choice = Prompt.ask("Choose vision model", choices=["1", "2"], default="1")
    vision_model = vision_models[v_choice][0]
    
    # Check if model needs to be pulled
    if vision_model not in available_models:
        console.print(f"[yellow]Model {vision_model} not found locally.[/yellow]")
        if Confirm.ask(f"Pull {vision_model} now?", default=True):
            console.print(f"[dim]Running: ollama pull {vision_model}[/dim]")
            os.system(f"ollama pull {vision_model}")
    
    # Analyzer model selection
    console.print("\n[bold]🧠 Analyzer/Labeling Model:[/bold]")
    analyzer_models = {
        "1": ("granite4", "Granite 4.0", "IBM enterprise model, balanced"),
        "2": ("llama3.2", "Llama 3.2", "Meta's latest, instruction-following"),
        "3": ("qwen2.5", "Qwen 2.5", "Alibaba, strong reasoning"),
        "4": ("mistral", "Mistral", "Fast, efficient"),
    }
    
    a_table = Table(show_header=True, header_style="bold magenta")
    a_table.add_column("ID", style="cyan", width=4)
    a_table.add_column("Model", style="green")
    a_table.add_column("Description")
    a_table.add_column("Status", style="yellow")
    
    for key, (model_name, display_name, desc) in analyzer_models.items():
        # Check both exact and partial matches
        is_available = any(model_name in m for m in available_models)
        status = "✓ Available" if is_available else "⚠ Not pulled"
        a_table.add_row(key, display_name, desc, status)
    
    console.print(a_table)
    a_choice = Prompt.ask("Choose analyzer model", choices=["1", "2", "3", "4"], default="1")
    analyzer_model = analyzer_models[a_choice][0]
    
    # Check if model needs to be pulled
    if not any(analyzer_model in m for m in available_models):
        console.print(f"[yellow]Model {analyzer_model} not found locally.[/yellow]")
        if Confirm.ask(f"Pull {analyzer_model} now?", default=True):
            console.print(f"[dim]Running: ollama pull {analyzer_model}[/dim]")
            os.system(f"ollama pull {analyzer_model}")
    
    # Step 3: Advanced Options
    console.print("\n[bold]⚙️ Step 3: Advanced Options[/bold]\n")
    
    show_advanced = Confirm.ask("Configure advanced options?", default=False)
    
    # Defaults
    chunk_size = 2500
    chunk_overlap = 600
    qa_per_chunk = 30
    generate_summaries = True
    generate_instructions = True
    dataset_name = "dataset"
    
    if show_advanced:
        dataset_name = Prompt.ask("Dataset name prefix", default="dataset")
        
        console.print("\n[dim]Chunking affects how documents are split for processing.[/dim]")
        chunk_size = int(Prompt.ask("Chunk size (characters)", default="2500"))
        chunk_overlap = int(Prompt.ask("Chunk overlap (characters)", default="600"))
        
        console.print("\n[dim]Generation settings affect output quality and speed.[/dim]")
        qa_per_chunk = int(Prompt.ask("Q&A pairs per chunk", default="30"))
        generate_summaries = Confirm.ask("Generate summaries?", default=True)
        generate_instructions = Confirm.ask("Generate instruction pairs?", default=True)
    
    # Step 4: Summary and Confirmation
    console.print("\n")
    summary_table = Table(title="📋 Configuration Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Setting", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Source Path", str(raw_path))
    summary_table.add_row("Output Directory", output_path)
    summary_table.add_row("Dataset Name", dataset_name)
    summary_table.add_row("Vision Model", vision_models[v_choice][1])
    summary_table.add_row("Analyzer Model", analyzer_models[a_choice][1])
    summary_table.add_row("Chunk Size", f"{chunk_size} chars")
    summary_table.add_row("Q&A per Chunk", str(qa_per_chunk))
    summary_table.add_row("Summaries", "Yes" if generate_summaries else "No")
    summary_table.add_row("Instructions", "Yes" if generate_instructions else "No")
    
    console.print(summary_table)
    console.print()
    
    if not Confirm.ask("[bold]Proceed with dataset creation?[/bold]", default=True):
        console.print("[yellow]Aborted by user.[/yellow]")
        return
    
    # Step 5: Run Pipeline
    console.print("\n[bold cyan]🚀 Starting Dataset Creation Pipeline...[/bold cyan]\n")
    
    # Build config
    config_path = "config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    
    # Apply all settings
    if 'pdf' not in config: config['pdf'] = {}
    if 'ollama' not in config: config['ollama'] = {}
    if 'output' not in config: config['output'] = {}
    if 'text' not in config: config['text'] = {}
    if 'labeling' not in config: config['labeling'] = {}
    
    config['pdf']['ocr_engine'] = vision_model
    config['ollama']['model'] = analyzer_model
    config['output']['directory'] = output_path
    config['text']['chunk_size'] = chunk_size
    config['text']['chunk_overlap'] = chunk_overlap
    config['labeling']['qa_per_chunk'] = qa_per_chunk
    config['labeling']['generate_summaries'] = generate_summaries
    config['labeling']['generate_instructions'] = generate_instructions
    
    # Initialize pipeline
    pipeline = get_pipeline(config)
    
    # Health check
    console.print("[dim]Checking pipeline health...[/dim]")
    if not pipeline.check_health():
        console.print("[red]❌ Health check failed. Please ensure Ollama is running with the selected models.[/red]")
        console.print(f"[dim]Try: ollama pull {analyzer_model}[/dim]")
        return
    
    # Process
    raw_path_obj = Path(raw_path)
    if raw_path_obj.is_file():
        result = pipeline.process_file(str(raw_path_obj), dataset_name)
    else:
        result = pipeline.process_directory(str(raw_path_obj), dataset_name)
    
    # Results
    if result.success:
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]✅ Dataset Creation Complete![/bold green]\n\n"
            f"Documents Processed: {result.documents_processed}\n"
            f"Total Chunks: {result.total_chunks}\n"
            f"Total Samples: {result.total_samples}\n"
            f"Duration: {result.duration_seconds:.1f}s",
            title="Success",
            border_style="green"
        ))
        
        console.print("\n[bold]📁 Generated Files:[/bold]")
        for dtype, files in result.output_files.items():
            if isinstance(files, dict):
                for fmt, fpath in files.items():
                    console.print(f"  • {dtype}/{fmt}: [cyan]{fpath}[/cyan]")
            else:
                console.print(f"  • {dtype}: [cyan]{files}[/cyan]")
        
        # Offer training
        console.print("\n")
        if Confirm.ask("Would you like to train a model on this dataset now?", default=False):
            # Find ShareGPT file
            sharegpt_file = f"{output_path}/{dataset_name}_sharegpt.jsonl"
            if not os.path.exists(sharegpt_file):
                sharegpt_files = list(Path(output_path).glob("*sharegpt*.jsonl"))
                if sharegpt_files:
                    sharegpt_file = str(sharegpt_files[0])
            
            run_training_wizard(default_data_path=sharegpt_file, config=config)
    else:
        console.print("\n[bold red]❌ Dataset creation failed[/bold red]")
        for error in result.errors:
            console.print(f"  • {error}")


def run_training_wizard(default_data_path: str = None, config: dict = None):
    """Run the interactive training setup."""
    console.print("\n[bold]Select Base Model to Train:[/bold]")
    t_table = Table(show_header=True, header_style="bold magenta")
    t_table.add_column("ID", style="cyan", width=4)
    t_table.add_column("Model", style="green")
    t_table.add_column("Type", style="yellow")
    
    t_table.add_row("1", "sarvamai/sarvam-1", "2B")
    t_table.add_row("2", "google/gemma-2b", "2B")
    t_table.add_row("3", "meta-llama/Llama-3.2-1B", "1B")
    t_table.add_row("4", "Qwen/Qwen2.5-7B", "7B")
    t_table.add_row("5", "mistralai/Mistral-7B-v0.1", "7B")
    t_table.add_row("6", "TinyLlama/TinyLlama-1.1B", "1.1B")
    t_table.add_row("7", "Other", "-")
    
    console.print(t_table)
    t_choice = Prompt.ask("Choose a base model", choices=["1", "2", "3", "4", "5", "6", "7"], default="2")
    
    model_id = "sarvamai/sarvam-1"
    if t_choice == "2":
        model_id = "google/gemma-2b"
    elif t_choice == "3":
        model_id = "meta-llama/Llama-3.2-1B"
    elif t_choice == "4":
        model_id = "Qwen/Qwen2.5-7B"
    elif t_choice == "5":
        model_id = "mistralai/Mistral-7B-v0.1"
    elif t_choice == "6":
        model_id = "TinyLlama/TinyLlama-1.1B"
    elif t_choice == "7":
        model_id = Prompt.ask("Enter HuggingFace Model ID (e.g. microsoft/phi-2)")
    
    console.print(f"[bold]Selected Model:[/bold] {model_id}")
    
    gated_models = ["google/gemma", "meta-llama/Llama-3", "mistralai/Mistral"]
    is_gated = any(gated in model_id for gated in gated_models)
    
    if is_gated:
        console.print("[yellow]⚠️ This model requires HuggingFace authentication.[/yellow]")
        if Confirm.ask("Do you want to login to HuggingFace now?", default=True):
            hf_token = Prompt.ask("Enter your HuggingFace token", password=True)
            try:
                from huggingface_hub import login
                login(token=hf_token)
                console.print("[green]✅ Successfully logged in![/green]")
            except Exception as e:
                console.print(f"[red]Login failed: {e}[/red]")
                return
    
    while True:
        if default_data_path:
            data_file = default_data_path
            default_data_path = None
        else:
            default_guess = "datasets/interactive_batch_sharegpt.jsonl"
            if not os.path.exists(default_guess):
                default_guess = "datasets/distilled_train.jsonl"
                
            data_file = Prompt.ask("Path to training dataset (.jsonl)", default=default_guess).strip('"\'')
            
        path_obj = Path(data_file)
        
        if path_obj.is_dir():
            jsonl_files = list(path_obj.glob("*.jsonl"))
            if jsonl_files:
                console.print(f"[green]Found {len(jsonl_files)} JSONL files.[/green]")
                
                sharegpt_files = [f for f in jsonl_files if 'sharegpt' in f.name.lower()]
                instruction_files = [f for f in jsonl_files if 'instruction' in f.name.lower()]
                qa_files = [f for f in jsonl_files if '_qa' in f.name.lower()]
                
                console.print("\n[bold]Select dataset type:[/bold]")
                console.print("  1. ShareGPT (Chat)")
                console.print("  2. Instruction")
                console.print("  3. Q&A")
                console.print("  4. All files")
                
                type_choice = Prompt.ask("Select type", choices=["1", "2", "3", "4"], default="1")
                
                if type_choice == "1":
                    selected_files = sharegpt_files
                elif type_choice == "2":
                    selected_files = instruction_files
                elif type_choice == "3":
                    selected_files = qa_files
                else:
                    selected_files = jsonl_files
                
                if not selected_files:
                    console.print("[red]No files of selected type found.[/red]")
                    continue
                
                data_file = [str(f) for f in selected_files]
                break
            else:
                console.print("[red]No .jsonl files found.[/red]")
                continue
        elif not path_obj.exists():
             console.print(f"[red]File or directory not found: {data_file}[/red]")
             default_data_path = None
             if not Confirm.ask("Try again?", default=True):
                 return
        else:
            break
        
    resume_path = None
    if Confirm.ask("Do you want to resume from a checkpoint?", default=False):
        resume_path = Prompt.ask("Enter path to checkpoint directory").strip('"\'')
    
    from saara.train import LLMTrainer
    
    if not config:
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

    trainer = LLMTrainer(model_id=model_id, config=config)
    try:
        trainer.train(data_file, resume_from_checkpoint=resume_path)
    except Exception as e:
        console.print(f"[bold red]Training failed:[/bold red] {e}")


def run_evaluation_wizard(config: dict = None):
    """Run the model evaluation wizard."""
    console.print(Panel.fit(
        "[bold cyan]🧪 Model Evaluation[/bold cyan]\n\n"
        "Test your fine-tuned model using Granite 4 as a judge.",
        title="Evaluation Mode",
        border_style="cyan"
    ))
    
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[red]No models directory found. Please train a model first.[/red]")
        return
    
    finetuned_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            adapter_path = model_dir / "final_adapter"
            if adapter_path.exists():
                finetuned_models.append({
                    "name": model_dir.name,
                    "path": str(adapter_path),
                })
    
    if not finetuned_models:
        console.print("[yellow]No fine-tuned models found.[/yellow]")
        return
    
    console.print("\n[bold]Available Models:[/bold]\n")
    for i, m in enumerate(finetuned_models, 1):
        console.print(f" {i}. {m['name']}")
    
    choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(finetuned_models)+1)], default="1")
    selected = finetuned_models[int(choice)-1]
    
    base_model = Prompt.ask("Enter base model ID", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    console.print("\n[bold]Select Mode:[/bold]")
    console.print("1. Standard Evaluation")
    console.print("2. Autonomous Learning")
    mode_choice = Prompt.ask("Select mode", choices=["1", "2"], default="1")
    
    from saara.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(config)
    
    if mode_choice == "1":
        num_samples = int(Prompt.ask("Number of test samples", default="10"))
        evaluator.evaluate_adapter(base_model, selected["path"], num_samples=num_samples)
    else:
        topic = Prompt.ask("Enter topic to learn about")
        iterations = int(Prompt.ask("Learning iterations", default="10"))
        
        # --- Teacher Model Selection ---
        console.print("\n[bold cyan]Select Teacher Model:[/bold cyan]")
        console.print("[1] Ollama (Local) - granite3.2-vision, llama3.2, qwen2.5, etc.")
        console.print("[2] OpenAI API - GPT-4o, GPT-4-turbo")
        console.print("[3] Google AI - Gemini 1.5 Flash/Pro")
        console.print("[4] DeepSeek API")
        console.print("[5] HuggingFace Inference API")
        
        provider_choice = Prompt.ask("Select provider", choices=["1", "2", "3", "4", "5"], default="1")
        
        teacher_config = {}
        
        if provider_choice == "1":
            # Ollama - list available models
            teacher_config["provider"] = "ollama"
            try:
                import ollama
                models_list = ollama.list()
                available = [m.model for m in models_list.models] if hasattr(models_list, 'models') else []
                if available:
                    console.print(f"[dim]Available: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}[/dim]")
            except:
                pass
            teacher_config["model"] = Prompt.ask("Enter Ollama model name", default="granite4:latest")
            
        elif provider_choice == "2":
            teacher_config["provider"] = "openai"
            teacher_config["api_key"] = Prompt.ask("Enter OpenAI API Key", password=True)
            teacher_config["model"] = Prompt.ask("Model name", default="gpt-4o-mini")
            
        elif provider_choice == "3":
            teacher_config["provider"] = "google"
            teacher_config["api_key"] = Prompt.ask("Enter Google AI API Key", password=True)
            teacher_config["model"] = Prompt.ask("Model name", default="gemini-1.5-flash")
            
        elif provider_choice == "4":
            teacher_config["provider"] = "deepseek"
            teacher_config["api_key"] = Prompt.ask("Enter DeepSeek API Key", password=True)
            teacher_config["base_url"] = "https://api.deepseek.com"
            teacher_config["model"] = Prompt.ask("Model name", default="deepseek-chat")
            
        else:
            teacher_config["provider"] = "huggingface"
            teacher_config["api_key"] = Prompt.ask("Enter HuggingFace Token", password=True)
            teacher_config["model"] = Prompt.ask("Model ID", default="meta-llama/Llama-3.3-70B-Instruct")
        
        evaluator.run_autonomous_learning(base_model, selected["path"], topic, num_iterations=iterations, teacher_config=teacher_config)


def run_deployment_wizard(config: dict = None):
    """Run the model deployment wizard."""
    from rich.panel import Panel
    from rich.prompt import Prompt
    from pathlib import Path
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Model Deployment[/bold cyan]\n"
        "[dim]Deploy models locally, to cloud, or export formats.[/dim]",
        title="Deployment Mode",
        border_style="green"
    ))
    
    # Lazy import to avoid startup lag
    from saara.deployer import ModelDeployer
    
    models_dir = Path("models")
    finetuned_models = []
    
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                adapter_path = model_dir / "final_adapter"
                if adapter_path.exists():
                    finetuned_models.append({
                        "name": model_dir.name,
                        "path": str(adapter_path),
                    })
    
    if not finetuned_models:
        console.print("[yellow]No fine-tuned models found. Please train a model first.[/yellow]")
        return
    
    console.print("\n[bold]Select Model to Deploy:[/bold]\n")
    for i, m in enumerate(finetuned_models, 1):
        console.print(f" {i}. {m['name']}")
        
    choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(finetuned_models)+1)], default="1")
    selected = finetuned_models[int(choice)-1]
    
    base_model = Prompt.ask("Enter base model ID", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    deployer = ModelDeployer(config)
    deployer.deploy_menu(base_model, selected["path"])


# --- Typer Commands ---

@app.command()
def run():
    """Start the interactive setup wizard."""
    interactive_mode()


@app.command()
def wizard():
    """Start the interactive setup wizard (alias for 'run')."""
    interactive_mode()


@app.command()
def version():
    """Show SAARA version and copyright information."""
    from saara import __version__, __copyright__, __license__
    from saara.splash import display_version
    
    # Display styled version info
    display_version()
    
    # Additional system info
    console.print(f"[dim]Python: {sys.version.split()[0]}[/dim]")
    console.print(f"[dim]License: {__license__}[/dim]")
    console.print()

@app.command()
def process(
    file: str = typer.Argument(..., help="Path to PDF file"),
    name: str = typer.Option(None, "--name", "-n", help="Dataset name"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    Process a single PDF file.
    """
    if not Path(file).exists():
        console.print(f"[red]Error: File not found: {file}[/red]")
        raise typer.Exit(code=1)
        
    pipeline = get_pipeline(config)
    if not pipeline.check_health():
        console.print("[red]Health check failed. Ensure Ollama is running.[/red]")
        raise typer.Exit(code=1)
    
    result = pipeline.process_file(file, name)
    if result.success:
        console.print(f"\n[bold green]✅ Success![/bold green] Processed in {result.duration_seconds:.1f}s")
        console.print(f"   Total samples generated: {result.total_samples}")
    else:
        console.print(f"\n[bold red]❌ Failed[/bold red]")
        for error in result.errors:
            console.print(f"   • {error}")
        raise typer.Exit(code=1)


@app.command()
def batch(
    directory: str = typer.Argument(..., help="Directory containing PDFs"),
    name: str = typer.Option("dataset", "--name", "-n", help="Dataset name"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    Process all PDFs in a directory.
    """
    if not Path(directory).is_dir():
        console.print(f"[red]Error: Directory not found: {directory}[/red]")
        raise typer.Exit(code=1)
        
    pipeline = get_pipeline(config)
    if not pipeline.check_health():
        console.print("[red]Health check failed. Ensure Ollama is running.[/red]")
        raise typer.Exit(code=1)
    
    result = pipeline.process_directory(directory, name)
    if result.success:
        console.print(f"\n[bold green]✅ Success![/bold green] Processed {result.documents_processed} docs in {result.duration_seconds:.1f}s")
        console.print(f"   Total samples generated: {result.total_samples}")
    else:
        console.print(f"\n[bold red]❌ Failed[/bold red]")
        for error in result.errors:
            console.print(f"   • {error}")
        raise typer.Exit(code=1)


@app.command()
def health(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    Check pipeline health (Ollama connection).
    """
    pipeline = get_pipeline(config)
    healthy = pipeline.check_health()
    raise typer.Exit(code=0 if healthy else 1)


@app.command()
def serve(
    host: str = typer.Option('0.0.0.0', help='Host to bind to'),
    port: int = typer.Option(8000, "--port", "-p", help='Port to bind to'),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    Start the Saara web interface.
    """
    console.print(f"[bold cyan]Starting Saara web interface on http://{host}:{port}[/bold cyan]")
    import uvicorn
    uvicorn.run("saara.api:app", host=host, port=port, reload=True)


@app.command()
def distill(
    input_path: str = typer.Argument(None, help="Path to input file (markdown/text) or directory"),
    output: str = typer.Option("datasets/synthetic", "--output", "-o", help="Output directory"),
    data_type: str = typer.Option("all", "--type", "-t", help="Data type: factual, reasoning, conversational, instruction, all"),
    pairs: int = typer.Option(3, "--pairs", "-p", help="Pairs per type per chunk"),
    clean: bool = typer.Option(True, "--clean/--no-clean", help="Enable text sanitization"),
    filter_quality: bool = typer.Option(True, "--filter/--no-filter", help="Enable quality filtering"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    Distill text into high-quality training data.
    
    Generates synthetic training samples with:
    - Text sanitization (removes OCR artifacts)
    - Semantic chunking (by headers)
    - Multi-type generation (factual, reasoning, conversational)
    - Quality filtering (removes low-quality samples)
    
    Examples:
        saara distill document.md --type reasoning
        saara distill ./texts --pairs 5 --output ./my_dataset
    """
    from saara.cleaner import TextCleaner, SemanticChunker
    from saara.synthetic_generator import SyntheticDataGenerator, DataType, QualityJudge
    import json
    
    console.print(Panel.fit(
        "[bold cyan]🔬 Synthetic Data Generation[/bold cyan]\n\n"
        "Creating high-quality training data with sanitization and quality control.",
        title="Distillation Pipeline",
        border_style="cyan"
    ))
    
    # Load config
    cfg = {}
    if os.path.exists(config):
        with open(config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    
    # Determine input
    if not input_path:
        # Interactive mode - ask for input
        input_path = Prompt.ask("Enter path to input file or directory").strip('"\'')
    
    input_obj = Path(input_path)
    if not input_obj.exists():
        console.print(f"[red]❌ Input path not found: {input_path}[/red]")
        raise typer.Exit(code=1)
    
    # Collect input files
    if input_obj.is_file():
        input_files = [input_obj]
    else:
        input_files = list(input_obj.glob("**/*.md")) + list(input_obj.glob("**/*.txt"))
        console.print(f"[green]Found {len(input_files)} text files[/green]")
    
    if not input_files:
        console.print("[red]No input files found[/red]")
        raise typer.Exit(code=1)
    
    # Initialize components
    cleaner = TextCleaner(cfg) if clean else None
    chunker = SemanticChunker(cfg)
    generator = SyntheticDataGenerator(cfg)
    
    # Determine data types
    type_map = {
        "factual": [DataType.FACTUAL],
        "reasoning": [DataType.REASONING],
        "conversational": [DataType.CONVERSATIONAL],
        "instruction": [DataType.INSTRUCTION],
        "all": [DataType.FACTUAL, DataType.REASONING, DataType.CONVERSATIONAL],
    }
    selected_types = type_map.get(data_type.lower(), [DataType.ALL])
    
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Data types: {[t.value for t in selected_types]}")
    console.print(f"  Pairs per type: {pairs}")
    console.print(f"  Sanitization: {'Enabled' if clean else 'Disabled'}")
    console.print(f"  Quality filter: {'Enabled' if filter_quality else 'Disabled'}")
    console.print()
    
    # Process
    all_samples = []
    total_generated = 0
    total_passed = 0
    total_rejected = 0
    rejection_stats = {}
    
    from tqdm import tqdm
    
    for file_path in tqdm(input_files, desc="Processing files"):
        console.print(f"\n[dim]Processing: {file_path.name}[/dim]")
        
        # Read file
        text = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Step 1: Sanitize
        if cleaner:
            result = cleaner.clean(text)
            text = result.cleaned
            if result.removed_phrases:
                console.print(f"  [dim]Removed {len(result.removed_phrases)} filler phrases[/dim]")
        
        # Step 2: Chunk
        chunks = chunker.chunk_by_headers(text)
        console.print(f"  [dim]Created {len(chunks)} semantic chunks[/dim]")
        
        # Step 3: Generate
        for chunk in chunks:
            gen_result = generator.generate(
                chunk['content'],
                data_types=selected_types,
                pairs_per_type=pairs
            )
            
            all_samples.extend(gen_result.samples)
            total_generated += gen_result.total_generated
            total_passed += gen_result.total_passed
            total_rejected += gen_result.total_rejected
            
            for reason, count in gen_result.rejection_stats.items():
                rejection_stats[reason] = rejection_stats.get(reason, 0) + count
    
    # Save results
    Path(output).mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL (Alpaca format)
    alpaca_path = Path(output) / "synthetic_alpaca.jsonl"
    with open(alpaca_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            entry = {
                "instruction": sample.instruction,
                "input": sample.input_context,
                "output": sample.output,
                "type": sample.data_type
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save as ShareGPT format
    sharegpt_path = Path(output) / "synthetic_sharegpt.jsonl"
    with open(sharegpt_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            entry = {
                "conversations": [
                    {"from": "human", "value": sample.instruction},
                    {"from": "gpt", "value": sample.output}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Summary
    console.print("\n")
    summary = Table(title="📊 Distillation Results", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    
    summary.add_row("Files Processed", str(len(input_files)))
    summary.add_row("Total Generated", str(total_generated))
    summary.add_row("Passed Quality Filter", str(total_passed))
    summary.add_row("Rejected", str(total_rejected))
    summary.add_row("Pass Rate", f"{(total_passed/max(total_generated,1))*100:.1f}%")
    
    console.print(summary)
    
    if rejection_stats:
        console.print("\n[bold]Rejection Reasons:[/bold]")
        for reason, count in sorted(rejection_stats.items(), key=lambda x: -x[1]):
            console.print(f"  • {reason}: {count}")
    
    console.print(f"\n[bold green]✅ Output saved to:[/bold green]")
    console.print(f"  • Alpaca format: [cyan]{alpaca_path}[/cyan]")
    console.print(f"  • ShareGPT format: [cyan]{sharegpt_path}[/cyan]")




@app.command()
def train(
    data: Annotated[Optional[str], typer.Option("--data", "-d", help="Path to training data (jsonl)")] = None,
    model: str = typer.Option(None, "--model", "-m", help='Base model ID'),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    Fine-tune model using SFT.
    """


    from saara.train import LLMTrainer
    from rich.prompt import Prompt
    from pathlib import Path
    import json
    
    # --- 1. Select Data ---
    if not data:
        dataset_dir = Path("datasets")
        if not dataset_dir.exists():
            dataset_dir.mkdir(exist_ok=True)
            
        candidates = list(dataset_dir.glob("*.jsonl"))
        
        console.print(Panel.fit("[bold cyan]Dataset Selection[/bold cyan]", border_style="cyan"))
        
        console.print("[0] 🔄 [bold yellow]Merge ALL files[/bold yellow] in ./datasets into one")
        for i, f in enumerate(candidates, 1):
             console.print(f"[{i}] {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        console.print(f"[{len(candidates)+1}] 📂 Select Custom File Path")
             
        choice = Prompt.ask("Choose option", choices=[str(i) for i in range(0, len(candidates)+2)], default="0")
        
        if choice == "0":
            if not candidates:
                console.print("[red]No files to merge.[/red]")
                raise typer.Exit(1)
                
            merged_data = []
            console.print(f"[dim]Merging {len(candidates)} files...[/dim]")
            for f in candidates:
                with open(f, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            try:
                                merged_data.append(json.loads(line))
                            except: pass
            
            merged_path = dataset_dir / "merged_training_data.jsonl"
            with open(merged_path, 'w', encoding='utf-8') as outfile:
                for entry in merged_data:
                    outfile.write(json.dumps(entry) + "\n")
            
            console.print(f"[green]✓ Merged {len(merged_data)} samples into {merged_path}[/green]\n")
            data = str(merged_path)
            
        elif choice == str(len(candidates)+1):
            data = Prompt.ask("Enter absolute path to .jsonl file")
            if not Path(data).exists():
                console.print(f"[red]File not found: {data}[/red]")
                raise typer.Exit(1)
        else:
            data = str(candidates[int(choice)-1])
            console.print(f"[green]Selected: {data}[/green]\n")

    # --- 2. Select Model ---
    if not model:
        # Curated list for consumer hardware
        models = [
            {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama 1.1B (Fastest, 2GB VRAM)"},
            {"id": "sarvamai/sarvam-1", "name": "Sarvam-1 (Good for Hindi/English, 2GB VRAM)"},
            {"id": "unsloth/llama-3-8b-Instruct-bnb-4bit", "name": "Llama 3 8B (4-bit, High Quality, 6GB VRAM)"},
            {"id": "mistralai/Mistral-7B-Instruct-v0.2", "name": "Mistral 7B (Solid Performer)"},
            {"id": "custom", "name": "Enter Custom Model ID"}
        ]
        
        console.print(Panel.fit("[bold cyan]Select Base Model[/bold cyan]", border_style="cyan"))
        for i, m in enumerate(models, 1):
            console.print(f"[{i}] [bold]{m['name']}[/bold]\n    [dim]{m['id']}[/dim]")
            
        choice = Prompt.ask("Choose base model", choices=[str(i) for i in range(1, len(models)+1)], default="1")
        
        selection = models[int(choice)-1]
        if selection["id"] == "custom":
            model = Prompt.ask("Enter HuggingFace Model ID")
        else:
            model = selection["id"]
        console.print(f"[green]Selected: {model}[/green]\n")

    # --- 3. Train ---
    config_obj = get_pipeline(config).config
    trainer = LLMTrainer(model_id=model, config=config_obj)
    trainer.train(data)
    
    # --- 4. Post-Training ---
    console.print(Panel.fit(
        "[bold green]🎉 Training Complete![/bold green]\n\n"
        "Your model is ready. What next?\n"
        "👉 Run [bold cyan]saara deploy[/bold cyan] to test or deploy it.",
        title="Next Steps",
        border_style="green"
    ))


@app.command()
def deploy():
    """
    Launch the Model Deployment Wizard (Local Chat, Cloud, Ollama Export).
    """
    run_deployment_wizard()


@app.command()
def evaluate(
    base_model: str = typer.Argument(..., help="Base model ID (e.g. TinyLlama/...)"),
    adapter_path: str = typer.Argument(..., help="Path to adapter checkpoint"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    Evaluate a fine-tuned model using Granite as a judge.
    """
    from saara.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(config)
    evaluator.evaluate_adapter(base_model, adapter_path)


# ============================================================================
# SETUP & MODEL MANAGEMENT COMMANDS
# ============================================================================

@app.command()
def setup():
    """
    First-time setup wizard. Detects hardware, recommends models, and installs them.
    """
    from saara.model_manager import HardwareDetector, ModelManager, MODEL_CATALOG
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Saara Setup Wizard[/bold cyan]\n\n"
        "Welcome! This wizard will help you set up Saara for your system.\n"
        "[dim]We'll detect your hardware and recommend optimal models.[/dim]",
        title="Setup",
        border_style="cyan"
    ))
    
    # Step 1: Check dependencies
    console.print("\n[bold]📦 Step 1: Checking Dependencies[/bold]\n")
    
    # Check Ollama
    manager = ModelManager()
    if not manager.check_ollama_running():
        console.print("[yellow]⚠ Ollama is not running.[/yellow]")
        console.print("[dim]Attempting to start Ollama...[/dim]")
        
        if manager.start_ollama():
            console.print("[green]✓ Ollama started successfully![/green]")
        else:
            console.print("[red]❌ Could not start Ollama.[/red]")
            console.print("\n[bold]Please install Ollama first:[/bold]")
            console.print("  1. Download from: [cyan]https://ollama.ai[/cyan]")
            console.print("  2. Install and run: [cyan]ollama serve[/cyan]")
            console.print("  3. Then run: [cyan]saara setup[/cyan] again")
            return
    else:
        console.print("[green]✓ Ollama is running[/green]")
    
    # Step 2: Detect hardware
    console.print("\n[bold]💻 Step 2: Detecting Hardware[/bold]\n")
    
    hardware = HardwareDetector.get_system_info()
    HardwareDetector.display_hardware_info(hardware)
    
    tier = HardwareDetector.get_recommended_tier(hardware)
    tier_names = {"minimal": "Lightweight", "light": "Light", "medium": "Medium", "heavy": "Full"}
    
    console.print(f"\n[bold]Recommended tier:[/bold] [cyan]{tier_names.get(tier, tier)}[/cyan]")
    
    # Step 3: Select Vision Model
    console.print("\n[bold]👁️ Step 3: Select Vision Model[/bold]\n")
    console.print("[dim]Vision models extract text from images/PDFs.[/dim]\n")
    
    vision_models = manager.get_model_catalog("vision", tier)
    manager.display_models("vision", tier)
    
    console.print()
    v_choices = [str(i) for i in range(1, len(vision_models) + 1)]
    v_choice = Prompt.ask("Select vision model", choices=v_choices + ["skip"], default="1")
    
    selected_vision = None
    if v_choice != "skip":
        selected_vision = vision_models[int(v_choice) - 1]
        if not selected_vision.is_installed:
            console.print(f"\n[cyan]Installing {selected_vision.display_name}...[/cyan]")
            if manager.install_model(selected_vision.name):
                console.print(f"[green]✓ {selected_vision.display_name} installed![/green]")
            else:
                console.print(f"[red]Failed to install {selected_vision.display_name}[/red]")
        else:
            console.print(f"[green]✓ {selected_vision.display_name} already installed[/green]")
    
    # Step 4: Select Analyzer Model
    console.print("\n[bold]🧠 Step 4: Select Analyzer Model[/bold]\n")
    console.print("[dim]Analyzer models generate training data from text.[/dim]\n")
    
    analyzer_models = manager.get_model_catalog("analyzer", tier)
    manager.display_models("analyzer", tier)
    
    console.print()
    a_choices = [str(i) for i in range(1, len(analyzer_models) + 1)]
    a_choice = Prompt.ask("Select analyzer model", choices=a_choices + ["skip"], default="1")
    
    selected_analyzer = None
    if a_choice != "skip":
        selected_analyzer = analyzer_models[int(a_choice) - 1]
        if not selected_analyzer.is_installed:
            console.print(f"\n[cyan]Installing {selected_analyzer.display_name}...[/cyan]")
            if manager.install_model(selected_analyzer.name):
                console.print(f"[green]✓ {selected_analyzer.display_name} installed![/green]")
            else:
                console.print(f"[red]Failed to install {selected_analyzer.display_name}[/red]")
        else:
            console.print(f"[green]✓ {selected_analyzer.display_name} already installed[/green]")
    
    # Step 5: Save configuration
    console.print("\n[bold]💾 Step 5: Saving Configuration[/bold]\n")
    
    config = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": selected_analyzer.name if selected_analyzer else "granite3.1-dense:8b"
        },
        "pdf": {
            "ocr_engine": selected_vision.name.split(":")[0] if selected_vision else "moondream"
        },
        "output": {
            "directory": "datasets"
        },
        "hardware": {
            "tier": tier,
            "vram_gb": hardware.get("vram_gb", 0),
            "ram_gb": hardware.get("ram_gb", 0)
        }
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print("[green]✓ Configuration saved to config.yaml[/green]")
    
    # Done!
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✅ Setup Complete![/bold green]\n\n"
        "You're ready to use Saara!\n\n"
        "[bold]Quick Start:[/bold]\n"
        "  saara run              - Interactive wizard\n"
        "  saara process doc.pdf  - Process a PDF\n"
        "  saara distill text.md  - Generate training data\n"
        "  saara train            - Fine-tune a model\n\n"
        "[bold]Model Management:[/bold]\n"
        "  saara models list      - Show all models\n"
        "  saara models install   - Install a model\n"
        "  saara models remove    - Uninstall a model",
        title="🎉 Ready!",
        border_style="green"
    ))


# Create models subcommand group
models_app = typer.Typer(help="Manage Ollama and fine-tuned models")
app.add_typer(models_app, name="models")


@models_app.callback(invoke_without_command=True)
def models_callback(ctx: typer.Context):
    """Manage Ollama and fine-tuned models."""
    if ctx.invoked_subcommand is None:
        # Default to listing models
        from saara.model_manager import ModelManager, TrainedModelManager
        manager = ModelManager()
        trained = TrainedModelManager()
        
        if not manager.check_ollama_running():
            console.print("[yellow]⚠ Ollama is not running.[/yellow]\n")
        
        manager.display_models("vision")
        manager.display_models("analyzer")
        console.print("\n")
        trained.display_trained_models()


@models_app.command("list")
def models_list(
    category: str = typer.Option(None, "--category", "-c", help="Filter: vision, analyzer"),
    installed_only: bool = typer.Option(False, "--installed", "-i", help="Show only installed models")
):
    """
    List available and installed models.
    """
    from saara.model_manager import ModelManager, TrainedModelManager
    
    console.print(Panel.fit(
        "[bold cyan]📋 Model Inventory[/bold cyan]",
        border_style="cyan"
    ))
    
    manager = ModelManager()
    
    # Check Ollama
    if not manager.check_ollama_running():
        console.print("[yellow]⚠ Ollama is not running. Install status may be inaccurate.[/yellow]\n")
    
    # Display Ollama models
    if category:
        manager.display_models(category)
    else:
        manager.display_models("vision")
        manager.display_models("analyzer")
    
    # Display trained models
    console.print("\n")
    trained = TrainedModelManager()
    trained.display_trained_models()


@models_app.command("install")
def models_install(
    model_name: str = typer.Argument(None, help="Model name to install (e.g., moondream, llama3.2:3b)")
):
    """
    Install an Ollama model.
    
    Examples:
        saara models install moondream
        saara models install llama3.2:3b
        saara models install qwen2.5vl:7b
    """
    from saara.model_manager import ModelManager, MODEL_CATALOG
    
    manager = ModelManager()
    
    if not manager.check_ollama_running():
        console.print("[red]❌ Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(code=1)
    
    if not model_name:
        # Interactive mode - show catalog and let user choose
        console.print("[bold]Available Models:[/bold]\n")
        
        all_models = []
        for cat in ["vision", "analyzer"]:
            models = manager.get_model_catalog(cat)
            all_models.extend(models)
        
        manager.display_models()
        
        # Let user pick
        console.print()
        model_name = Prompt.ask("Enter model name to install (e.g., moondream)")
    
    console.print(f"\n[cyan]Installing {model_name}...[/cyan]")
    
    if manager.install_model(model_name):
        console.print(f"\n[bold green]✅ Successfully installed {model_name}[/bold green]")
    else:
        console.print(f"\n[bold red]❌ Failed to install {model_name}[/bold red]")
        raise typer.Exit(code=1)


@models_app.command("remove")
def models_remove(
    model_name: str = typer.Argument(None, help="Model name to remove")
):
    """
    Remove/uninstall an Ollama model.
    
    Examples:
        saara models remove moondream
        saara models remove llama3.2:3b
    """
    from saara.model_manager import ModelManager, TrainedModelManager
    
    manager = ModelManager()
    trained = TrainedModelManager()
    
    if not model_name:
        # Interactive mode
        installed = manager.get_installed_models()
        trained_models = trained.list_trained_models()
        
        if not installed and not trained_models:
            console.print("[yellow]No models installed.[/yellow]")
            return
        
        console.print("[bold]Installed Ollama Models:[/bold]")
        for i, m in enumerate(installed, 1):
            console.print(f"  {i}. {m}")
        
        if trained_models:
            console.print("\n[bold]Fine-tuned Models:[/bold]")
            for i, m in enumerate(trained_models, len(installed) + 1):
                console.print(f"  {i}. {m['name']} (trained)")
        
        console.print()
        model_name = Prompt.ask("Enter model name to remove")
    
    # Check if it's a trained model
    trained_models = [m["name"] for m in trained.list_trained_models()]
    
    if model_name in trained_models:
        if Confirm.ask(f"Delete fine-tuned model '{model_name}'?", default=False):
            if trained.delete_trained_model(model_name):
                console.print(f"[green]✓ Deleted trained model: {model_name}[/green]")
            else:
                console.print(f"[red]Failed to delete: {model_name}[/red]")
    else:
        # Ollama model
        if Confirm.ask(f"Remove Ollama model '{model_name}'?", default=False):
            if manager.uninstall_model(model_name):
                console.print(f"[green]✓ Removed: {model_name}[/green]")
            else:
                console.print(f"[red]Failed to remove: {model_name}[/red]")


@models_app.command("status")
def models_status():
    """
    Show status of installed models and disk usage.
    """
    from saara.model_manager import ModelManager, TrainedModelManager, HardwareDetector
    
    console.print(Panel.fit(
        "[bold cyan]📊 Models Status[/bold cyan]",
        border_style="cyan"
    ))
    
    # Hardware info
    hardware = HardwareDetector.get_system_info()
    HardwareDetector.display_hardware_info(hardware)
    
    console.print()
    
    # Ollama status
    manager = ModelManager()
    if manager.check_ollama_running():
        console.print("[green]✓ Ollama: Running[/green]")
        installed = manager.get_installed_models()
        console.print(f"  Installed models: {len(installed)}")
        for m in installed:
            console.print(f"    • {m}")
    else:
        console.print("[red]✗ Ollama: Not running[/red]")
    
    console.print()
    
    # Trained models
    trained = TrainedModelManager()
    trained_models = trained.list_trained_models()
    
    console.print(f"[bold]Fine-tuned Models:[/bold] {len(trained_models)}")
    total_size = 0
    for m in trained_models:
        console.print(f"  • {m['name']} ({m['size_mb']:.1f} MB)")
        total_size += m['size_mb']
    
    if trained_models:
        console.print(f"  [dim]Total: {total_size:.1f} MB[/dim]")


def main():
    """Main entry point."""
    app()

if __name__ == "__main__":
    main()


