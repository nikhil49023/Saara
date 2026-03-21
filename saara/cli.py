"""
Command Line Interface for the Data Pipeline.

🪔 SAARA - ज्ञानस्य सारः
© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
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
from saara.model_manager import TrainedModelManager
from saara.deployer import run_deployment_wizard

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
    mode_table.add_row("5", "🏗️ Pre-training", "Build & train a model from scratch")
    
    console.print(mode_table)
    console.print()
    
    mode_choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"], default="1")
    
    if mode_choice == "2":
        run_training_wizard()
        return
    elif mode_choice == "3":
        run_evaluation_wizard()
        return
    elif mode_choice == "4":
        run_deployment_wizard()
        return
    elif mode_choice == "5":
        run_pretrain_wizard()
        return

    # --- Comprehensive Dataset Creation Flow ---
    run_dataset_creation_wizard()


def tokenizer_selection_wizard() -> dict:
    """
    Interactive wizard for selecting tokenizer type.
    
    Returns dict with:
        - type: "default", "custom", "ai", "huggingface"
        - path: path to tokenizer (if applicable)
        - config: additional config (for AI tokenizer)
    """
    from rich.panel import Panel
    from pathlib import Path
    
    console.print("\n[bold]🔤 Tokenizer Selection[/bold]\n")
    
    # List available custom tokenizers
    tokenizers_dir = Path("tokenizers")
    custom_tokenizers = []
    
    if tokenizers_dir.exists():
        for tok_dir in tokenizers_dir.iterdir():
            if tok_dir.is_dir() and (tok_dir / "vocab.json").exists():
                custom_tokenizers.append(tok_dir.name)
    
    console.print("  1. 🦙 Default LLaMA Tokenizer")
    console.print("  2. 🤗 HuggingFace Tokenizer (by name/path)")
    console.print("  [bold green]3. 🤖 Train NEW AI-Enhanced Tokenizer[/bold green]")
    
    if custom_tokenizers:
        console.print(f"  4. 📁 Use Existing Custom Tokenizer ({len(custom_tokenizers)} available)")
        choices = ["1", "2", "3", "4"]
    else:
        choices = ["1", "2", "3"]
    
    tok_choice = Prompt.ask("Select tokenizer", choices=choices, default="1")
    
    result = {"type": "default", "path": None, "config": {}}
    
    if tok_choice == "1":
        result["type"] = "default"
        console.print("[green]Using default LLaMA tokenizer[/green]")
        
    elif tok_choice == "2":
        hf_name = Prompt.ask("Enter HuggingFace tokenizer name/path", default="meta-llama/Llama-2-7b-hf")
        result["type"] = "huggingface"
        result["path"] = hf_name
        console.print(f"[green]Using HuggingFace tokenizer: {hf_name}[/green]")
        
    elif tok_choice == "3":
        # Train new AI tokenizer
        console.print(Panel(
            "[bold cyan]🤖 AI-Enhanced Tokenizer[/bold cyan]\n\n"
            "Train a custom tokenizer that uses AI to:\n"
            "• Extract domain-specific vocabulary\n"
            "• Protect medical/legal/scientific terms from being split\n"
            "• Optimize BPE merges for semantic coherence",
            title="AI Tokenizer",
            border_style="cyan"
        ))
        
        # Domain selection
        console.print("\n[bold]Select domain for vocabulary optimization:[/bold]")
        console.print("  1. General (default)")
        console.print("  2. Medical/Healthcare")
        console.print("  3. Legal")
        console.print("  4. Code/Programming")
        console.print("  5. Scientific")
        
        domain_choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5"], default="1")
        domain_map = {"1": "general", "2": "medical", "3": "legal", "4": "code", "5": "scientific"}
        domain = domain_map[domain_choice]
        
        # Vocab size
        vocab_size = int(Prompt.ask("Vocabulary size", default="32000"))
        
        # AI provider selection
        console.print("\n[bold]Select AI provider for vocabulary extraction:[/bold]")
        console.print("  1. Auto-detect (recommended)")
        console.print("  2. Ollama (local)")
        console.print("  3. Gemini API (cloud)")
        console.print("  4. OpenAI API (cloud)")
        console.print("  5. No AI (rule-based only)")
        
        provider_choice = Prompt.ask("Provider", choices=["1", "2", "3", "4", "5"], default="1")
        provider_map = {"1": "auto", "2": "ollama", "3": "gemini", "4": "openai", "5": "none"}
        provider = provider_map[provider_choice]
        
        # Output directory
        output_name = Prompt.ask("Tokenizer name", default=f"ai_{domain}_tokenizer")
        output_dir = f"tokenizers/{output_name}"
        
        result["type"] = "ai"
        result["path"] = output_dir
        result["config"] = {
            "domain": domain,
            "vocab_size": vocab_size,
            "provider": provider,
            "needs_training": True,  # Flag to train during model setup
        }
        
        console.print(f"\n[green]AI tokenizer will be trained at: {output_dir}[/green]")
        console.print(f"[dim]Domain: {domain}, Vocab: {vocab_size}, Provider: {provider}[/dim]")
        
    elif tok_choice == "4" and custom_tokenizers:
        # Show available tokenizers
        console.print("\n[bold]Available custom tokenizers:[/bold]")
        for i, tok in enumerate(custom_tokenizers, 1):
            console.print(f"  {i}. {tok}")
        
        tok_idx = Prompt.ask("Select tokenizer", choices=[str(i) for i in range(1, len(custom_tokenizers)+1)], default="1")
        selected = custom_tokenizers[int(tok_idx) - 1]
        
        result["type"] = "custom"
        result["path"] = str(tokenizers_dir / selected)
        console.print(f"[green]Using custom tokenizer: {selected}[/green]")
    
    return result


def train_ai_tokenizer_if_needed(tokenizer_config: dict, data_path: str) -> str:
    """
    Train AI tokenizer if needed based on config.
    Returns path to tokenizer.
    """
    if tokenizer_config.get("type") != "ai" or not tokenizer_config.get("config", {}).get("needs_training"):
        return tokenizer_config.get("path")
    
    from saara.ai_tokenizer import train_tokenizer_on_files
    
    config = tokenizer_config["config"]
    output_dir = tokenizer_config["path"]
    
    console.print("\n[bold cyan]Training AI-Enhanced Tokenizer...[/bold cyan]\n")
    
    use_ai = config.get("provider") != "none"
    
    tokenizer = train_tokenizer_on_files(
        input_path=data_path,
        output_dir=output_dir,
        vocab_size=config.get("vocab_size", 32000),
        domain=config.get("domain", "general"),
        use_ai=use_ai
    )
    
    return output_dir




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
    
    # Vision-only mode (skip parser, always use OCR)
    console.print("\n[bold]📷 Extraction Mode:[/bold]")
    console.print("  1. 🔀 Hybrid (parser first, vision fallback) - Faster for digital PDFs")
    console.print("  2. 👁️ Vision Only (always use OCR) - Better for Sanskrit/complex layouts")
    extraction_mode = Prompt.ask("Choose extraction mode", choices=["1", "2"], default="2")
    force_vision = (extraction_mode == "2")
    
    if force_vision:
        console.print("[green]✓ Using Vision OCR only (best for Sanskrit/scanned docs)[/green]")
    
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
    config['pdf']['force_ocr'] = force_vision  # Vision-only mode
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


def run_autonomous_finetune_wizard(config: dict = None):
    """Run autonomous fine-tuning wizard - AI generates training data."""
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from saara.train import AutonomousFineTuner
    
    console.print(Panel.fit(
        "[bold cyan]🤖 Autonomous Fine-tuning[/bold cyan]\n\n"
        "[green]AI Teacher will:[/green]\n"
        "• Generate curriculum for your domain\n"
        "• Create high-quality Q&A pairs\n"
        "• Format data for fine-tuning\n"
        "• Train your model automatically",
        title="AI-Powered Fine-tuning",
        border_style="green"
    ))
    
    # Step 1: Teacher Selection
    console.print("\n[bold cyan]Step 1: Configure Teacher Model[/bold cyan]")
    console.print("[1] Google AI - Gemini 2.0 Flash (Recommended)")
    console.print("[2] [bold yellow]Sarvam AI - Sanskrit & Indian Languages[/bold yellow]")
    console.print("[3] OpenAI API - GPT-4o")
    console.print("[4] Ollama (Local)")
    
    provider = Prompt.ask("Select teacher", choices=["1", "2", "3", "4"], default="1")
    
    teacher_config = {}
    if provider == "1":
        teacher_config["provider"] = "google"
        teacher_config["api_key"] = Prompt.ask("Enter Google AI API Key", password=True)
        teacher_config["model"] = "gemini-2.0-flash-exp"
    elif provider == "2":
        teacher_config["provider"] = "sarvam"
        teacher_config["api_key"] = Prompt.ask("Enter Sarvam AI API Key", password=True)
        teacher_config["model"] = "sarvam-1"
        console.print("[green]✓ Sarvam AI - Perfect for Sanskrit/Indian languages![/green]")
    elif provider == "3":
        teacher_config["provider"] = "openai"
        teacher_config["api_key"] = Prompt.ask("Enter OpenAI API Key", password=True)
        teacher_config["model"] = "gpt-4o"
    else:
        teacher_config["provider"] = "ollama"
        teacher_config["model"] = Prompt.ask("Ollama model", default="granite4:latest")
    
    # Step 2: Domain
    console.print("\n[bold cyan]Step 2: Define Knowledge Domain[/bold cyan]")
    domain = Prompt.ask(
        "What domain should the model learn?",
        default="Sanskrit Language and Texts"
    )
    
    # Step 3: Base Model
    console.print("\n[bold cyan]Step 3: Choose Base Model to Fine-tune[/bold cyan]")
    
    # Standard base models
    base_models = [
        ("google/gemma-2-2b", "⭐ Recommended - Gemma 2", "huggingface"),
        ("google/gemma-2-9b", "Gemma 2 Large", "huggingface"),
        ("sarvamai/sarvam-1", "Sarvam (Indic)", "huggingface"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama (Fast)", "huggingface"),
    ]
    
    # Add custom pre-trained models
    from saara.pretrain import list_pretrained_models
    pretrained = list_pretrained_models()
    for pm in pretrained:
        base_models.append((pm["path"], f"🏗️ {pm['name']} ({pm['params']})", "pretrained"))
    
    model_table = Table(show_header=True)
    model_table.add_column("#")
    model_table.add_column("Model")
    model_table.add_column("Notes")
    
    for i, (model_id, notes, _) in enumerate(base_models, 1):
        # Shorten long paths
        display_id = model_id if len(model_id) < 40 else f"...{model_id[-35:]}"
        model_table.add_row(str(i), display_id, notes)
    console.print(model_table)
    
    model_idx = int(Prompt.ask("Select base model", default="1")) - 1
    selected = base_models[min(model_idx, len(base_models)-1)]
    base_model = selected[0]
    
    if selected[2] == "pretrained":
        console.print(f"[green]✓ Using your pre-trained model: {selected[1]}[/green]")
    
    # Step 4: Configuration
    console.print("\n[bold cyan]Step 4: Training Configuration[/bold cyan]")
    target_pairs = int(Prompt.ask("Number of Q&A pairs to generate", default="500"))
    train_immediately = Confirm.ask("Train model after generating data?", default=True)
    
    # Run autonomous pipeline
    console.print("\n[bold green]Starting Autonomous Fine-tuning...[/bold green]")
    
    try:
        finetuner = AutonomousFineTuner(
            base_model=base_model,
            teacher_config=teacher_config
        )
        
        if train_immediately:
            result = finetuner.run_full_pipeline(
                domain=domain,
                target_pairs=target_pairs,
                train_model=True
            )
            console.print(f"\n[bold green]✅ Model fine-tuned and saved![/bold green]")
        else:
            result = finetuner.run_autonomous_generation(
                domain=domain,
                target_pairs=target_pairs
            )
            console.print(f"\n[bold green]✅ Dataset saved: {result}[/bold green]")
            console.print("[dim]Run training wizard again to train on this data.[/dim]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


def run_training_wizard(default_data_path: str = None, config: dict = None):
    """Run the interactive training setup."""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold cyan]🧠 Model Training[/bold cyan]\n\n"
        "Fine-tune language models on your data.",
        title="Training Mode",
        border_style="cyan"
    ))
    
    # Sub-menu for training options
    console.print("\n[bold]Choose Training Mode:[/bold]\n")
    console.print("  1. 📚 Train with Existing Dataset")
    console.print("  [bold green]2. 🤖 Autonomous Fine-tuning (AI generates training data)[/bold green]")
    console.print("  3. ↩️ Back to Main Menu")
    
    from rich.prompt import Prompt
    train_mode = Prompt.ask("Select mode", choices=["1", "2", "3"], default="1")
    
    if train_mode == "3":
        return
    elif train_mode == "2":
        # Autonomous Fine-tuning
        run_autonomous_finetune_wizard(config)
        return
    
    # Regular training flow continues...
    # 0. Hardware & Swarm Check
    from saara.model_manager import HardwareDetector
    from saara.gpu_workers import WorkerManager, display_swarm_status
    
    console.print("\n[bold]Step 0: Hardware & Swarm Analysis[/bold]\n")
    
    # Show Swarm Status
    worker_manager = WorkerManager()
    worker_manager.start()
    try:
        display_swarm_status(worker_manager)
    except Exception as e:
        console.print(f"[dim]Could not display swarm status: {e}[/dim]")

    console.print("[dim]Analyzing your system to recommend base models for fine-tuning...[/dim]\n")
    
    hardware_info = HardwareDetector.get_system_info()
    HardwareDetector.display_hardware_info(hardware_info)
    
    # Get fine-tuning recommendations
    finetune_recs = HardwareDetector.get_finetune_recommendations(hardware_info)
    
    console.print()
    HardwareDetector.display_finetune_recommendations(hardware_info, show_cloud=False)
    
    # 1. Fetch Models
    tm_manager = TrainedModelManager()
    trained_models = tm_manager.list_trained_models()
    
    # Also fetch pre-trained models
    from saara.pretrain import list_pretrained_models
    pretrained_models = list_pretrained_models()
    
    console.print("\n[bold]Select Model to Fine-tune:[/bold]")
    console.print(f"[dim]Maximum recommended size based on your hardware: {finetune_recs.get('max_model_size', 'Unknown')}[/dim]\n")
    
    t_table = Table(show_header=True, header_style="bold magenta")
    t_table.add_column("ID", style="cyan", width=4)
    t_table.add_column("Model/Adapter", style="green")
    t_table.add_column("Size", style="yellow")
    t_table.add_column("Status", width=18)
    t_table.add_column("Type/Details", style="dim")
    
    # Base Models - filtered/marked based on hardware recommendations
    # Models are categorized by approximate VRAM requirements for QLoRA fine-tuning
    base_models = [
        ("google/gemma-2-2b", "2B", "⭐ Gemma 2", 4.0),
        ("google/gemma-2-9b", "9B", "⭐ Gemma 2", 10.0),
        ("google/gemma-2b", "2B", "Gemma", 4.0),
        ("google/gemma-7b", "7B", "Gemma", 8.0),
        ("sarvamai/sarvam-1", "2B", "Indic", 4.0),
        ("meta-llama/Llama-3.2-1B", "1B", "Llama", 3.0),
        ("Qwen/Qwen2.5-7B", "7B", "Qwen", 8.0),
        ("TinyLlama/TinyLlama-1.1B", "1.1B", "Tiny", 3.0),
    ]
    
    options = []
    user_vram = hardware_info.get("vram_gb", 0)
    
    # Add Base Models with hardware status
    for i, (mid, size, type_, min_vram) in enumerate(base_models, 1):
        # Determine status based on VRAM
        if user_vram >= min_vram:
            status = "[bold green]✓ OK[/bold green]"
        elif user_vram >= min_vram * 0.7:  # Might work with optimization
            status = "[yellow]⚠ Might work[/yellow]"
        else:
            status = "[red]☁️ Cloud[/red]"
        
        t_table.add_row(str(i), mid, size, status, type_)
        options.append({"type": "base", "id": mid, "min_vram": min_vram})
        
    start_idx = len(base_models) + 1
    
    # Add Pre-trained Models (custom built)
    if pretrained_models:
        t_table.add_section()
        t_table.add_row("", "[bold]Custom Pre-trained Models[/bold]", "", "", "")
        
        for i, pm in enumerate(pretrained_models, start_idx):
            t_table.add_row(
                str(i), 
                pm["name"], 
                pm["params"],
                "[green]✓ Local[/green]",
                f"Arch: {pm['architecture']}"
            )
            options.append({
                "type": "pretrained", 
                "id": pm["path"],
                "name": pm["name"]
            })
        start_idx = len(options) + 1
    
    # Add Fine-Tuned Models (adapters)
    if trained_models:
        t_table.add_section()
        t_table.add_row("", "[bold]Fine-Tuned Adapters[/bold]", "", "", "")
        
        for i, tm in enumerate(trained_models, start_idx):
            t_table.add_row(
                str(i), 
                tm["name"], 
                f"{tm['size_mb']:.1f}MB",
                "[green]✓ Local[/green]",
                f"Base: {tm['base_model'].split('/')[-1]}"
            )
            options.append({
                "type": "adapter", 
                "id": tm["base_model"], 
                "path": tm["path"],
                "name": tm["name"]
            })
            
    # Add "Other" option
    other_idx = len(options) + 1
    t_table.add_section()
    t_table.add_row(str(other_idx), "Other (HuggingFace ID)", "-", "[dim]Manual[/dim]", "-")
    
    console.print(t_table)
    
    choice_idx = int(Prompt.ask("Choose a model", choices=[str(i) for i in range(1, other_idx + 1)], default="1")) - 1
    
    model_id = None
    adapter_path = None
    
    if choice_idx < len(options):
        selection = options[choice_idx]
        model_id = selection["id"]
        
        # Check if selected model requires more VRAM than available
        if selection["type"] == "base" and selection.get("min_vram"):
            min_vram = selection["min_vram"]
            if user_vram < min_vram:
                console.print(f"\n[bold red]⚠️ Warning: This model requires ~{min_vram}GB VRAM for fine-tuning![/bold red]")
                console.print(f"[red]Your VRAM: {user_vram}GB[/red]\n")
                
                # Show cloud options
                HardwareDetector.display_cloud_options()
                
                # Re-display swarm since user might be confused
                console.print("\n[bold]Checking decentralized worker swarm...[/bold]")
                display_swarm_status(worker_manager)
                
                console.print()
                if not Confirm.ask("[bold yellow]Continue using LOCAL GPU? (Training may fail)[/bold yellow]", default=False):
                    console.print("[yellow]Aborted. Please connect a cloud worker or choose a smaller model.[/yellow]")
                    return
                console.print("[dim]Proceeding with selected model. Use QLoRA and low batch size to reduce memory.[/dim]\n")
            elif user_vram < min_vram * 1.3:
                console.print(f"\n[bold yellow]💡 Note: This model will use most of your VRAM ({min_vram}GB needed, {user_vram}GB available).[/bold yellow]")
                console.print("[dim]Consider using QLoRA (4-bit) and batch_size=1 for stability.[/dim]\n")
        
        if selection["type"] == "pretrained":
            # Custom pre-trained model - use the path as model_id
            console.print(f"[bold]Selected Pre-trained Model:[/bold] {selection['name']}")
            
        elif selection["type"] == "adapter":
            # Handle unknown base model
            if model_id == "Unknown" or not model_id:
                model_id = Prompt.ask("Could not detect base model. Please enter Base Model ID")
            
            adapter_path = selection["path"]
            console.print(f"[bold]Selected Adapter:[/bold] {selection['name']} (on {model_id})")
        else:
            console.print(f"[bold]Selected Base Model:[/bold] {model_id}")
    else:
        # Other - warn about hardware check for unknown models
        model_id = Prompt.ask("Enter HuggingFace Model ID (e.g. microsoft/phi-2)")
        console.print(f"[bold]Selected Model:[/bold] {model_id}")
        console.print(f"[dim]Note: Max recommended model size for your hardware: {finetune_recs.get('max_model_size', 'Unknown')}[/dim]")
    
    gated_models = ["google/gemma", "meta-llama/Llama-3", "mistralai/Mistral"]
    is_gated = any(gated in model_id for gated in gated_models)
    
    if is_gated:
        console.print("[yellow]⚠️ This model requires HuggingFace authentication.[/yellow]")
        if Confirm.ask("Do you want to login to HuggingFace now?", default=True):
            import getpass
            console.print("[dim]Get your token from: https://huggingface.co/settings/tokens[/dim]")
            try:
                hf_token = getpass.getpass("Enter your HuggingFace token: ")
                if not hf_token.strip():
                    console.print("[red]No token provided.[/red]")
                    return
                from huggingface_hub import login
                login(token=hf_token.strip())
                console.print("[green]✅ Successfully logged in to HuggingFace![/green]")
            except Exception as e:
                console.print(f"[red]Login failed: {e}[/red]")
                return
    
    # --- New: Config Output & Checkpoints ---
    console.print("\n[bold]Configuration:[/bold]")
    
    # 1. Output Path
    default_output = f"models/{model_id.split('/')[-1]}-finetuned"
    if adapter_path:
        default_output = f"{adapter_path}-v2"
        
    output_dir = Prompt.ask("Output model directory", default=default_output).strip('"\'')
    
    # 2. Resume Checkpoint
    resume_checkpoint = None
    if Confirm.ask("Resume training from a checkpoint?", default=False):
        resume_checkpoint = Prompt.ask("Path to checkpoint folder (e.g. ./models/xyz/checkpoint-100)").strip('"\'')
        if not os.path.exists(resume_checkpoint):
            console.print(f"[yellow]Warning: Checkpoint path does not exist: {resume_checkpoint}[/yellow]")
            if not Confirm.ask("Continue without checkpoint?", default=True):
                 return
            resume_checkpoint = None

    # Dataset Selection
    while True:
        if default_data_path:
            data_file = default_data_path
            default_data_path = None
        else:
            default_guess = "datasets/interactive_batch_sharegpt.jsonl"
            if not os.path.exists(default_guess):
                default_guess = "datasets/distilled_train.jsonl"
                
            data_file = Prompt.ask("Path to dataset file OR Hugging Face Dataset ID (e.g. 'mlabonne/guanaco-llama2-1k')", default=default_guess).strip('"\'')
            
        path_obj = Path(data_file)
        
        # Check if it looks like a HF dataset ID (no extension, not existing file)
        if not path_obj.exists() and not path_obj.suffix:
             # Assume HF dataset
             if Confirm.ask(f"Use Hugging Face dataset '{data_file}'?", default=True):
                 data_paths = data_file
                 # Ask for optional config name
                 dataset_config = Prompt.ask("Dataset config name (optional, press Enter to skip)", default="")
                 if not dataset_config: 
                     dataset_config = None
                 break
        
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
        
    # Optional: Custom tokenizer for fine-tuning
    custom_tokenizer_path = None
    if Confirm.ask("Use a custom tokenizer? (default: use model's tokenizer)", default=False):
        console.print("[bold]🔤 Custom Tokenizer for Fine-tuning[/bold]")
        tokenizer_config = tokenizer_selection_wizard()
        
        if tokenizer_config.get("type") != "default":
            if tokenizer_config.get("type") == "ai" and tokenizer_config.get("config", {}).get("needs_training"):
                # Train AI tokenizer on the training data
                data_for_tok = data_file if isinstance(data_file, str) else data_file[0]
                custom_tokenizer_path = train_ai_tokenizer_if_needed(tokenizer_config, data_for_tok)
            else:
                custom_tokenizer_path = tokenizer_config.get("path")
    
    from saara.train import LLMTrainer
    
    if not config:
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    # Inject output dir into config
    config['output_dir'] = output_dir

    # ---------------------------------------------------------
    # CHECK FOR CLOUD WORKERS
    # ---------------------------------------------------------
    use_cloud = False
    worker_id = None
    
    try:
        import requests
        resp = requests.get("http://localhost:8765/status", timeout=0.2)
        if resp.ok:
            stats = resp.json()
            if stats["workers"]["connected"] > 0 or stats["workers"]["busy"] > 0:
                console.print(f"\n[bold cyan]☁️ Cloud Workers Detected: {stats['workers']['connected']} ready[/bold cyan]")
                if Confirm.ask("🚀 Run training on Cloud Worker?", default=True):
                    use_cloud = True
    except:
        pass

    if use_cloud:
        # Submit to cloud
        console.print("[dim]Submitting job to cloud swarm...[/dim]")
        
        # Prepare payload
        payload = {
            "model_name": model_id,
            "dataset_path": data_file, # If local path, this might fail unless notebook handles uploads. Assuming HF dataset or public URL for now for simplicity.
            "output_dir": output_dir,
            "epochs": 1, # Should ask user
            "learning_rate": 2e-4,
            "batch_size": 4
        }
        
        # If data_file is local path, warn user
        if os.path.exists(str(data_file)) and not str(data_file).startswith("http"):
             console.print("[yellow]⚠️ Note: Local datasets are not yet automatically uploaded to cloud workers.[/yellow]")
             console.print("[yellow]For cloud training, please use a HuggingFace Dataset ID or a public URL.[/yellow]")
             if not Confirm.ask("Continue anyway (worker might fail if it can't reach file)?", default=False):
                 return

        try:
            resp = requests.post("http://localhost:8765/jobs/submit", json={
                "job_type": "training",
                "payload": payload
            })
            
            if resp.ok:
                job_id = resp.json()["job_id"]
                console.print(f"[green]✅ Job Submitted: {job_id}[/green]")
                
                # Poll for progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=False,
                ) as progress:
                    task = progress.add_task(f"Job {job_id}: Queued...", total=100)
                    
                    while True:
                        import time
                        time.sleep(2)
                        status_resp = requests.get(f"http://localhost:8765/jobs/{job_id}", timeout=1)
                        if status_resp.ok:
                            job_data = status_resp.json()
                            status = job_data["status"]
                            prog = job_data["progress"]
                            
                            progress.update(task, completed=prog, description=f"Job {job_id}: {status.upper()} ({prog}%)")
                            
                            if status in ["completed", "failed", "cancelled"]:
                                break
                
                final_status = job_data.get("status")
                if final_status == "completed":
                    console.print(f"\n[bold green]🎉 Remote Training Completed![/bold green]")
                    console.print("[dim]The model was trained on the worker. Check worker logs/notebook for artifact location.[/dim]")
                    # In future: implement artifact download
                else:
                    console.print(f"\n[bold red]❌ Job Failed: {job_data.get('error')}[/bold red]")
                    
            else:
                console.print(f"[red]Failed to submit job: {resp.text}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error submitting job: {e}[/red]")
            
        return # Skip local training
        
    # Local Training Fallback
    trainer = LLMTrainer(model_id=model_id, adapter_path=adapter_path, config=config)
    
    # Pass custom tokenizer if specified
    if custom_tokenizer_path:
        console.print(f"[dim]Using custom tokenizer: {custom_tokenizer_path}[/dim]")
        # Note: Custom tokenizer support would need to be added to LLMTrainer
    
    try:
        # data_file could be a path string, a list of paths, or a huggingface dataset ID
        # variable 'dataset_config' might not exist if we took the file path route, so define it safely
        if 'dataset_config' not in locals():
            dataset_config = None
            
        trainer.train(data_file, resume_from_checkpoint=resume_checkpoint, dataset_config_name=dataset_config)
    except Exception as e:
        console.print(f"[bold red]Training failed:[/bold red] {e}")



def run_evaluation_wizard(config: dict = None):
    """Run the model evaluation wizard."""
    console.print(Panel.fit(
        "[bold cyan]🧪 Model Evaluation[/bold cyan]\n\n"
        "Test your pre-trained or fine-tuned model using Granite 4 as a judge.",
        title="Evaluation Mode",
        border_style="cyan"
    ))
    
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[red]No models directory found. Please train a model first.[/red]")
        return
    
    all_models = []
    
    # Find fine-tuned models (with adapters)
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            adapter_path = model_dir / "final_adapter"
            if adapter_path.exists():
                all_models.append({
                    "name": model_dir.name,
                    "path": str(adapter_path),
                    "type": "fine-tuned",
                    "is_pretrained": False
                })
    
    # Find pre-trained models (with 'final' subfolder or config.json)
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            final_path = model_dir / "final"
            # Check for pre-trained model (has config.json, no adapter_config.json)
            if final_path.exists() and (final_path / "config.json").exists():
                if not (final_path / "adapter_config.json").exists():
                    # Also check architecture info
                    arch_info_path = final_path / "architecture_info.json"
                    arch_name = "Pre-trained"
                    if arch_info_path.exists():
                        try:
                            import json
                            with open(arch_info_path) as f:
                                info = json.load(f)
                            arch_name = info.get("architecture", "Pre-trained")
                        except:
                            pass
                    all_models.append({
                        "name": model_dir.name,
                        "path": str(final_path),
                        "type": f"pre-trained ({arch_name})",
                        "is_pretrained": True
                    })
    
    if not all_models:
        console.print("[yellow]No models found. Train or pre-train a model first.[/yellow]")
        return
    
    console.print("\n[bold]Available Models:[/bold]\n")
    
    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("#", style="cyan", width=3)
    model_table.add_column("Name", style="green")
    model_table.add_column("Type", style="yellow")
    
    for i, m in enumerate(all_models, 1):
        model_table.add_row(str(i), m['name'], m['type'])
    
    console.print(model_table)
    
    choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(all_models)+1)], default="1")
    selected = all_models[int(choice)-1]
    
    # Handle pre-trained vs fine-tuned differently
    if selected["is_pretrained"]:
        # Pre-trained model
        console.print(f"\n[green]Selected pre-trained model: {selected['name']}[/green]")
        console.print(f"[dim]Path: {selected['path']}[/dim]\n")
        
        console.print("[bold]Test Options:[/bold]")
        console.print("1. 🧪 Interactive Text Completion")
        console.print("2. 📊 Perplexity Evaluation")
        console.print("3. 📋 Standard Evaluation (with test prompts)")
        console.print("4. 🎓 Autonomous Learning")
        console.print("5. ↩️ Back")
        
        test_choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], default="1")
        
        if test_choice == "1":
            # Use the PretrainedModelTester
            from saara.pretrain import PretrainedModelTester
            tester = PretrainedModelTester(selected["path"])
            tester.interactive_test()
        elif test_choice == "2":
            console.print("[yellow]Computing perplexity on sample text...[/yellow]")
            from saara.pretrain import PretrainedModelTester
            tester = PretrainedModelTester(selected["path"])
            # Simple perplexity test
            test_text = Prompt.ask("Enter test text (or press Enter for default)", 
                                   default="The three doshas in Ayurveda are vata, pitta, and kapha.")
            perplexity = tester.calculate_perplexity(test_text)
            console.print(f"\n[bold cyan]Perplexity: {perplexity:.2f}[/bold cyan]")
            console.print("[dim]Lower is better. < 50 is good, < 20 is excellent.[/dim]")
        elif test_choice == "3":
            # Standard Evaluation for pre-trained model
            console.print("\n[bold cyan]Select Evaluator/Judge Model:[/bold cyan]")
            console.print("[1] Google AI - Gemini 2.0 Flash (Recommended)")
            console.print("[2] Ollama (Local) - granite4, llama3.2, qwen2.5, etc.")
            console.print("[3] OpenAI API - GPT-4o, GPT-4-turbo")
            console.print("[4] DeepSeek API")
            console.print("[5] HuggingFace Inference API")
            
            eval_provider = Prompt.ask("Select provider", choices=["1", "2", "3", "4", "5"], default="1")
            
            eval_config = {}
            
            if eval_provider == "1":
                eval_config["provider"] = "google"
                eval_config["api_key"] = Prompt.ask("Enter Google AI API Key", password=True)
                eval_config["model"] = Prompt.ask("Model name", default="gemini-2.0-flash-exp")
            elif eval_provider == "2":
                eval_config["provider"] = "ollama"
                try:
                    import ollama
                    models_list = ollama.list()
                    available = [m.model for m in models_list.models] if hasattr(models_list, 'models') else []
                    if available:
                        console.print(f"[dim]Available: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}[/dim]")
                except:
                    pass
                eval_config["model"] = Prompt.ask("Enter Ollama model name", default="granite4:latest")
            elif eval_provider == "3":
                eval_config["provider"] = "openai"
                eval_config["api_key"] = Prompt.ask("Enter OpenAI API Key", password=True)
                eval_config["model"] = Prompt.ask("Model name", default="gpt-4o-mini")
            elif eval_provider == "4":
                eval_config["provider"] = "deepseek"
                eval_config["api_key"] = Prompt.ask("Enter DeepSeek API Key", password=True)
                eval_config["base_url"] = "https://api.deepseek.com"
                eval_config["model"] = Prompt.ask("Model name", default="deepseek-chat")
            else:
                eval_config["provider"] = "huggingface"
                eval_config["api_key"] = Prompt.ask("Enter HuggingFace Token", password=True)
                eval_config["model"] = Prompt.ask("Model ID", default="meta-llama/Llama-3.3-70B-Instruct")
            
            from saara.evaluator import ModelEvaluator
            evaluator = ModelEvaluator(config)
            evaluator.default_teacher_config = eval_config
            num_samples = int(Prompt.ask("Number of test samples", default="10"))
            # For pre-trained models, evaluate directly without adapter
            evaluator.evaluate_pretrained(selected["path"], num_samples=num_samples)
        elif test_choice == "4":
            # Autonomous Learning
            topic = Prompt.ask("Enter topic to learn about")
            iterations = int(Prompt.ask("Learning iterations", default="10"))
            
            # --- Teacher Model Selection ---
            console.print("\n[bold cyan]Select Teacher Model:[/bold cyan]")
            console.print("[1] Google AI - Gemini 2.0 Flash (Recommended)")
            console.print("[2] Ollama (Local) - granite4, llama3.2, qwen2.5, etc.")
            console.print("[3] OpenAI API - GPT-4o, GPT-4-turbo")
            console.print("[4] DeepSeek API")
            console.print("[5] HuggingFace Inference API")
            
            provider_choice = Prompt.ask("Select provider", choices=["1", "2", "3", "4", "5"], default="1")
            
            teacher_config = {}
            
            if provider_choice == "1":
                # Google Gemini (default)
                teacher_config["provider"] = "google"
                teacher_config["api_key"] = Prompt.ask("Enter Google AI API Key", password=True)
                teacher_config["model"] = Prompt.ask("Model name", default="gemini-2.0-flash")
            elif provider_choice == "2":
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
            elif provider_choice == "3":
                teacher_config["provider"] = "openai"
                teacher_config["api_key"] = Prompt.ask("Enter OpenAI API Key", password=True)
                teacher_config["model"] = Prompt.ask("Model name", default="gpt-4o-mini")
            elif provider_choice == "4":
                teacher_config["provider"] = "deepseek"
                teacher_config["api_key"] = Prompt.ask("Enter DeepSeek API Key", password=True)
                teacher_config["base_url"] = "https://api.deepseek.com"
                teacher_config["model"] = Prompt.ask("Model name", default="deepseek-chat")
            else:
                teacher_config["provider"] = "huggingface"
                teacher_config["api_key"] = Prompt.ask("Enter HuggingFace Token", password=True)
                teacher_config["model"] = Prompt.ask("Model ID", default="meta-llama/Llama-3.3-70B-Instruct")
            
            from saara.evaluator import ModelEvaluator
            evaluator = ModelEvaluator(config)
            evaluator.run_autonomous_learning_pretrained(
                selected["path"], 
                topic, 
                iterations=iterations,
                teacher_config=teacher_config
            )
        return
    
    # Fine-tuned model flow
    base_model = Prompt.ask("Enter base model ID", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    console.print("\n[bold]Select Mode:[/bold]")
    console.print("1. Standard Evaluation")
    console.print("2. Autonomous Learning")
    mode_choice = Prompt.ask("Select mode", choices=["1", "2"], default="1")
    
    # --- Evaluator/Judge Model Selection ---
    console.print("\n[bold cyan]Select Evaluator/Judge Model:[/bold cyan]")
    console.print("[1] Google AI - Gemini 2.0 Flash (Recommended)")
    console.print("[2] Ollama (Local) - granite4, llama3.2, qwen2.5, etc.")
    console.print("[3] OpenAI API - GPT-4o, GPT-4-turbo")
    console.print("[4] DeepSeek API")
    console.print("[5] HuggingFace Inference API")
    
    eval_provider = Prompt.ask("Select provider", choices=["1", "2", "3", "4", "5"], default="1")
    
    eval_config = {}
    
    if eval_provider == "1":
        eval_config["provider"] = "google"
        eval_config["api_key"] = Prompt.ask("Enter Google AI API Key", password=True)
        eval_config["model"] = Prompt.ask("Model name", default="gemini-2.0-flash-exp")
    elif eval_provider == "2":
        eval_config["provider"] = "ollama"
        try:
            import ollama
            models_list = ollama.list()
            available = [m.model for m in models_list.models] if hasattr(models_list, 'models') else []
            if available:
                console.print(f"[dim]Available: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}[/dim]")
        except:
            pass
        eval_config["model"] = Prompt.ask("Enter Ollama model name", default="granite4:latest")
    elif eval_provider == "3":
        eval_config["provider"] = "openai"
        eval_config["api_key"] = Prompt.ask("Enter OpenAI API Key", password=True)
        eval_config["model"] = Prompt.ask("Model name", default="gpt-4o-mini")
    elif eval_provider == "4":
        eval_config["provider"] = "deepseek"
        eval_config["api_key"] = Prompt.ask("Enter DeepSeek API Key", password=True)
        eval_config["base_url"] = "https://api.deepseek.com"
        eval_config["model"] = Prompt.ask("Model name", default="deepseek-chat")
    else:
        eval_config["provider"] = "huggingface"
        eval_config["api_key"] = Prompt.ask("Enter HuggingFace Token", password=True)
        eval_config["model"] = Prompt.ask("Model ID", default="meta-llama/Llama-3.3-70B-Instruct")
    
    from saara.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(config)
    evaluator.default_teacher_config = eval_config  # Override with user selection
    
    if mode_choice == "1":
        num_samples = int(Prompt.ask("Number of test samples", default="10"))
        evaluator.evaluate_adapter(base_model, selected["path"], num_samples=num_samples)
    else:
        topic = Prompt.ask("Enter topic to learn about")
        iterations = int(Prompt.ask("Learning iterations", default="10"))
        
        # --- Teacher Model Selection ---
        console.print("\n[bold cyan]Select Teacher Model:[/bold cyan]")
        console.print("[1] Google AI - Gemini 2.0 Flash (Recommended)")
        console.print("[2] Ollama (Local) - granite4, llama3.2, qwen2.5, etc.")
        console.print("[3] OpenAI API - GPT-4o, GPT-4-turbo")
        console.print("[4] DeepSeek API")
        console.print("[5] HuggingFace Inference API")
        
        provider_choice = Prompt.ask("Select provider", choices=["1", "2", "3", "4", "5"], default="1")
        
        teacher_config = {}
        
        if provider_choice == "1":
            # Google Gemini (default)
            teacher_config["provider"] = "google"
            teacher_config["api_key"] = Prompt.ask("Enter Google AI API Key", password=True)
            teacher_config["model"] = Prompt.ask("Model name", default="gemini-2.0-flash")
            
        elif provider_choice == "2":
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
            
        elif provider_choice == "3":
            teacher_config["provider"] = "openai"
            teacher_config["api_key"] = Prompt.ask("Enter OpenAI API Key", password=True)
            teacher_config["model"] = Prompt.ask("Model name", default="gpt-4o-mini")
            
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
    from saara.pretrain import list_pretrained_models
    
    models_dir = Path("models")
    
    all_models = []
    
    # Collect pre-trained models
    pretrained_models = list_pretrained_models()
    for pm in pretrained_models:
        all_models.append({
            "name": pm["name"],
            "path": pm["path"],
            "type": "pretrained",
            "details": f"Pre-trained ({pm['params']})"
        })
    
    # Collect fine-tuned models
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                adapter_path = model_dir / "final_adapter"
                if adapter_path.exists():
                    all_models.append({
                        "name": model_dir.name,
                        "path": str(adapter_path),
                        "type": "adapter",
                        "details": "Fine-tuned Adapter"
                    })
    
    if not all_models:
        console.print("[yellow]No models found. Please train a model first.[/yellow]")
        return
    
    console.print("\n[bold]Select Model to Deploy:[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Name", style="green")
    table.add_column("Type", style="yellow")
    
    for i, m in enumerate(all_models, 1):
        table.add_row(str(i), m["name"], m["details"])
        
    console.print(table)
    
    choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(all_models)+1)], default="1")
    selected = all_models[int(choice)-1]
    
    deployer = ModelDeployer(config)
    
    if selected["type"] == "pretrained":
        # Pre-trained model - deploy full model
        deployer.deploy_menu(selected["path"], None)
    else:
        # Adapter - need base model
        base_model = Prompt.ask("Enter base model ID", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        deployer.deploy_menu(base_model, selected["path"])


def run_model_expansion_wizard(config: dict = None):
    """
    Wizard to expand model parameters - scale up a small model to a larger architecture.
    
    This enables progressive training:
    1. Start with a small model (fast iteration, low VRAM)
    2. Train until convergence
    3. Expand to larger architecture (more capacity)
    4. Continue training with inherited knowledge
    """
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from pathlib import Path
    
    console.print(Panel.fit(
        "[bold cyan]📈 Model Parameter Expansion[/bold cyan]\n\n"
        "[dim]Scale up your model to a larger architecture while preserving learned knowledge.[/dim]\n"
        "[dim]This enables progressive training: start small, expand as you go.[/dim]",
        title="Progressive Scaling",
        border_style="cyan"
    ))
    
    from saara.pretrain import list_pretrained_models, ARCHITECTURES, ModelExpander
    
    # List available models
    models = list_pretrained_models()
    
    if not models:
        console.print("[yellow]No pre-trained models found. Train a model first![/yellow]")
        console.print("[dim]Use 'saara pretrain' → 'Build & Train New Model'[/dim]")
        return
    
    console.print("\n[bold]Step 1: Select Model to Expand[/bold]\n")
    
    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("#", style="cyan", width=3)
    model_table.add_column("Name", style="green")
    model_table.add_column("Current Architecture", style="yellow")
    model_table.add_column("Parameters", width=10)
    model_table.add_column("Expansion Path", style="dim")
    
    for i, m in enumerate(models, 1):
        # Get expansion path
        expansion_options = ModelExpander.get_expansion_path(m["architecture"])
        expansion_str = " → ".join(expansion_options[:3]) if expansion_options else "[green]Already largest[/green]"
        
        model_table.add_row(
            str(i),
            m["name"],
            m["architecture"],
            m["params"],
            expansion_str
        )
    
    console.print(model_table)
    
    choice = Prompt.ask("Select model to expand", choices=[str(i) for i in range(1, len(models)+1)], default="1")
    selected_model = models[int(choice) - 1]
    
    # Check if already at largest
    current_arch = selected_model["architecture"]
    expansion_options = ModelExpander.get_expansion_path(current_arch)
    
    if not expansion_options:
        console.print(f"\n[yellow]'{selected_model['name']}' is already at the largest architecture ({current_arch}).[/yellow]")
        console.print("[dim]Consider fine-tuning instead to improve quality.[/dim]")
        return
    
    # Select target architecture
    console.print(f"\n[bold]Step 2: Select Target Architecture[/bold]")
    console.print(f"[dim]Current: {current_arch} ({selected_model['params']})[/dim]\n")
    
    arch_table = Table(show_header=True, header_style="bold magenta")
    arch_table.add_column("#", style="cyan", width=3)
    arch_table.add_column("Architecture", style="green", width=15)
    arch_table.add_column("Parameters", width=10)
    arch_table.add_column("VRAM Required", width=12)
    arch_table.add_column("Description", width=45)
    
    for i, arch_name in enumerate(expansion_options, 1):
        arch = ARCHITECTURES[arch_name]
        arch_table.add_row(
            str(i),
            arch.display_name,
            arch.estimated_params,
            f"{arch.min_vram_gb} GB+",
            arch.description
        )
    
    console.print(arch_table)
    console.print()
    
    target_choice = Prompt.ask(
        "Select target architecture",
        choices=[str(i) for i in range(1, len(expansion_options)+1)],
        default="1"
    )
    target_arch = expansion_options[int(target_choice) - 1]
    target_arch_info = ARCHITECTURES[target_arch]
    
    # Summary
    console.print("\n")
    summary_table = Table(title="📋 Expansion Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Setting", style="green")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Source Model", selected_model["name"])
    summary_table.add_row("Current Size", f"{current_arch} ({selected_model['params']})")
    summary_table.add_row("Target Size", f"{target_arch} ({target_arch_info.estimated_params})")
    summary_table.add_row("VRAM Required", f"{target_arch_info.min_vram_gb} GB+")
    
    console.print(summary_table)
    console.print()
    
    console.print("[dim]The expansion will:[/dim]")
    console.print("[dim]  • Create a new model with larger capacity[/dim]")
    console.print("[dim]  • Transfer learned weights from smaller model[/dim]")
    console.print("[dim]  • Interpolate weights where dimensions differ[/dim]")
    console.print("[dim]  • Initialize new parameters smartly[/dim]")
    console.print()
    
    if not Confirm.ask("[bold]Proceed with expansion?[/bold]", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        return
    
    # Run expansion
    try:
        expander = ModelExpander(selected_model["path"], target_arch)
        expanded_path = expander.expand()
        
        # Offer next steps
        console.print("\n[bold]What would you like to do next?[/bold]\n")
        console.print("  1. 🏗️ Continue pre-training on expanded model")
        console.print("  2. 🧪 Test the expanded model")
        console.print("  3. ✅ Done")
        
        next_action = Prompt.ask("Select action", choices=["1", "2", "3"], default="1")
        
        if next_action == "1":
            # Continue pre-training
            console.print("\n[bold cyan]Continue training the expanded model:[/bold cyan]")
            console.print(f"[dim]Model path: {expanded_path}[/dim]\n")
            
            data_path = Prompt.ask("Path to training data").strip('"\'')
            
            if Path(data_path).exists():
                from saara.pretrain import PreTrainer
                
                # Get model name from path
                model_name = Path(expanded_path).name
                
                pretrainer = PreTrainer(
                    architecture=target_arch,
                    model_name=model_name + "-continued",
                    output_dir="models",
                    config=config
                )
                
                # Use expanded model's tokenizer
                pretrainer.pretrain(data_path, tokenizer_path=expanded_path)
            else:
                console.print(f"[red]Path not found: {data_path}[/red]")
                
        elif next_action == "2":
            from saara.pretrain import PretrainedModelTester
            tester = PretrainedModelTester(expanded_path)
            tester.interactive_test()
            
    except Exception as e:
        console.print(f"[bold red]Expansion failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()


def run_autonomous_pretrain_wizard(config: dict = None):
    """Run autonomous pre-training wizard powered by Gemini."""
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from saara.pretrain import AutonomousPretrainer, ARCHITECTURES
    
    console.print(Panel.fit(
        "[bold cyan]🤖 Autonomous Pre-training[/bold cyan]\n\n"
        "[green]Gemini will help you:[/green]\n"
        "• Generate a learning curriculum\n"
        "• Create synthetic training data\n"
        "• Quality-filter all samples\n"
        "• Train your model automatically",
        title="Gemini-Powered Pre-training",
        border_style="green"
    ))
    
    # Get teacher model config
    console.print("\n[bold cyan]Step 1: Configure Teacher Model[/bold cyan]")
    console.print("[1] Google AI - Gemini 2.0 Flash (Recommended)")
    console.print("[2] [bold yellow]Sarvam AI - Sanskrit & Indian Languages[/bold yellow]")
    console.print("[3] OpenAI API - GPT-4o")
    console.print("[4] Ollama (Local)")
    
    provider_choice = Prompt.ask("Select teacher", choices=["1", "2", "3", "4"], default="1")
    
    teacher_config = {}
    if provider_choice == "1":
        teacher_config["provider"] = "google"
        teacher_config["api_key"] = Prompt.ask("Enter Google AI API Key", password=True)
        teacher_config["model"] = Prompt.ask("Model", default="gemini-2.0-flash-exp")
    elif provider_choice == "2":
        # Sarvam AI for Sanskrit
        teacher_config["provider"] = "sarvam"
        teacher_config["api_key"] = Prompt.ask("Enter Sarvam AI API Key", password=True)
        teacher_config["model"] = "sarvam-1"
        console.print("[green]✓ Sarvam AI selected - Great for Sanskrit and Indian languages![/green]")
    elif provider_choice == "3":
        teacher_config["provider"] = "openai"
        teacher_config["api_key"] = Prompt.ask("Enter OpenAI API Key", password=True)
        teacher_config["model"] = Prompt.ask("Model", default="gpt-4o")
    else:
        teacher_config["provider"] = "ollama"
        teacher_config["model"] = Prompt.ask("Ollama model name", default="granite4:latest")
    
    # Get domain
    console.print("\n[bold cyan]Step 2: Define Knowledge Domain[/bold cyan]")
    domain = Prompt.ask("Enter the knowledge domain to pre-train on", 
                        default="Ayurveda and Traditional Indian Medicine")
    
    # Get model config
    console.print("\n[bold cyan]Step 3: Choose Model Architecture[/bold cyan]")
    from rich.table import Table
    arch_table = Table(show_header=True, header_style="bold")
    arch_table.add_column("#")
    arch_table.add_column("Name")
    arch_table.add_column("Params")
    arch_table.add_column("VRAM")
    
    arch_list = list(ARCHITECTURES.keys())
    for i, key in enumerate(arch_list, 1):
        arch = ARCHITECTURES[key]
        arch_table.add_row(str(i), arch.display_name, arch.estimated_params, f"{arch.min_vram_gb}GB+")
    console.print(arch_table)
    
    arch_idx = int(Prompt.ask("Select architecture", default="1")) - 1
    architecture = arch_list[min(arch_idx, len(arch_list)-1)]
    
    # Model name
    model_name = Prompt.ask("Model name", default=f"{domain.split()[0]}-{architecture}")
    
    # Sample count
    console.print("\n[bold cyan]Step 4: Training Configuration[/bold cyan]")
    target_samples = int(Prompt.ask("Number of training samples to generate", default="100"))
    quality_threshold = int(Prompt.ask("Quality threshold (1-10)", default="7"))
    
    train_immediately = Confirm.ask("Train model after generating data?", default=True)
    
    # Run autonomous pipeline
    console.print("\n[bold green]Starting Autonomous Pre-training...[/bold green]")
    
    try:
        pretrainer = AutonomousPretrainer(
            model_name=model_name,
            architecture=architecture,
            teacher_config=teacher_config,
        )
        
        if train_immediately:
            result = pretrainer.run_full_autonomous_pipeline(
                domain=domain,
                target_samples=target_samples,
                train_model=True
            )
            console.print(f"\n[bold green]✅ Model trained and saved to: {result}[/bold green]")
        else:
            result = pretrainer.run_autonomous_generation(
                domain=domain,
                target_samples=target_samples,
                quality_threshold=quality_threshold
            )
            console.print(f"\n[bold green]✅ Dataset saved to: {result}[/bold green]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


def run_pretrain_wizard(config: dict = None):
    """Run the pre-training wizard to build models from scratch."""
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from pathlib import Path
    
    console.print(Panel.fit(
        "[bold cyan]🏗️ Pre-training from Scratch[/bold cyan]\n\n"
        "[dim]Build and train your own language model from the ground up.[/dim]\n"
        "[dim]You can then fine-tune, evaluate, and deploy it like any other model.[/dim]",
        title="Pre-training Mode",
        border_style="cyan"
    ))
    
    # Sub-menu
    console.print("\n[bold]What would you like to do?[/bold]\n")
    console.print("  1. 📚 Create Pre-training Dataset")
    console.print("  2. 🏗️ Build & Train New Model")
    console.print("  3. 🔤 Train Custom Tokenizer")
    console.print("  4. 🧪 Test Pre-trained Model")
    console.print("  5. 📋 List Pre-trained Models")
    console.print("  6. 📈 Expand Model Parameters")
    console.print("  [bold green]7. 🤖 Autonomous Pre-training (Gemini-Powered)[/bold green]")
    console.print("  8. ↩️ Back to Main Menu")
    
    action = Prompt.ask("Select action", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")
    
    if action == "8":
        return
    elif action == "1":
        # Create pre-training dataset
        from saara.pretrain_data import run_pretrain_dataset_wizard
        run_pretrain_dataset_wizard(config)
        return
    elif action == "7":
        # Autonomous Pre-training with Gemini
        run_autonomous_pretrain_wizard(config)
        return
    elif action == "6":
        # Expand model parameters
        run_model_expansion_wizard(config)
        return
    elif action == "5":
        # List models
        from saara.pretrain import list_pretrained_models
        models = list_pretrained_models()
        
        if not models:
            console.print("[yellow]No pre-trained models found.[/yellow]")
            return
            
        table = Table(title="🎯 Pre-trained Models", show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="green")
        table.add_column("Architecture", style="yellow")
        table.add_column("Parameters", width=10)
        table.add_column("Path", style="dim")
        
        for i, m in enumerate(models, 1):
            table.add_row(str(i), m["name"], m["architecture"], m["params"], m["path"])
            
        console.print(table)
        return
        
    elif action == "4":
        # Test model
        from saara.pretrain import list_pretrained_models, PretrainedModelTester
        
        models = list_pretrained_models()
        if not models:
            console.print("[yellow]No pre-trained models found. Train one first![/yellow]")
            return
            
        console.print("\n[bold]Select Model to Test:[/bold]\n")
        for i, m in enumerate(models, 1):
            console.print(f"  {i}. {m['name']} ({m['architecture']})")
            
        choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(models)+1)], default="1")
        selected = models[int(choice)-1]
        
        tester = PretrainedModelTester(selected["path"])
        tester.interactive_test()
        return
        
    elif action == "3":
        # Train tokenizer
        from saara.pretrain import TokenizerTrainer
        
        console.print("\n[bold]🔤 Custom Tokenizer Training[/bold]\n")
        
        vocab_size = int(Prompt.ask("Vocabulary size", default="32000"))
        data_path = Prompt.ask("Path to training data (text files)", default="./data").strip('"\'')
        output_dir = Prompt.ask("Output directory", default="./tokenizers/custom").strip('"\'')
        
        path_obj = Path(data_path)
        if not path_obj.exists():
            console.print(f"[red]Path not found: {data_path}[/red]")
            return
            
        # Collect text files
        if path_obj.is_file():
            data_files = [str(path_obj)]
        else:
            data_files = [str(f) for f in path_obj.glob("**/*.txt")]
            data_files += [str(f) for f in path_obj.glob("**/*.md")]
            
        if not data_files:
            console.print("[red]No text files found![/red]")
            return
            
        console.print(f"[green]Found {len(data_files)} text files[/green]")
        
        trainer = TokenizerTrainer(vocab_size=vocab_size)
        trainer.train(data_files, output_dir)
        return
    
    # Action 2: Build & Train New Model
    if action != "2":
        return
        
    from saara.pretrain import ARCHITECTURES, PreTrainer, list_pretrained_models
    from saara.model_manager import HardwareDetector
    
    # Step 0: Hardware Check
    console.print("\n[bold]Step 0: Hardware Analysis[/bold]\n")
    console.print("[dim]Analyzing your system to recommend optimal architectures...[/dim]\n")
    
    hardware_info = HardwareDetector.get_system_info()
    HardwareDetector.display_hardware_info(hardware_info)
    
    # Get recommendations
    pretrain_recs = HardwareDetector.get_pretrain_recommendations(hardware_info)
    
    console.print()
    HardwareDetector.display_pretrain_recommendations(hardware_info, show_cloud=False)
    
    console.print("\n[bold]Step 1: Select Model Architecture[/bold]\n")
    
    # Show architectures with recommendation status
    arch_table = Table(show_header=True, header_style="bold magenta")
    arch_table.add_column("#", style="cyan", width=3)
    arch_table.add_column("Name", style="green", width=15)
    arch_table.add_column("Params", width=8)
    arch_table.add_column("VRAM", width=8)
    arch_table.add_column("Status", width=20)
    arch_table.add_column("Description", width=40)
    
    arch_keys = list(ARCHITECTURES.keys())
    for i, key in enumerate(arch_keys, 1):
        arch = ARCHITECTURES[key]
        
        # Determine status based on hardware recommendations
        if key in pretrain_recs.get("recommended", []):
            status = "[bold green]✓ Recommended[/bold green]"
        elif key in pretrain_recs.get("possible_with_optimization", []):
            status = "[yellow]⚠ With optimization[/yellow]"
        else:
            status = "[red]☁️ Needs cloud[/red]"
        
        arch_table.add_row(
            str(i),
            arch.display_name,
            arch.estimated_params,
            f"{arch.min_vram_gb}GB+",
            status,
            arch.description[:40] + "..." if len(arch.description) > 40 else arch.description
        )
        
    console.print(arch_table)
    
    # Suggest default based on hardware
    default_arch = "2"  # micro by default
    if pretrain_recs.get("recommended"):
        # Find the index of the largest recommended architecture
        for i, key in enumerate(arch_keys):
            if key in pretrain_recs["recommended"]:
                default_arch = str(i + 1)
    
    arch_choice = int(Prompt.ask("Select architecture", choices=[str(i) for i in range(1, len(arch_keys)+1)], default=default_arch)) - 1
    selected_arch = arch_keys[arch_choice]
    
    # Warn if selected architecture requires cloud
    if selected_arch in pretrain_recs.get("requires_cloud", []):
        console.print(f"\n[bold red]⚠️ Warning: {ARCHITECTURES[selected_arch].display_name} requires more powerful hardware![/bold red]")
        console.print(f"[red]Your hardware: VRAM={hardware_info.get('vram_gb', 0)}GB, RAM={hardware_info.get('ram_gb', 0)}GB[/red]")
        console.print(f"[red]Required: VRAM={ARCHITECTURES[selected_arch].min_vram_gb}GB+[/red]\n")
        
        # Show cloud options
        HardwareDetector.display_cloud_options()
        
        console.print()
        if not Confirm.ask("[bold yellow]Continue anyway? (Training may fail or be very slow)[/bold yellow]", default=False):
            console.print("[yellow]Aborted. Consider using a smaller architecture or cloud GPU.[/yellow]")
            return
        console.print("[dim]Proceeding with selected architecture. Watch for OOM errors.[/dim]")
    elif selected_arch in pretrain_recs.get("possible_with_optimization", []):
        console.print(f"\n[bold yellow]💡 Note: {ARCHITECTURES[selected_arch].display_name} will require optimizations.[/bold yellow]")
        console.print("[dim]We'll enable gradient checkpointing and use fp16 to reduce memory usage.[/dim]")
    
    console.print(f"\n[green]Selected: {ARCHITECTURES[selected_arch].display_name}[/green]")
    
    # Model name
    model_name = Prompt.ask("\nEnter a name for your model", default="my-custom-model")
    
    # Data path
    console.print("\n[bold]Step 2: Training Data[/bold]")
    console.print("[dim]Provide text files (.txt, .md) or JSONL with 'text' field.[/dim]\n")
    
    data_path = Prompt.ask("Path to training data").strip('"\'')
    
    if not Path(data_path).exists():
        console.print(f"[red]Path not found: {data_path}[/red]")
        return
    
    # Tokenizer - use the new enhanced selection wizard
    console.print("\n[bold]Step 3: Tokenizer[/bold]")
    tokenizer_config = tokenizer_selection_wizard()
    
    # Handle tokenizer path based on selection
    tokenizer_path = tokenizer_config.get("path")
    
    # If AI tokenizer selected, it will be trained before model training
    if tokenizer_config.get("type") == "ai":
        console.print("[dim]AI tokenizer will be trained on your data before model training.[/dim]")
    elif tokenizer_config.get("type") == "huggingface":
        console.print(f"[dim]Using HuggingFace tokenizer: {tokenizer_path}[/dim]")

    
    # Advanced options
    console.print("\n[bold]Step 4: Training Parameters[/bold]\n")
    
    show_advanced = Confirm.ask("Configure advanced training options?", default=False)
    
    epochs = 1
    batch_size = 8
    learning_rate = 3e-4
    max_seq_length = 1024
    
    if show_advanced:
        epochs = int(Prompt.ask("Number of epochs", default="1"))
        batch_size = int(Prompt.ask("Batch size", default="8"))
        learning_rate = float(Prompt.ask("Learning rate", default="3e-4"))
        max_seq_length = int(Prompt.ask("Max sequence length", default="1024"))
    
    # Summary
    console.print("\n")
    summary_table = Table(title="📋 Pre-training Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Setting", style="green")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Model Name", model_name)
    summary_table.add_row("Architecture", ARCHITECTURES[selected_arch].display_name)
    summary_table.add_row("Parameters", ARCHITECTURES[selected_arch].estimated_params)
    summary_table.add_row("Training Data", data_path)
    
    # Enhanced tokenizer info
    tok_type = tokenizer_config.get("type", "default")
    if tok_type == "ai":
        tok_info = f"🤖 AI-Enhanced ({tokenizer_config['config'].get('domain', 'general')})"
    elif tok_type == "huggingface":
        tok_info = f"🤗 HuggingFace: {tokenizer_path}"
    elif tok_type == "custom":
        tok_info = f"📁 Custom: {tokenizer_path}"
    else:
        tok_info = "🦙 Default (LLaMA)"
    
    summary_table.add_row("Tokenizer", tok_info)
    summary_table.add_row("Epochs", str(epochs))
    summary_table.add_row("Batch Size", str(batch_size))
    summary_table.add_row("Learning Rate", str(learning_rate))
    
    console.print(summary_table)
    console.print()
    
    if not Confirm.ask("[bold]Start pre-training?[/bold]", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        return
    
    # Create and run PreTrainer
    pretrainer = PreTrainer(
        architecture=selected_arch,
        model_name=model_name,
        output_dir="models",
        config=config
    )
    
    # Override training params
    pretrainer.train_params["num_train_epochs"] = epochs
    pretrainer.train_params["per_device_train_batch_size"] = batch_size
    pretrainer.train_params["learning_rate"] = learning_rate
    pretrainer.train_params["max_seq_length"] = max_seq_length
    
    try:
        # Train AI tokenizer first if needed
        if tokenizer_config.get("type") == "ai" and tokenizer_config.get("config", {}).get("needs_training"):
            tokenizer_path = train_ai_tokenizer_if_needed(tokenizer_config, data_path)
        
        model_path = pretrainer.pretrain(data_path, tokenizer_path)
        
        # Offer next steps
        console.print("\n[bold]What would you like to do next?[/bold]\n")
        console.print("  1. 🧪 Test the model")
        console.print("  2. 🎯 Fine-tune the model")
        console.print("  3. 🚀 Deploy the model")
        console.print("  4. ✅ Done")
        
        next_action = Prompt.ask("Select action", choices=["1", "2", "3", "4"], default="4")
        
        if next_action == "1":
            from saara.pretrain import PretrainedModelTester
            tester = PretrainedModelTester(model_path)
            tester.interactive_test()
        elif next_action == "2":
            run_training_wizard_for_pretrained(model_path, config)
        elif next_action == "3":
            run_deployment_wizard_for_pretrained(model_path, config)
            
    except Exception as e:
        console.print(f"[bold red]Pre-training failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()


def run_training_wizard_for_pretrained(model_path: str, config: dict = None):
    """Fine-tune a custom pre-trained model."""
    console.print(f"\n[bold cyan]🎯 Fine-tuning {model_path}[/bold cyan]\n")
    
    # Get data path
    data_path = Prompt.ask("Path to fine-tuning data (.jsonl)", default="datasets/").strip('"\'')
    
    from saara.train import LLMTrainer
    
    if not config:
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    trainer = LLMTrainer(model_id=model_path, config=config)
    try:
        trainer.train(data_path)
    except Exception as e:
        console.print(f"[bold red]Training failed:[/bold red] {e}")


def run_deployment_wizard_for_pretrained(model_path: str, config: dict = None):
    """Deploy a custom pre-trained model."""
    console.print(f"\n[bold cyan]🚀 Deploying {model_path}[/bold cyan]\n")
    
    from saara.deployer import ModelDeployer
    
    deployer = ModelDeployer(config)
    deployer.deploy_menu(model_path, None)  # No adapter for full models

def run_worker_wizard(config: dict = None):
    """Wizard to generate cloud worker notebooks."""
    from saara.gpu_workers import TokenManager, generate_kaggle_notebook, generate_colab_notebook, WorkerManager
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold cyan]🕸️ Connect Cloud GPUs[/bold cyan]\n\n"
        "Generate setup scripts to connect free GPUs (Kaggle/Colab) to your local training.",
        title="Decentralized Compute",
        border_style="cyan"
    ))

    # We leave the URL blank for the user to fill in the notebook
    server_url = "ENTER_YOUR_NGROK_URL_HERE"
    
    # Generate Token
    token_manager = TokenManager()
    token = token_manager.generate_token(name="Cloud Worker")
    
    console.print(f"\n[green]✅ Generated Worker Token:[/green] {token}")
    console.print("[dim]This token has been pre-filled in the notebook for you.[/dim]")
    
    console.print("\n[bold]Select Cloud Provider:[/bold]")
    console.print("  1. Kaggle (2x T4 GPU - Free)")
    console.print("  2. Google Colab (T4 GPU - Free)")
    console.print("  3. RunPod / Other (Shell Script)")
    
    choice = Prompt.ask("Choice", choices=["1", "2", "3"], default="1")
    
    output_filename = ""
    
    if choice == "1":
        content = generate_kaggle_notebook(server_url, token)
        output_filename = "saara_kaggle_worker.ipynb"
    elif choice == "2":
        content = generate_colab_notebook(server_url, token)
        output_filename = "saara_colab_worker.ipynb"
    else:
        # Generic script
        content = f"""
# SAARA Worker Setup
pip install -U saara-ai
saara worker connect --url {server_url} --token {token}
"""
        output_filename = "saara_worker.sh"

    # Ask for destination
    console.print("\n[bold]Save Notebook[/bold]")
    default_path = os.getcwd()
    dest_path = Prompt.ask("Where should I save the notebook?", default=default_path)
    
    save_path = Path(dest_path)
    if save_path.is_dir():
        save_path = save_path / output_filename
    
    try:
        with open(save_path, "w") as f:
            f.write(content)
        console.print(f"\n[bold green]✅ Saved notebook to: {save_path}[/bold green]")
    except Exception as e:
        console.print(f"[red]Failed to save file: {e}[/red]")
        return
        
    console.print("\n[yellow]Instructions:[/yellow]")
    console.print(f"1. Upload '{save_path.name}' to your cloud platform")
    console.print(f"2. [bold red]IMPORTANT:[/bold red] In the notebook, replace '{server_url}' with your Ngrok/Public URL")
    console.print(f"   (Token is already pre-filled: [bold cyan]{token}[/bold cyan])")
    console.print("3. Run all cells to connect back to this machine")
    
    if Confirm.ask("Start Worker Server now?", default=True):
        console.print("\n[bold yellow]⚠️  The server will run in this terminal.[/bold yellow]")
        console.print("[bold green]👉 Please open a NEW terminal window to continue with Training/Dataset tasks.[/bold green]\n")
        
        console.print("[dim]Starting server... (Press Ctrl+C to stop)[/dim]")
        
        # Start server logic (Blocking)
        from saara.gpu_workers import create_worker_server, WorkerManager
        import uvicorn
        
        manager = WorkerManager()
        manager.start()
        app = create_worker_server(manager)
        
        if app:
            try:
                # Listen on all interfaces
                console.print(f"[bold cyan]🚀 Server listening on 0.0.0.0:8765[/bold cyan]")
                console.print("[dim]Connect your ngrok/tunnel to port 8765[/dim]")
                uvicorn.run(app, host="0.0.0.0", port=8765)
            except KeyboardInterrupt:
                console.print("[yellow]Server stopped.[/yellow]")
            finally:
                manager.stop()

# Add the worker-serve command to CLI
@app.command()
def worker_serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8765, help="Port to bind to")
):
    """Start the GPU worker server (background task)."""
    from saara.gpu_workers import create_worker_server, WorkerManager
    import uvicorn
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("saara.server")
    logger.info(f"Starting SAARA Worker Server on {host}:{port}")
    
    manager = WorkerManager()
    manager.start()
    app = create_worker_server(manager)
    
    if app:
        try:
            uvicorn.run(app, host=host, port=port)
        finally:
            manager.stop()

@app.command()
def run():
    """Start the interactive workflow wizard."""
    console.print(Panel.fit(
        "[bold magenta]SAARA AI - CLI[/bold magenta]\n"
        "[dim]Super Autonomous Artificial Reasoning Agent[/dim]",
        border_style="magenta"
    ))
    
    # Check for config
    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
    # Interactive Menu
    while True:
        console.print("\n[bold cyan]Choose Your Workflow[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="dim", width=6)
        table.add_column("Mode", style="bold")
        table.add_column("Description")
        
        table.add_row("1", "📄 Dataset Creation", "Extract data from PDFs → Generate training datasets")
        table.add_row("2", "🧠 Model Training", "Fine-tune LLMs on your prepared data")
        table.add_row("3", "🧪 Model Evaluation", "Test & improve trained models")
        table.add_row("4", "🚀 Model Deployment", "Deploy models locally or to cloud")
        table.add_row("5", "🏗️ Pre-training", "Build & train a model from scratch")
        table.add_row("6", "🕸️ Cloud Workers", "Generate notebooks for Kaggle/Colab GPUs")
        
        console.print(table)
        
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6"], default="1")
        
        if choice == "1":
            run_dataset_wizard(config)
        elif choice == "2":
             # Refresh config in case it changed
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            run_training_wizard(config=config)
        elif choice == "3":
            run_evaluation_wizard(config)
        elif choice == "4":
            # Deployment
            base_model = config.get("model", {}).get("name", "sarvamai/sarvam-1")
            adapter = config.get("model", {}).get("adapter_path", "models/latest")
            run_deployment_wizard(base_model, adapter, config)
        elif choice == "5":
            run_pretrain_wizard(config)
        elif choice == "6":
            run_worker_wizard(config)
        
        if not Confirm.ask("\nPerform another task?", default=False):
            console.print("[bold green]Goodbye! 👋[/bold green]")
            break



@app.command()
def pretrain():
    """
    Build and train a language model from scratch.
    
    Launch the pre-training wizard to:
    - Create pre-training datasets from PDFs/text
    - Select model architecture (15M to 3B parameters)
    - Train custom tokenizers
    - Pre-train on your data
    - Test and evaluate
    """
    run_pretrain_wizard()


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

# ============================================================================
# Data-Centric AI Protocols (Saara Refinery)
# ============================================================================

@app.command()
def clean(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    strict: bool = typer.Option(False, "--strict", help="Apply strict filtering rules")
):
    """
    🧹 Protocol 1: Clean data using heuristic filters.
    """
    from saara.protocols import ProtocolEngine
    engine = ProtocolEngine()
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "cleaned"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]🧹 Saara Cleaner Protocol[/bold cyan]")
    console.print(f"   Strict Mode: {'Enabled' if strict else 'Disabled'}")
    
    # Simple logic to handle file/dir
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt")) + list(input_path.glob("*.jsonl"))
    
    processed_count = 0
    with console.status("Cleaning data..."):
        for f in files:
            try:
                text = f.read_text(encoding='utf-8')
                
                if strict:
                    # In strict mode, we filter out docs that fail heuristics
                    ok, reason = engine.cleaner.check_heuristics(text)
                    if not ok:
                        console.print(f"   [yellow]Filtered {f.name}: {reason}[/yellow]")
                        continue
                
                # Check perplexity if text is long enough
                if len(text) > 100:
                    ok, score = engine.cleaner.check_perplexity(text)
                    if not ok:
                         console.print(f"   [yellow]Filtered {f.name}: High Perplexity ({score:.2f})[/yellow]")
                         continue

                out_file = output_path / f.name
                out_file.write_text(text, encoding='utf-8')
                processed_count += 1
            except Exception as e:
                console.print(f"[red]Failed to process {f.name}: {e}[/red]")
                
    console.print(f"[green]✅ Cleaned {processed_count} files. Output: {output_path}[/green]")


@app.command()
def dedupe(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    fuzzy: float = typer.Option(0.8, "--fuzzy", help="MinHash similarity threshold (0.0-1.0)")
):
    """
    👯 Protocol 2: Deduplicate data using MinHash LSH.
    """
    from saara.protocols import ProtocolEngine
    # Initialize engine with custom threshold
    engine = ProtocolEngine()
    engine.deduper.threshold = fuzzy
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "deduped"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]👯 Saara Deduplication Protocol[/bold cyan]")
    console.print(f"   Fuzzy Threshold: {fuzzy}")
    
    # Collect all texts
    texts = []
    filenames = []
    
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt"))
    
    with console.status("Indexing documents..."):
        for f in files:
            try:
                texts.append(f.read_text(encoding='utf-8'))
                filenames.append(f)
            except: pass
            
    if not texts:
        console.print("[red]No valid text files found.[/red]")
        return

    with console.status(f"Deduplicating {len(texts)} documents..."):
        unique_indices = engine.deduper.find_duplicates(texts)
    
    # Save unique
    count = 0
    for idx in unique_indices:
        out_file = output_path / filenames[idx].name
        out_file.write_text(texts[idx], encoding='utf-8')
        count += 1
        
    console.print(f"[green]✅ Deduped: {len(texts)} -> {count} unique docs. Output: {output_path}[/green]")


@app.command()
def verify(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    engine: str = typer.Option("sympy", "--engine", help="Verification engine")
):
    """
    ✅ Protocol 3: Verify math/logic consistency.
    """
    from saara.protocols import ProtocolEngine
    protocol = ProtocolEngine()
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "verified"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]✅ Saara Verification Protocol[/bold cyan]")
    console.print(f"   Engine: {engine}")
    
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt"))
    
    pass_count = 0
    with console.status("Verifying logic..."):
        for f in files:
            text = f.read_text(encoding='utf-8')
            if protocol.verifier.verify_math_consistency(text):
                out_file = output_path / f.name
                out_file.write_text(text, encoding='utf-8')
                pass_count += 1
            else:
                console.print(f"   [yellow]⚠️ Failed verification: {f.name}[/yellow]")

    console.print(f"[green]✅ Verified {pass_count}/{len(files)} docs. Output: {output_path}[/green]")


@app.command()
def redact(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    all_pii: bool = typer.Option(True, "--all", help="Redact all PII types")
):
    """
    🔒 Protocol 4: Redact Sensitive PII.
    """
    from saara.protocols import ProtocolEngine
    engine = ProtocolEngine()
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "redacted"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]🔒 Saara PII Redaction Protocol[/bold cyan]")
    
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt"))
    
    count = 0
    with console.status("Redacting PII..."):
        for f in files:
            text = f.read_text(encoding='utf-8')
            redacted = engine.guard.redact(text)
            
            if redacted != text:
                console.print(f"   [dim]Redacted info in {f.name}[/dim]")
            
            out_file = output_path / f.name
            out_file.write_text(redacted, encoding='utf-8')
            count += 1
            
    console.print(f"[green]✅ Secured {count} docs. Output: {output_path}[/green]")


# ============================================================================
# Data-Centric AI Protocols (Saara Refinery)
# ============================================================================

@app.command()
def clean(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    strict: bool = typer.Option(False, "--strict", help="Apply strict filtering rules")
):
    """
    🧹 Protocol 1: Clean data using heuristic filters.
    """
    from saara.protocols import ProtocolEngine
    engine = ProtocolEngine()
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "cleaned"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]🧹 Saara Cleaner Protocol[/bold cyan]")
    console.print(f"   Strict Mode: {'Enabled' if strict else 'Disabled'}")
    
    # Simple logic to handle file/dir
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt")) + list(input_path.glob("*.jsonl"))
    
    processed_count = 0
    with console.status("Cleaning data..."):
        for f in files:
            try:
                text = f.read_text(encoding='utf-8')
                
                if strict:
                    # In strict mode, we filter out docs that fail heuristics
                    ok, reason = engine.cleaner.check_heuristics(text)
                    if not ok:
                        console.print(f"   [yellow]Filtered {f.name}: {reason}[/yellow]")
                        continue
                
                # Check perplexity if text is long enough
                if len(text) > 100:
                    ok, score = engine.cleaner.check_perplexity(text)
                    if not ok:
                         console.print(f"   [yellow]Filtered {f.name}: High Perplexity ({score:.2f})[/yellow]")
                         continue

                out_file = output_path / f.name
                out_file.write_text(text, encoding='utf-8')
                processed_count += 1
            except Exception as e:
                console.print(f"[red]Failed to process {f.name}: {e}[/red]")
                
    console.print(f"[green]✅ Cleaned {processed_count} files. Output: {output_path}[/green]")


@app.command()
def dedupe(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    fuzzy: float = typer.Option(0.8, "--fuzzy", help="MinHash similarity threshold (0.0-1.0)")
):
    """
    👯 Protocol 2: Deduplicate data using MinHash LSH.
    """
    from saara.protocols import ProtocolEngine
    # Initialize engine with custom threshold
    engine = ProtocolEngine()
    engine.deduper.threshold = fuzzy
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "deduped"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]👯 Saara Deduplication Protocol[/bold cyan]")
    console.print(f"   Fuzzy Threshold: {fuzzy}")
    
    # Collect all texts
    texts = []
    filenames = []
    
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt"))
    
    with console.status("Indexing documents..."):
        for f in files:
            try:
                texts.append(f.read_text(encoding='utf-8'))
                filenames.append(f)
            except: pass
            
    if not texts:
        console.print("[red]No valid text files found.[/red]")
        return

    with console.status(f"Deduplicating {len(texts)} documents..."):
        unique_indices = engine.deduper.find_duplicates(texts)
    
    # Save unique
    count = 0
    for idx in unique_indices:
        out_file = output_path / filenames[idx].name
        out_file.write_text(texts[idx], encoding='utf-8')
        count += 1
        
    console.print(f"[green]✅ Deduped: {len(texts)} -> {count} unique docs. Output: {output_path}[/green]")


@app.command()
def verify(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    engine: str = typer.Option("sympy", "--engine", help="Verification engine")
):
    """
    ✅ Protocol 3: Verify math/logic consistency.
    """
    from saara.protocols import ProtocolEngine
    protocol = ProtocolEngine()
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "verified"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]✅ Saara Verification Protocol[/bold cyan]")
    console.print(f"   Engine: {engine}")
    
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt"))
    
    pass_count = 0
    with console.status("Verifying logic..."):
        for f in files:
            text = f.read_text(encoding='utf-8')
            if protocol.verifier.verify_math_consistency(text):
                out_file = output_path / f.name
                out_file.write_text(text, encoding='utf-8')
                pass_count += 1
            else:
                console.print(f"   [yellow]⚠️ Failed verification: {f.name}[/yellow]")

    console.print(f"[green]✅ Verified {pass_count}/{len(files)} docs. Output: {output_path}[/green]")


@app.command()
def redact(
    file: str = typer.Argument(..., help="Input file or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    all_pii: bool = typer.Option(True, "--all", help="Redact all PII types")
):
    """
    🔒 Protocol 4: Redact Sensitive PII.
    """
    from saara.protocols import ProtocolEngine
    engine = ProtocolEngine()
    
    input_path = Path(file)
    output_path = Path(output) if output else input_path.parent / "redacted"
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]🔒 Saara PII Redaction Protocol[/bold cyan]")
    
    files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt"))
    
    count = 0
    with console.status("Redacting PII..."):
        for f in files:
            text = f.read_text(encoding='utf-8')
            redacted = engine.guard.redact(text)
            
            if redacted != text:
                console.print(f"   [dim]Redacted info in {f.name}[/dim]")
            
            out_file = output_path / f.name
            out_file.write_text(redacted, encoding='utf-8')
            count += 1
            
    console.print(f"[green]✅ Secured {count} docs. Output: {output_path}[/green]")


@app.command()
def deploy(
    model: str = typer.Option("sarvamai/sarvam-1", help="Base model ID"),
    adapter: str = typer.Option("models/latest", help="Path to adapter"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path")
):
    """
    🚀 Deploy a model (Cloud, Docker, or HuggingFace).
    """
    cfg = {}
    if Path(config).exists():
        with open(config) as f:
            cfg = yaml.safe_load(f)
            
    from saara.deployer import run_deployment_wizard
    run_deployment_wizard(model, adapter, cfg)


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
    
    # Step 5: HuggingFace Authentication (optional)
    console.print("\n[bold]🤗 Step 5: HuggingFace Authentication (Optional)[/bold]\n")
    console.print("[dim]Required to access gated models like Llama 3, Gemma, etc.[/dim]\n")
    
    # Check if already logged in
    hf_logged_in = False
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        console.print(f"[green]✓ Already logged in as: {user_info.get('name', user_info.get('fullname', 'Unknown'))}[/green]")
        hf_logged_in = True
    except:
        pass
    
    if not hf_logged_in:
        if Confirm.ask("Do you want to configure HuggingFace access?", default=True):
            console.print("\n[bold]Get your token from:[/bold] [cyan]https://huggingface.co/settings/tokens[/cyan]")
            console.print("[dim]Make sure to create a token with 'read' access.[/dim]\n")
            
            import getpass
            try:
                hf_token = getpass.getpass("Enter your HuggingFace token (or press Enter to skip): ")
                
                if hf_token.strip():
                    from huggingface_hub import login
                    login(token=hf_token.strip())
                    console.print("[green]✓ Successfully logged in to HuggingFace![/green]")
                    hf_logged_in = True
                else:
                    console.print("[dim]Skipped HuggingFace login.[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not login: {e}[/yellow]")
                console.print("[dim]You can login later using: huggingface-cli login[/dim]")
    
    # Step 6: Save configuration
    console.print("\n[bold]💾 Step 6: Saving Configuration[/bold]\n")
    
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
        },
        "huggingface": {
            "logged_in": hf_logged_in
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
        "  saara train            - Fine-tune a model\n"
        "  saara rag create       - Build RAG agent\n\n"
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
    
    # Storage usage
    console.print()
    trained.display_storage_usage()


@models_app.command("hardware")
def models_hardware(
    mode: str = typer.Option("both", "--mode", "-m", help="Check for: pretrain, finetune, or both")
):
    """
    Check hardware capabilities for model training.
    
    Shows GPU/RAM info and recommends model architectures based on your hardware.
    If your hardware isn't sufficient for larger models, cloud options are displayed.
    
    Examples:
        saara models hardware              # Check for both pretraining and finetuning
        saara models hardware -m pretrain  # Check for pretraining only
        saara models hardware -m finetune  # Check for finetuning only
    """
    from saara.model_manager import HardwareDetector
    
    if mode not in ["pretrain", "finetune", "both"]:
        console.print(f"[red]Invalid mode: {mode}. Use: pretrain, finetune, or both[/red]")
        raise typer.Exit(code=1)
    
    HardwareDetector.run_hardware_check_wizard(mode=mode)


@models_app.command("info")
def models_info(
    model_name: str = typer.Argument(None, help="Model name to get info for")
):
    """
    Show detailed information about a specific model.
    
    Examples:
        saara models info my-model-finetuned
    """
    from saara.model_manager import TrainedModelManager
    
    trained = TrainedModelManager()
    
    if not model_name:
        # Interactive mode - select from list
        models = trained.list_trained_models()
        
        if not models:
            console.print("[yellow]No trained models found.[/yellow]")
            return
        
        console.print("[bold]Select a model to inspect:[/bold]")
        for i, m in enumerate(models, 1):
            console.print(f"  {i}. {m['name']}")
        
        choice = Prompt.ask("Enter model number", choices=[str(i) for i in range(1, len(models)+1)])
        model_name = models[int(choice)-1]["name"]
    
    trained.display_model_info(model_name)


@models_app.command("clear")
def models_clear(
    target: str = typer.Argument("checkpoints", help="What to clear: checkpoints, models, datasets, tokenizers, all"),
    model_name: str = typer.Option(None, "--model", "-m", help="Specific model name (for checkpoints)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Clear/delete models, checkpoints, datasets, or tokenizers.
    
    Examples:
        saara models clear checkpoints           # Clear all checkpoints
        saara models clear checkpoints -m mymodel  # Clear specific model's checkpoints
        saara models clear models --yes          # Delete ALL trained models
        saara models clear datasets --yes        # Delete all datasets
        saara models clear all --yes             # Factory reset (everything)
    """
    from saara.model_manager import TrainedModelManager
    
    trained = TrainedModelManager()
    
    target = target.lower()
    
    if target == "checkpoints":
        count = trained.clear_all_checkpoints(model_name)
        if count == 0 and model_name is None:
            console.print("[yellow]No checkpoints found.[/yellow]")
            
    elif target == "models":
        if not yes:
            models = trained.list_trained_models()
            if not models:
                console.print("[yellow]No models to delete.[/yellow]")
                return
            
            console.print(f"[red]⚠️  This will delete {len(models)} trained models:[/red]")
            for m in models:
                console.print(f"  • {m['name']} ({m['size_mb']:.1f} MB)")
            
            if not Confirm.ask("Are you sure?", default=False):
                console.print("[dim]Cancelled.[/dim]")
                return
        
        trained.clear_all_models(confirm=True)
        
    elif target == "datasets":
        if not yes:
            if not Confirm.ask("Delete all datasets?", default=False):
                console.print("[dim]Cancelled.[/dim]")
                return
        
        trained.clear_datasets(confirm=True)
        
    elif target == "tokenizers":
        if not yes:
            if not Confirm.ask("Delete all custom tokenizers?", default=False):
                console.print("[dim]Cancelled.[/dim]")
                return
        
        trained.clear_tokenizers(confirm=True)
        
    elif target == "all":
        if not yes:
            console.print(Panel(
                "[bold red]⚠️  FACTORY RESET[/bold red]\n\n"
                "This will delete:\n"
                "• All trained models\n"
                "• All generated datasets\n"
                "• All custom tokenizers\n\n"
                "[yellow]This action cannot be undone![/yellow]",
                title="⚠️  Warning",
                border_style="red"
            ))
            
            if not Confirm.ask("Are you absolutely sure?", default=False):
                console.print("[dim]Cancelled.[/dim]")
                return
        
        trained.reset_all(confirm=True)
        
    else:
        console.print(f"[red]Unknown target: {target}[/red]")
        console.print("Use: checkpoints, models, datasets, tokenizers, or all")


@models_app.command("storage")
def models_storage():
    """
    Show disk storage usage for models, datasets, and tokenizers.
    """
    from saara.model_manager import TrainedModelManager
    
    trained = TrainedModelManager()
    trained.display_storage_usage()
    
    # Also show breakdown by model
    models = trained.list_trained_models()
    
    if models:
        console.print("\n[bold]Storage by Model:[/bold]")
        for m in sorted(models, key=lambda x: x['size_mb'], reverse=True):
            bar_width = int(min(m['size_mb'] / 100, 30))  # Scale to max 30 chars
            bar = "█" * bar_width + "░" * (30 - bar_width)
            console.print(f"  {m['name'][:25]:25} [{bar}] {m['size_mb']:.1f} MB")


@models_app.command("retrain")
def models_retrain(
    model_name: str = typer.Argument(None, help="Model to retrain from scratch")
):
    """
    Delete a model and retrain it from scratch using the same settings.
    
    This will:
    1. Save the model's configuration (base model, LoRA settings)
    2. Delete the old model
    3. Start fresh training with the same configuration
    
    Examples:
        saara models retrain my-finetuned-model
    """
    from saara.model_manager import TrainedModelManager
    
    trained = TrainedModelManager()
    
    if not model_name:
        # Interactive mode
        models = trained.list_trained_models()
        
        if not models:
            console.print("[yellow]No models to retrain.[/yellow]")
            return
        
        console.print("[bold]Select a model to retrain from scratch:[/bold]")
        for i, m in enumerate(models, 1):
            console.print(f"  {i}. {m['name']} ({m['base_model'].split('/')[-1]})")
        
        choice = Prompt.ask("Enter model number", choices=[str(i) for i in range(1, len(models)+1)])
        model_name = models[int(choice)-1]["name"]
    
    # Get model info before deletion
    retrain_config = trained.prepare_for_retrain(model_name)
    
    if not retrain_config:
        console.print(f"[red]Model not found: {model_name}[/red]")
        return
    
    console.print(Panel(
        f"[bold]Retrain Configuration[/bold]\n\n"
        f"Model: {retrain_config['original_name']}\n"
        f"Base: {retrain_config.get('base_model', 'Unknown')}\n"
        f"Type: {retrain_config.get('type', 'Unknown')}\n"
        + (f"LoRA Rank: {retrain_config.get('lora_rank')}\n" if retrain_config.get('lora_rank') else ""),
        title="🔄 Retrain From Scratch",
        border_style="yellow"
    ))
    
    console.print(f"\n[yellow]This will delete '{model_name}' and start fresh training.[/yellow]")
    
    if not Confirm.ask("Continue?", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return
    
    # Delete old model
    trained.delete_trained_model(model_name)
    
    # Show next steps
    base_model = retrain_config.get('base_model', 'Unknown')
    
    console.print(Panel(
        f"[green]✓ Model deleted. Ready to retrain![/green]\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"1. Prepare your training data\n"
        f"2. Run: [cyan]saara train --model {base_model}[/cyan]\n\n"
        f"Or use the interactive wizard:\n"
        f"   [cyan]saara train[/cyan]",
        title="🚀 Ready to Retrain",
        border_style="green"
    ))


# ============================================================================
# ACCELERATOR & VISUALIZER COMMANDS
# ============================================================================

@app.command()
def accelerator():
    """
    Show neural accelerator status and GPU optimization info.
    
    Displays:
    - Device detection (CUDA/CPU/MPS)
    - GPU information and memory
    - Optimization settings
    - Recommended training parameters
    """
    from saara.accelerator import NeuralAccelerator, AcceleratorConfig
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Neural Accelerator Status[/bold cyan]\n\n"
        "Hardware acceleration for neural network training.",
        title="SAARA Accelerator",
        border_style="cyan"
    ))
    
    # Create accelerator and display info
    config = AcceleratorConfig()
    accelerator = NeuralAccelerator(config)
    accelerator.display_status()
    
    # Show recommended settings
    console.print("\n[bold]Recommended Training Settings:[/bold]")
    
    device_info = accelerator.get_device_info()
    
    if device_info.get("device") == "cuda":
        vram = device_info.get("gpu_memory_total_gb", 0)
        
        if vram >= 40:
            settings = {"batch_size": 16, "grad_accum": 2, "max_length": 4096, "precision": "bf16"}
        elif vram >= 20:
            settings = {"batch_size": 8, "grad_accum": 4, "max_length": 2048, "precision": "fp16"}
        elif vram >= 12:
            settings = {"batch_size": 4, "grad_accum": 8, "max_length": 2048, "precision": "fp16"}
        else:
            settings = {"batch_size": 2, "grad_accum": 16, "max_length": 1024, "precision": "fp16+4bit"}
    else:
        settings = {"batch_size": 1, "grad_accum": 32, "max_length": 512, "precision": "fp32"}
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Batch Size", str(settings["batch_size"]))
    table.add_row("Gradient Accumulation", str(settings["grad_accum"]))
    table.add_row("Max Sequence Length", str(settings["max_length"]))
    table.add_row("Precision", settings["precision"])
    table.add_row("Gradient Checkpointing", "✅ Enabled")
    
    console.print(table)


@app.command()
def visualize(
    model_path: str = typer.Argument(None, help="Path to model or adapter to visualize"),
    report: bool = typer.Option(False, "--report", "-r", help="Generate HTML report")
):
    """
    Visualize neural network architecture and training metrics.
    
    Examples:
        saara visualize                      # Show demo visualization
        saara visualize models/my-model      # Analyze specific model
        saara visualize --report             # Generate HTML report
    """
    from saara.visualizer import (
        create_visualizer, 
        ModelAnalyzer, 
        NetworkArchitecture,
        LayerInfo,
        HTMLReportGenerator,
        TrainingHistory,
        TrainingSnapshot
    )
    
    console.print(Panel.fit(
        "[bold cyan]🔍 Neural Network Visualizer[/bold cyan]",
        border_style="cyan"
    ))
    
    if model_path and Path(model_path).exists():
        # Analyze actual model
        try:
            import torch
            from transformers import AutoModel
            
            console.print(f"[dim]Loading model from {model_path}...[/dim]")
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            
            analyzer = ModelAnalyzer()
            arch = analyzer.analyze(model, model_name=Path(model_path).name)
            
            visualizer = create_visualizer()
            visualizer.visualize_architecture(arch)
            
        except Exception as e:
            console.print(f"[yellow]Could not load model: {e}[/yellow]")
            console.print("[dim]Showing demo architecture...[/dim]\n")
            _show_demo_architecture()
    else:
        # Show demo
        _show_demo_architecture()
    
    if report:
        # Generate HTML report
        from saara.visualizer import HTMLReportGenerator, TrainingHistory, TrainingSnapshot
        import random
        
        # Create sample data
        arch = NetworkArchitecture(
            name="Demo Neural Network",
            layers=[
                LayerInfo("embed", "Embedding", (), (512,), 32000 * 512, True),
                LayerInfo("attn.0", "MultiHeadAttention", (512,), (512,), 512 * 512 * 4, True),
                LayerInfo("ffn.0", "FeedForward", (512,), (512,), 512 * 2048 * 2, True),
            ],
            total_parameters=50_000_000,
            trainable_parameters=50_000_000,
            model_size_mb=200
        )
        
        history = TrainingHistory()
        for i in range(100):
            history.add(TrainingSnapshot(
                step=i,
                epoch=i // 20,
                loss=2.0 * (0.95 ** i) + random.uniform(-0.1, 0.1),
                learning_rate=2e-4 * (0.98 ** i),
                gpu_memory_gb=11.5,
                throughput=120
            ))
        
        generator = HTMLReportGenerator("reports")
        report_path = generator.generate_training_report(arch, history)
        console.print(f"\n[green]✅ Report saved to: {report_path}[/green]")


def _show_demo_architecture():
    """Show a demo neural network architecture."""
    from saara.visualizer import create_visualizer, NetworkArchitecture, LayerInfo
    
    # Create demo architecture
    demo_arch = NetworkArchitecture(
        name="Gemma-2-2B (Demo)",
        layers=[
            LayerInfo("model.embed_tokens", "Embedding", (1, 2048), (1, 2048, 2048), 262_144_000, True),
            LayerInfo("model.layers.0.self_attn", "GemmaAttention", (1, 2048, 2048), (1, 2048, 2048), 16_777_216, True),
            LayerInfo("model.layers.0.mlp", "GemmaMLP", (1, 2048, 2048), (1, 2048, 2048), 33_554_432, True),
            LayerInfo("model.layers.1.self_attn", "GemmaAttention", (1, 2048, 2048), (1, 2048, 2048), 16_777_216, True),
            LayerInfo("model.layers.1.mlp", "GemmaMLP", (1, 2048, 2048), (1, 2048, 2048), 33_554_432, True),
            LayerInfo("model.norm", "GemmaRMSNorm", (1, 2048, 2048), (1, 2048, 2048), 2048, True),
            LayerInfo("lm_head", "Linear", (1, 2048, 2048), (1, 2048, 256000), 524_288_000, True),
        ],
        total_parameters=2_500_000_000,
        trainable_parameters=2_500_000_000,
        model_size_mb=4800
    )
    
    visualizer = create_visualizer()
    visualizer.visualize_architecture(demo_arch)


# ============================================================================
# CLOUD RUNTIME COMMANDS
# ============================================================================

# Create cloud subcommand group for better organization
cloud_app = typer.Typer(help="Cloud runtime and GPU worker management")
app.add_typer(cloud_app, name="cloud")


@cloud_app.callback(invoke_without_command=True)
def cloud_callback(ctx: typer.Context):
    """Cloud runtime management for Colab/Kaggle/cloud environments."""
    if ctx.invoked_subcommand is None:
        # Default to showing info
        from saara.cloud_runtime import CloudRuntime, get_environment_info
        
        runtime = CloudRuntime()
        runtime.display_info()
        
        # Show API status
        console.print("\n[bold]API Provider Status:[/bold]")
        
        providers = [
            ("GOOGLE_API_KEY / GEMINI_API_KEY", "Gemini", "☁️ Recommended for Colab"),
            ("GROQ_API_KEY", "Groq", "⚡ Ultra-fast inference"),
            ("DEEPSEEK_API_KEY", "DeepSeek", "💰 Cost-effective"),
            ("OPENAI_API_KEY", "OpenAI", "🤖 GPT models"),
            ("HF_TOKEN", "HuggingFace", "🤗 Open models"),
        ]
        
        for env_var, name, desc in providers:
            keys = env_var.split(" / ")
            available = any(os.environ.get(k) for k in keys)
            status = "[green]✓ Configured[/green]" if available else "[dim]○ Not set[/dim]"
            console.print(f"  {status} {name} - {desc}")
        
        console.print("\n[dim]Use 'saara cloud --help' to see all cloud commands.[/dim]")


@cloud_app.command("info")
def cloud_info():
    """Show cloud environment information and API status."""
    from saara.cloud_runtime import CloudRuntime, get_environment_info
    
    runtime = CloudRuntime()
    runtime.display_info()
    
    # Show API status
    console.print("\n[bold]API Provider Status:[/bold]")
    
    providers = [
        ("GOOGLE_API_KEY / GEMINI_API_KEY", "Gemini", "☁️ Recommended for Colab"),
        ("GROQ_API_KEY", "Groq", "⚡ Ultra-fast inference"),
        ("DEEPSEEK_API_KEY", "DeepSeek", "💰 Cost-effective"),
        ("OPENAI_API_KEY", "OpenAI", "🤖 GPT models"),
        ("HF_TOKEN", "HuggingFace", "🤗 Open models"),
    ]
    
    for env_var, name, desc in providers:
        keys = env_var.split(" / ")
        available = any(os.environ.get(k) for k in keys)
        status = "[green]✓ Configured[/green]" if available else "[dim]○ Not set[/dim]"
        console.print(f"  {status} {name} - {desc}")


@cloud_app.command("setup")
def cloud_setup():
    """Configure cloud API providers (Gemini, Groq, OpenAI, etc.)."""
    from saara.cloud_runtime import CloudRuntime
    
    console.print(Panel.fit(
        "[bold cyan]☁️ Cloud API Setup[/bold cyan]\n\n"
        "Configure API providers for cloud environments.\n"
        "[dim]In Colab, use Secrets to set these securely.[/dim]",
        title="Cloud Setup",
        border_style="cyan"
    ))
    
    # Interactive API key setup
    console.print("\n[bold]Select API provider to configure:[/bold]")
    console.print("  1. Google AI (Gemini) - [green]Recommended, free tier available[/green]")
    console.print("  2. Groq - [cyan]Free, ultra-fast inference[/cyan]")
    console.print("  3. DeepSeek - [yellow]Very cheap, good quality[/yellow]")
    console.print("  4. OpenAI - GPT models")
    console.print("  5. HuggingFace - Open source models")
    console.print("  0. Skip")
    
    choice = Prompt.ask("Choose provider", choices=["0", "1", "2", "3", "4", "5"], default="1")
    
    if choice == "0":
        return
    
    provider_map = {
        "1": ("GOOGLE_API_KEY", "https://aistudio.google.com/apikey"),
        "2": ("GROQ_API_KEY", "https://console.groq.com/keys"),
        "3": ("DEEPSEEK_API_KEY", "https://platform.deepseek.com/api_keys"),
        "4": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
        "5": ("HF_TOKEN", "https://huggingface.co/settings/tokens"),
    }
    
    env_var, url = provider_map[choice]
    console.print(f"\n[dim]Get your API key from: {url}[/dim]")
    
    api_key = Prompt.ask("Enter API key", password=True)
    
    if api_key:
        os.environ[env_var] = api_key
        console.print(f"[green]✓ {env_var} set for this session[/green]")
        
        console.print("\n[bold]To persist this key:[/bold]")
        console.print(f"  • In Colab: Add to Secrets as '{env_var}'")
        console.print(f"  • In terminal: export {env_var}='your-key'")
        
        # Test the connection
        console.print("\n[dim]Testing connection...[/dim]")
        runtime = CloudRuntime()
        runtime.setup(api_key=api_key)


@cloud_app.command("quickstart")
def cloud_quickstart():
    """Display quickstart guide for Colab/Kaggle."""
    from saara.cloud_runtime import colab_quickstart
    colab_quickstart()


@cloud_app.command("connect")
def cloud_connect(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8765, "--port", "-p", help="Port to bind to"),
    use_ngrok: bool = typer.Option(False, "--ngrok", help="Use ngrok for public URL"),
    ngrok_token: str = typer.Option(None, "--ngrok-token", help="ngrok auth token")
):
    """
    Start the GPU worker server to accept connections from cloud GPUs.
    
    This allows Kaggle/Colab notebooks to connect and execute training jobs.
    
    Examples:
        saara cloud connect                    # Start on localhost:8765
        saara cloud connect --ngrok            # Use ngrok for public URL
        saara cloud connect -p 9000            # Use custom port
    """
    from saara.gpu_workers import (
        WorkerManager, TokenManager, create_worker_server, display_workers
    )
    
    console.print(Panel.fit(
        "[bold cyan]🌐 GPU Worker Server[/bold cyan]\n\n"
        "Start a server to accept connections from cloud GPU workers.\n"
        "[dim]Workers from Kaggle, Colab, etc. can connect to run training jobs.[/dim]",
        title="Cloud GPU Connect",
        border_style="cyan"
    ))
    
    # Initialize managers
    token_manager = TokenManager()
    worker_manager = WorkerManager(token_manager)
    worker_manager.start()
    
    # Generate a token for workers
    token = token_manager.generate_token(name="CLI Session Token", expires_hours=24)
    
    # Create server
    app_server = create_worker_server(worker_manager, host=host, port=port)
    
    if app_server is None:
        console.print("[red]❌ Failed to create server. Install FastAPI: pip install fastapi uvicorn[/red]")
        return
    
    # Determine public URL
    public_url = f"http://{host}:{port}"
    
    if use_ngrok:
        try:
            from pyngrok import ngrok
            
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
            
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.public_url
            console.print(f"[green]✓ ngrok tunnel created: {public_url}[/green]")
        except ImportError:
            console.print("[yellow]⚠ pyngrok not installed. Install with: pip install pyngrok[/yellow]")
            console.print("[yellow]  Falling back to localhost URL.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ ngrok error: {e}[/yellow]")
            console.print("[yellow]  Falling back to localhost URL.[/yellow]")
    
    if host == "0.0.0.0" and not use_ngrok:
        # Try to get local IP
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            public_url = f"http://{local_ip}:{port}"
        except:
            public_url = f"http://localhost:{port}"
    
    # Display connection info
    console.print("\n")
    console.print(Panel(
        f"[bold green]✅ Server Ready![/bold green]\n\n"
        f"[bold]Server URL:[/bold] [cyan]{public_url}[/cyan]\n"
        f"[bold]Worker Token:[/bold] [yellow]{token}[/yellow]\n\n"
        f"[bold]Connect from Kaggle/Colab:[/bold]\n"
        f"  1. Run: [cyan]saara cloud generate --url {public_url}[/cyan]\n"
        f"  2. Download and open the generated notebook\n"
        f"  3. Run all cells to connect the GPU worker\n\n"
        f"[dim]Press Ctrl+C to stop the server.[/dim]",
        title="🚀 GPU Worker Server",
        border_style="green"
    ))
    
    # Start the server
    try:
        import uvicorn
        console.print("\n[dim]Starting server...[/dim]\n")
        uvicorn.run(app_server, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    finally:
        worker_manager.stop()


@cloud_app.command("token")
def cloud_token(
    action: str = typer.Argument("generate", help="Action: generate, list, revoke"),
    name: str = typer.Option(None, "--name", "-n", help="Token name/description"),
    expires: int = typer.Option(24, "--expires", "-e", help="Hours until expiration (0 = never)"),
    token_id: str = typer.Option(None, "--token", "-t", help="Token to revoke")
):
    """
    Manage GPU worker authentication tokens.
    
    Actions:
        generate - Create a new worker token
        list     - List all tokens
        revoke   - Revoke a token
    
    Examples:
        saara cloud token generate --name "Kaggle Worker"
        saara cloud token list
        saara cloud token revoke --token saara_worker_abc123
    """
    from saara.gpu_workers import TokenManager
    
    token_manager = TokenManager()
    
    if action == "generate":
        token = token_manager.generate_token(name=name, expires_hours=expires)
        
        console.print(Panel(
            f"[bold green]✅ Token Generated![/bold green]\n\n"
            f"[bold]Token:[/bold] [cyan]{token}[/cyan]\n"
            f"[bold]Name:[/bold] {name or 'Unnamed'}\n"
            f"[bold]Expires:[/bold] {'Never' if expires == 0 else f'In {expires} hours'}\n\n"
            f"[dim]Use this token in your Kaggle/Colab worker notebook.[/dim]",
            title="🔑 Worker Token",
            border_style="green"
        ))
        
    elif action == "list":
        tokens = token_manager.list_tokens()
        
        if not tokens:
            console.print("[yellow]No tokens found.[/yellow]")
            return
        
        table = Table(title="🔑 Worker Tokens", show_header=True, header_style="bold cyan")
        table.add_column("Token", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Created", style="dim")
        table.add_column("Status", style="yellow")
        table.add_column("Used By", style="magenta")
        
        for t in tokens:
            status = "[green]✓ Valid[/green]" if t["is_valid"] else "[red]✗ Expired[/red]"
            if t.get("used"):
                status = "[yellow]● In Use[/yellow]"
            
            table.add_row(
                t["token"][:25] + "...",
                t.get("name", "Unnamed"),
                t.get("created_at", "")[:10],
                status,
                t.get("worker_id", "-") or "-"
            )
        
        console.print(table)
        
    elif action == "revoke":
        if not token_id:
            tokens = token_manager.list_tokens()
            if not tokens:
                console.print("[yellow]No tokens to revoke.[/yellow]")
                return
            
            console.print("[bold]Select token to revoke:[/bold]")
            for i, t in enumerate(tokens, 1):
                console.print(f"  {i}. {t['token'][:25]}... ({t.get('name', 'Unnamed')})")
            
            choice = Prompt.ask("Token number", choices=[str(i) for i in range(1, len(tokens)+1)])
            token_id = tokens[int(choice)-1]["token"]
        
        if token_manager.revoke_token(token_id):
            console.print(f"[green]✓ Token revoked: {token_id[:25]}...[/green]")
        else:
            console.print(f"[red]Token not found: {token_id}[/red]")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Use: generate, list, or revoke")


@cloud_app.command("workers")
def cloud_workers(
    all_workers: bool = typer.Option(False, "--all", "-a", help="Show disconnected workers too")
):
    """
    List connected GPU workers.
    
    Shows all workers that are currently connected or have been connected.
    
    Examples:
        saara cloud workers          # Show connected workers only
        saara cloud workers --all    # Include disconnected workers
    """
    from saara.gpu_workers import WorkerManager, TokenManager, display_workers
    
    token_manager = TokenManager()
    worker_manager = WorkerManager(token_manager)
    
    workers = worker_manager.list_workers(include_disconnected=all_workers)
    
    if not workers:
        console.print(Panel(
            "[yellow]No GPU workers connected.[/yellow]\n\n"
            "[bold]To connect a worker:[/bold]\n"
            "1. Run: [cyan]saara cloud connect[/cyan] to start the server\n"
            "2. Run: [cyan]saara cloud generate[/cyan] to create a notebook\n"
            "3. Open the notebook in Kaggle/Colab and run it",
            title="🖥️ GPU Workers",
            border_style="yellow"
        ))
        return
    
    display_workers(workers)
    
    # Show stats
    stats = worker_manager.get_stats()
    console.print(f"\n[dim]Connected: {stats['workers']['connected']} | "
                  f"Busy: {stats['workers']['busy']} | "
                  f"Disconnected: {stats['workers']['disconnected']}[/dim]")


@cloud_app.command("generate")
def cloud_generate(
    platform: str = typer.Option("colab", "--platform", "-p", help="Platform: colab, kaggle"),
    url: str = typer.Option(None, "--url", "-u", help="Server URL (if running saara cloud connect)"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path")
):
    """
    Generate a notebook for connecting cloud GPU to SAARA.
    
    Creates a ready-to-run Jupyter notebook for Kaggle or Colab
    that connects to your SAARA instance as a GPU worker.
    
    Examples:
        saara cloud generate --platform colab --url http://your-server:8765
        saara cloud generate -p kaggle -u https://your-ngrok-url.ngrok.io
    """
    from saara.gpu_workers import (
        TokenManager, generate_colab_notebook, generate_kaggle_notebook
    )
    
    if not url:
        console.print("[yellow]⚠ No server URL provided.[/yellow]")
        console.print("[dim]Start the server first with: saara cloud connect[/dim]\n")
        url = Prompt.ask("Enter server URL", default="http://localhost:8765")
    
    # Generate token
    token_manager = TokenManager()
    token = token_manager.generate_token(name=f"{platform.capitalize()} Worker", expires_hours=48)
    
    # Generate notebook
    if platform.lower() == "kaggle":
        notebook_content = generate_kaggle_notebook(url, token)
        default_filename = "saara_kaggle_worker.ipynb"
    else:
        notebook_content = generate_colab_notebook(url, token)
        default_filename = "saara_colab_worker.ipynb"
    
    # Save notebook
    output_path = Path(output) if output else Path.cwd() / default_filename
    
    with open(output_path, "w") as f:
        f.write(notebook_content)
    
    console.print(Panel(
        f"[bold green]✅ Notebook Generated![/bold green]\n\n"
        f"[bold]File:[/bold] [cyan]{output_path}[/cyan]\n"
        f"[bold]Platform:[/bold] {platform.capitalize()}\n"
        f"[bold]Server URL:[/bold] {url}\n"
        f"[bold]Token:[/bold] {token[:20]}...\n\n"
        f"[bold]Next Steps:[/bold]\n"
        f"  1. Upload this notebook to {platform.capitalize()}\n"
        f"  2. Enable GPU in runtime settings\n"
        f"  3. Run all cells to connect as a GPU worker\n\n"
        f"[dim]The notebook is pre-configured with your server URL and token.[/dim]",
        title=f"📓 {platform.capitalize()} Worker Notebook",
        border_style="green"
    ))


@cloud_app.command("jobs")
def cloud_jobs(
    action: str = typer.Argument("list", help="Action: list, create, cancel"),
    job_type: str = typer.Option("training", "--type", "-t", help="Job type: training, evaluation"),
    model: str = typer.Option(None, "--model", "-m", help="Model name for job"),
    job_id: str = typer.Option(None, "--job", "-j", help="Job ID for cancel action")
):
    """
    Manage GPU worker jobs (training, evaluation tasks).
    
    Actions:
        list   - List all jobs in queue
        create - Create a new job
        cancel - Cancel a queued/running job
    
    Examples:
        saara cloud jobs list
        saara cloud jobs create --type training --model google/gemma-2-2b
        saara cloud jobs cancel --job job_abc123
    """
    from saara.gpu_workers import (
        WorkerManager, TokenManager, JobType, JobStatus, display_jobs
    )
    
    token_manager = TokenManager()
    worker_manager = WorkerManager(token_manager)
    
    if action == "list":
        jobs = worker_manager.list_jobs()
        
        if not jobs:
            console.print("[yellow]No jobs in queue.[/yellow]")
            console.print("[dim]Create a job with: saara cloud jobs create[/dim]")
            return
        
        display_jobs(jobs)
        
        # Show stats
        stats = worker_manager.get_stats()
        console.print(f"\n[dim]Queued: {stats['jobs']['queued']} | "
                      f"Running: {stats['jobs']['running']} | "
                      f"Completed: {stats['jobs']['completed']} | "
                      f"Failed: {stats['jobs']['failed']}[/dim]")
        
    elif action == "create":
        if not model:
            console.print("[bold]Select model for job:[/bold]")
            model = Prompt.ask("Model name", default="google/gemma-2-2b")
        
        try:
            jt = JobType(job_type)
        except ValueError:
            console.print(f"[red]Invalid job type: {job_type}[/red]")
            console.print("Use: training or evaluation")
            return
        
        payload = {
            "model_name": model,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "max_length": 512
        }
        
        if jt == JobType.EVALUATION:
            payload = {"model_name": model}
        
        job = worker_manager.create_job(jt, payload)
        
        console.print(Panel(
            f"[bold green]✅ Job Created![/bold green]\n\n"
            f"[bold]Job ID:[/bold] [cyan]{job.job_id}[/cyan]\n"
            f"[bold]Type:[/bold] {job.job_type.value}\n"
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Status:[/bold] Queued\n\n"
            f"[dim]Job will be executed by the next available GPU worker.[/dim]",
            title="📋 New Job",
            border_style="green"
        ))
        
    elif action == "cancel":
        if not job_id:
            jobs = [j for j in worker_manager.list_jobs() 
                   if j.status in [JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING]]
            
            if not jobs:
                console.print("[yellow]No cancellable jobs.[/yellow]")
                return
            
            console.print("[bold]Select job to cancel:[/bold]")
            for i, j in enumerate(jobs, 1):
                console.print(f"  {i}. {j.job_id[:16]}... ({j.job_type.value} - {j.status.value})")
            
            choice = Prompt.ask("Job number", choices=[str(i) for i in range(1, len(jobs)+1)])
            job_id = jobs[int(choice)-1].job_id
        
        if worker_manager.cancel_job(job_id):
            console.print(f"[green]✓ Job cancelled: {job_id}[/green]")
        else:
            console.print(f"[red]Failed to cancel job: {job_id}[/red]")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Use: list, create, or cancel")


@app.command()
def benchmark(
    model_id: str = typer.Option("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--model", "-m", help="Model to benchmark"),
    steps: int = typer.Option(10, "--steps", "-s", help="Number of benchmark steps")
):
    """
    Benchmark training performance with the accelerator.
    
    Tests throughput, memory usage, and optimal batch sizes.
    """
    from saara.accelerator import create_accelerator
    from saara.visualizer import create_dashboard
    import time
    
    console.print(Panel.fit(
        "[bold cyan]⚡ Performance Benchmark[/bold cyan]\n\n"
        f"Model: {model_id}\n"
        f"Steps: {steps}",
        title="SAARA Benchmark",
        border_style="cyan"
    ))
    
    # Create accelerator
    accelerator = create_accelerator(
        mixed_precision=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True
    )
    
    accelerator.display_status()
    
    # Simulated benchmark (actual benchmark would require loading model)
    console.print("\n[bold]Running benchmark...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Benchmarking...", total=steps)
        
        times = []
        for i in range(steps):
            start = time.time()
            time.sleep(0.1)  # Simulated step
            elapsed = time.time() - start
            times.append(elapsed)
            progress.advance(task)
    
    avg_time = sum(times) / len(times)
    
    # Results
    console.print("\n[bold]Benchmark Results:[/bold]")
    
    results = Table(show_header=True, header_style="bold green")
    results.add_column("Metric", style="cyan")
    results.add_column("Value", style="green")
    
    results.add_row("Average Step Time", f"{avg_time*1000:.1f} ms")
    results.add_row("Estimated Throughput", f"{1/avg_time:.1f} steps/sec")
    results.add_row("Device", accelerator.device.upper())
    results.add_row("Mixed Precision", "✅ FP16" if accelerator.config.mixed_precision else "❌")
    
    console.print(results)
    
    console.print("\n[dim]Note: Full benchmark requires loading the actual model.[/dim]")


# ============================================================================
# AI TOKENIZER COMMANDS
# ============================================================================

@app.command()
def tokenizer(
    action: str = typer.Argument("train", help="Action: train, info, test"),
    input_path: str = typer.Option(None, "--input", "-i", help="Input text file or directory"),
    output_dir: str = typer.Option("tokenizers/ai_tokenizer", "--output", "-o", help="Output directory"),
    vocab_size: int = typer.Option(32000, "--vocab-size", "-v", help="Vocabulary size"),
    domain: str = typer.Option("general", "--domain", "-d", help="Domain: general, medical, legal, code, scientific"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Disable AI extraction (use rule-based only)")
):
    """
    Train AI-enhanced tokenizers with domain-aware vocabulary.
    
    The AI tokenizer uses LLMs to:
    - Extract domain-specific terms that should not be split
    - Identify technical vocabulary, abbreviations, compound words
    - Optimize BPE merges for semantic coherence
    
    Actions:
        train - Train a new tokenizer on text data
        info  - Show info about an existing tokenizer
        test  - Test tokenization on sample text
    
    Examples:
        saara tokenizer train -i datasets/my_data.jsonl --domain medical
        saara tokenizer info -o tokenizers/my_tokenizer
        saara tokenizer test -o tokenizers/my_tokenizer
    """
    from saara.ai_tokenizer import (
        create_ai_tokenizer, 
        train_tokenizer_on_files,
        AIEnhancedTokenizer
    )
    
    if action == "train":
        console.print(Panel.fit(
            "[bold cyan]🤖 AI-Enhanced Tokenizer Training[/bold cyan]\n\n"
            f"Domain: {domain}\n"
            f"Vocab Size: {vocab_size:,}\n"
            f"AI Extraction: {'❌ Disabled' if no_ai else '✅ Enabled'}",
            title="AI Tokenizer",
            border_style="cyan"
        ))
        
        if not input_path:
            # Interactive mode
            console.print("\n[bold]Select input data source:[/bold]")
            console.print("  1. Text file (.txt)")
            console.print("  2. JSONL dataset")
            console.print("  3. Directory of files")
            
            choice = Prompt.ask("Choice", choices=["1", "2", "3"], default="2")
            
            if choice == "1":
                input_path = Prompt.ask("Path to text file")
            elif choice == "2":
                input_path = Prompt.ask("Path to JSONL file", default="datasets")
            else:
                input_path = Prompt.ask("Path to directory", default="datasets")
        
        if not Path(input_path).exists():
            console.print(f"[red]Error: Path not found: {input_path}[/red]")
            return
        
        # Domain selection
        if domain == "general":
            console.print("\n[bold]Select domain for vocabulary optimization:[/bold]")
            console.print("  1. General (default)")
            console.print("  2. Medical/Healthcare")
            console.print("  3. Legal")
            console.print("  4. Code/Programming")
            console.print("  5. Scientific")
            
            domain_choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5"], default="1")
            domain_map = {"1": "general", "2": "medical", "3": "legal", "4": "code", "5": "scientific"}
            domain = domain_map[domain_choice]
        
        console.print(f"\n[dim]Training tokenizer on {input_path}...[/dim]\n")
        
        try:
            tokenizer = train_tokenizer_on_files(
                input_path=input_path,
                output_dir=output_dir,
                vocab_size=vocab_size,
                domain=domain,
                use_ai=not no_ai
            )
            
            console.print(Panel(
                f"[green]✅ Tokenizer trained successfully![/green]\n\n"
                f"Saved to: {output_dir}\n"
                f"Vocabulary: {len(tokenizer.vocab):,} tokens\n"
                f"Protected terms: {len(tokenizer.protected_tokens)}",
                title="Training Complete",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]Error training tokenizer: {e}[/red]")
            raise
    
    elif action == "info":
        if not Path(output_dir).exists():
            console.print(f"[red]Tokenizer not found at: {output_dir}[/red]")
            return
        
        tokenizer = AIEnhancedTokenizer.load(output_dir)
        tokenizer.display_info()
        
    elif action == "test":
        if not Path(output_dir).exists():
            console.print(f"[red]Tokenizer not found at: {output_dir}[/red]")
            return
        
        tokenizer = AIEnhancedTokenizer.load(output_dir)
        tokenizer.display_info()
        
        console.print("\n[bold]Test Tokenization[/bold]")
        console.print("[dim]Enter text to tokenize (or 'quit' to exit):[/dim]\n")
        
        while True:
            text = Prompt.ask("Text")
            
            if text.lower() in ["quit", "exit", "q"]:
                break
            
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            console.print(f"\n[cyan]Token IDs:[/cyan] {tokens}")
            console.print(f"[cyan]Token count:[/cyan] {len(tokens)}")
            console.print(f"[cyan]Decoded:[/cyan] {decoded}\n")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Use: train, info, or test")


# ============================================================================
# RAG AGENT COMMANDS
# ============================================================================

# Create RAG subcommand group
rag_app = typer.Typer(help="Build and manage RAG (Retrieval-Augmented Generation) agents")
app.add_typer(rag_app, name="rag")


@rag_app.callback(invoke_without_command=True)
def rag_callback(ctx: typer.Context):
    """
    Build and manage RAG (Retrieval-Augmented Generation) agents.
    
    RAG agents can search and reason over your documents to answer questions.
    """
    if ctx.invoked_subcommand is None:
        # Show help if no subcommand
        console.print(Panel.fit(
            "[bold cyan]🔍 RAG Agent Builder[/bold cyan]\n\n"
            "Build intelligent agents that can search and reason over your documents.\n\n"
            "[bold]Available Commands:[/bold]\n"
            "  • [cyan]saara rag create[/cyan]  - Create a new knowledge base\n"
            "  • [cyan]saara rag add[/cyan]     - Add documents to a knowledge base\n"
            "  • [cyan]saara rag chat[/cyan]    - Chat with your knowledge base\n"
            "  • [cyan]saara rag search[/cyan]  - Search the knowledge base\n"
            "  • [cyan]saara rag list[/cyan]    - List all knowledge bases\n"
            "  • [cyan]saara rag info[/cyan]    - Show knowledge base info\n"
            "  • [cyan]saara rag serve[/cyan]   - Start RAG API server\n"
            "  • [cyan]saara rag delete[/cyan]  - Delete a knowledge base",
            title="🤖 SAARA RAG",
            border_style="cyan"
        ))


@rag_app.command("create")
def rag_create(
    name: str = typer.Argument(None, help="Name for the knowledge base"),
    description: str = typer.Option("", "--desc", "-d", help="Description of the knowledge base"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", "--embedding", "-e", help="Embedding model"),
    llm_model: str = typer.Option("granite4", "--llm", "-l", help="LLM model for generation"),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Chunk size for document splitting"),
    chunk_overlap: int = typer.Option(50, "--overlap", help="Chunk overlap")
):
    """
    Create a new RAG knowledge base.
    
    Examples:
        saara rag create my_docs
        saara rag create medical_kb --desc "Medical knowledge base" --llm llama3.2
    """
    from saara.rag_engine import RAGManager, KnowledgeBaseConfig
    
    console.print(Panel.fit(
        "[bold cyan]🔍 Create RAG Knowledge Base[/bold cyan]\n\n"
        "Build an intelligent agent that can search and answer questions\n"
        "about your documents using semantic understanding.\n\n"
        "[dim]Type 'back' at any step to return to the previous step.[/dim]",
        title="RAG Agent Builder",
        border_style="cyan"
    ))
    
    # Step tracking for back navigation
    current_step = 1
    collected = {}
    
    while current_step <= 4:
        # Step indicator
        step_colors = ["green" if i < current_step else "cyan" if i == current_step else "dim" for i in range(1, 5)]
        step_display = f"[{step_colors[0]}]●[/{step_colors[0]}]━[{step_colors[1]}]●[/{step_colors[1]}]━[{step_colors[2]}]●[/{step_colors[2]}]━[{step_colors[3]}]●[/{step_colors[3]}]"
        console.print(f"\n{step_display}  [bold]Step {current_step}/4[/bold]")
        
        if current_step == 1:
            # Step 1: Name
            console.print("[bold magenta]📛 Knowledge Base Name[/bold magenta]")
            if name and not collected.get('name_prompted'):
                collected['name'] = name
                collected['name_prompted'] = True
                current_step += 1
                continue
            
            response = Prompt.ask("Enter name (or 'back' to cancel)", default="my_knowledge_base")
            if response.lower() == 'back':
                console.print("[dim]Cancelled.[/dim]")
                return
            collected['name'] = response
            current_step += 1
            
        elif current_step == 2:
            # Step 2: Description
            console.print("[bold magenta]📝 Description[/bold magenta]")
            response = Prompt.ask("Enter description (optional, 'back' to go back)", default="")
            if response.lower() == 'back':
                current_step -= 1
                continue
            collected['description'] = response
            current_step += 1
            
        elif current_step == 3:
            # Step 3: Embedding Model
            console.print("[bold magenta]🧠 Embedding Model[/bold magenta]")
            console.print("  [green]1.[/green] all-MiniLM-L6-v2 (384d) - [green]Fast, lightweight, good quality[/green]")
            console.print("  [cyan]2.[/cyan] all-mpnet-base-v2 (768d) - [cyan]Higher quality, slower[/cyan]")
            console.print("  [yellow]3.[/yellow] nomic-embed-text - [yellow]Requires Ollama[/yellow]")
            console.print("  [blue]4.[/blue] Custom model")
            console.print("  [dim]0.[/dim] [dim]← Back[/dim]")
            
            emb_choice = Prompt.ask("Choice", choices=["0", "1", "2", "3", "4"], default="1")
            
            if emb_choice == "0":
                current_step -= 1
                continue
            
            embedding_models = {
                "1": "all-MiniLM-L6-v2",
                "2": "all-mpnet-base-v2",
                "3": "nomic-embed-text",
            }
            
            if emb_choice == "4":
                custom_model = Prompt.ask("Enter model name")
                if custom_model.lower() == 'back':
                    continue
                collected['embedding_model'] = custom_model
            else:
                collected['embedding_model'] = embedding_models[emb_choice]
            
            current_step += 1
            
        elif current_step == 4:
            # Step 4: LLM Model
            console.print("[bold magenta]🤖 LLM for Generation[/bold magenta]")
            
            # Try to get installed models
            installed = []
            try:
                from saara.model_manager import ModelManager
                mm = ModelManager()
                installed = mm.get_installed_models()
            except:
                pass
            
            if installed:
                console.print("[dim]Installed Ollama models:[/dim]")
                for i, m in enumerate(installed[:5], 1):
                    console.print(f"  [cyan]{i}.[/cyan] {m}")
                if len(installed) > 5:
                    console.print(f"  [dim]... and {len(installed) - 5} more[/dim]")
            
            console.print("  [dim]0.[/dim] [dim]← Back[/dim]")
            
            response = Prompt.ask("Enter LLM model name", default=installed[0] if installed else "granite4")
            
            if response.lower() == 'back' or response == '0':
                current_step -= 1
                continue
            
            collected['llm_model'] = response
            current_step += 1
    
    # Show summary before creating
    console.print("\n" + "═" * 50)
    console.print("[bold green]📋 Configuration Summary[/bold green]")
    console.print("═" * 50)
    
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Key", style="cyan")
    summary_table.add_column("Value", style="white")
    summary_table.add_row("Name", collected['name'])
    summary_table.add_row("Description", collected.get('description', '') or "[dim]None[/dim]")
    summary_table.add_row("Embedding Model", collected['embedding_model'])
    summary_table.add_row("LLM Model", collected['llm_model'])
    summary_table.add_row("Chunk Size", str(chunk_size))
    console.print(summary_table)
    
    console.print()
    if not Confirm.ask("[bold]Create this knowledge base?[/bold]", default=True):
        console.print("[dim]Cancelled.[/dim]")
        return
    
    # Create the knowledge base with visual feedback
    console.print()
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=30, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Creating knowledge base...", total=100)
        
        progress.update(task, advance=30)
        
        try:
            rag_manager = RAGManager()
            
            progress.update(task, description="Initializing vector store...", advance=30)
            
            engine = rag_manager.create(
                name=collected['name'],
                description=collected.get('description', ''),
                embedding_model=collected['embedding_model'],
                llm_model=collected['llm_model'],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            progress.update(task, description="Finalizing...", advance=40)
            
        except ValueError as e:
            console.print(f"\n[red]Error: {e}[/red]")
            return
        except Exception as e:
            console.print(f"\n[red]Failed to create knowledge base: {e}[/red]")
            return
    
    # Success panel with colorful output
    console.print(Panel(
        f"[bold green]✅ Knowledge base created successfully![/bold green]\n\n"
        f"[cyan]╭───────────────────────────────────────╮[/cyan]\n"
        f"[cyan]│[/cyan]  [bold]Name:[/bold] {collected['name']:<28} [cyan]│[/cyan]\n"
        f"[cyan]│[/cyan]  [bold]Embedding:[/bold] {collected['embedding_model']:<22} [cyan]│[/cyan]\n"
        f"[cyan]│[/cyan]  [bold]LLM:[/bold] {collected['llm_model']:<29} [cyan]│[/cyan]\n"
        f"[cyan]╰───────────────────────────────────────╯[/cyan]\n\n"
        f"[bold yellow]🚀 Next steps:[/bold yellow]\n"
        f"  [green]1.[/green] Add documents: [cyan]saara rag add {collected['name']} path/to/docs[/cyan]\n"
        f"  [green]2.[/green] Start chatting: [cyan]saara rag chat {collected['name']}[/cyan]",
        title="✓ Created",
        border_style="green"
    ))


@rag_app.command("add")
def rag_add(
    kb_name: str = typer.Argument(None, help="Knowledge base name"),
    source: str = typer.Argument(None, help="Path to file or directory to add")
):
    """
    Add documents to a knowledge base.
    
    Supports: PDF, TXT, MD, JSONL, JSON files and directories.
    
    Examples:
        saara rag add my_docs ./documents/
        saara rag add my_docs paper.pdf
        saara rag add my_docs dataset.jsonl
    """
    from saara.rag_engine import RAGManager
    
    console.print(Panel.fit(
        "[bold cyan]📄 Add Documents to Knowledge Base[/bold cyan]",
        border_style="cyan"
    ))
    
    manager = RAGManager()
    
    # Select knowledge base if not provided
    if not kb_name:
        kbs = manager.list()
        
        if not kbs:
            console.print("[yellow]No knowledge bases found. Create one first:[/yellow]")
            console.print("  saara rag create <name>")
            return
        
        console.print("[bold]Available knowledge bases:[/bold]")
        for i, kb in enumerate(kbs, 1):
            console.print(f"  {i}. {kb['name']} ({kb['documents']} docs, {kb['chunks']} chunks)")
        
        choice = Prompt.ask("Select knowledge base", choices=[str(i) for i in range(1, len(kbs)+1)])
        kb_name = kbs[int(choice)-1]['name']
    
    # Get source path if not provided
    if not source:
        console.print("\n[bold]Select source type:[/bold]")
        console.print("  1. Single file")
        console.print("  2. Directory of files")
        
        src_choice = Prompt.ask("Choice", choices=["1", "2"], default="1")
        
        if src_choice == "1":
            source = Prompt.ask("Path to file")
        else:
            source = Prompt.ask("Path to directory")
    
    # Validate source
    source_path = Path(source)
    if not source_path.exists():
        console.print(f"[red]Error: Path not found: {source}[/red]")
        return
    
    try:
        engine = manager.get(kb_name)
        
        console.print(f"\n[dim]Indexing documents from {source}...[/dim]\n")
        
        num_chunks = engine.add_documents(source, show_progress=True)
        
        stats = engine.get_stats()
        
        console.print(Panel(
            f"[green]✅ Documents indexed successfully![/green]\n\n"
            f"[bold]Added:[/bold] {num_chunks} chunks\n"
            f"[bold]Total documents:[/bold] {stats['documents']}\n"
            f"[bold]Total chunks:[/bold] {stats['chunks']}\n\n"
            f"[bold]Start chatting:[/bold]\n"
            f"  [cyan]saara rag chat {kb_name}[/cyan]",
            title="✓ Indexed",
            border_style="green"
        ))
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to add documents: {e}[/red]")
        raise


@rag_app.command("chat")
def rag_chat(
    kb_name: str = typer.Argument(None, help="Knowledge base name"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream responses")
):
    """
    Interactive chat with a knowledge base.
    
    Ask questions and get answers based on your indexed documents.
    
    Examples:
        saara rag chat my_docs
        saara rag chat medical_kb --stream
    """
    from saara.rag_engine import RAGManager
    
    manager = RAGManager()
    
    # Select knowledge base if not provided
    if not kb_name:
        kbs = manager.list()
        
        if not kbs:
            console.print("[yellow]No knowledge bases found. Create one first:[/yellow]")
            console.print("  saara rag create <name>")
            return
        
        console.print("[bold]Available knowledge bases:[/bold]")
        for i, kb in enumerate(kbs, 1):
            console.print(f"  {i}. {kb['name']} ({kb['chunks']} chunks)")
        
        choice = Prompt.ask("Select knowledge base", choices=[str(i) for i in range(1, len(kbs)+1)])
        kb_name = kbs[int(choice)-1]['name']
    
    try:
        engine = manager.get(kb_name)
        stats = engine.get_stats()
        
        console.print(Panel.fit(
            f"[bold cyan]💬 RAG Chat[/bold cyan]\n\n"
            f"[bold]Knowledge Base:[/bold] {stats['name']}\n"
            f"[bold]Documents:[/bold] {stats['documents']} | [bold]Chunks:[/bold] {stats['chunks']}\n"
            f"[bold]LLM:[/bold] {stats['llm_model']}\n\n"
            f"[dim]Type your questions below. Type 'exit' to quit.[/dim]",
            title="🤖 RAG Agent",
            border_style="cyan"
        ))
        
        conversation = []
        
        while True:
            console.print()
            question = Prompt.ask("[bold green]You[/bold green]")
            
            if question.lower() in ['exit', 'quit', 'q', 'bye']:
                console.print("\n[dim]Goodbye! 👋[/dim]")
                break
            
            if not question.strip():
                continue
            
            # Special commands
            if question.startswith('/'):
                cmd = question[1:].lower().strip()
                if cmd == 'clear':
                    conversation = []
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue
                elif cmd == 'sources':
                    console.print("[dim]Use /sources after a query to see sources.[/dim]")
                    continue
                elif cmd == 'help':
                    console.print("[dim]Commands: /clear, /sources, /help, /stats[/dim]")
                    continue
                elif cmd == 'stats':
                    stats = engine.get_stats()
                    console.print(f"[dim]Documents: {stats['documents']}, Chunks: {stats['chunks']}[/dim]")
                    continue
            
            # Add to conversation
            conversation.append({"role": "user", "content": question})
            
            # Query the knowledge base
            with console.status("[bold cyan]Thinking...[/bold cyan]"):
                response = engine.query(question)
            
            # Display response
            console.print(f"\n[bold cyan]🤖 Agent[/bold cyan]")
            console.print(response.answer)
            
            # Show sources if available
            if response.sources:
                console.print(f"\n[dim]📚 Sources ({len(response.sources)}):[/dim]")
                for i, src in enumerate(response.sources[:3], 1):
                    source_name = src.doc_metadata.get('filename', src.doc_metadata.get('source', 'Unknown'))
                    console.print(f"  [dim]{i}. {source_name} (score: {src.score:.2f})[/dim]")
            
            # Show latency
            console.print(f"\n[dim]⏱️ {response.latency_ms:.0f}ms | Confidence: {response.confidence:.1%}[/dim]")
            
            # Add to conversation
            conversation.append({"role": "assistant", "content": response.answer})
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to start chat: {e}[/red]")
        raise


@rag_app.command("search")
def rag_search(
    kb_name: str = typer.Argument(None, help="Knowledge base name"),
    query: str = typer.Argument(None, help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results")
):
    """
    Search the knowledge base without generating an answer.
    
    Returns the most relevant document chunks.
    
    Examples:
        saara rag search my_docs "What is machine learning?"
        saara rag search my_docs "neural networks" --top-k 10
    """
    from saara.rag_engine import RAGManager
    
    manager = RAGManager()
    
    # Select knowledge base if not provided
    if not kb_name:
        kbs = manager.list()
        
        if not kbs:
            console.print("[yellow]No knowledge bases found.[/yellow]")
            return
        
        console.print("[bold]Available knowledge bases:[/bold]")
        for i, kb in enumerate(kbs, 1):
            console.print(f"  {i}. {kb['name']}")
        
        choice = Prompt.ask("Select knowledge base", choices=[str(i) for i in range(1, len(kbs)+1)])
        kb_name = kbs[int(choice)-1]['name']
    
    if not query:
        query = Prompt.ask("Search query")
    
    try:
        engine = manager.get(kb_name)
        
        console.print(f"\n[dim]Searching for: {query}[/dim]\n")
        
        results = engine.search(query, top_k=top_k)
        
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        console.print(f"[bold]Found {len(results)} results:[/bold]\n")
        
        for i, result in enumerate(results, 1):
            source = result.doc_metadata.get('filename', result.doc_metadata.get('source', 'Unknown'))
            score = result.score
            
            # Truncate content for display
            content = result.chunk.content
            if len(content) > 300:
                content = content[:300] + "..."
            
            console.print(Panel(
                f"[dim]{content}[/dim]",
                title=f"[bold]#{i}[/bold] {source} [cyan](score: {score:.3f})[/cyan]",
                border_style="dim"
            ))
    
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@rag_app.command("list")
def rag_list():
    """
    List all RAG knowledge bases.
    """
    from saara.rag_engine import RAGManager
    
    manager = RAGManager()
    kbs = manager.list()
    
    if not kbs:
        console.print(Panel(
            "[yellow]No knowledge bases found.[/yellow]\n\n"
            "Create one with:\n"
            "  [cyan]saara rag create my_knowledge_base[/cyan]",
            title="📚 Knowledge Bases",
            border_style="yellow"
        ))
        return
    
    console.print(Panel.fit(
        "[bold cyan]📚 RAG Knowledge Bases[/bold cyan]",
        border_style="cyan"
    ))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Documents", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Embedding Model")
    table.add_column("LLM")
    table.add_column("Created")
    
    for kb in kbs:
        created = kb.get('created_at', '')[:10] if kb.get('created_at') else 'Unknown'
        table.add_row(
            kb['name'],
            str(kb['documents']),
            str(kb['chunks']),
            kb['embedding_model'][:20],
            kb['llm_model'],
            created
        )
    
    console.print(table)


@rag_app.command("info")
def rag_info(
    kb_name: str = typer.Argument(None, help="Knowledge base name")
):
    """
    Show detailed information about a knowledge base.
    """
    from saara.rag_engine import RAGManager
    
    manager = RAGManager()
    
    if not kb_name:
        kbs = manager.list()
        
        if not kbs:
            console.print("[yellow]No knowledge bases found.[/yellow]")
            return
        
        console.print("[bold]Select a knowledge base:[/bold]")
        for i, kb in enumerate(kbs, 1):
            console.print(f"  {i}. {kb['name']}")
        
        choice = Prompt.ask("Choice", choices=[str(i) for i in range(1, len(kbs)+1)])
        kb_name = kbs[int(choice)-1]['name']
    
    try:
        engine = manager.get(kb_name)
        stats = engine.get_stats()
        
        console.print(Panel(
            f"[bold]Name:[/bold] {stats['name']}\n"
            f"[bold]Description:[/bold] {stats.get('description', 'N/A')}\n"
            f"[bold]Created:[/bold] {stats.get('created_at', 'Unknown')}\n\n"
            f"[bold]Documents:[/bold] {stats['documents']}\n"
            f"[bold]Chunks:[/bold] {stats['chunks']}\n"
            f"[bold]Chunk Size:[/bold] {stats['chunk_size']}\n\n"
            f"[bold]Embedding Model:[/bold] {stats['embedding_model']}\n"
            f"[bold]LLM Model:[/bold] {stats['llm_model']}",
            title=f"📊 {stats['name']}",
            border_style="cyan"
        ))
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@rag_app.command("serve")
def rag_serve(
    kb_name: str = typer.Argument(None, help="Knowledge base name"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8001, "--port", "-p", help="Port to bind to")
):
    """
    Start a REST API server for the RAG knowledge base.
    
    Endpoints:
        POST /query - Query the knowledge base
        POST /search - Search without generation
        GET /health - Health check
    
    Examples:
        saara rag serve my_docs
        saara rag serve my_docs --port 8080
    """
    from saara.rag_engine import RAGManager
    
    manager = RAGManager()
    
    if not kb_name:
        kbs = manager.list()
        
        if not kbs:
            console.print("[yellow]No knowledge bases found.[/yellow]")
            return
        
        console.print("[bold]Select a knowledge base to serve:[/bold]")
        for i, kb in enumerate(kbs, 1):
            console.print(f"  {i}. {kb['name']}")
        
        choice = Prompt.ask("Choice", choices=[str(i) for i in range(1, len(kbs)+1)])
        kb_name = kbs[int(choice)-1]['name']
    
    try:
        engine = manager.get(kb_name)
        stats = engine.get_stats()
        
        console.print(Panel.fit(
            f"[bold cyan]🚀 RAG API Server[/bold cyan]\n\n"
            f"[bold]Knowledge Base:[/bold] {stats['name']}\n"
            f"[bold]Documents:[/bold] {stats['documents']} | [bold]Chunks:[/bold] {stats['chunks']}\n\n"
            f"[bold]Starting server at:[/bold] http://{host}:{port}",
            title="RAG Server",
            border_style="cyan"
        ))
        
        # Create FastAPI app
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        api = FastAPI(title=f"SAARA RAG API - {kb_name}", version="1.0.0")
        
        class QueryRequest(BaseModel):
            question: str
            top_k: int = 5
        
        class SearchRequest(BaseModel):
            query: str
            top_k: int = 5
        
        @api.get("/health")
        def health():
            return {"status": "healthy", "knowledge_base": kb_name, "chunks": stats['chunks']}
        
        @api.post("/query")
        def query(request: QueryRequest):
            try:
                response = engine.query(request.question, top_k=request.top_k)
                return {
                    "answer": response.answer,
                    "sources": [
                        {
                            "content": s.chunk.content[:500],
                            "score": s.score,
                            "source": s.doc_metadata.get('source', 'Unknown')
                        }
                        for s in response.sources
                    ],
                    "latency_ms": response.latency_ms,
                    "confidence": response.confidence
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @api.post("/search")
        def search(request: SearchRequest):
            try:
                results = engine.search(request.query, top_k=request.top_k)
                return {
                    "results": [
                        {
                            "content": r.chunk.content,
                            "score": r.score,
                            "source": r.doc_metadata.get('source', 'Unknown')
                        }
                        for r in results
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        console.print(f"\n[green]API Endpoints:[/green]")
        console.print(f"  POST http://{host}:{port}/query   - Query with RAG")
        console.print(f"  POST http://{host}:{port}/search  - Semantic search")
        console.print(f"  GET  http://{host}:{port}/health  - Health check")
        console.print(f"\n[dim]Press Ctrl+C to stop the server.[/dim]\n")
        
        uvicorn.run(api, host=host, port=port)
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        raise


@rag_app.command("delete")
def rag_delete(
    kb_name: str = typer.Argument(None, help="Knowledge base name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Delete a RAG knowledge base.
    
    Examples:
        saara rag delete my_docs
        saara rag delete my_docs --yes
    """
    from saara.rag_engine import RAGManager
    
    manager = RAGManager()
    
    if not kb_name:
        kbs = manager.list()
        
        if not kbs:
            console.print("[yellow]No knowledge bases found.[/yellow]")
            return
        
        console.print("[bold]Select a knowledge base to delete:[/bold]")
        for i, kb in enumerate(kbs, 1):
            console.print(f"  {i}. {kb['name']} ({kb['chunks']} chunks)")
        
        choice = Prompt.ask("Choice", choices=[str(i) for i in range(1, len(kbs)+1)])
        kb_name = kbs[int(choice)-1]['name']
    
    if not yes:
        console.print(f"\n[red]⚠️  This will permanently delete '{kb_name}' and all its data.[/red]")
        if not Confirm.ask("Are you sure?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return
    
    try:
        manager.delete(kb_name)
        console.print(f"[green]✓ Deleted knowledge base: {kb_name}[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@rag_app.command("clear")
def rag_clear(
    kb_name: str = typer.Argument(None, help="Knowledge base name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Clear all documents from a knowledge base (keep the KB).
    
    Examples:
        saara rag clear my_docs
    """
    from saara.rag_engine import RAGManager
    
    manager = RAGManager()
    
    if not kb_name:
        kbs = manager.list()
        
        if not kbs:
            console.print("[yellow]No knowledge bases found.[/yellow]")
            return
        
        console.print("[bold]Select a knowledge base to clear:[/bold]")
        for i, kb in enumerate(kbs, 1):
            console.print(f"  {i}. {kb['name']} ({kb['chunks']} chunks)")
        
        choice = Prompt.ask("Choice", choices=[str(i) for i in range(1, len(kbs)+1)])
        kb_name = kbs[int(choice)-1]['name']
    
    if not yes:
        console.print(f"\n[yellow]⚠️  This will remove all documents from '{kb_name}'.[/yellow]")
        if not Confirm.ask("Are you sure?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return
    
    try:
        engine = manager.get(kb_name)
        engine.clear()
        console.print(f"[green]✓ Cleared all documents from: {kb_name}[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    app()

if __name__ == "__main__":
    main()




