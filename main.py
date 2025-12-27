"""
Command Line Interface for the Data Pipeline.
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from src.pipeline import DataPipeline

console = Console()


def interactive_mode():
    """Run the interactive setup wizard."""
    console.print(Panel.fit(
        "[bold cyan]🧠 NeuroPipe Data Engine[/bold cyan]\n\n"
        "Autonomous Document-to-LLM Data Factory\n"
        "[dim]Transform PDFs into training-ready datasets & fine-tuned models[/dim]",
        title="✨ Welcome",
        border_style="cyan"
    ))
    
    # Selection Mode with Table
    console.print("\n")
    mode_table = Table(title="Choose Your Workflow", show_header=True, header_style="bold magenta")
    mode_table.add_column("Option", style="cyan", width=8)
    mode_table.add_column("Mode", style="green")
    mode_table.add_column("Description", style="dim")
    
    mode_table.add_row("1", "📄 Dataset Creation", "Extract data from PDFs → Generate training datasets")
    mode_table.add_row("2", "🧠 Model Training", "Fine-tune LLMs on your prepared data")
    mode_table.add_row("3", "🧪 Model Evaluation", "Test & improve trained models with Granite 4")
    mode_table.add_row("4", "🚀 Model Deployment", "Deploy models locally (Ollama) or to cloud")
    
    console.print(mode_table)
    console.print()
    
    mode_choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4"], default="1")
    
    if mode_choice == "2":
        # Direct Training Flow
        run_training_wizard()
        return
    elif mode_choice == "3":
        # Model Evaluation Flow
        run_evaluation_wizard()
        return
    elif mode_choice == "4":
        # Model Deployment Flow
        run_deployment_wizard()
        return

    # --- Dataset Creation Flow ---
    
    # 1. Ask for paths
    base_dir = os.getcwd()
    raw_path = Prompt.ask("Enter path to raw data folder (PDFs)", default=base_dir).strip('"\'')
    output_path = Prompt.ask("Enter path for output datasets", default="./datasets").strip('"\'')
    
    # Validation
    raw_path_obj = Path(raw_path)
    if not raw_path_obj.exists():
        console.print(f"[yellow]Warning: Source directory '{raw_path}' does not exist.[/yellow]")
        if not Confirm.ask("Continue anyway?"):
            sys.exit(0)
            
    # Display formats
    formats = "JSONL, CSV, ShareGPT, HuggingFace"
    console.print(Panel(
        f"Data will be formatted in [bold]{formats}[/bold]\n"
        f"Output storage: [bold]{output_path}[/bold]",
        title="Configuration",
        style="green"
    ))
    
    # 2. Select Vision Model
    console.print("\n[bold]Select Vision OCR Model:[/bold]")
    v_table = Table(show_header=True, header_style="bold magenta")
    v_table.add_column("ID", style="cyan", width=4)
    v_table.add_column("Model", style="green")
    v_table.add_column("VRAM Required", style="yellow")
    v_table.add_column("Description")
    
    v_table.add_row("1", "Moondream", "~2 GB", "Fast, lightweight, good for simple layouts")
    v_table.add_row("2", "Qwen2.5-VL", "~4 GB", "High accuracy, handles complex tables/diagrams")
    
    console.print(v_table)
    v_choice = Prompt.ask("Choose a vision model", choices=["1", "2"], default="1")
    vision_model = "moondream" if v_choice == "1" else "qwen"
    
    # 3. Select Analyzer Model
    console.print("\n[bold]Select Analyzer/Labeling Model:[/bold]")
    a_table = Table(show_header=True, header_style="bold magenta")
    a_table.add_column("ID", style="cyan", width=4)
    a_table.add_column("Model", style="green")
    a_table.add_column("VRAM Required", style="yellow")
    a_table.add_column("Description")
    
    a_table.add_row("1", "Granite 4.0", "~4 GB", "IBM's enterprise model, balanced performance")
    a_table.add_row("2", "Llama 3.2", "~4 GB", "Strong instruction following capabilities")
    
    console.print(a_table)
    a_choice = Prompt.ask("Choose an analyzer model", choices=["1", "2"], default="1")
    analyzer_model = "granite4" if a_choice == "1" else "llama3.2"
    
    # 4. Proceed
    console.print(f"\n[dim]Selected Configuration: Vision=[bold]{vision_model}[/bold], Analyzer=[bold]{analyzer_model}[/bold][/dim]")
    
    if not Confirm.ask("Proceed with processing?", default=True):
        console.print("[yellow]Aborted by user.[/yellow]")
        sys.exit(0)
        
    console.print("\n[bold cyan]🚀 Starting Pipeline...[/bold cyan]\n")
    
    # 5. Run Pipeline
    # Load default config and override
    config_path = "config.yaml"
    config = {}
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    # Ensure keys exist
    if 'pdf' not in config: config['pdf'] = {}
    if 'ollama' not in config: config['ollama'] = {}
    if 'output' not in config: config['output'] = {}
    
    # Apply overrides
    config['pdf']['ocr_engine'] = vision_model
    config['ollama']['model'] = analyzer_model
    config['output']['directory'] = output_path
    
    # Initialize pipeline
    pipeline = DataPipeline(config)
    
    # Check health
    if not pipeline.check_health():
        console.print("[red]Pipeline health check failed. Please ensure Ollama is running and models are pulled.[/red]")
        sys.exit(1)
        
    # Run
    result = pipeline.process_directory(raw_path, dataset_name="interactive_batch")
    
    if result.success:
        console.print(f"\n[bold green]✅ Success! Batch Processing Complete.[/bold green]")
        console.print(f"Total Samples Generated: {result.total_samples}")
        
        # 6. Training Section (Offer to continue)
        console.print(Panel("\n[bold]Model Training[/bold]\nNow that you have a dataset, you can fine-tune a model.", style="cyan"))
        
        if Confirm.ask("Do you want to start training now?", default=True):
            # Try to find the ShareGPT formatted file which is best for training
            data_file = f"{output_path}/interactive_batch_sharegpt.jsonl"
            if not os.path.exists(data_file):
                 # Failover to looking for any reasonable jsonl
                 data_file = Prompt.ask("Path to training dataset (.jsonl)", default=f"{output_path}/dataset_sharegpt.jsonl")
            
            # Pass the just-generated file and current config
            run_training_wizard(default_data_path=data_file, config=config)
                
    else:
        console.print(f"\n[bold red]❌ One or more errors occurred during processing.[/bold red]")


def run_training_wizard(default_data_path: str = None, config: dict = None):
    """Run the interactive training setup."""
    console.print("\n[bold]Select Base Model to Train:[/bold]")
    t_table = Table(show_header=True, header_style="bold magenta")
    t_table.add_column("ID", style="cyan", width=4)
    t_table.add_column("Model", style="green")
    t_table.add_column("Type", style="yellow")
    t_table.add_column("Description")
    
    t_table.add_row("1", "sarvamai/sarvam-1", "2B", "Optimized for Indian Languages")
    t_table.add_row("2", "google/gemma-2b", "2B", "Google's efficient model, good balance")
    t_table.add_row("3", "meta-llama/Llama-3.2-1B", "1B", "Fast, lightweight, good English")
    t_table.add_row("4", "Qwen/Qwen2.5-7B", "7B", "Strong reasoning, larger VRAM")
    t_table.add_row("5", "mistralai/Mistral-7B-v0.1", "7B", "High performance, industry standard")
    t_table.add_row("6", "TinyLlama/TinyLlama-1.1B", "1.1B", "Very fast, runs on most CPUs")
    t_table.add_row("7", "Other", "-", "Enter custom HuggingFace model ID")
    
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
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif t_choice == "7":
        model_id = Prompt.ask("Enter HuggingFace Model ID (e.g. microsoft/phi-2)")
    
    console.print(f"[bold]Selected Model:[/bold] {model_id}")
    
    # Check for gated models that require HuggingFace login
    gated_models = ["google/gemma", "meta-llama/Llama-3", "mistralai/Mistral"]
    is_gated = any(gated in model_id for gated in gated_models)
    
    if is_gated:
        console.print("[yellow]⚠️ This model requires HuggingFace authentication.[/yellow]")
        console.print("[dim]You need to: 1) Accept the model license on huggingface.co, 2) Login with your HF token.[/dim]")
        
        if Confirm.ask("Do you want to login to HuggingFace now?", default=True):
            hf_token = Prompt.ask("Enter your HuggingFace token (from hf.co/settings/tokens)", password=True)
            try:
                from huggingface_hub import login
                login(token=hf_token)
                console.print("[green]✅ Successfully logged in to HuggingFace![/green]")
            except Exception as e:
                console.print(f"[red]Login failed: {e}[/red]")
                console.print("[yellow]You can also run 'huggingface-cli login' in terminal.[/yellow]")
                return
    
    # Data Path
    while True:
        if default_data_path:
            data_file = default_data_path
            default_data_path = None # Only use once
        else:
            # Suggest a default file if it exists
            default_guess = "datasets/interactive_batch_sharegpt.jsonl"
            if not os.path.exists(default_guess):
                default_guess = "datasets/distilled_train.jsonl"
                
            data_file = Prompt.ask("Path to training dataset (.jsonl)", default=default_guess).strip('"\'')
            
        # Validation Logic
        path_obj = Path(data_file)
        
        if path_obj.is_dir():
            # Find all JSONL files (skip CSV as they need conversion)
            jsonl_files = list(path_obj.glob("*.jsonl"))
            
            if jsonl_files:
                console.print(f"[green]Found {len(jsonl_files)} JSONL files in directory.[/green]")
                
                # Show file summary
                file_table = Table(title="Training Files", show_header=True, header_style="bold cyan")
                file_table.add_column("#", width=4)
                file_table.add_column("File", style="dim")
                file_table.add_column("Type", style="yellow")
                
                # Group by type
                sharegpt_files = [f for f in jsonl_files if 'sharegpt' in f.name.lower()]
                instruction_files = [f for f in jsonl_files if 'instruction' in f.name.lower()]
                qa_files = [f for f in jsonl_files if '_qa' in f.name.lower()]
                other_files = [f for f in jsonl_files if f not in sharegpt_files + instruction_files + qa_files]
                
                console.print(f"\n[bold]Dataset Summary:[/bold]")
                console.print(f"  ShareGPT files: {len(sharegpt_files)}")
                console.print(f"  Instruction files: {len(instruction_files)}")
                console.print(f"  Q&A files: {len(qa_files)}")
                console.print(f"  Other files: {len(other_files)}")
                
                # Ask which type to use
                console.print("\n[bold]Select dataset type to train on:[/bold]")
                console.print("  1. ShareGPT only (recommended for chat)")
                console.print("  2. Instruction only")
                console.print("  3. Q&A only")
                console.print("  4. All files")
                
                type_choice = Prompt.ask("Select type", choices=["1", "2", "3", "4"], default="1")
                
                if type_choice == "1":
                    selected_files = sharegpt_files
                    console.print(f"[green]Selected {len(selected_files)} ShareGPT files[/green]")
                elif type_choice == "2":
                    selected_files = instruction_files
                    console.print(f"[green]Selected {len(selected_files)} Instruction files[/green]")
                elif type_choice == "3":
                    selected_files = qa_files
                    console.print(f"[green]Selected {len(selected_files)} Q&A files[/green]")
                else:
                    selected_files = jsonl_files
                    console.print(f"[green]Selected all {len(selected_files)} files[/green]")
                
                if not selected_files:
                    console.print("[red]No files of selected type found.[/red]")
                    continue
                
                # Store file list for batch loading
                data_file = [str(f) for f in selected_files]
                console.print(f"[green]Will load {len(data_file)} files for training.[/green]")
                break
            else:
                console.print("[red]No .jsonl files found in that directory.[/red]")
                continue
                 
        elif not path_obj.exists():
             console.print(f"[red]File or directory not found: {data_file}[/red]")
             default_data_path = None
             if not Confirm.ask("Try again?", default=True):
                 return # Exit wizard
        else:
            break # Valid file
        
    # Checkpoint logic
    resume_path = None
    if Confirm.ask("Do you want to resume from a checkpoint?", default=False):
        resume_path = Prompt.ask("Enter path to checkpoint directory (e.g. models/sarvam-finetuned/checkpoint-50)").strip('"\'')
    
    console.print("[dim]The tokenizer will be loaded and the model will be pulled automatically if not present.[/dim]")
    
    from src.train import LLMTrainer
    
    # Load config if not provided
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
    """Run the model evaluation and improvement wizard."""
    console.print(Panel.fit(
        "[bold cyan]🧪 Model Evaluation & Improvement[/bold cyan]\n\n"
        "Test your fine-tuned model using Granite 4 as a judge.\n"
        "Low-scoring responses will be used to generate improvement data.",
        title="Evaluation Mode",
        border_style="cyan"
    ))
    
    # Scan for fine-tuned models
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[red]No models directory found. Please train a model first.[/red]")
        return
    
    # Find all fine-tuned models
    finetuned_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            adapter_path = model_dir / "final_adapter"
            if adapter_path.exists():
                # Try to get base model from config
                adapter_config_path = adapter_path / "adapter_config.json"
                base_model = None
                if adapter_config_path.exists():
                    import json
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model = adapter_config.get("base_model_name_or_path", "")
                
                finetuned_models.append({
                    "name": model_dir.name,
                    "path": str(adapter_path),
                    "base_model": base_model or "Unknown"
                })
    
    if not finetuned_models:
        console.print("[yellow]No fine-tuned models found in models/ directory.[/yellow]")
        console.print("[dim]Train a model first using option 2.[/dim]")
        return
    
    # Display available models
    console.print("\n[bold]Available Fine-tuned Models:[/bold]\n")
    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("#", style="cyan", width=4)
    model_table.add_column("Model Name", style="green")
    model_table.add_column("Base Model", style="yellow")
    
    for i, m in enumerate(finetuned_models, 1):
        model_table.add_row(str(i), m["name"], m["base_model"])
    
    console.print(model_table)
    console.print()
    
    # Select model
    choice = Prompt.ask("Select model to evaluate", choices=[str(i) for i in range(1, len(finetuned_models)+1)], default="1")
    selected = finetuned_models[int(choice)-1]
    
    base_model = selected["base_model"]
    adapter_path = selected["path"]
    
    if base_model == "Unknown":
        base_model = Prompt.ask("Enter base model ID", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    else:
        console.print(f"[green]Using base model: {base_model}[/green]")
    
    # Number of test samples
    num_samples = int(Prompt.ask("Number of test samples", default="10"))
    
    # Run evaluation
    from src.evaluator import ModelEvaluator
    
    if not config:
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    evaluator = ModelEvaluator(config)
    
    try:
        results = evaluator.evaluate_adapter(
            base_model_id=base_model,
            adapter_path=adapter_path,
            num_samples=num_samples
        )
        
        # Offer to retrain with improvement data
        if results.get("improvement_data"):
            console.print("\n[yellow]Improvement data has been generated from low-scoring responses.[/yellow]")
            
            if Confirm.ask("Would you like to retrain with the improvement data?", default=False):
                improvement_file = "evaluations/corrections_for_training.jsonl"
                if os.path.exists(improvement_file):
                    run_training_wizard(default_data_path=improvement_file, config=config)
                    
    except Exception as e:
        console.print(f"[bold red]Evaluation failed:[/bold red] {e}")


def run_deployment_wizard(config: dict = None):
    """Run the model deployment wizard."""
    console.print(Panel.fit(
        "[bold cyan]🚀 Model Deployment[/bold cyan]\n\n"
        "Deploy your fine-tuned model locally or to the cloud.",
        title="Deployment Mode",
        border_style="green"
    ))
    
    # Scan for fine-tuned models
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[red]No models directory found. Please train a model first.[/red]")
        return
    
    # Find all fine-tuned models (directories with 'final_adapter' subfolder)
    finetuned_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            adapter_path = model_dir / "final_adapter"
            if adapter_path.exists():
                finetuned_models.append({
                    "name": model_dir.name,
                    "path": str(adapter_path),
                    "dir": str(model_dir)
                })
    
    if not finetuned_models:
        console.print("[yellow]No fine-tuned models found in models/ directory.[/yellow]")
        console.print("[dim]Train a model first using option 2.[/dim]")
        return
    
    # Display available models
    console.print("\n[bold]Available Fine-tuned Models:[/bold]\n")
    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("#", style="cyan", width=4)
    model_table.add_column("Model Name", style="green")
    model_table.add_column("Adapter Path", style="dim")
    
    for i, m in enumerate(finetuned_models, 1):
        model_table.add_row(str(i), m["name"], m["path"])
    
    console.print(model_table)
    console.print()
    
    # Select model
    choice = Prompt.ask("Select model to deploy", choices=[str(i) for i in range(1, len(finetuned_models)+1)], default="1")
    selected = finetuned_models[int(choice)-1]
    
    # Infer base model from adapter config
    adapter_config_path = Path(selected["path"]) / "adapter_config.json"
    base_model = None
    
    if adapter_config_path.exists():
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path", "")
    
    if not base_model:
        # Ask user for base model
        console.print("[yellow]Could not detect base model. Please specify:[/yellow]")
        base_model = Prompt.ask("Enter base model ID", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    else:
        console.print(f"[green]Detected base model: {base_model}[/green]")
    
    # Load config
    if not config:
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    # Run deployment menu
    from src.deployer import ModelDeployer
    deployer = ModelDeployer(config)
    deployer.deploy_menu(base_model, selected["path"])


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🧠 NeuroPipe - Autonomous Document-to-LLM Data Factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF file
  python main.py process document.pdf --name my_dataset
  
  # Process all PDFs in a directory
  python main.py batch ./documents --name combined_dataset
  
  # Run Interactive Wizard
  python main.py
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single PDF file')
    process_parser.add_argument('file', type=str, help='Path to PDF file')
    process_parser.add_argument('--name', '-n', type=str, help='Dataset name', default=None)
    process_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process all PDFs in a directory')
    batch_parser.add_argument('directory', type=str, help='Directory containing PDFs')
    batch_parser.add_argument('--name', '-n', type=str, help='Dataset name', default='dataset')
    batch_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check pipeline health')
    health_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web interface')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', '-p', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')

    # Distill command
    distill_parser = subparsers.add_parser('distill', help='Distill generated data for training')
    distill_parser.add_argument('--name', '-n', type=str, help='Batch name to process', default='combined')
    distill_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')

    # Train command
    train_parser = subparsers.add_parser('train', help='Fine-tune Sarvam-1 model')
    train_parser.add_argument('--data', '-d', type=str, help='Path to training data (jsonl)', default=None)
    train_parser.add_argument('--model', '-m', type=str, help='Base model ID', default='sarvamai/sarvam-1')
    train_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    # Interactive command
    subparsers.add_parser('wizard', help='Run interactive setup wizard')
    
    args = parser.parse_args()
    
    # Default to interactive mode if no command
    if not args.command or args.command == 'wizard':
        interactive_mode()
        return
    
    if args.command == 'health':
        pipeline = DataPipeline(args.config)
        healthy = pipeline.check_health()
        sys.exit(0 if healthy else 1)
    
    elif args.command == 'process':
        if not Path(args.file).exists():
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            sys.exit(1)
            
        pipeline = DataPipeline(args.config)
        
        if not pipeline.check_health():
            console.print("[red]Pipeline health check failed. Please ensure Ollama is running.[/red]")
            sys.exit(1)
        
        result = pipeline.process_file(args.file, args.name)
        
        if result.success:
            console.print(f"\n[bold green]✅ Success![/bold green] Processed in {result.duration_seconds:.1f}s")
            console.print(f"   Total samples generated: {result.total_samples}")
        else:
            console.print(f"\n[bold red]❌ Failed[/bold red]")
            for error in result.errors:
                console.print(f"   • {error}")
            sys.exit(1)
    
    elif args.command == 'batch':
        if not Path(args.directory).is_dir():
            console.print(f"[red]Error: Directory not found: {args.directory}[/red]")
            sys.exit(1)
            
        pipeline = DataPipeline(args.config)
        
        if not pipeline.check_health():
            console.print("[red]Pipeline health check failed. Please ensure Ollama is running.[/red]")
            sys.exit(1)
        
        result = pipeline.process_directory(args.directory, args.name)
        
        if result.success:
            console.print(f"\n[bold green]✅ Success![/bold green] Processed {result.documents_processed} documents in {result.duration_seconds:.1f}s")
            console.print(f"   Total samples generated: {result.total_samples}")
        else:
            console.print(f"\n[bold red]❌ Failed[/bold red]")
            for error in result.errors:
                console.print(f"   • {error}")
            sys.exit(1)
    
    elif args.command == 'serve':
        console.print(f"[bold cyan]Starting web interface on http://{args.host}:{args.port}[/bold cyan]")
        import uvicorn
        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=True)

    elif args.command == 'distill':
        from src.distiller import DataDistiller
        
        distiller = DataDistiller(DataPipeline(args.config).config) 
        distiller.distill_batch(args.name)

    elif args.command == 'train':
        from src.train import LLMTrainer
        
        # Check for data path, default to distilled path if not provided
        data_path = args.data
        if not data_path:
            # Try to infer based on name or defaults
            data_path = "datasets/distilled_train.jsonl"
            
        trainer = LLMTrainer(model_id=args.model, config=DataPipeline(args.config).config)
        trainer.train(data_path)


if __name__ == "__main__":
    main()
