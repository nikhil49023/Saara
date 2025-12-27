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
        "Welcome to the Interactive Setup Wizard!",
        title="Welcome"
    ))
    
    # 1. Ask for paths
    base_dir = os.getcwd()
    raw_path = Prompt.ask("Enter path to raw data folder (PDFs)", default=base_dir)
    output_path = Prompt.ask("Enter path for output datasets", default="./datasets")
    
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
    else:
        console.print(f"\n[bold red]❌ One or more errors occurred during processing.[/bold red]")


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
        from src.train import SarvamTrainer
        
        # Check for data path, default to distilled path if not provided
        data_path = args.data
        if not data_path:
            # Try to infer based on name or defaults
            data_path = "datasets/distilled_train.jsonl"
            
        trainer = SarvamTrainer(DataPipeline(args.config).config)
        trainer.train(data_path)


if __name__ == "__main__":
    main()
