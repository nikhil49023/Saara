"""
Main Pipeline Module
Orchestrates the complete document-to-dataset pipeline.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from .pdf_extractor import PDFExtractor, BatchExtractor, ExtractedDocument
from .chunker import TextChunker, ChunkProcessor, TextChunk
from .labeler import DataLabeler, LabeledDocument
from .dataset_generator import DatasetGenerator, AlpacaFormatGenerator, ShareGPTFormatGenerator
from .ollama_client import OllamaClient
from .translator import Translator

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    documents_processed: int
    total_chunks: int
    total_samples: int
    output_files: Dict[str, Any]
    errors: List[str]
    duration_seconds: float
    labeled_document: Optional[LabeledDocument] = None


class DataPipeline:
    """
    Main pipeline for converting documents to training datasets.
    Uses Granite 4.0 via Ollama for intelligent labeling.
    Uses Saara AI for translation of Indian languages.
    """
    
    def __init__(self, config_source: Union[str, Dict[str, Any]] = "config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_source: Path to configuration YAML file or configuration dictionary
        """
        if isinstance(config_source, dict):
            self.config = config_source
        else:
            self.config = self._load_config(config_source)
        self._setup_logging()
        
        # Initialize components
        self.extractor = PDFExtractor(self.config.get('pdf', {}))
        self.chunker = TextChunker(self.config.get('text', {}))
        self.labeler = DataLabeler(self.config)
        self.generator = DatasetGenerator(self.config)
        self.ollama_client = OllamaClient(self.config.get('ollama', {}))
        self.translator = Translator(self.config, self.ollama_client)
        
        # Create directories
        self._setup_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Setup file handler if configured
        log_file = log_config.get('file')
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
        
        logging.getLogger().setLevel(level)
    
    def _setup_directories(self):
        """Create necessary directories."""
        output_dir = self.config.get('output', {}).get('directory', './datasets')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path('./logs').mkdir(parents=True, exist_ok=True)
        Path('./uploads').mkdir(parents=True, exist_ok=True)
    
    def check_health(self) -> bool:
        """Check if all services are healthy."""
        console.print("\n[bold]🔍 Checking pipeline health...[/bold]\n")
        
        # Check Ollama
        ollama_ok = self.ollama_client.check_health()
        
        if ollama_ok:
            model_info = self.ollama_client.get_model_info()
            console.print(f"✅ Ollama: Connected")
            console.print(f"   Model: {model_info.get('name', 'unknown')}")
            console.print(f"   Parameters: {model_info.get('parameter_size', 'unknown')}")
        else:
            console.print("❌ Ollama: Not available")
            console.print("   Run: ollama pull granite4")
            return False
        
        console.print("")
        return True
    
    def process_file(self, file_path: str, dataset_name: str = None) -> PipelineResult:
        """
        Process a single PDF file through the pipeline.
        
        Args:
            file_path: Path to PDF file
            dataset_name: Name for output dataset
            
        Returns:
            PipelineResult with processing details
        """
        start_time = datetime.now()
        errors = []
        
        if not dataset_name:
            dataset_name = Path(file_path).stem
        
        console.print(Panel.fit(
            f"[bold cyan]Processing: {Path(file_path).name}[/bold cyan]",
            title="📄 Document Pipeline"
        ))
        
        try:
            # Step 1: Extract
            with console.status("[bold green]Extracting text from PDF..."):
                document = self.extractor.extract(file_path)
                console.print(f"✅ Extracted {document.metadata.page_count} pages")
            
            # Step 1.5: Translate if needed (Disabled as per user request: all English)
            # with console.status("[bold green]Checking language & Translating..."):
            #     self.translator.translate_document(document)
            
            # Step 2: Chunk
            with console.status("[bold green]Chunking document..."):
                chunks = self.chunker.chunk_document(
                    document.full_text,
                    document.sections
                )
                chunks = ChunkProcessor.filter_chunks(chunks)
                console.print(f"✅ Created {len(chunks)} chunks")
            
            # Step 3: Label
            console.print("[bold green]Labeling with Granite 4.0...[/bold green]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing chunks...", total=len(chunks))
                
                def update_progress(current, total):
                    progress.update(task, completed=current)
                
                labeled_doc = self.labeler.label_document(
                    document, chunks, update_progress
                )
            
            console.print(f"✅ Generated {labeled_doc.total_qa_pairs} Q&A pairs")
            console.print(f"✅ Generated {labeled_doc.total_instruction_pairs} instruction pairs")
            
            # Step 4: Generate datasets
            with console.status("[bold green]Generating datasets..."):
                output_files = self.generator.generate_all([labeled_doc], dataset_name)
            
            # Display results
            self._display_results(labeled_doc, output_files)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                success=True,
                documents_processed=1,
                total_chunks=len(chunks),
                total_samples=labeled_doc.total_qa_pairs + labeled_doc.total_instruction_pairs,
                output_files=output_files,
                errors=errors,
                duration_seconds=duration,
                labeled_document=labeled_doc
            )
            
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            errors.append(str(e))
            duration = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                success=False,
                documents_processed=0,
                total_chunks=0,
                total_samples=0,
                output_files={},
                errors=errors,
                duration_seconds=duration,
                labeled_document=None
            )
    
    def process_directory(self, 
                          directory: str, 
                          dataset_name: str = "dataset") -> PipelineResult:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Path to directory containing PDFs
            dataset_name: Name for the combined dataset
            
        Returns:
            PipelineResult with processing details
        """
        start_time = datetime.now()
        errors = []
        
        # Find all PDFs
        pdf_files = list(Path(directory).glob("**/*.pdf"))
        
        if not pdf_files:
            console.print("[yellow]No PDF files found in directory[/yellow]")
            return PipelineResult(
                success=False,
                documents_processed=0,
                total_chunks=0,
                total_samples=0,
                output_files={},
                errors=["No PDF files found"],
                duration_seconds=0
            )
        
        console.print(Panel.fit(
            f"[bold cyan]Processing {len(pdf_files)} PDF files[/bold cyan]",
            title="📁 Batch Processing"
        ))
        
        all_labeled_docs = []
        total_chunks = 0
        successful_files = 0
        failed_files = 0
        
        for i, pdf_path in enumerate(pdf_files):
            console.print(f"\n[bold]Document {i+1}/{len(pdf_files)}: {pdf_path.name}[/bold]")
            
            try:
                # Process single file (Extraction -> Translation -> Chunking -> Labeling -> Individual Dataset Generation)
                # The process_file method now handles individual dataset generation and display
                result = self.process_file(str(pdf_path))
                
                if result.success:
                    successful_files += 1
                    total_chunks += result.total_chunks
                    if result.labeled_document:
                        all_labeled_docs.append(result.labeled_document)
                else:
                    failed_files += 1
                    console.print(f"[red]Failed to process {pdf_path.name}[/red]")
                    errors.extend(result.errors)
                    
            except Exception as e:
                failed_files += 1
                logger.error(f"Batch processing error for {pdf_path.name}: {e}")
                console.print(f"[red]Error: {e}[/red]")
                errors.append(f"{pdf_path.name}: {e}")
        
        # Display batch results summary
        if all_labeled_docs:
            with console.status("[bold green]Generating combined dataset..."):
                output_files = self.generator.generate_all(all_labeled_docs, dataset_name)
            
            self._display_batch_results(all_labeled_docs, output_files)
        else:
            output_files = {}
        
        duration = (datetime.now() - start_time).total_seconds()
        
        total_samples = sum(d.total_qa_pairs + d.total_instruction_pairs for d in all_labeled_docs)
        
        return PipelineResult(
            success=len(all_labeled_docs) > 0,
            documents_processed=len(all_labeled_docs),
            total_chunks=total_chunks,
            total_samples=total_samples,
            output_files=output_files,
            errors=errors,
            duration_seconds=duration
        )
    
    def _display_results(self, doc: LabeledDocument, output_files: Dict):
        """Display results for a single document."""
        console.print("\n")
        
        table = Table(title="📊 Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Document", doc.title)
        table.add_row("Category", doc.category)
        table.add_row("Confidence", f"{doc.confidence:.2%}")
        table.add_row("Main Topic", doc.topics.get('main_topic', 'N/A'))
        table.add_row("Chunks", str(len(doc.chunks)))
        table.add_row("Q&A Pairs", str(doc.total_qa_pairs))
        table.add_row("Instruction Pairs", str(doc.total_instruction_pairs))
        
        console.print(table)
        
        if output_files:
            console.print("\n[bold]📁 Output Files:[/bold]")
            for dataset_type, files in output_files.items():
                if isinstance(files, dict):
                    for fmt, path in files.items():
                        console.print(f"  • {dataset_type}/{fmt}: {path}")
                else:
                    console.print(f"  • {dataset_type}: {files}")
    
    def _display_batch_results(self, docs: List[LabeledDocument], output_files: Dict):
        """Display results for batch processing."""
        console.print("\n")
        
        # Summary table
        table = Table(title="📊 Batch Processing Summary")
        table.add_column("Document", style="cyan")
        table.add_column("Category", style="yellow")
        table.add_column("Chunks", style="green")
        table.add_column("Q&A", style="green")
        table.add_column("Instructions", style="green")
        
        for doc in docs:
            table.add_row(
                doc.title[:40] + "..." if len(doc.title) > 40 else doc.title,
                doc.category,
                str(len(doc.chunks)),
                str(doc.total_qa_pairs),
                str(doc.total_instruction_pairs)
            )
        
        # Totals
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{len(docs)} docs[/bold]",
            f"[bold]{sum(len(d.chunks) for d in docs)}[/bold]",
            f"[bold]{sum(d.total_qa_pairs for d in docs)}[/bold]",
            f"[bold]{sum(d.total_instruction_pairs for d in docs)}[/bold]"
        )
        
        console.print(table)
        
        if output_files:
            console.print("\n[bold]📁 Output Files:[/bold]")
            for dataset_type, files in output_files.items():
                if isinstance(files, dict):
                    for fmt, path in files.items():
                        console.print(f"  • {dataset_type}/{fmt}: {path}")


def load_pipeline(config_path: str = "config.yaml") -> DataPipeline:
    """Factory function to create a pipeline instance."""
    return DataPipeline(config_path)
