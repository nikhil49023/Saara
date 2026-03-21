"""
Pre-training Dataset Generator

Creates datasets specifically optimized for language model pre-training.
Unlike fine-tuning datasets (Q&A, instruction pairs), pre-training needs:
- Large volumes of clean, raw text
- Minimal structure (just text)
- Quality filtering for coherence
- Deduplication

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import logging
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class PretrainDatasetStats:
    """Statistics for pre-training dataset."""
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens_estimate: int = 0
    total_characters: int = 0
    filtered_chunks: int = 0
    duplicate_chunks: int = 0
    output_files: List[str] = field(default_factory=list)
    
    def display(self):
        """Display stats in a nice table."""
        table = Table(title="📊 Pre-training Dataset Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="green")
        table.add_column("Value", style="yellow")
        
        table.add_row("Documents Processed", str(self.total_documents))
        table.add_row("Total Chunks", str(self.total_chunks))
        table.add_row("Filtered (low quality)", str(self.filtered_chunks))
        table.add_row("Duplicates Removed", str(self.duplicate_chunks))
        table.add_row("Final Chunks", str(self.total_chunks - self.filtered_chunks - self.duplicate_chunks))
        table.add_row("Total Characters", f"{self.total_characters:,}")
        table.add_row("Estimated Tokens", f"~{self.total_tokens_estimate:,}")
        
        console.print(table)
        
        if self.output_files:
            console.print("\n[bold]Output Files:[/bold]")
            for f in self.output_files:
                console.print(f"  • [cyan]{f}[/cyan]")


class PretrainDatasetGenerator:
    """
    Generates datasets specifically for pre-training language models.
    
    Key differences from fine-tuning dataset generation:
    1. Outputs raw text, not instruction pairs
    2. Larger chunk sizes for context
    3. Focus on text quality and coherence
    4. Deduplication to prevent memorization
    5. Optional LLM-enhanced processing
    """
    
    # LLM Prompts for text processing
    CLEAN_TEXT_PROMPT = """You are a text cleaner for language model pre-training data.

Your task is to clean and improve the following text while preserving ALL information:

1. Fix OCR errors and typos
2. Expand abbreviations where appropriate
3. Fix broken sentences and paragraphs
4. Remove artifacts like page numbers, headers, footers
5. Normalize formatting (consistent spacing, punctuation)
6. Keep the text natural and readable

IMPORTANT: 
- Do NOT summarize or remove any content
- Do NOT add new information
- Keep the original meaning exactly
- Output ONLY the cleaned text, nothing else

Text to clean:
{text}

Cleaned text:"""

    REFORMAT_PROMPT = """You are a text reformatter for language model pre-training.

Reformat the following text to be clean, well-structured prose suitable for training a language model:

1. Convert bullet points and lists into flowing paragraphs
2. Expand abbreviations to full words
3. Write out numbers in words where appropriate  
4. Fix any grammatical issues
5. Ensure proper paragraph breaks
6. Remove any references to "this document" or "page X"

Keep ALL the factual information. Do NOT summarize.

Text:
{text}

Reformatted text:"""

    QUALITY_SCORE_PROMPT = """You are a text quality evaluator for pre-training data.

Evaluate the following text chunk for suitability in training a language model.

Score the text from 1-10 on these criteria:
- Coherence: Does it make sense? Is it complete thoughts?
- Quality: Is it well-written? Free of errors?
- Information: Does it contain useful knowledge?
- Formatting: Is it clean and readable?

Text:
{text}

Respond with ONLY a JSON object like this:
{{"score": 7, "reason": "Brief explanation"}}"""

    EXPAND_PROMPT = """You are a content expander for language model pre-training data.

Take the following text and expand it into more detailed, educational prose:

1. Add relevant context and explanations
2. Define technical terms when they appear
3. Provide examples where helpful
4. Maintain factual accuracy
5. Write in clear, instructional style

Text:
{text}

Expanded text:"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Pre-training specific settings
        self.chunk_size = self.config.get("pretrain", {}).get("chunk_size", 4096)  # Larger chunks
        self.chunk_overlap = self.config.get("pretrain", {}).get("chunk_overlap", 256)
        self.min_chunk_length = self.config.get("pretrain", {}).get("min_chunk_length", 200)
        self.max_chunk_length = self.config.get("pretrain", {}).get("max_chunk_length", 8192)
        self.deduplicate = self.config.get("pretrain", {}).get("deduplicate", True)
        self.quality_filter = self.config.get("pretrain", {}).get("quality_filter", True)
        
        # LLM settings
        self.use_llm = self.config.get("pretrain", {}).get("use_llm", False)
        self.llm_model = self.config.get("pretrain", {}).get("llm_model", "granite4:latest")
        self.ollama_url = self.config.get("ollama", {}).get("base_url", "http://localhost:11434")
        
        # Vision OCR settings (default to vision for better extraction quality)
        self.use_vision = self.config.get("pretrain", {}).get("use_vision", True)
        self.ocr_model = self.config.get("pretrain", {}).get("ocr_model", "qwen2.5vl:3b")
        
        # Track seen hashes for deduplication
        self.seen_hashes: Set[str] = set()
        
        # Patterns that indicate bad OCR output (vision model describing instead of extracting)
        self.ocr_artifact_patterns = [
            "the image shows",
            "this image shows", 
            "the page shows",
            "the document shows",
            "the text appears to be",
            "appears to be written",
            "appears to be in",
            "seems to be",
            "this appears to be",
            "the image contains",
            "displays a",
            "difficult to read",
            "foreign language",
            "i can see",
            "i cannot",
            "looking at this",
            "the page contains",
            "written in black ink",
            "printed on paper",
            "predominantly white",
            "bullet points",
        ]
    
    def _is_ocr_artifact(self, text: str) -> bool:
        """Check if text contains OCR artifacts (vision model descriptions instead of content)."""
        if not text or len(text.strip()) < 50:
            return True
        
        text_lower = text.lower()
        
        # Check for multiple bad patterns
        artifact_count = sum(1 for pattern in self.ocr_artifact_patterns if pattern in text_lower)
        
        # If 3 or more patterns found, likely an OCR artifact
        if artifact_count >= 3:
            return True
        
        # Check for high ratio of pattern matches to text length
        if artifact_count >= 2 and len(text) < 500:
            return True
        
        return False
    
    def _clean_ocr_artifacts(self, text: str) -> str:
        """Remove OCR artifact sentences from text."""
        if not text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip lines that are primarily OCR artifacts
            artifact_count = sum(1 for pattern in self.ocr_artifact_patterns if pattern in line_lower)
            
            if artifact_count >= 2:
                continue  # Skip this line
            
            # Skip short lines that are just descriptions
            if len(line.strip()) < 50 and artifact_count >= 1:
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
        
    def _call_llm(self, prompt: str, timeout: int = 60) -> Optional[str]:
        """Call Ollama LLM with the given prompt."""
        import requests
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048,
                    }
                },
                timeout=timeout
            )
            
            if response.ok:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
        
        return None
    
    def llm_clean_text(self, text: str) -> str:
        """Use LLM to clean and improve text quality."""
        if not self.use_llm or len(text) > 3000:  # Limit size for LLM
            return text
            
        prompt = self.CLEAN_TEXT_PROMPT.format(text=text[:3000])
        result = self._call_llm(prompt)
        
        if result and len(result) > len(text) * 0.5:  # Sanity check
            return result
        return text
    
    def llm_reformat_text(self, text: str) -> str:
        """Use LLM to reformat text into clean prose."""
        if not self.use_llm or len(text) > 3000:
            return text
            
        prompt = self.REFORMAT_PROMPT.format(text=text[:3000])
        result = self._call_llm(prompt)
        
        if result and len(result) > len(text) * 0.5:
            return result
        return text
    
    def llm_score_quality(self, text: str) -> tuple:
        """Use LLM to score text quality. Returns (score, reason)."""
        if not self.use_llm:
            return (7, "LLM scoring disabled")
            
        prompt = self.QUALITY_SCORE_PROMPT.format(text=text[:2000])
        result = self._call_llm(prompt, timeout=30)
        
        if result:
            try:
                # Try to parse JSON response
                import json as json_module
                data = json_module.loads(result)
                return (data.get("score", 5), data.get("reason", ""))
            except:
                # Try to extract score from text
                import re
                match = re.search(r'(\d+)', result)
                if match:
                    return (int(match.group(1)), result)
        
        return (5, "Could not score")
    
    def llm_expand_text(self, text: str) -> str:
        """Use LLM to expand text with more detail."""
        if not self.use_llm or len(text) > 2000:
            return text
            
        prompt = self.EXPAND_PROMPT.format(text=text[:2000])
        result = self._call_llm(prompt, timeout=90)
        
        if result and len(result) > len(text):  # Should be longer
            return result
        return text

    def extract_text_from_pdfs(self, input_path: str, use_ocr: bool = None) -> List[Dict[str, Any]]:
        """
        Extract raw text from PDF files.
        
        Args:
            input_path: Path to PDF file or directory
            use_ocr: Whether to use Vision OCR for extraction (defaults to self.use_vision)
            
        Returns:
            List of documents with text and metadata
        """
        # Use instance setting if not explicitly specified
        if use_ocr is None:
            use_ocr = self.use_vision
        
        path_obj = Path(input_path)
        pdf_files = []
        
        if path_obj.is_file() and path_obj.suffix.lower() == '.pdf':
            pdf_files = [path_obj]
        elif path_obj.is_dir():
            pdf_files = list(path_obj.glob("**/*.pdf"))
        else:
            console.print(f"[red]Invalid path: {input_path}[/red]")
            return []
        
        console.print(f"[cyan]Found {len(pdf_files)} PDF files[/cyan]")
        
        # Use Vision OCR if enabled (default)
        if use_ocr:
            return self._extract_with_vision(pdf_files)
        else:
            return self._extract_with_parser(pdf_files)
    
    def _extract_with_vision(self, pdf_files: List[Path]) -> List[Dict[str, Any]]:
        """Extract text using Vision OCR (higher quality, slower)."""
        from saara.qwen_ocr import QwenVisionOCR
        from saara.moondream_ocr import MoondreamOCR
        
        console.print(f"[cyan]📷 Using Vision Model ({self.ocr_model}) for extraction...[/cyan]")
        
        documents = []
        
        # Initialize OCR based on model
        try:
            if 'moondream' in self.ocr_model.lower():
                ocr = MoondreamOCR(self.config)
            else:
                ocr = QwenVisionOCR(self.config, model_name=self.ocr_model)
        except Exception as e:
            console.print(f"[yellow]⚠ Vision OCR init failed: {e}. Falling back to parser.[/yellow]")
            return self._extract_with_parser(pdf_files)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Vision OCR extraction...", total=len(pdf_files))
            
            for pdf_path in pdf_files:
                try:
                    full_text = ocr.extract_text_from_pdf(str(pdf_path))
                    
                    if full_text and full_text.strip():
                        # Clean OCR artifacts (vision model descriptions)
                        cleaned_text = self._clean_ocr_artifacts(full_text)
                        
                        # Check if result is still usable
                        if cleaned_text and len(cleaned_text.strip()) > 100 and not self._is_ocr_artifact(cleaned_text):
                            documents.append({
                                "source": str(pdf_path),
                                "text": cleaned_text,
                                "extraction_method": "vision",
                            })
                        else:
                            # OCR output was mostly artifacts, try parser
                            logger.warning(f"Vision output for {pdf_path.name} was mostly OCR artifacts, using parser...")
                            fallback_docs = self._extract_with_parser([pdf_path])
                            documents.extend(fallback_docs)
                        
                except Exception as e:
                    logger.warning(f"Vision extraction failed for {pdf_path}: {e}")
                    # Try parser fallback for this file
                    try:
                        fallback_docs = self._extract_with_parser([pdf_path])
                        documents.extend(fallback_docs)
                    except:
                        pass
                    
                progress.advance(task)
        
        return documents
    
    def _extract_with_parser(self, pdf_files: List[Path]) -> List[Dict[str, Any]]:
        """Extract text using PyMuPDF parser (faster, works for digital PDFs)."""
        import fitz  # PyMuPDF
        
        console.print(f"[cyan]📄 Using PDF Parser for extraction...[/cyan]")
        
        documents = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting text...", total=len(pdf_files))
            
            for pdf_path in pdf_files:
                try:
                    doc = fitz.open(str(pdf_path))
                    full_text = ""
                    
                    for page in doc:
                        text = page.get_text()
                        if text.strip():
                            full_text += text + "\n\n"
                    
                    num_pages = len(doc)
                    doc.close()
                    
                    if full_text.strip():
                        documents.append({
                            "source": str(pdf_path),
                            "text": full_text,
                            "num_pages": num_pages,
                            "extraction_method": "parser",
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to extract {pdf_path}: {e}")
                    
                progress.advance(task)
        
        return documents
    
    def extract_text_from_files(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from various file formats (.txt, .md, .json, .jsonl).
        
        Args:
            input_path: Path to file or directory
            
        Returns:
            List of documents with text
        """
        path_obj = Path(input_path)
        documents = []
        
        # Supported extensions
        extensions = [".txt", ".md", ".json", ".jsonl"]
        
        if path_obj.is_file():
            files = [path_obj]
        elif path_obj.is_dir():
            files = []
            for ext in extensions:
                files.extend(path_obj.glob(f"**/*{ext}"))
        else:
            return []
        
        console.print(f"[cyan]Found {len(files)} text files[/cyan]")
        
        for file_path in files:
            try:
                suffix = file_path.suffix.lower()
                
                if suffix == ".jsonl":
                    # JSONL - extract text field from each line
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                text = data.get("text") or data.get("content") or ""
                                if text:
                                    documents.append({
                                        "source": str(file_path),
                                        "text": text,
                                    })
                            except:
                                pass
                                
                elif suffix == ".json":
                    # JSON - try to find text content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        for item in data:
                            text = item.get("text") or item.get("content") or ""
                            if text:
                                documents.append({"source": str(file_path), "text": text})
                    elif isinstance(data, dict):
                        text = data.get("text") or data.get("content") or ""
                        if text:
                            documents.append({"source": str(file_path), "text": text})
                            
                else:
                    # Plain text or markdown
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    if text.strip():
                        documents.append({
                            "source": str(file_path),
                            "text": text,
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
        
        return documents
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for pre-training.
        
        Operations:
        - Remove excessive whitespace
        - Normalize unicode
        - Remove control characters
        - Fix common OCR artifacts
        """
        if not text:
            return ""
        
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        
        # Remove control characters (except newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix common OCR artifacts
        text = re.sub(r'[|]{2,}', '', text)  # Multiple pipes
        text = re.sub(r'[_]{3,}', '', text)  # Multiple underscores
        text = re.sub(r'\.{4,}', '...', text)  # Excessive dots
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
        
        # Remove lines that are just punctuation or numbers
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep line if it has enough alphabetic content
            alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
            if alpha_ratio > 0.3 or len(stripped) < 5:  # Short lines are likely headers
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        Split text into chunks suitable for pre-training.
        
        Uses sentence-aware splitting to avoid cutting mid-sentence.
        """
        if not text or len(text) < self.min_chunk_length:
            return []
        
        chunks = []
        
        # Try to split on paragraph boundaries first
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                # Save current chunk if it's long enough
                if len(current_chunk) >= self.min_chunk_length:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source,
                        "length": len(current_chunk),
                    })
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last N characters as overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # Try to start at a sentence boundary
                    sentence_start = overlap_text.rfind('. ')
                    if sentence_start > 0:
                        overlap_text = overlap_text[sentence_start + 2:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if len(current_chunk) >= self.min_chunk_length:
            chunks.append({
                "text": current_chunk.strip(),
                "source": source,
                "length": len(current_chunk),
            })
        
        return chunks
    
    def is_quality_chunk(self, text: str) -> bool:
        """
        Check if a chunk meets quality standards for pre-training.
        
        Filters out:
        - Too short or too long
        - Too much repetition
        - Not enough real words
        - Too many special characters
        """
        if len(text) < self.min_chunk_length:
            return False
            
        if len(text) > self.max_chunk_length:
            return False
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) < 10:
            return False
            
        word_counts = Counter(words)
        most_common_ratio = word_counts.most_common(1)[0][1] / len(words)
        if most_common_ratio > 0.3:  # Single word is more than 30% of text
            return False
        
        # Check for enough alphabetic content
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / len(text)
        if alpha_ratio < 0.5:
            return False
        
        # Check for too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:\'"()-')
        special_ratio = special_chars / len(text)
        if special_ratio > 0.1:
            return False
        
        return True
    
    def get_text_hash(self, text: str) -> str:
        """Get hash of text for deduplication."""
        # Normalize for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if this chunk is a duplicate."""
        text_hash = self.get_text_hash(text)
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def generate_dataset(
        self,
        input_path: str,
        output_dir: str,
        dataset_name: str = "pretrain_corpus",
        formats: List[str] = None
    ) -> PretrainDatasetStats:
        """
        Generate a pre-training dataset from input files.
        
        Args:
            input_path: Path to input files (PDFs, text, etc.)
            output_dir: Output directory
            dataset_name: Name prefix for output files
            formats: Output formats ("txt", "jsonl", "parquet")
            
        Returns:
            Dataset statistics
        """
        if formats is None:
            formats = ["txt", "jsonl"]
        
        stats = PretrainDatasetStats()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        console.print(Panel.fit(
            "[bold cyan]🔄 Pre-training Dataset Generation[/bold cyan]\n\n"
            f"Input: {input_path}\n"
            f"Output: {output_dir}\n"
            f"Chunk Size: {self.chunk_size} chars",
            border_style="cyan"
        ))
        
        # Step 1: Extract text from all sources
        console.print("\n[bold]Step 1: Extracting text...[/bold]")
        
        path_obj = Path(input_path)
        documents = []
        
        # Check for PDFs
        if path_obj.is_file() and path_obj.suffix.lower() == '.pdf':
            documents.extend(self.extract_text_from_pdfs(input_path))
        elif path_obj.is_dir():
            # Extract from all file types
            pdf_docs = self.extract_text_from_pdfs(input_path)
            text_docs = self.extract_text_from_files(input_path)
            documents = pdf_docs + text_docs
        else:
            documents = self.extract_text_from_files(input_path)
        
        stats.total_documents = len(documents)
        console.print(f"[green]✓ Extracted text from {len(documents)} documents[/green]")
        
        if not documents:
            console.print("[red]No documents found![/red]")
            return stats
        
        # Step 2: Clean and chunk
        console.print("\n[bold]Step 2: Cleaning and chunking...[/bold]")
        
        all_chunks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(documents))
            
            for doc in documents:
                # Clean text
                cleaned_text = self.clean_text(doc["text"])
                
                # Chunk
                chunks = self.chunk_text(cleaned_text, doc["source"])
                all_chunks.extend(chunks)
                
                progress.advance(task)
        
        stats.total_chunks = len(all_chunks)
        console.print(f"[green]✓ Created {len(all_chunks)} chunks[/green]")
        
        # Step 2.5: LLM Enhancement (if enabled)
        if self.use_llm:
            console.print("\n[bold]Step 2.5: LLM-enhanced text processing...[/bold]")
            console.print(f"[dim]Using model: {self.llm_model}[/dim]")
            
            enhanced_chunks = []
            llm_cleaned = 0
            llm_failed = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("LLM cleaning...", total=len(all_chunks))
                
                for chunk in all_chunks:
                    original_text = chunk["text"]
                    
                    # Try LLM cleaning
                    cleaned = self.llm_clean_text(original_text)
                    
                    if cleaned != original_text:
                        chunk["text"] = cleaned
                        chunk["llm_enhanced"] = True
                        llm_cleaned += 1
                    else:
                        llm_failed += 1
                    
                    enhanced_chunks.append(chunk)
                    progress.advance(task)
            
            all_chunks = enhanced_chunks
            console.print(f"[green]✓ LLM enhanced {llm_cleaned} chunks[/green]")
            if llm_failed > 0:
                console.print(f"[dim]  {llm_failed} chunks unchanged (too long or LLM unavailable)[/dim]")
        
        # Step 3: Quality filtering
        if self.quality_filter:
            console.print("\n[bold]Step 3: Quality filtering...[/bold]")
            
            quality_chunks = []
            
            # Use LLM scoring if enabled, otherwise use heuristics
            if self.use_llm:
                console.print("[dim]Using LLM for quality scoring (slower but more accurate)[/dim]")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Scoring quality...", total=len(all_chunks))
                    
                    for chunk in all_chunks:
                        score, reason = self.llm_score_quality(chunk["text"])
                        chunk["quality_score"] = score
                        chunk["quality_reason"] = reason
                        
                        if score >= 5:  # Keep chunks with score >= 5
                            quality_chunks.append(chunk)
                        else:
                            stats.filtered_chunks += 1
                        
                        progress.advance(task)
            else:
                for chunk in all_chunks:
                    if self.is_quality_chunk(chunk["text"]):
                        quality_chunks.append(chunk)
                    else:
                        stats.filtered_chunks += 1
            
            all_chunks = quality_chunks
            console.print(f"[green]✓ Kept {len(all_chunks)} high-quality chunks[/green]")
            console.print(f"[dim]  Filtered out {stats.filtered_chunks} low-quality chunks[/dim]")
        
        # Step 4: Deduplication
        if self.deduplicate:
            console.print("\n[bold]Step 4: Deduplicating...[/bold]")
            
            unique_chunks = []
            for chunk in all_chunks:
                if not self.is_duplicate(chunk["text"]):
                    unique_chunks.append(chunk)
                else:
                    stats.duplicate_chunks += 1
            
            all_chunks = unique_chunks
            console.print(f"[green]✓ Kept {len(all_chunks)} unique chunks[/green]")
            console.print(f"[dim]  Removed {stats.duplicate_chunks} duplicates[/dim]")
        
        # Calculate final stats
        for chunk in all_chunks:
            stats.total_characters += len(chunk["text"])
        
        # Estimate tokens (rough: 1 token ≈ 4 characters for English)
        stats.total_tokens_estimate = stats.total_characters // 4
        
        # Step 5: Save outputs
        console.print("\n[bold]Step 5: Saving dataset...[/bold]")
        
        if "txt" in formats:
            # Plain text format (one file, chunks separated by newlines)
            txt_path = output_path / f"{dataset_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for chunk in all_chunks:
                    f.write(chunk["text"])
                    f.write("\n\n" + "="*50 + "\n\n")
            stats.output_files.append(str(txt_path))
            console.print(f"  [green]✓[/green] Saved {txt_path}")
        
        if "jsonl" in formats:
            # JSONL format (standard for HuggingFace datasets)
            jsonl_path = output_path / f"{dataset_name}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for chunk in all_chunks:
                    record = {
                        "text": chunk["text"],
                        "source": chunk.get("source", ""),
                        "length": chunk.get("length", len(chunk["text"])),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.output_files.append(str(jsonl_path))
            console.print(f"  [green]✓[/green] Saved {jsonl_path}")
        
        if "parquet" in formats:
            try:
                import pandas as pd
                
                df = pd.DataFrame([{
                    "text": c["text"],
                    "source": c.get("source", ""),
                    "length": c.get("length", len(c["text"])),
                } for c in all_chunks])
                
                parquet_path = output_path / f"{dataset_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                stats.output_files.append(str(parquet_path))
                console.print(f"  [green]✓[/green] Saved {parquet_path}")
            except ImportError:
                console.print("  [yellow]⚠ Parquet format requires pandas and pyarrow[/yellow]")
        
        # Display final stats
        console.print("\n")
        stats.display()
        
        return stats
    
    def create_train_val_split(
        self,
        dataset_path: str,
        output_dir: str,
        val_ratio: float = 0.05,
        seed: int = 42
    ) -> Dict[str, str]:
        """
        Split a pre-training dataset into train and validation sets.
        
        Args:
            dataset_path: Path to .jsonl dataset
            output_dir: Output directory
            val_ratio: Fraction for validation (default 5%)
            seed: Random seed
            
        Returns:
            Paths to train and val files
        """
        import random
        
        random.seed(seed)
        
        # Load dataset
        chunks = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
        
        # Shuffle
        random.shuffle(chunks)
        
        # Split
        val_size = int(len(chunks) * val_ratio)
        val_chunks = chunks[:val_size]
        train_chunks = chunks[val_size:]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save train
        train_path = output_path / "train.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for chunk in train_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        
        # Save val
        val_path = output_path / "validation.jsonl"
        with open(val_path, 'w', encoding='utf-8') as f:
            for chunk in val_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        
        console.print(f"[green]✓ Train set: {len(train_chunks)} samples → {train_path}[/green]")
        console.print(f"[green]✓ Val set: {len(val_chunks)} samples → {val_path}[/green]")
        
        return {
            "train": str(train_path),
            "validation": str(val_path),
        }


def run_pretrain_dataset_wizard(config: dict = None):
    """Interactive wizard for creating pre-training datasets."""
    from rich.prompt import Prompt, Confirm
    
    console.print(Panel.fit(
        "[bold cyan]📚 Pre-training Dataset Creation[/bold cyan]\n\n"
        "[dim]Create large-scale text datasets optimized for language model pre-training.[/dim]\n"
        "[dim]Supports PDFs, text files, markdown, and JSONL.[/dim]",
        title="Pre-training Data Wizard",
        border_style="cyan"
    ))
    
    # Step 1: Input path
    console.print("\n[bold]Step 1: Select Input Data[/bold]")
    console.print("[dim]Supported: PDFs, .txt, .md, .jsonl files[/dim]\n")
    
    input_path = Prompt.ask("Path to input files or directory").strip('"\'')
    
    if not Path(input_path).exists():
        console.print(f"[red]Path not found: {input_path}[/red]")
        return None
    
    # Step 2: Output configuration
    console.print("\n[bold]Step 2: Configure Output[/bold]\n")
    
    output_dir = Prompt.ask("Output directory", default="datasets/pretrain").strip('"\'')
    dataset_name = Prompt.ask("Dataset name", default="corpus")
    
    # Step 3: PDF Extraction Method
    console.print("\n[bold]Step 3: PDF Extraction Method[/bold]")
    console.print("[dim]Choose how to extract text from PDF files.[/dim]\n")
    console.print("  1. 📷 Vision Model (recommended) - Better quality, handles scanned PDFs")
    console.print("  2. 📄 PDF Parser - Faster, works well for digital PDFs")
    
    extraction_choice = Prompt.ask("Select method", choices=["1", "2"], default="1")
    use_vision = extraction_choice == "1"
    
    # Vision model selection
    ocr_model = "qwen2.5vl:3b"
    if use_vision:
        from saara.model_manager import ModelManager, MODEL_CATALOG
        
        # Get installed models
        manager = ModelManager()
        installed_models = manager.get_installed_models()
        
        # Display available vision models
        console.print("\n[bold]👁️ Available Vision Models:[/bold]\n")
        
        vision_models = []
        
        # Add models from catalog
        for model_info in MODEL_CATALOG.get("vision", []):
            vision_models.append({
                "name": model_info.name,
                "display": model_info.display_name,
                "size": f"{model_info.size_gb} GB",
                "vram": f"{model_info.vram_required} GB",
                "desc": model_info.description[:45] + "..." if len(model_info.description) > 45 else model_info.description,
                "installed": any(model_info.name.split(":")[0] in m for m in installed_models)
            })
        
        # Display table
        vision_table = Table(show_header=True, header_style="bold magenta")
        vision_table.add_column("#", style="cyan", width=3)
        vision_table.add_column("Model", style="green", width=18)
        vision_table.add_column("Size", width=8)
        vision_table.add_column("VRAM", width=8)
        vision_table.add_column("Description", width=42)
        vision_table.add_column("Status", width=12)
        
        for i, model in enumerate(vision_models, 1):
            status = "[green]✓ Ready[/green]" if model["installed"] else "[dim]Not installed[/dim]"
            vision_table.add_row(
                str(i),
                model["display"],
                model["size"],
                model["vram"],
                model["desc"],
                status
            )
        
        console.print(vision_table)
        console.print()
        
        # Let user select
        console.print(f"[dim]Enter a number (1-{len(vision_models)}) or 'other' for custom model[/dim]")
        ocr_choice = Prompt.ask("Select vision model", default="3")  # Default to qwen2.5vl:3b
        
        if ocr_choice.lower() == "other":
            ocr_model = Prompt.ask("Enter Ollama vision model name")
        elif ocr_choice.isdigit() and 1 <= int(ocr_choice) <= len(vision_models):
            selected = vision_models[int(ocr_choice) - 1]
            ocr_model = selected["name"]
            
            # Offer to install if not present
            if not selected["installed"]:
                console.print(f"\n[yellow]⚠ Model '{ocr_model}' is not installed.[/yellow]")
                if Confirm.ask(f"Install {ocr_model} now?", default=True):
                    console.print(f"[cyan]Installing {ocr_model}...[/cyan]")
                    import os
                    os.system(f"ollama pull {ocr_model}")
        else:
            console.print("[yellow]Invalid choice, using default (qwen2.5vl:3b).[/yellow]")
            ocr_model = "qwen2.5vl:3b"
            
        console.print(f"\n[green]✓ Using Vision Model: {ocr_model}[/green]")
    else:
        console.print("\n[green]✓ Using PDF Parser (PyMuPDF)[/green]")
    
    # Step 4: LLM Enhancement
    console.print("\n[bold]Step 4: LLM Enhancement (Optional)[/bold]")
    console.print("[dim]Use a local LLM to improve text quality, fix OCR errors, and score chunks.[/dim]\n")
    
    use_llm = Confirm.ask("Enable LLM-enhanced processing?", default=False)
    
    llm_model = "granite4:latest"
    if use_llm:
        from saara.model_manager import ModelManager, MODEL_CATALOG
        
        # Get installed models
        manager = ModelManager()
        installed_models = manager.get_installed_models()
        
        # Display available analyzer models
        console.print("\n[bold]🧠 Available LLM Models for Enhancement:[/bold]\n")
        
        llm_models = []
        
        # Add granite4 as first option (recommended)
        llm_models.append({
            "name": "granite4:latest",
            "display": "Granite 4 (Recommended)",
            "size": "~5 GB",
            "vram": "~6 GB",
            "desc": "IBM's latest. Excellent for text enhancement.",
            "installed": any("granite4" in m for m in installed_models)
        })
        
        # Add models from catalog
        for model_info in MODEL_CATALOG.get("analyzer", []):
            llm_models.append({
                "name": model_info.name,
                "display": model_info.display_name,
                "size": f"{model_info.size_gb} GB",
                "vram": f"{model_info.vram_required} GB",
                "desc": model_info.description[:45] + "..." if len(model_info.description) > 45 else model_info.description,
                "installed": any(model_info.name.split(":")[0] in m for m in installed_models)
            })
        
        # Display table
        llm_table = Table(show_header=True, header_style="bold magenta")
        llm_table.add_column("#", style="cyan", width=3)
        llm_table.add_column("Model", style="green", width=22)
        llm_table.add_column("Size", width=8)
        llm_table.add_column("VRAM", width=8)
        llm_table.add_column("Description", width=45)
        llm_table.add_column("Status", width=12)
        
        for i, model in enumerate(llm_models, 1):
            status = "[green]✓ Ready[/green]" if model["installed"] else "[dim]Not installed[/dim]"
            llm_table.add_row(
                str(i),
                model["display"],
                model["size"],
                model["vram"],
                model["desc"],
                status
            )
        
        console.print(llm_table)
        console.print()
        
        # Let user select
        valid_choices = [str(i) for i in range(1, len(llm_models) + 1)] + ["other"]
        console.print(f"[dim]Enter a number (1-{len(llm_models)}) or 'other' for custom model[/dim]")
        model_choice = Prompt.ask("Select model", default="1")
        
        if model_choice.lower() == "other":
            llm_model = Prompt.ask("Enter Ollama model name")
        elif model_choice.isdigit() and 1 <= int(model_choice) <= len(llm_models):
            selected = llm_models[int(model_choice) - 1]
            llm_model = selected["name"]
            
            # Offer to install if not present
            if not selected["installed"]:
                console.print(f"\n[yellow]⚠ Model '{llm_model}' is not installed.[/yellow]")
                if Confirm.ask(f"Install {llm_model} now?", default=True):
                    console.print(f"[cyan]Installing {llm_model}...[/cyan]")
                    import os
                    os.system(f"ollama pull {llm_model}")
        else:
            console.print("[yellow]Invalid choice, using default.[/yellow]")
            llm_model = "granite4:latest"
            
        console.print(f"\n[green]✓ Using LLM: {llm_model}[/green]")
        console.print("[yellow]⚠ LLM processing is slower but produces higher quality data[/yellow]")
    
    # Step 5: Processing options
    console.print("\n[bold]Step 5: Processing Options[/bold]\n")
    
    show_advanced = Confirm.ask("Configure advanced options?", default=False)
    
    chunk_size = 4096
    min_length = 200
    deduplicate = True
    quality_filter = True
    
    if show_advanced:
        chunk_size = int(Prompt.ask("Chunk size (characters)", default="4096"))
        min_length = int(Prompt.ask("Minimum chunk length", default="200"))
        deduplicate = Confirm.ask("Remove duplicates?", default=True)
        quality_filter = Confirm.ask("Filter low-quality chunks?", default=True)
    
    # Step 6: Output formats
    console.print("\n[bold]Step 6: Output Formats[/bold]")
    console.print("  1. JSONL only (recommended for HuggingFace)")
    console.print("  2. TXT only (plain text)")
    console.print("  3. Both JSONL and TXT")
    
    format_choice = Prompt.ask("Select format", choices=["1", "2", "3"], default="1")
    
    formats = []
    if format_choice == "1":
        formats = ["jsonl"]
    elif format_choice == "2":
        formats = ["txt"]
    else:
        formats = ["jsonl", "txt"]
    
    # Summary
    console.print("\n")
    summary_table = Table(title="📋 Configuration Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Setting", style="green")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Input Path", input_path)
    summary_table.add_row("Output Directory", output_dir)
    summary_table.add_row("Dataset Name", dataset_name)
    summary_table.add_row("PDF Extraction", f"Vision ({ocr_model})" if use_vision else "Parser (PyMuPDF)")
    summary_table.add_row("LLM Enhancement", f"Yes ({llm_model})" if use_llm else "No")
    summary_table.add_row("Chunk Size", f"{chunk_size} chars")
    summary_table.add_row("Min Chunk Length", f"{min_length} chars")
    summary_table.add_row("Deduplication", "Yes" if deduplicate else "No")
    summary_table.add_row("Quality Filter", "Yes" if quality_filter else "No")
    summary_table.add_row("Output Formats", ", ".join(formats))
    
    console.print(summary_table)
    console.print()
    
    if not Confirm.ask("[bold]Start dataset generation?[/bold]", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        return None
    
    # Create generator with config
    gen_config = {
        "pretrain": {
            "chunk_size": chunk_size,
            "min_chunk_length": min_length,
            "deduplicate": deduplicate,
            "quality_filter": quality_filter,
            "use_llm": use_llm,
            "llm_model": llm_model,
            "use_vision": use_vision,
            "ocr_model": ocr_model,
        }
    }
    
    generator = PretrainDatasetGenerator(gen_config)
    stats = generator.generate_dataset(input_path, output_dir, dataset_name, formats)
    
    # Offer train/val split
    if stats.output_files:
        console.print()
        if Confirm.ask("Create train/validation split?", default=True):
            val_ratio = float(Prompt.ask("Validation ratio", default="0.05"))
            
            # Find JSONL file
            jsonl_files = [f for f in stats.output_files if f.endswith('.jsonl')]
            if jsonl_files:
                split_paths = generator.create_train_val_split(
                    jsonl_files[0],
                    output_dir,
                    val_ratio=val_ratio
                )
                stats.output_files.extend(split_paths.values())
    
    # Success message
    console.print(Panel.fit(
        f"[bold green]✅ Pre-training Dataset Ready![/bold green]\n\n"
        f"Total samples: {stats.total_chunks - stats.filtered_chunks - stats.duplicate_chunks}\n"
        f"Estimated tokens: ~{stats.total_tokens_estimate:,}\n\n"
        f"[yellow]Next step:[/yellow] Use this with `saara pretrain` to train your model!",
        title="🎉 Success",
        border_style="green"
    ))
    
    return stats


if __name__ == "__main__":
    run_pretrain_dataset_wizard()
