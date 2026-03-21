"""
SAARA QuickAPI - AI-Powered Data Processing Pipeline
Dead simple document-to-dataset conversion in 2-3 lines.

This is a DATA PROCESSING tool, not a tokenization tool.
Pipeline: Documents → Extract → Label → Distill → Format

=== SIMPLE USAGE (2-3 lines) ===

    from saara import quickapi

    # Option 1: Full auto pipeline
    quickapi.setup("ollama")  # or "vllm"
    dataset = quickapi.pdf_to_dataset("doc.pdf", format="alpaca")

    # Option 2: Step by step
    quickapi.setup("vllm", model="mistral")
    data = quickapi.extract("document.pdf")
    labeled = quickapi.label(data)
    result = quickapi.convert(labeled, "sharegpt")

=== FORMAT SELECTION ===

| Use Case                      | Format       |
|-------------------------------|--------------|
| Domain adaptation (PDFs)      | alpaca       |
| Chatbot / assistant           | chatml       |
| Multi-turn with history       | sharegpt     |
| Base model continuation       | completion   |
| Quality alignment / RLHF      | dpo          |
| Function calling / tools      | chatml_tools |
"""

import os
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions (Clear Error Messages)
# ============================================================================

class SaaraError(Exception):
    """Base exception for SAARA errors."""
    pass


class SetupError(SaaraError):
    """Configuration/setup errors."""
    pass


class BackendError(SaaraError):
    """Backend (Ollama/vLLM) errors."""
    pass


class FileError(SaaraError):
    """File handling errors."""
    pass


class FormatError(SaaraError):
    """Format conversion errors."""
    pass


class ValidationError(SaaraError):
    """Input validation errors."""
    pass


# ============================================================================
# Constants
# ============================================================================

VALID_BACKENDS = ["ollama", "vllm", "auto"]
VALID_FORMATS = ["alpaca", "chatml", "sharegpt", "completion", "dpo", "chatml_tools"]
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".json", ".jsonl", ".csv"]


# ============================================================================
# Global Configuration
# ============================================================================

@dataclass
class QuickConfig:
    """Global SAARA quick configuration."""
    model: str = "mistral"
    backend: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 2048
    output_dir: str = "./saara_output"
    use_cache: bool = True
    device: str = "auto"
    verbose: bool = True
    system_prompt: str = ""
    initialized: bool = False
    backend_available: bool = False
    error_message: str = ""


_config = QuickConfig()
_inference_engine = None


# ============================================================================
# Validation Helpers
# ============================================================================

def _validate_file(filepath: str, allowed_types: List[str] = None) -> Path:
    """Validate file exists and has correct type."""
    if not filepath:
        raise FileError("File path cannot be empty")

    path = Path(filepath)

    if not path.exists():
        raise FileError(f"File not found: {filepath}\n  Check the path and try again.")

    if not path.is_file():
        raise FileError(f"Not a file: {filepath}\n  Provide a file path, not a directory.")

    if allowed_types:
        ext = path.suffix.lower()
        if ext not in allowed_types:
            raise FileError(
                f"Unsupported file type: {ext}\n"
                f"  Supported types: {', '.join(allowed_types)}"
            )

    return path


def _validate_format(format_name: str) -> str:
    """Validate output format name."""
    fmt = format_name.lower().strip()
    if fmt not in VALID_FORMATS:
        raise FormatError(
            f"Unknown format: '{format_name}'\n"
            f"  Valid formats: {', '.join(VALID_FORMATS)}\n"
            f"  Use quickapi.list_formats() to see all options."
        )
    return fmt


def _validate_backend(backend: str) -> str:
    """Validate backend name."""
    be = backend.lower().strip()
    if be not in VALID_BACKENDS:
        raise SetupError(
            f"Unknown backend: '{backend}'\n"
            f"  Valid backends: {', '.join(VALID_BACKENDS)}\n"
            f"  Recommended: 'ollama' (simple) or 'vllm' (fast)"
        )
    return be


def _check_ollama_available() -> tuple:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "Ollama is running"
        return False, "Ollama installed but not responding"
    except FileNotFoundError:
        return False, "Ollama not installed. Install from: https://ollama.ai"
    except subprocess.TimeoutExpired:
        return False, "Ollama not responding (timeout)"
    except Exception as e:
        return False, f"Ollama check failed: {e}"


def _check_vllm_available() -> tuple:
    """Check if vLLM is available."""
    try:
        import vllm
        return True, f"vLLM {vllm.__version__} available"
    except ImportError:
        return False, "vLLM not installed. Install with: pip install vllm"
    except Exception as e:
        return False, f"vLLM check failed: {e}"


def _ensure_setup():
    """Ensure setup() was called before using pipeline functions."""
    if not _config.initialized:
        raise SetupError(
            "SAARA not configured!\n"
            "  Call quickapi.setup() first:\n"
            "    >>> quickapi.setup('ollama')  # or 'vllm'\n"
            "    >>> quickapi.setup('ollama', model='llama3')"
        )

    if not _config.backend_available:
        raise BackendError(
            f"Backend '{_config.backend}' not available!\n"
            f"  Error: {_config.error_message}\n"
            f"  Try: quickapi.check_backends() to see available options"
        )


# ============================================================================
# Setup & Configuration
# ============================================================================

def check_backends() -> Dict[str, Any]:
    """
    Check which backends are available.

    Returns:
        Dict with status of each backend

    Example:
        >>> quickapi.check_backends()
        {'ollama': {'available': True, 'message': 'Ollama is running'},
         'vllm': {'available': False, 'message': 'vLLM not installed'}}
    """
    ollama_ok, ollama_msg = _check_ollama_available()
    vllm_ok, vllm_msg = _check_vllm_available()

    result = {
        "ollama": {"available": ollama_ok, "message": ollama_msg},
        "vllm": {"available": vllm_ok, "message": vllm_msg},
        "recommended": "ollama" if ollama_ok else ("vllm" if vllm_ok else None)
    }

    print("\nBackend Status:")
    print(f"  Ollama: {'Available' if ollama_ok else 'Not available'} - {ollama_msg}")
    print(f"  vLLM:   {'Available' if vllm_ok else 'Not available'} - {vllm_msg}")

    if result["recommended"]:
        print(f"\nRecommended: quickapi.setup('{result['recommended']}')")
    else:
        print("\nNo backends available! Install Ollama or vLLM first.")

    return result


def setup(
    backend: str = "ollama",
    model: str = "mistral",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    output_dir: str = "./saara_output",
    system_prompt: str = "",
    verbose: bool = True,
    auto_check: bool = True
) -> Dict[str, Any]:
    """
    Configure SAARA - call once at start.

    Args:
        backend: "ollama" (recommended), "vllm" (faster), or "auto"
        model: Model name (e.g., "mistral", "llama3", "granite3.1-dense:8b")
        temperature: 0-1, creativity level
        max_tokens: Maximum response length
        output_dir: Where to save results
        system_prompt: Default system prompt
        verbose: Print status messages
        auto_check: Auto-check backend availability

    Returns:
        Configuration summary

    Example:
        >>> quickapi.setup("ollama")
        >>> quickapi.setup("ollama", model="llama3")
        >>> quickapi.setup("vllm", model="mistral-7b")

    Raises:
        SetupError: If backend is invalid
        BackendError: If backend is not available
    """
    global _config, _inference_engine

    # Validate inputs
    backend = _validate_backend(backend)

    if not 0 <= temperature <= 1:
        raise ValidationError(f"Temperature must be 0-1, got {temperature}")

    if max_tokens < 1:
        raise ValidationError(f"max_tokens must be positive, got {max_tokens}")

    # Check backend availability
    backend_available = True
    error_message = ""

    if auto_check:
        if backend == "ollama" or backend == "auto":
            available, msg = _check_ollama_available()
            if available:
                backend = "ollama"
                backend_available = True
            elif backend == "auto":
                # Try vLLM as fallback
                available, msg = _check_vllm_available()
                if available:
                    backend = "vllm"
                    backend_available = True
                else:
                    backend_available = False
                    error_message = "No backends available. Install Ollama or vLLM."
            else:
                backend_available = False
                error_message = msg

        elif backend == "vllm":
            available, msg = _check_vllm_available()
            backend_available = available
            error_message = "" if available else msg

    # Create config
    _config = QuickConfig(
        model=model,
        backend=backend,
        temperature=temperature,
        max_tokens=max_tokens,
        output_dir=output_dir,
        verbose=verbose,
        device="auto",
        system_prompt=system_prompt,
        initialized=True,
        backend_available=backend_available,
        error_message=error_message
    )

    # Create output directory
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise SetupError(f"Cannot create output directory: {output_dir}\n  Check permissions.")
    except Exception as e:
        raise SetupError(f"Failed to create output directory: {e}")

    # Initialize inference engine if backend is available
    engine_name = "None"
    if backend_available:
        try:
            from .local_inference import LocalInferenceEngine, InferenceConfig
            inference_config = InferenceConfig(
                backend=backend,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_auto_fallback=True
            )
            _inference_engine = LocalInferenceEngine(inference_config)
            engine_name = _inference_engine.backend_name
        except Exception as e:
            _config.backend_available = False
            _config.error_message = str(e)
            engine_name = f"Error: {e}"

    config_summary = {
        "status": "Ready" if backend_available else "Backend unavailable",
        "backend": engine_name,
        "model": model,
        "output_dir": os.path.abspath(output_dir),
        "formats": VALID_FORMATS.copy()
    }

    if verbose:
        if backend_available:
            print(f"SAARA Ready | Backend: {engine_name} | Model: {model}")
        else:
            print(f"SAARA Warning | Backend not available: {error_message}")
            print(f"  Run quickapi.check_backends() to see options")

    return config_summary


# ============================================================================
# SIMPLE API (2-3 lines) - With Full Error Handling
# ============================================================================

def pdf_to_dataset(
    filepath: str,
    format: str = "alpaca",
    system_prompt: str = "",
    min_quality: float = 0.7,
    save_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Convert PDF to training dataset in one call.

    Args:
        filepath: Path to PDF file
        format: Output format (alpaca, chatml, sharegpt, completion, dpo)
        system_prompt: System prompt for chat formats
        min_quality: Quality threshold for filtering (0-1)
        save_intermediate: Save intermediate extraction/labeling files

    Returns:
        Dict with converted items and output file path

    Example:
        >>> quickapi.setup("ollama")
        >>> result = quickapi.pdf_to_dataset("manual.pdf", format="alpaca")
        >>> print(f"Created {result['total_items']} training samples")

    Raises:
        SetupError: If setup() not called
        FileError: If file not found or wrong type
        FormatError: If format is invalid
    """
    _ensure_setup()
    _validate_file(filepath, [".pdf"])
    format = _validate_format(format)

    if not 0 <= min_quality <= 1:
        raise ValidationError(f"min_quality must be 0-1, got {min_quality}")

    try:
        # Extract
        if _config.verbose:
            print(f"Extracting: {filepath}")
        extracted = dataExtract_PDF(filepath, save_output=save_intermediate)

        # Label
        if _config.verbose:
            print("Labeling with LLM...")
        labeled = dataLabel_Dataset(extracted, save_output=save_intermediate)

        # Distill
        if _config.verbose:
            print("Filtering quality...")
        distilled = dataDistill_Dataset(labeled, save_output=save_intermediate)

        # Convert to format
        result = dataConvert_Format(
            distilled,
            target_format=format,
            system_prompt=system_prompt or _config.system_prompt,
            save_output=True
        )

        if _config.verbose:
            print(f"Created {result['total_items']} samples in {format} format")
            if result.get('output_file'):
                print(f"Saved to: {result['output_file']}")

        return result

    except SaaraError:
        raise
    except Exception as e:
        raise SaaraError(f"Pipeline failed: {e}\n  Check your PDF file and try again.")


def text_to_dataset(
    filepath: str,
    format: str = "alpaca",
    system_prompt: str = "",
    min_quality: float = 0.7
) -> Dict[str, Any]:
    """
    Convert text file to training dataset.

    Args:
        filepath: Path to text file (.txt, .md, .json, .jsonl)
        format: Output format
        system_prompt: System prompt for chat formats
        min_quality: Quality threshold

    Example:
        >>> quickapi.setup("ollama")
        >>> result = quickapi.text_to_dataset("notes.txt", format="sharegpt")

    Raises:
        SetupError: If setup() not called
        FileError: If file not found
        FormatError: If format is invalid
    """
    _ensure_setup()
    _validate_file(filepath, [".txt", ".md", ".json", ".jsonl", ".csv"])
    format = _validate_format(format)

    try:
        extracted = dataExtract_Text(filepath, save_output=False)
        labeled = dataLabel_Dataset(extracted, save_output=False)
        distilled = dataDistill_Dataset(labeled, save_output=False)
        return dataConvert_Format(distilled, target_format=format, system_prompt=system_prompt)
    except SaaraError:
        raise
    except Exception as e:
        raise SaaraError(f"Pipeline failed: {e}")


def convert_existing(
    filepath: str,
    target_format: str = "alpaca",
    system_prompt: str = ""
) -> Dict[str, Any]:
    """
    Convert existing JSONL dataset to different format (no LLM needed).

    This is useful when you already have labeled data and just need
    to change the format.

    Args:
        filepath: Path to JSONL file with instruction/response pairs
        target_format: Target format
        system_prompt: System prompt for chat formats

    Example:
        >>> # No setup needed for format conversion only!
        >>> result = quickapi.convert_existing("data.jsonl", "sharegpt")

    Returns:
        Dict with converted items and output path
    """
    _validate_file(filepath, [".jsonl", ".json"])
    target_format = _validate_format(target_format)

    # Load data
    items = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    except json.JSONDecodeError as e:
        raise FileError(f"Invalid JSON in file: {e}")
    except UnicodeDecodeError:
        raise FileError("File encoding error. Ensure file is UTF-8 encoded.")

    if not items:
        raise ValidationError("File is empty or contains no valid JSON lines")

    # Convert using formats module
    from .formats import convert_dataset

    converted = convert_dataset(items, target_format, system_prompt=system_prompt)

    # Save output
    output_dir = Path(_config.output_dir if _config.initialized else "./saara_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"converted_{target_format}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} items to {target_format} format")
    print(f"Saved to: {output_file}")

    return {
        "format": target_format,
        "total_items": len(converted),
        "items": converted,
        "output_file": str(output_file)
    }


# ============================================================================
# Aliases for Simple API
# ============================================================================

def extract(filepath: str, **kwargs) -> Dict[str, Any]:
    """
    Extract text from PDF or text file.

    Args:
        filepath: Path to file

    Returns:
        Dict with extracted text and metadata

    Example:
        >>> data = quickapi.extract("document.pdf")
        >>> print(data['text'][:100])
    """
    _ensure_setup()
    path = _validate_file(filepath, SUPPORTED_FILE_TYPES)

    if path.suffix.lower() == ".pdf":
        return dataExtract_PDF(filepath, **kwargs)
    return dataExtract_Text(filepath, **kwargs)


def label(data: Union[Dict, str], **kwargs) -> Dict[str, Any]:
    """
    Label data using LLM to generate instruction/response pairs.

    Args:
        data: Extracted data dict or path to JSON file

    Returns:
        Dict with labeled items

    Example:
        >>> data = quickapi.extract("doc.pdf")
        >>> labeled = quickapi.label(data)
    """
    _ensure_setup()
    return dataLabel_Dataset(data, **kwargs)


def distill(data: Union[Dict, str], **kwargs) -> Dict[str, Any]:
    """
    Filter and clean data by removing duplicates and low-quality items.

    Args:
        data: Labeled data dict or path to JSONL file

    Returns:
        Dict with distilled items

    Example:
        >>> labeled = quickapi.label(data)
        >>> clean = quickapi.distill(labeled)
    """
    return dataDistill_Dataset(data, **kwargs)


def convert(
    data: Union[Dict, str, List],
    format: str = "alpaca",
    **kwargs
) -> Dict[str, Any]:
    """
    Convert data to training format.

    Args:
        data: Data dict, list, or path to JSONL file
        format: Target format (alpaca, chatml, sharegpt, completion, dpo)

    Returns:
        Dict with converted items

    Example:
        >>> clean = quickapi.distill(labeled)
        >>> result = quickapi.convert(clean, "sharegpt")
    """
    format = _validate_format(format)
    return dataConvert_Format(data, target_format=format, **kwargs)


# ============================================================================
# Feature 1: PDF/Document Extraction
# ============================================================================

def dataExtract_PDF(
    filename: str,
    use_ocr: bool = True,
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Extract text from PDF with OCR support.

    Args:
        filename: Path to PDF file
        use_ocr: Use vision models for complex layouts
        save_output: Save extracted data to JSON

    Returns:
        Dict with extracted text and metadata

    Raises:
        FileError: If file not found or not a PDF
    """
    path = _validate_file(filename, [".pdf"])

    try:
        from .pdf_extractor import PDFExtractor

        extractor = PDFExtractor(
            use_ocr=use_ocr,
            ocr_model="qwen"
        )

        result = extractor.extract_from_pdf(str(path))

        output_data = {
            "filename": str(path),
            "total_pages": result.get("num_pages", 0),
            "text": result.get("text", ""),
            "structured_content": result.get("blocks", []),
            "metadata": {
                "use_ocr": use_ocr,
                "extraction_method": "qwen" if use_ocr else "pymupdf",
                "extracted_at": datetime.now().isoformat()
            },
            "output_file": None
        }

        if not output_data["text"].strip():
            raise FileError(
                f"No text extracted from PDF: {filename}\n"
                "  The PDF may be scanned/image-based. Try with use_ocr=True"
            )

        if save_output and _config.initialized:
            output_file = Path(_config.output_dir) / f"{path.stem}_extracted.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            output_data["output_file"] = str(output_file)

            if _config.verbose:
                print(f"Extracted {output_data['total_pages']} pages from {path.name}")

        return output_data

    except ImportError:
        raise SetupError(
            "PDF extraction requires PyMuPDF.\n"
            "  Install with: pip install pymupdf"
        )
    except SaaraError:
        raise
    except Exception as e:
        raise FileError(f"PDF extraction failed: {e}")


def dataExtract_Text(
    filename: str,
    encoding: str = "utf-8",
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Extract text from plain text files.

    Args:
        filename: Path to text file
        encoding: File encoding (default: utf-8)
        save_output: Save extracted data

    Returns:
        Dict with extracted text
    """
    path = _validate_file(filename, [".txt", ".md", ".json", ".jsonl", ".csv"])

    try:
        with open(path, "r", encoding=encoding) as f:
            text = f.read()

        if not text.strip():
            raise FileError(f"File is empty: {filename}")

        output_data = {
            "filename": str(path),
            "text": text,
            "char_count": len(text),
            "line_count": text.count('\n') + 1,
            "output_file": None
        }

        if save_output and _config.initialized:
            output_file = Path(_config.output_dir) / f"{path.stem}_extracted.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            output_data["output_file"] = str(output_file)

        return output_data

    except UnicodeDecodeError:
        raise FileError(
            f"Cannot read file with {encoding} encoding: {filename}\n"
            f"  Try: dataExtract_Text('{filename}', encoding='latin-1')"
        )
    except SaaraError:
        raise
    except Exception as e:
        raise FileError(f"Text extraction failed: {e}")


# ============================================================================
# Feature 2: Dataset Labeling (AI-Powered)
# ============================================================================

def dataLabel_Dataset(
    content: Union[Dict, str],
    label_types: List[str] = None,
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Label content using LLM to generate instruction/response pairs.

    Args:
        content: Extracted data dict or path to JSON file
        label_types: Types of labels ["qa", "instruction"]
        save_output: Save labeled data

    Returns:
        Dict with labeled items

    Raises:
        BackendError: If LLM backend not available
    """
    _ensure_setup()

    if label_types is None:
        label_types = ["qa", "instruction"]

    if not _inference_engine:
        raise BackendError(
            "No inference engine available.\n"
            f"  Backend: {_config.backend}\n"
            f"  Error: {_config.error_message}"
        )

    try:
        # Extract text if dict is passed
        if isinstance(content, dict):
            text = content.get("text", "")
        elif isinstance(content, str) and Path(content).exists():
            with open(content, "r") as f:
                data = json.load(f)
            text = data.get("text", "")
        else:
            text = str(content)

        if not text.strip():
            raise ValidationError("No text content to label")

        chunks = _chunk_text(text, chunk_size=512)
        labeled_items = []

        system_prompt = """You are a data labeling expert. Generate high-quality instruction-response pairs
from the provided text. Return ONLY valid JSON with fields: instruction, response"""

        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            if _config.verbose:
                progress = (i + 1) / total_chunks * 100
                print(f"\r  Labeling: {progress:.0f}% ({i+1}/{total_chunks})", end="", flush=True)

            for label_type in label_types:
                if label_type == "qa":
                    prompt = f"""Generate a factual question-answer pair from this text:
TEXT: {chunk[:500]}

Return JSON: {{"instruction": "question here", "response": "answer here"}}"""
                elif label_type == "instruction":
                    prompt = f"""Generate an instruction-following pair from this text:
TEXT: {chunk[:500]}

Return JSON: {{"instruction": "instruction here", "response": "response here"}}"""
                else:
                    continue

                try:
                    response = _inference_engine.generate(prompt, system_prompt=system_prompt)

                    # Parse JSON from response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        data = json.loads(json_str)

                        if data.get("instruction") and data.get("response"):
                            labeled_items.append({
                                "text": chunk[:300],
                                "instruction": data["instruction"],
                                "response": data["response"],
                                "label_type": label_type
                            })
                except (json.JSONDecodeError, KeyError):
                    pass  # Skip malformed responses
                except Exception:
                    pass

        if _config.verbose:
            print()  # New line after progress

        if not labeled_items:
            raise SaaraError(
                "No labeled items generated.\n"
                "  The LLM may not be responding correctly.\n"
                "  Try a different model or check your content."
            )

        output_data = {
            "total_chunks": len(chunks),
            "labeled_items": labeled_items,
            "label_types": label_types,
            "output_file": None
        }

        if save_output:
            output_file = Path(_config.output_dir) / "labeled_dataset.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in labeled_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_data["output_file"] = str(output_file)

            if _config.verbose:
                print(f"  Generated {len(labeled_items)} labeled items")

        return output_data

    except SaaraError:
        raise
    except Exception as e:
        raise SaaraError(f"Labeling failed: {e}")


# ============================================================================
# Feature 3: Data Distillation (Quality Filtering)
# ============================================================================

def dataDistill_Dataset(
    labeled_data: Union[Dict, str],
    min_instruction_len: int = 10,
    min_response_len: int = 20,
    remove_duplicates: bool = True,
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Clean and distill dataset by removing low-quality items and duplicates.

    Args:
        labeled_data: Labeled data dict or path to JSONL file
        min_instruction_len: Minimum instruction length
        min_response_len: Minimum response length
        remove_duplicates: Remove duplicate entries
        save_output: Save distilled data

    Returns:
        Dict with distilled items
    """
    try:
        # Load data
        if isinstance(labeled_data, str):
            path = _validate_file(labeled_data, [".jsonl", ".json"])
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        else:
            items = labeled_data.get("labeled_items", [])

        if not items:
            raise ValidationError("No items to distill")

        original_count = len(items)

        # Remove duplicates
        seen = set()
        unique_items = []
        for item in items:
            key = f"{item.get('instruction', '')}|{item.get('response', '')}"
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        duplicates_removed = original_count - len(unique_items)

        # Quality filtering
        distilled = []
        for item in unique_items:
            instruction_len = len(item.get("instruction", ""))
            response_len = len(item.get("response", ""))

            if instruction_len >= min_instruction_len and response_len >= min_response_len:
                distilled.append(item)

        low_quality_removed = len(unique_items) - len(distilled)

        output_data = {
            "original_count": original_count,
            "distilled_count": len(distilled),
            "removed_duplicates": duplicates_removed,
            "removed_low_quality": low_quality_removed,
            "items": distilled,
            "output_file": None
        }

        if save_output and _config.initialized:
            output_file = Path(_config.output_dir) / "distilled_dataset.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in distilled:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_data["output_file"] = str(output_file)

            if _config.verbose:
                print(f"  Distilled: {original_count} -> {len(distilled)} items")
                if duplicates_removed or low_quality_removed:
                    print(f"  Removed: {duplicates_removed} duplicates, {low_quality_removed} low-quality")

        return output_data

    except SaaraError:
        raise
    except Exception as e:
        raise SaaraError(f"Distillation failed: {e}")


# ============================================================================
# Feature 4: Format Conversion
# ============================================================================

def dataConvert_Format(
    dataset: Union[Dict, str, List[Dict]],
    target_format: str = "alpaca",
    system_prompt: str = None,
    tool_schemas: List[Dict] = None,
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Convert dataset to standard training formats.

    Supported Formats:
    - alpaca: {"instruction", "input", "output"} - Domain adaptation
    - chatml: {"messages": [...]} - Chatbots/assistants
    - sharegpt: {"conversations": [...]} - Multi-turn
    - completion: {"text": "..."} - Base model
    - dpo: {"prompt", "chosen", "rejected"} - RLHF
    - chatml_tools: {"messages", "tools"} - Function calling

    Args:
        dataset: Data items (dict with 'items', list, or JSONL file path)
        target_format: Target format name
        system_prompt: Optional system prompt for chat formats
        tool_schemas: Optional tool definitions for chatml_tools
        save_output: Save formatted data

    Returns:
        Dict with converted items and output file path
    """
    target_format = _validate_format(target_format)

    try:
        from .formats import convert_dataset

        # Load data
        if isinstance(dataset, str):
            path = _validate_file(dataset, [".jsonl", ".json"])
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        elif isinstance(dataset, list):
            items = dataset
        else:
            items = dataset.get("items", []) or dataset.get("labeled_items", [])

        if not items:
            raise ValidationError("No items to convert")

        # Use config system prompt if not provided
        sys_prompt = system_prompt if system_prompt is not None else _config.system_prompt

        # Convert using formats module
        converted = convert_dataset(
            items,
            target_format,
            system_prompt=sys_prompt,
            tool_schemas=tool_schemas
        )

        output_data = {
            "format": target_format,
            "total_items": len(converted),
            "items": converted,
            "output_file": None
        }

        if save_output and _config.initialized:
            output_file = Path(_config.output_dir) / f"dataset_{target_format}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in converted:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_data["output_file"] = str(output_file)

            if _config.verbose:
                print(f"  Converted to {target_format}: {len(converted)} items")

        return output_data

    except SaaraError:
        raise
    except Exception as e:
        raise FormatError(f"Format conversion failed: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def get_format_recommendation(use_case: str) -> str:
    """
    Get recommended format for a use case.

    Args:
        use_case: Description like "chatbot", "domain", "rlhf", etc.

    Returns:
        Recommended format name

    Example:
        >>> quickapi.get_format_recommendation("chatbot")
        'chatml'
    """
    from .formats import FormatRegistry
    return FormatRegistry.get_recommendation(use_case)


def list_formats() -> List[str]:
    """List all available output formats."""
    return VALID_FORMATS.copy()


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def get_status() -> Dict[str, Any]:
    """
    Get current configuration and status.

    Returns:
        Dict with configuration details
    """
    return {
        "initialized": _config.initialized,
        "backend": _config.backend,
        "backend_available": _config.backend_available,
        "model": _config.model,
        "output_dir": _config.output_dir,
        "formats": list_formats(),
        "error": _config.error_message if not _config.backend_available else None
    }


def reset():
    """Reset all configuration."""
    global _config, _inference_engine
    _config = QuickConfig()
    _inference_engine = None
    print("SAARA reset complete")


# ============================================================================
# Full Pipeline Helper
# ============================================================================

def process_document(
    filepath: str,
    output_format: str = "alpaca",
    system_prompt: str = "",
    label_types: List[str] = None,
    save_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Full pipeline: Extract -> Label -> Distill -> Format

    Args:
        filepath: Path to document (PDF or text file)
        output_format: Target format (alpaca, chatml, sharegpt, completion, dpo)
        system_prompt: System prompt for chat formats
        label_types: Label types to generate
        save_intermediate: Save intermediate files

    Returns:
        Final formatted dataset
    """
    _ensure_setup()
    path = _validate_file(filepath, SUPPORTED_FILE_TYPES)
    output_format = _validate_format(output_format)

    # Determine file type and extract
    if path.suffix.lower() == ".pdf":
        extracted = dataExtract_PDF(filepath, save_output=save_intermediate)
    else:
        extracted = dataExtract_Text(filepath, save_output=save_intermediate)

    # Label
    labeled = dataLabel_Dataset(extracted, label_types=label_types, save_output=save_intermediate)

    # Distill
    distilled = dataDistill_Dataset(labeled, save_output=save_intermediate)

    # Format
    formatted = dataConvert_Format(
        distilled,
        target_format=output_format,
        system_prompt=system_prompt or _config.system_prompt,
        save_output=True
    )

    return formatted
