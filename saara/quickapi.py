"""
SAARA QuickAPI - Dead Simple End-to-End Pipeline
One import, then use simple functions. Everything pre-configured and working.

Usage:
    from saara import quickapi
    
    # Configure once
    quickapi.setup(model="mistral", backend="vllm")
    
    # Use simple functions
    pdf_data = quickapi.dataExtract_PDF("document.pdf")
    dataset = quickapi.dataLabel_Dataset(pdf_data)
    clean_data = quickapi.dataDistill_Dataset(dataset)
    tokens = quickapi.dataTokenize_Dataset(clean_data)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import tempfile

logger = logging.getLogger(__name__)


# ============================================================================
# Global Configuration
# ============================================================================

@dataclass
class QuickConfig:
    """Global SAARA quick configuration."""
    model: str = "mistral"
    backend: str = "auto"  # auto, vllm, ollama
    temperature: float = 0.7
    max_tokens: int = 2048
    output_dir: str = "./saara_output"
    tokenizer: str = "auto"  # auto, bpe, sentencepiece
    use_cache: bool = True
    device: str = "auto"  # auto, cuda, cpu
    verbose: bool = True


_config = QuickConfig()
_inference_engine = None
_tokenizer = None


def setup(
    model: str = "mistral",
    backend: str = "auto",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    output_dir: str = "./saara_output",
    tokenizer: str = "auto",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Configure SAARA QuickAPI - call this once at the start.
    
    Args:
        model: Model name (e.g., "mistral", "neural-chat", "llama2")
        backend: "auto" (smart select), "vllm" (fast), or "ollama" (simple)
        temperature: 0-1, where 0=deterministic, 1=creative
        max_tokens: Maximum response length
        output_dir: Where to save results
        tokenizer: "auto" or specific tokenizer name
        verbose: Print debug info
        
    Returns:
        Configuration summary
    """
    global _config, _inference_engine, _tokenizer
    
    _config = QuickConfig(
        model=model,
        backend=backend,
        temperature=temperature,
        max_tokens=max_tokens,
        output_dir=output_dir,
        tokenizer=tokenizer,
        verbose=verbose,
        device="auto"
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    try:
        from saara.local_inference import LocalInferenceEngine, InferenceConfig
        inference_config = InferenceConfig(
            backend=backend if backend != "auto" else None,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_auto_fallback=True
        )
        _inference_engine = LocalInferenceEngine(inference_config)
        engine_name = _inference_engine.backend_name
    except Exception as e:
        logger.warning(f"Inference engine init failed: {e}")
        engine_name = "None (API required)"
    
    # Initialize tokenizer
    try:
        _tokenizer = _init_tokenizer(tokenizer)
        tokenizer_name = type(_tokenizer).__name__
    except Exception as e:
        logger.warning(f"Tokenizer init failed: {e}")
        tokenizer_name = "None"
    
    config_summary = {
        "status": "✅ Ready",
        "model": model,
        "inference_backend": engine_name,
        "tokenizer": tokenizer_name,
        "output_directory": os.path.abspath(output_dir),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if verbose:
        print("\n🚀 SAARA QuickAPI Configuration:")
        print("=" * 50)
        for key, val in config_summary.items():
            print(f"  {key}: {val}")
        print("=" * 50 + "\n")
    
    return config_summary


def _init_tokenizer(tokenizer_name: str = "auto"):
    """Initialize tokenizer."""
    if tokenizer_name == "auto":
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(_config.model)
        except:
            return None
    else:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name)


# ============================================================================
# Feature 1: PDF Extraction
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
        {
            "filename": "document.pdf",
            "total_pages": 10,
            "text": "Extracted text...",
            "structured_content": [...],
            "metadata": {...},
            "output_file": "path/to/extracted.json"
        }
    """
    try:
        from saara.pdf_extractor import PDFExtractor
        
        extractor = PDFExtractor(
            use_ocr=use_ocr,
            ocr_model="qwen"  # or "moondream"
        )
        
        result = extractor.extract_from_pdf(filename)
        
        output_data = {
            "filename": str(filename),
            "total_pages": result.get("num_pages", 0),
            "text": result.get("text", ""),
            "structured_content": result.get("blocks", []),
            "metadata": {
                "use_ocr": use_ocr,
                "extraction_method": "qwen" if use_ocr else "pymupdf"
            },
            "output_file": None
        }
        
        if save_output:
            output_file = Path(_config.output_dir) / f"{Path(filename).stem}_extracted.json"
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            output_data["output_file"] = str(output_file)
            
            if _config.verbose:
                print(f"✅ PDF Extracted: {output_file}")
        
        return output_data
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise


# ============================================================================
# Feature 2: Dataset Labeling (AI-Powered)
# ============================================================================

def dataLabel_Dataset(
    pdf_content: Union[Dict, str],
    label_types: List[str] = None,
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Label PDF content using vLLM/Ollama to generate Q&A pairs.
    
    Args:
        pdf_content: Output from dataExtract_PDF or text string
        label_types: Types of labels to generate
                    ["qa", "summarization", "classification"]
        save_output: Save labeled data
        
    Returns:
        {
            "total_chunks": 50,
            "labeled_items": [
                {
                    "text": "Original chunk...",
                    "instruction": "What is...",
                    "response": "It is...",
                    "label_type": "qa"
                },
                ...
            ],
            "output_file": "path/to/labeled.jsonl"
        }
    """
    if label_types is None:
        label_types = ["qa", "summarization"]
    
    if not _inference_engine:
        raise RuntimeError("Inference engine not initialized. Call setup() first.")
    
    try:
        # Extract text if dict is passed
        if isinstance(pdf_content, dict):
            text = pdf_content.get("text", "")
            chunks = _chunk_text(text, chunk_size=512)
        else:
            chunks = _chunk_text(pdf_content, chunk_size=512)
        
        labeled_items = []
        
        system_prompt = """You are a data labeling expert. Generate high-quality instruction-response pairs 
        from the provided text. Return ONLY valid JSON array with fields: instruction, response, label_type"""
        
        for i, chunk in enumerate(chunks):
            if _config.verbose and i % 5 == 0:
                print(f"  Labeling chunk {i+1}/{len(chunks)}...", end="\r")
            
            for label_type in label_types:
                prompt = f"""Generate a {label_type} pair from this text:
                TEXT: {chunk[:200]}...
                
                Return JSON object with: instruction (string), response (string)"""
                
                try:
                    response = _inference_engine.generate(prompt, system_prompt=system_prompt)
                    
                    # Parse JSON from response
                    import json
                    json_str = response[response.find("{"):response.rfind("}")+1]
                    data = json.loads(json_str)
                    
                    labeled_items.append({
                        "text": chunk[:300],
                        "instruction": data.get("instruction", ""),
                        "response": data.get("response", ""),
                        "label_type": label_type
                    })
                except:
                    pass  # Skip malformed responses
        
        output_data = {
            "total_chunks": len(chunks),
            "labeled_items": labeled_items,
            "label_types": label_types,
            "output_file": None
        }
        
        if save_output:
            output_file = Path(_config.output_dir) / "labeled_dataset.jsonl"
            with open(output_file, "w") as f:
                for item in labeled_items:
                    f.write(json.dumps(item) + "\n")
            output_data["output_file"] = str(output_file)
            
            if _config.verbose:
                print(f"✅ Dataset Labeled: {len(labeled_items)} items → {output_file}")
        
        return output_data
        
    except Exception as e:
        logger.error(f"Dataset labeling failed: {e}")
        raise


# ============================================================================
# Feature 3: Data Distillation (Quality Filtering)
# ============================================================================

def dataDistill_Dataset(
    labeled_data: Union[Dict, str],
    min_quality: float = 0.7,
    remove_duplicates: bool = True,
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Clean and distill dataset by removing low-quality items and duplicates.
    
    Args:
        labeled_data: Output from dataLabel_Dataset or path to JSONL file
        min_quality: 0-1, quality threshold for filtering
        remove_duplicates: Remove duplicate entries
        save_output: Save distilled data
        
    Returns:
        {
            "original_count": 100,
            "distilled_count": 85,
            "removed_duplicates": 10,
            "quality_score": 0.92,
            "items": [...],
            "output_file": "path/to/distilled.jsonl"
        }
    """
    try:
        # Load data
        if isinstance(labeled_data, str):
            items = []
            with open(labeled_data, "r") as f:
                for line in f:
                    items.append(json.loads(line))
        else:
            items = labeled_data.get("labeled_items", [])
        
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
            # Simple quality heuristic
            instruction_len = len(item.get("instruction", ""))
            response_len = len(item.get("response", ""))
            
            if instruction_len > 10 and response_len > 20:  # Basic length check
                distilled.append(item)
        
        distilled_count = len(distilled)
        quality_score = distilled_count / max(original_count, 1)
        
        output_data = {
            "original_count": original_count,
            "distilled_count": distilled_count,
            "removed_duplicates": duplicates_removed,
            "removed_low_quality": len(unique_items) - distilled_count,
            "quality_score": quality_score,
            "items": distilled,
            "output_file": None
        }
        
        if save_output:
            output_file = Path(_config.output_dir) / "distilled_dataset.jsonl"
            with open(output_file, "w") as f:
                for item in distilled:
                    f.write(json.dumps(item) + "\n")
            output_data["output_file"] = str(output_file)
            
            if _config.verbose:
                print(f"✅ Dataset Distilled: {original_count} → {distilled_count} items")
                print(f"   Removed: {duplicates_removed} duplicates, {len(unique_items) - distilled_count} low-quality")
                print(f"   Quality Score: {quality_score:.1%}")
        
        return output_data
        
    except Exception as e:
        logger.error(f"Dataset distillation failed: {e}")
        raise


# ============================================================================
# Feature 4: Tokenization
# ============================================================================

def dataTokenize_Dataset(
    cleaned_data: Union[Dict, str],
    max_length: int = 1024,
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Tokenize dataset using configured tokenizer.
    Pre-formats for training (instruction + response concatenation).
    
    Args:
        cleaned_data: Output from dataDistill_Dataset or JSONL file path
        max_length: Maximum token sequence length
        save_output: Save tokenized data
        
    Returns:
        {
            "total_sequences": 100,
            "total_tokens": 102400,
            "avg_length": 1024,
            "tokenizer_name": "mistral",
            "tokens": [(ids), (ids), ...],
            "output_file": "path/to/tokens.arrow"
        }
    """
    try:
        # Load data
        if isinstance(cleaned_data, str):
            items = []
            with open(cleaned_data, "r") as f:
                for line in f:
                    items.append(json.loads(line))
        else:
            items = cleaned_data.get("items", [])
        
        if not _tokenizer:
            raise RuntimeError("Tokenizer not initialized. Call setup() first.")
        
        all_tokens = []
        total_tokens = 0
        
        for i, item in enumerate(items):
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            
            # Combine instruction + response with special tokens
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            
            # Tokenize
            tokens = _tokenizer.encode(text)[:max_length]
            all_tokens.append(tokens)
            total_tokens += len(tokens)
            
            if _config.verbose and i % 20 == 0:
                print(f"  Tokenizing item {i+1}/{len(items)}...", end="\r")
        
        avg_length = total_tokens / len(all_tokens) if all_tokens else 0
        
        output_data = {
            "total_sequences": len(all_tokens),
            "total_tokens": total_tokens,
            "avg_length": int(avg_length),
            "max_length": max_length,
            "tokenizer_name": _config.tokenizer or "auto-selected",
            "tokens": all_tokens,
            "output_file": None
        }
        
        if save_output:
            # Save as JSONL (can be converted to .arrow for memory mapping)
            output_file = Path(_config.output_dir) / "tokenized_dataset.jsonl"
            with open(output_file, "w") as f:
                for tokens in all_tokens:
                    f.write(json.dumps({"input_ids": tokens}) + "\n")
            output_data["output_file"] = str(output_file)
            
            if _config.verbose:
                print(f"✅ Dataset Tokenized: {len(all_tokens)} sequences, {total_tokens:,} total tokens")
                print(f"   Saved to: {output_file}")
        
        return output_data
        
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise


# ============================================================================
# Feature 5: Format Conversion (ShareGPT, Alpaca)
# ============================================================================

def dataConvert_Format(
    dataset: Union[Dict, str],
    target_format: str = "sharegpt",
    save_output: bool = True
) -> Dict[str, Any]:
    """
    Convert dataset to standard training formats.
    
    Args:
        dataset: Output from any previous step
        target_format: "sharegpt" (default), "alpaca", "openai"
        save_output: Save formatted data
        
    Returns:
        {
            "format": "sharegpt",
            "total_items": 100,
            "items": [...],
            "output_file": "path/to/formatted.jsonl"
        }
    """
    try:
        # Load data
        if isinstance(dataset, str):
            items = []
            with open(dataset, "r") as f:
                for line in f:
                    items.append(json.loads(line))
        else:
            items = dataset.get("items", []) or dataset.get("labeled_items", [])
        
        formatted = []
        
        for item in items:
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            
            if target_format == "sharegpt":
                formatted_item = {
                    "conversations": [
                        {"from": "user", "value": instruction},
                        {"from": "assistant", "value": response}
                    ]
                }
            elif target_format == "alpaca":
                formatted_item = {
                    "instruction": instruction,
                    "input": "",
                    "output": response
                }
            elif target_format == "openai":
                formatted_item = {
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response}
                    ]
                }
            else:
                formatted_item = item
            
            formatted.append(formatted_item)
        
        output_data = {
            "format": target_format,
            "total_items": len(formatted),
            "items": formatted,
            "output_file": None
        }
        
        if save_output:
            output_file = Path(_config.output_dir) / f"dataset_{target_format}.jsonl"
            with open(output_file, "w") as f:
                for item in formatted:
                    f.write(json.dumps(item) + "\n")
            output_data["output_file"] = str(output_file)
            
            if _config.verbose:
                print(f"✅ Dataset Converted: {target_format.upper()} format → {output_file}")
        
        return output_data
        
    except Exception as e:
        logger.error(f"Format conversion failed: {e}")
        raise


# ============================================================================
# Utility Functions
# ============================================================================

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
    """Get current configuration and status."""
    return {
        "configured": _config is not None,
        "model": _config.model,
        "backend": "✅ " + (_inference_engine.backend_name if _inference_engine else "❌ None"),
        "tokenizer": "✅ " + (type(_tokenizer).__name__ if _tokenizer else "❌ None"),
        "output_dir": _config.output_dir,
        "verbose": _config.verbose
    }


def reset():
    """Reset all configuration."""
    global _config, _inference_engine, _tokenizer
    _config = QuickConfig()
    _inference_engine = None
    _tokenizer = None
    if _config.verbose:
        print("✅ SAARA QuickAPI reset")
