# SAARA QuickAPI - Complete Reference

> **Complete function signatures, parameters, return values, and error handling**

---

## 📑 Table of Contents

1. [Core Functions](#core-functions)
2. [Parameter Guide](#parameter-guide)
3. [Return Value Formats](#return-value-formats)
4. [Error Handling](#error-handling)
5. [Configuration Reference](#configuration-reference)
6. [Advanced Usage](#advanced-usage)

---

## Core Functions

### `setup(backend="ollama", model="mistral", **kwargs)`

Initialize QuickAPI with a backend and language model.

#### **Signature**
```python
def setup(
    backend: str = "ollama",
    model: str = "mistral",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    output_dir: str = "./saara_output",
    system_prompt: str = "",
    verbose: bool = True,
    auto_check: bool = True
) -> Dict[str, Any]
```

#### **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"ollama"` | Inference backend: `"ollama"` (recommended), `"vllm"` (faster), `"auto"` |
| `model` | `str` | `"mistral"` | Model name, e.g. `"mistral"`, `"llama3"`, `"granite3.1-dense:8b"` |
| `temperature` | `float` | `0.7` | Creativity level (0.0-1.0). Higher = more creative |
| `max_tokens` | `int` | `2048` | Maximum response length |
| `output_dir` | `str` | `"./saara_output"` | Where to save results |
| `system_prompt` | `str` | `""` | Default system prompt |
| `verbose` | `bool` | `True` | Print status messages |
| `auto_check` | `bool` | `True` | Auto-check backend availability on setup |

#### **Returns**

A `dict` summarising the configuration: `{"status", "backend", "model", "output_dir", "formats"}`.

#### **Examples**

```python
# Minimal setup — uses Ollama with mistral
quickapi.setup("ollama")

# Specify model
quickapi.setup("ollama", model="granite3.1-dense:8b")

# GPU-accelerated with vLLM
quickapi.setup("vllm", model="mistral")

# Auto-select best available backend
quickapi.setup("auto", model="mistral", temperature=0.5)
```

#### **Raises**

| Exception | When |
|-----------|------|
| `SetupError` | Invalid backend name or cannot create output directory |
| `BackendError` | Selected backend is not available (e.g. Ollama not running) |
| `ValidationError` | temperature out of 0-1 range, or max_tokens < 1 |

#### **Usage Notes**
- 📍 Must call before `pdf_to_dataset()`, `extract()`, `label()`, etc.
- ✅ Safe to call multiple times (re-configures in place)
- 🔍 Run `quickapi.check_backends()` first to see what is available

---

### `dataExtract_PDF(file_path, use_ocr=True, extract_tables=True, max_pages=None)`

Extract text, tables, and images from PDF documents.

#### **Signature**
```python
def dataExtract_PDF(
    file_path: str,
    use_ocr: bool = True,
    extract_tables: bool = True,
    max_pages: int = None,
    include_metadata: bool = True
) -> dict
```

#### **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | **required** | Path to PDF file |
| `use_ocr` | `bool` | `True` | Enable OCR for scanned pages |
| `extract_tables` | `bool` | `True` | Extract structured tables |
| `max_pages` | `int` | `None` | Limit pages (None = all) |
| `include_metadata` | `bool` | `True` | Include document metadata |

#### **Returns**

```python
{
    "success": bool,
    "file_path": str,
    "pages": [
        {
            "number": int,
            "text": str,
            "confidence": float,      # 0.0-1.0 OCR confidence
            "has_images": bool
        },
        ...
    ],
    "tables": [
        {
            "page": int,
            "content": list,           # Structured table data
            "format": str             # "csv" or "markdown"
        },
        ...
    ],
    "images": [
        {
            "page": int,
            "description": str,        # AI-generated description
            "format": str             # "png", "jpg", etc
        },
        ...
    ],
    "metadata": {
        "total_pages": int,
        "title": str,
        "author": str,
        "creation_date": str,
        "processing_time": float
    }
}
```

#### **Examples**

```python
# Basic extraction
extracted = quickapi.dataExtract_PDF("paper.pdf")

# Scanned PDF (has OCR)
extracted = quickapi.dataExtract_PDF(
    "scanned_document.pdf",
    use_ocr=True
)

# First 10 pages only
extracted = quickapi.dataExtract_PDF(
    "large_file.pdf",
    max_pages=10
)

# Access results
print(f"Pages: {len(extracted['pages'])}")
print(f"Tables: {len(extracted['tables'])}")
for page in extracted['pages'][:3]:
    print(f"Page {page['number']}: {page['text'][:100]}...")
```

#### **Raises**

| Exception | When |
|-----------|------|
| `FileNotFoundError` | PDF file doesn't exist |
| `ValueError` | File is not a valid PDF |
| `PermissionError` | PDF is encrypted/password-protected |
| `RuntimeError` | OCR fails on corrupted pages |

#### **Performance**

| Document Size | Time | Notes |
|---------------|------|-------|
| 1 page | ~1 sec | Instant |
| 10 pages | ~5 sec | With OCR |
| 100 pages | ~30 sec | Scanned docs slower |
| 1000 pages | ~5 min | Process in batches |

---

### `dataLabel_Dataset(extracted_data, labels_per_chunk=1, quality_threshold=0.6)`

Generate synthetic question-answer pairs from extracted content.

#### **Signature**
```python
def dataLabel_Dataset(
    extracted_data: dict,
    labels_per_chunk: int = 1,
    quality_threshold: float = 0.6,
    max_samples: int = None,
    prompt_template: str = None
) -> dict
```

#### **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extracted_data` | `dict` | **required** | Output from `dataExtract_PDF()` |
| `labels_per_chunk` | `int` | `1` | Q&A pairs per ~100 tokens |
| `quality_threshold` | `float` | `0.6` | Min relevance score (0.0-1.0) |
| `max_samples` | `int` | `None` | Limit total Q&A pairs |
| `prompt_template` | `str` | `None` | Custom instruction template |

#### **Returns**

```python
{
    "success": bool,
    "samples": [
        {
            "id": str,
            "question": str,
            "answer": str,
            "context": str,               # Source text
            "relevance_score": float,     # 0.0-1.0
            "source_page": int,
            "tokens": int                 # Approximate token count
        },
        ...
    ],
    "statistics": {
        "total_samples": int,
        "avg_relevance": float,
        "quality_distribution": {
            "high": int,
            "medium": int,
            "low": int
        }
    },
    "metadata": {
        "model_used": str,
        "processing_time": float
    }
}
```

#### **Examples**

```python
# Basic labeling
labeled = quickapi.dataLabel_Dataset(extracted)

# More Q&A pairs per chunk
labeled = quickapi.dataLabel_Dataset(
    extracted,
    labels_per_chunk=3  # 3x more samples
)

# Only high-quality samples
labeled = quickapi.dataLabel_Dataset(
    extracted,
    quality_threshold=0.8  # Stricter
)

# Access results
print(f"Generated {len(labeled['samples'])} Q&A pairs")
for sample in labeled['samples'][:3]:
    print(f"Q: {sample['question']}")
    print(f"A: {sample['answer']}")
    print(f"Relevance: {sample['relevance_score']:.2f}\n")
```

#### **Raises**

| Exception | When |
|-----------|------|
| `ValueError` | Invalid extracted_data format |
| `RuntimeError` | LLM inference fails |
| `MemoryError` | Dataset too large for memory |

#### **Performance**

| Input | Time | Notes |
|-------|------|-------|
| 1 page | ~10 sec | Single Q&A |
| 10 pages | ~90 sec | 10 Q&As |
| 100 pages | ~15 min | Parallel processing |

---

### `dataDistill_Dataset(labeled_data, remove_duplicates=True, quality_filter=True)`

Filter and clean the dataset, removing low-quality and duplicate samples.

#### **Signature**
```python
def dataDistill_Dataset(
    labeled_data: dict,
    remove_duplicates: bool = True,
    quality_filter: bool = True,
    min_length: int = 10,
    max_length: int = 2048,
    diversity_threshold: float = 0.3
) -> dict
```

#### **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labeled_data` | `dict` | **required** | Output from `dataLabel_Dataset()` |
| `remove_duplicates` | `bool` | `True` | Remove duplicate Q&As |
| `quality_filter` | `bool` | `True` | Filter by quality score |
| `min_length` | `int` | `10` | Minimum answer length (tokens) |
| `max_length` | `int` | `2048` | Maximum answer length (tokens) |
| `diversity_threshold` | `float` | `0.3` | Min semantic difference |

#### **Returns**

```python
{
    "success": bool,
    "samples": [
        # Same format as dataLabel_Dataset samples
    ],
    "statistics": {
        "original_count": int,
        "final_count": int,
        "removed": {
            "duplicates": int,
            "low_quality": int,
            "too_short": int,
            "too_long": int,
            "low_diversity": int
        },
        "quality_metrics": {
            "avg_relevance": float,
            "avg_length": float,
            "diversity_score": float
        }
    }
}
```

#### **Examples**

```python
# Standard distillation
clean = quickapi.dataDistill_Dataset(labeled)

# Strict filtering
clean = quickapi.dataDistill_Dataset(
    labeled,
    quality_filter=True,
    min_length=50,
    diversity_threshold=0.5
)

# See what was removed
print(f"Kept: {clean['statistics']['final_count']} samples")
print(f"Removed {clean['statistics']['removed']['duplicates']} duplicates")

# Access stats
stats = clean['statistics']['quality_metrics']
print(f"Average relevance: {stats['avg_relevance']:.2f}")
print(f"Diversity: {stats['diversity_score']:.2f}")
```

#### **Raises**

| Exception | When |
|-----------|------|
| `ValueError` | Invalid labeled_data format |
| `EmptyDatasetError` | No samples survive filtering |

#### **Performance**

- ⚡ Instant (all local processing)
- 💾 Memory: ~100MB per 10,000 samples

---

---

### `dataGenerate_Distillation(seed_prompts, format, responses_per_prompt, diversity_mode, system_prompt, save_output)`

Generate a model distillation dataset by running seed prompts through the configured teacher model.

#### **Signature**
```python
def dataGenerate_Distillation(
    seed_prompts: Union[List[str], str],
    format: str = "alpaca",
    responses_per_prompt: int = 1,
    diversity_mode: bool = True,
    system_prompt: str = "",
    save_output: bool = True
) -> dict
```

#### **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed_prompts` | `list[str]` or `str` | **required** | List of instruction strings OR path to `.txt`/`.jsonl` file |
| `format` | `str` | `"alpaca"` | Output format: `alpaca`, `chatml`, `sharegpt`, `dpo`, `completion` |
| `responses_per_prompt` | `int` | `1` | Responses per prompt (use `2` + `format="dpo"` for DPO pairs) |
| `diversity_mode` | `bool` | `True` | Vary temperature slightly between responses for diversity |
| `system_prompt` | `str` | `""` | System prompt injected with each generation |
| `save_output` | `bool` | `True` | Save output JSONL to `output_dir` |

#### **Returns**

```python
{
    "total_prompts": int,
    "total_samples": int,
    "format": str,
    "items": [dict, ...],
    "output_file": str or None
}
```

#### **Examples**

```python
# Ollama — basic distillation
quickapi.setup("ollama", model="mistral")
result = quickapi.dataGenerate_Distillation(
    seed_prompts=["Explain gradient descent.", "What is backpropagation?"],
    format="alpaca"
)

# vLLM — large-batch distillation from file
quickapi.setup("vllm", model="mistralai/Mistral-7B-Instruct-v0.2")
result = quickapi.dataGenerate_Distillation(
    seed_prompts="my_prompts.txt",   # one per line
    format="chatml",
    responses_per_prompt=3,
    diversity_mode=True
)

# DPO pairs (chosen / rejected)
result = quickapi.dataGenerate_Distillation(
    seed_prompts=["Compare SQL vs NoSQL."],
    format="dpo",
    responses_per_prompt=2
)

# Short alias
result = quickapi.synthesize(prompts, format="alpaca")
```

#### **Raises**

| Exception | When |
|-----------|------|
| `SetupError` | `setup()` not called first |
| `ValidationError` | No valid prompts found |
| `BackendError` | LLM backend not available |

---

### `dataConvert_Format(data, target_format, output_path=None)`

Export dataset in standard training formats.

#### **Signature**
```python
def dataConvert_Format(
    data: dict,
    target_format: str,
    output_path: str = None,
    split_ratio: dict = None
) -> dict
```

#### **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict` | **required** | Output from distill/tokenize functions |
| `target_format` | `str` | **required** | `"alpaca"`, `"chatml"`, `"sharegpt"`, `"jsonl"` |
| `output_path` | `str` | `None` | Save to file (None = in-memory) |
| `split_ratio` | `dict` | `None` | Train/val/test split |

#### **Format Specifications**

##### **Alpaca Format**
```json
[
  {
    "instruction": "What is photosynthesis?",
    "input": "",
    "output": "Photosynthesis is the process..."
  }
]
```

##### **ChatML Format**
```json
[
  {
    "messages": [
      {"role": "user", "content": "What is photosynthesis?"},
      {"role": "assistant", "content": "Photosynthesis is..."}
    ]
  }
]
```

##### **ShareGPT Format**
```json
[
  {
    "conversations": [
      {"from": "user", "value": "What is photosynthesis?"},
      {"from": "assistant", "value": "Photosynthesis is..."}
    ]
  }
]
```

#### **Returns**

```python
{
    "success": bool,
    "format": str,
    "sample_count": int,
    "output_file": str,           # Path if saved
    "file_size_mb": float,
    "splits": {
        "train": int,
        "val": int,
        "test": int
    }
}
```

#### **Examples**

```python
# Export to Alpaca format (in-memory)
alpaca = quickapi.dataConvert_Format(clean, "alpaca")

# Save to file
result = quickapi.dataConvert_Format(
    clean,
    "sharegpt",
    output_path="./data/dataset.jsonl"
)
print(f"Saved to: {result['output_file']}")

# With train/val/test split
result = quickapi.dataConvert_Format(
    clean,
    "chatml",
    output_path="./data/dataset.jsonl",
    split_ratio={"train": 0.8, "val": 0.1, "test": 0.1}
)
```

#### **Raises**

| Exception | When |
|-----------|------|
| `ValueError` | Invalid target_format |
| `IOError` | Cannot write to output_path |

---

## Parameter Guide

### Model Names

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `phi2` | 2.7B | ⚡⚡⚡ | ⭐⭐ | Testing, fast iteration |
| `granite` | 3B | ⚡⚡⚡ | ⭐⭐⭐ | Production, offline |
| `mistral` | 7B | ⚡⚡ | ⭐⭐⭐⭐ | **Recommended default** |
| `llama2` | 7B | ⚡⚡ | ⭐⭐⭐⭐ | High-quality output |

### Inference Backends

| Backend | Speed | Quality | Requirements |
|---------|-------|---------|--------------|
| `vllm` | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | GPU (NVIDIA/AMD), ~24GB VRAM |
| `ollama` | ⚡⚡ | ⭐⭐⭐⭐ | CPU or GPU, requires daemon |
| `auto` | Medium | Good | Auto-selects best available |

---

## Return Value Formats

All functions return dictionaries with consistent structure:

```python
{
    "success": bool,           # Always present
    "samples": [...] or {...}, # Main data
    "statistics": {...},       # Metrics
    "metadata": {...}          # Context
}
```

Check `result["success"]` before accessing data.

---

## Error Handling

### Try-Except Pattern

```python
from saara import quickapi

try:
    quickapi.setup("ollama")
    result = quickapi.dataExtract_PDF("missing.pdf")
except FileNotFoundError:
    print("PDF not found")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not found` | Not downloaded | `ollama pull mistral` |
| `FileNotFoundError` | PDF path wrong | Check path exists |
| `RuntimeError` | LLM fails | Check model loaded |
| `MemoryError` | Dataset too large | Process in batches |
| `ValueError` | Bad data format | Check function signature |

---

## Configuration Reference

### Utility Functions

```python
from saara import quickapi

# Check which backends are available before setup
quickapi.check_backends()

# View current configuration and status after setup
status = quickapi.get_status()
print(status)
# {"initialized": True, "backend": "ollama", "model": "mistral", ...}

# Reset global state (clears setup, useful for testing)
quickapi.reset()

# List supported output formats
print(quickapi.list_formats())
# → ['alpaca', 'chatml', 'sharegpt', 'completion', 'dpo', 'chatml_tools']
```

---

## Advanced Usage

### Batch Processing

```python
from saara import quickapi

quickapi.setup("ollama", model="mistral")

pdf_files = [f"doc_{i}.pdf" for i in range(100)]

all_data = []
for pdf_file in pdf_files:
    extracted = quickapi.dataExtract_PDF(pdf_file)
    labeled = quickapi.dataLabel_Dataset(extracted)
    all_data.append(labeled)

# Combine all
combined = {
    "samples": [s for d in all_data for s in d["samples"]]
}

final = quickapi.dataDistill_Dataset(combined)
quickapi.dataConvert_Format(final, "alpaca", "./combined.jsonl")
```

### Custom Quality Metrics

```python
# Check dataset quality
result = quickapi.dataDistill_Dataset(labeled)
stats = result["statistics"]["quality_metrics"]

if stats["avg_relevance"] < 0.8:
    print("⚠️  Low relevance, retrying with better model")
    labeled = quickapi.dataLabel_Dataset(extracted, quality_threshold=0.8)
```

### Streaming Large Datasets

```python
# Process in chunks to save memory
chunk_size = 10  # pages
for i in range(0, 100, chunk_size):
    extracted = quickapi.dataExtract_PDF(
        "large_file.pdf",
        max_pages=chunk_size
    )
    labeled = quickapi.dataLabel_Dataset(extracted)
    final = quickapi.dataDistill_Dataset(labeled)
    # Save each chunk
    quickapi.dataConvert_Format(
        final,
        "alpaca",
        f"./chunk_{i}.jsonl"
    )
```

---

## 📞 Support

- 📖 **Guide**: See [QUICKAPI_START_HERE.md](QUICKAPI_START_HERE.md)
- 🔗 **Examples**: `examples/quickapi_tutorial.ipynb`
- 🐛 **Issues**: GitHub issues or discussions

---

**Last Updated**: 2026  
**Version**: SAARA 1.7.0+
