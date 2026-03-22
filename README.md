# SAARA — Document-to-Dataset Factory

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/saara-ai.svg)](https://pypi.org/project/saara-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SAARA** is an open-source framework that turns raw documents into fine-tuning-ready datasets using local LLMs. It handles the full data pipeline: extract → label → synthesize → filter → format. Training is explicitly out of scope — plug SAARA's outputs directly into [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), or [Unsloth](https://github.com/unslothai/unsloth).

```
Documents / Prompts
      │
      ├─▶  Extract     (PDF, TXT, MD, JSONL)
      ├─▶  Label       (LLM-generated Q&A + instruction pairs)
      ├─▶  Synthesize  (Teacher model → distillation dataset)
      ├─▶  Filter      (Quality gate, dedup)
      └─▶  Format      (Alpaca · ChatML · ShareGPT · DPO · Completion)
```

---

## Installation

```bash
pip install saara-ai
```

Optional PDF support:
```bash
pip install pdfplumber pymupdf
```

For vLLM (GPU-accelerated inference):
```bash
pip install vllm   # requires CUDA GPU
```

---

## Quick Start

### With Ollama (CPU-friendly)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. ollama serve
# 3. ollama pull mistral
```

```python
from saara import quickapi

quickapi.setup("ollama", model="mistral")

# PDF → Alpaca dataset
result = quickapi.pdf_to_dataset("document.pdf", format="alpaca")
print(f"Generated {result['total_items']} samples → {result['output_file']}")
```

### With vLLM (GPU, 5-10× faster throughput)

```python
from saara import quickapi

quickapi.setup("vllm", model="mistralai/Mistral-7B-Instruct-v0.2")

# Text → ChatML dataset
result = quickapi.text_to_dataset("notes.txt", format="chatml")
```

### Auto-detect backend

```python
quickapi.setup("auto")   # tries Ollama first, falls back to vLLM
```

---

## Core Workflows

### 1 · PDF / Text Extraction

```python
# PDF
pdf_data = quickapi.dataExtract_PDF("paper.pdf", use_ocr=True)

# Text / Markdown / JSONL
text_data = quickapi.dataExtract_Text("notes.txt")

# Short alias
data = quickapi.extract("document.pdf")
```

### 2 · AI Data Labeling

```python
labeled = quickapi.dataLabel_Dataset(data, label_types=["qa", "instruction"])
# → generates instruction-response pairs using the configured LLM
```

### 3 · Model Distillation — Teacher → Dataset

Run seed prompts through the teacher model and collect responses as distillation training data:

```python
# From a list of prompts
result = quickapi.dataGenerate_Distillation(
    seed_prompts=[
        "Explain gradient descent.",
        "What is attention in transformers?",
        "Describe the vanishing gradient problem.",
    ],
    format="alpaca",
    responses_per_prompt=1
)

# From a .txt file (one prompt per line) — great for large batches with vLLM
result = quickapi.dataGenerate_Distillation(
    seed_prompts="prompts.txt",
    format="chatml",
    responses_per_prompt=3,
    diversity_mode=True,
    system_prompt="You are an expert ML educator."
)

# DPO pairs (chosen / rejected) — pass responses_per_prompt=2 + format="dpo"
result = quickapi.dataGenerate_Distillation(
    seed_prompts=["Compare SQL vs NoSQL.", "Explain REST vs GraphQL."],
    format="dpo",
    responses_per_prompt=2
)

# Short alias
result = quickapi.synthesize(prompts, format="alpaca")
```

### 4 · Quality Filtering

```python
clean = quickapi.dataDistill_Dataset(labeled)
# Removes low-quality, too-short, and duplicate samples

# Short alias
clean = quickapi.distill(labeled)
```

### 5 · Format Conversion

```python
# Convert to any of: alpaca | chatml | sharegpt | dpo | completion
final = quickapi.dataConvert_Format(clean, target_format="sharegpt",
                                    system_prompt="You are a helpful assistant.")

# Convert existing JSONL file (no LLM needed)
quickapi.convert_existing("existing.jsonl", target_format="alpaca")

# Short alias
final = quickapi.convert(clean, format="alpaca")
```

---

## Synthetic Data Generation (4 Types)

```python
from saara import SyntheticDataGenerator, DataType

synth = SyntheticDataGenerator({"model": "mistral", "base_url": "http://localhost:11434"})

samples = synth.generate(
    text=source_text,
    data_types=[DataType.FACTUAL, DataType.REASONING,
                DataType.CONVERSATIONAL, DataType.INSTRUCTION],
    pairs_per_type=3,
    min_quality=0.65
)
```

---

## RAG Engine

```python
from saara import create_rag_engine

rag = create_rag_engine(
    documents=["paper.pdf", "notes.txt"],
    config={"model": "mistral", "base_url": "http://localhost:11434", "top_k": 3}
)

answer = rag.query("What is attention in transformers?")
dataset = rag.generate_dataset(num_questions=20, format="alpaca")
```

---

## DataPipeline Class

```python
from saara import DataPipeline, PipelineConfig

cfg = PipelineConfig(
    model="mistral", backend="ollama",
    output_dir="./output", output_format="alpaca",
    chunk_size=512, min_quality=0.7
)

pipeline = DataPipeline(cfg.to_dict())
result = pipeline.run("document.pdf")

# Or via YAML config
pipeline = DataPipeline("pipeline.yaml")
```

---

## Utilities

```python
from saara import load_jsonl, save_jsonl, split_dataset

data = load_jsonl("dataset.jsonl")
splits = split_dataset(data, train=0.8, val=0.1, test=0.1)
save_jsonl(splits["train"], "train.jsonl")
```

---

## Output Formats

| Format | Use Case |
|---|---|
| `alpaca` | General instruction fine-tuning |
| `chatml` | OpenAI-style chat fine-tuning |
| `sharegpt` | Multi-turn conversation fine-tuning |
| `dpo` | Direct Preference Optimization |
| `completion` | Text completion / continuation |

---

## Public API

```python
import saara
print(saara.__all__)
```

**Data extraction:** `DataPipeline`, `PipelineConfig`, `DataLabeler`, `PDFExtractor`, `TextChunker`, `TextCleaner`, `SemanticChunker`

**Synthetic generation:** `SyntheticDataGenerator`, `DataType`, `QualityJudge`

**RAG:** `RAGEngine`, `RAGManager`, `create_rag_engine`, `quick_rag`

**Formats:** `FormatType`, `FormatConfig`, `convert_dataset`, `AlpacaFormat`, `ChatMLFormat`, `ShareGPTFormat`, `CompletionFormat`, `DPOFormat`

**QuickAPI:** `quickapi`, `dataGenerate_Distillation`, `synthesize`

**Utilities:** `load_jsonl`, `save_jsonl`, `split_dataset`, `load_from_file`, `save_to_file`

> **Note:** Training APIs (`LLMTrainer`, `ModelEvaluator`, `TrainingPipeline`, etc.) are intentionally excluded from SAARA's public surface. SAARA is a data tool — use its output with your preferred training framework.

---

## Documentation

- [Quick Start Guide](QUICKAPI_START_HERE.md)
- [Full API Reference](QUICKAPI_REFERENCE.md)
- [Detailed Documentation](DETAILED_DOCUMENTATION.md)
- [Complete Notebook](examples/saara_complete_guide.ipynb)

## License

MIT — see [LICENSE](LICENSE).
