# SAARA Documentation

## Complete Module, API, and Usage Reference

Version scope: SAARA 1.6.x  
Audience: Developers building data pipelines, fine-tuning workflows, and RAG applications with SAARA.

---

## Table of Contents

1. Executive Summary
2. Installation and Environment Setup
3. Quick Start
4. Architecture Overview
5. Public Top-Level API Reference
6. Module Reference (Module-by-Module)
7. Sample Workflows
8. Error Handling Guidance
9. Validation and Quality Checks
10. Optional Dependencies
11. Best Practices

---

## 1) Executive Summary

SAARA is a modular framework for:

- Document ingestion and preprocessing
- Dataset construction and synthetic data generation
- Tokenization and token storage
- Fine-tuning orchestration
- Evaluation and deployment
- Retrieval-Augmented Generation (RAG)
- Multi-provider LLM inference

It exposes a high-level public API via saara package imports and lazily loads heavyweight modules where possible.

---

## 2) Installation and Environment Setup

### Core installation

```bash
pip install saara-ai
```

### Recommended local runtime (Ollama)

```bash
ollama serve
ollama pull granite4
```

### Optional development tools

```bash
pip install -e '.[dev]'
```

### Optional feature dependencies

```bash
pip install torch transformers peft trl
pip install vllm ollama
pip install pymupdf pdfplumber pyarrow pandas
```

---

## 3) Quick Start

### Minimal pipeline example

```python
from saara import DataPipeline, PipelineConfig

config = PipelineConfig(
    output_directory="./datasets",
    model="granite",
    use_ocr=True,
)

pipeline = DataPipeline(config)
result = pipeline.process_file("sample.pdf", dataset_name="demo")

print("success:", result.success)
print("outputs:", result.output_files)
```

### Minimal unified LLM example

```python
from saara import create_llm

llm = create_llm(provider="ollama", model="granite4")
print(llm.generate("Summarize this framework in 3 bullets."))
```

---

## 4) Architecture Overview

| Layer | Purpose | Primary Modules |
|---|---|---|
| Config and Contracts | Type-safe configuration and errors | saara/config.py, saara/exceptions.py, saara/protocols.py |
| Ingestion and Processing | Extract, clean, chunk, label, generate datasets | saara/pdf_extractor.py, saara/cleaner.py, saara/chunker.py, saara/labeler.py, saara/dataset_generator.py |
| Training Stack | Fine-tuning, modular training pipeline, token reuse | saara/train.py, saara/training_pipeline.py, saara/token_storage.py |
| Inference and RAG | Querying and retrieval-augmented generation | saara/rag_engine.py, saara/llm_providers.py |
| Runtime and Utilities | Cloud setup, acceleration, visualization, file utilities | saara/cloud_runtime.py, saara/accelerator.py, saara/visualizer.py, saara/file_utils.py, saara/quickstart.py |

---

## 5) Public Top-Level API Reference

These are importable directly from saara.

### 5.1 Configuration APIs

| Symbol | Type | Description |
|---|---|---|
| TrainConfig | dataclass | Training configuration |
| PipelineConfig | dataclass | Data pipeline configuration |
| EvaluatorConfig | dataclass | Evaluation configuration |
| DeployerConfig | dataclass | Deployment configuration |
| RAGConfig | dataclass | RAG configuration |
| PretrainConfig | dataclass | Pretraining configuration |
| convert_config | function | Converts dict-style config to typed config |

### 5.2 Exception APIs

| Symbol | Type | Description |
|---|---|---|
| SaaraException | exception | Base SAARA exception |
| ConfigurationError | exception | Invalid or missing config |
| ModelNotFoundError | exception | Model lookup/load failure |
| OllamaConnectionError | exception | Local Ollama connection failure |
| TrainingError | exception | Training runtime failure |
| EvaluationError | exception | Evaluation runtime failure |
| DeploymentError | exception | Deployment/export failure |
| PDFExtractionError | exception | PDF extraction failure |
| DatasetGenerationError | exception | Dataset generation failure |

### 5.3 Pipeline and Data APIs

| Symbol | Type | Description |
|---|---|---|
| DataPipeline | class | End-to-end document-to-dataset pipeline |
| PipelineResult | dataclass/class | Pipeline execution result metadata |
| DatasetGenerator | class | Converts processed data into trainable formats |
| DataLabeler | class | Labels and structures extracted content |
| PDFExtractor | class | PDF text and metadata extraction |
| TextChunker | class | Chunking by size/overlap |
| TextCleaner | class | OCR/noise cleanup |
| SemanticChunker | class | Semantic chunking helpers |
| SyntheticDataGenerator | class | Synthetic sample generation |
| DataType | enum | Synthetic data categories |
| QualityJudge | class | Synthetic sample quality scoring |

### 5.4 Training and Model APIs

| Symbol | Type | Description |
|---|---|---|
| LLMTrainer | class | Fine-tuning entrypoint |
| ModelEvaluator | class | Model quality/evaluation workflows |
| ModelDeployer | class | Deployment/export workflows |

### 5.5 Runtime and Visualization APIs

| Symbol | Type | Description |
|---|---|---|
| NeuralAccelerator | class | Performance optimization utilities |
| create_accelerator | function | Accelerator factory |
| TrainingDashboard | class | Training visualization dashboard |
| ModelAnalyzer | class | Model analysis helpers |
| create_visualizer | function | Visualizer factory |
| CloudRuntime | class | Cloud runtime setup helper |
| setup_colab | function | Colab environment setup |
| is_cloud_environment | function | Cloud environment detection |

### 5.6 Tokenization and Modular Training APIs

| Symbol | Type | Description |
|---|---|---|
| AIEnhancedTokenizer | class | AI-assisted tokenization |
| create_ai_tokenizer | function | AI tokenizer factory |
| TokenStorage | class | Persist tokenized dataset |
| TokenStorageConfig | dataclass | Token storage settings |
| quick_tokenize | function | Fast tokenization shortcut |
| TrainingPipeline | class | Stage-based modular training |
| TrainingPipelineConfig | dataclass | Training pipeline settings |
| quick_train | function | Fast modular training shortcut |

### 5.7 RAG APIs

| Symbol | Type | Description |
|---|---|---|
| RAGEngine | class | Retrieval + generation engine |
| RAGManager | class | RAG orchestration helper |
| create_rag_engine | function | RAG engine factory |
| quick_rag | function | Fast RAG setup helper |

### 5.8 Unified LLM Provider APIs

| Symbol | Type | Description |
|---|---|---|
| UnifiedLLM | class | Provider-agnostic inference client |
| create_llm | function | Unified client factory |
| quick_generate | function | One-line generation helper |

### 5.9 Flexible Tokenizer APIs

| Symbol | Type | Description |
|---|---|---|
| create_tokenizer | function | Tokenizer factory |
| TokenizerRegistry | class | Register/create tokenizer implementations |
| BPETokenizer | class | BPE tokenizer |
| WordPieceTokenizer | class | WordPiece tokenizer |
| ByteTokenizer | class | Byte-level tokenizer |

### 5.10 File Utility APIs

| Symbol | Type | Description |
|---|---|---|
| load_from_file | function | Auto-detect file loader |
| save_to_file | function | Auto-detect file saver |
| load_jsonl | function | Load JSONL records |
| save_jsonl | function | Save JSONL records |
| extract_texts | function | Extract text field from records |
| split_dataset | function | Train/val/test split helper |

### 5.11 Quickstart APIs

| Symbol | Type | Description |
|---|---|---|
| QuickLLM | class | Simplified LLM wrapper |
| QuickTokenizer | class | Simplified tokenizer wrapper |
| QuickDataset | class | Simplified dataset wrapper |
| QuickFineTune | class | Simplified fine-tune wrapper |
| ollama_local | function | Quick local Ollama client |
| vllm_local | function | Quick local vLLM client |

---

## 6) Module Reference (Module-by-Module)

| Module | Responsibility | Typical Use |
|---|---|---|
| saara/__init__.py | Public exports and lazy loading | Stable top-level imports |
| saara/config.py | Config dataclasses and conversion | Strongly typed config setup |
| saara/exceptions.py | Exception hierarchy | Granular error handling |
| saara/protocols.py | Shared interfaces/contracts | Cross-module consistency |
| saara/pipeline.py | End-to-end processing orchestration | Single entrypoint processing |
| saara/pdf_extractor.py | PDF extraction | Document ingestion |
| saara/cleaner.py | Text cleanup | OCR and noisy text cleanup |
| saara/chunker.py | Text chunking | Context-window-safe dataset prep |
| saara/labeler.py | Data labeling/structuring | Supervised data preparation |
| saara/dataset_generator.py | Dataset formatting | Build training corpora |
| saara/synthetic_generator.py | Synthetic data generation | Data augmentation |
| saara/train.py | Core fine-tuning | Adapter/QLoRA tuning |
| saara/pretrain.py | Pretraining orchestration | Full pretraining workflows |
| saara/pretrain_data.py | Pretraining data helpers | Pretraining data prep |
| saara/evaluator.py | Evaluation workflows | Score model outputs |
| saara/deployer.py | Deployment/export tooling | Package and deploy artifacts |
| saara/model_manager.py | Model lifecycle handling | Manage model paths/artifacts |
| saara/rag_engine.py | RAG orchestration | Retrieval-based QA apps |
| saara/token_storage.py | Reusable token storage | Faster iterative training |
| saara/training_pipeline.py | Stage-based training | Modular training execution |
| saara/llm_providers.py | Multi-provider LLM abstraction | Provider-agnostic inference |
| saara/tokenizers.py | Tokenizer implementations/registry | Custom tokenization setup |
| saara/file_utils.py | Manual IO utilities | Controlled dataset IO |
| saara/quickstart.py | Beginner-friendly wrappers | Fast prototyping |
| saara/cloud_runtime.py | Cloud-specific setup | Colab/Kaggle workflows |
| saara/accelerator.py | Hardware acceleration helpers | Performance optimization |
| saara/visualizer.py | Dashboards and analysis | Training introspection |
| saara/api.py | Programmatic API layer | Integrate SAARA services |
| saara/ollama_client.py | Ollama transport/client logic | Local model access |
| saara/moondream_ocr.py | Moondream OCR integration | OCR backend selection |
| saara/qwen_ocr.py | Qwen OCR integration | OCR backend selection |
| saara/translator.py | Translation helpers | Multilingual preprocessing |
| saara/distiller.py | Distillation logic | Teacher-student pipelines |
| saara/gpu_workers.py | GPU worker management | Parallel GPU tasks |
| saara/workspace.py | Workspace path/state utilities | Project-level management |

---

## 7) Sample Workflows

### 7.1 PDF to trainable dataset

```python
from saara import DataPipeline, PipelineConfig

config = PipelineConfig(
    output_directory="./datasets",
    model="granite",
    use_ocr=True,
    ocr_model="qwen",
    chunk_size=1500,
    chunk_overlap=200,
)

pipeline = DataPipeline(config)
result = pipeline.process_directory("./pdfs", dataset_name="my_corpus")

if result.success:
    print("processed:", result.documents_processed)
    print("samples:", result.total_samples)
    print("files:", result.output_files)
else:
    print("errors:", result.errors)
```

### 7.2 Tokenize once, train many times

```python
from saara import quick_tokenize, quick_train

# Stage 1: pre-tokenize data
store_path = quick_tokenize("train.jsonl", "./token_store")

# Stage 2: train using token store
run_info = quick_train(store_path, output_dir="./models/run_01")
print(run_info)
```

### 7.3 Unified provider usage with fallback

```python
from saara import create_llm

primary = create_llm(provider="vllm", model="mistral")
fallback = create_llm(provider="ollama", model="mistral")

prompt = "Write a concise release note for SAARA improvements."

try:
    print(primary.generate(prompt))
except Exception:
    print(fallback.generate(prompt))
```

### 7.4 RAG setup

```python
from saara import create_rag_engine

rag = create_rag_engine(index_path="./index", model="granite4")
answer = rag.query("What are the mandatory compliance controls?")
print(answer)
```

### 7.5 Manual file processing utilities

```python
from saara import load_jsonl, save_jsonl, extract_texts, split_dataset

records = load_jsonl("dataset.jsonl")
texts = extract_texts(records, "text")
train, val, test = split_dataset(records, train_ratio=0.8, val_ratio=0.1)

save_jsonl(train, "train.jsonl")
save_jsonl(val, "val.jsonl")
save_jsonl(test, "test.jsonl")

print("texts:", len(texts))
print("splits:", len(train), len(val), len(test))
```

### 7.6 Quickstart-first development pattern

```python
from saara import QuickLLM, QuickTokenizer, QuickDataset

llm = QuickLLM("ollama", model="granite4")
tok = QuickTokenizer("bpe", vocab_size=32000)
ds = QuickDataset.from_file("input.jsonl")

texts = ds.get_texts("text")
tok.train(texts[:1000])

print(llm.generate("Summarize tokenization strategy in one paragraph."))
```

---

## 8) Error Handling Guidance

Use typed exceptions for resilient automation.

```python
from saara import (
    DataPipeline,
    PipelineConfig,
    OllamaConnectionError,
    PDFExtractionError,
    DatasetGenerationError,
)

config = PipelineConfig(output_directory="./datasets", model="granite")

try:
    result = DataPipeline(config).process_file("sample.pdf", dataset_name="demo")
except OllamaConnectionError:
    print("Local model service unavailable. Start Ollama and retry.")
except PDFExtractionError:
    print("Input PDF could not be parsed. Check file integrity.")
except DatasetGenerationError:
    print("Pipeline reached dataset stage but failed formatting/output.")
```

---

## 9) Validation and Quality Checks

Run these after setup or major changes:

```bash
python test_comprehensive.py
python test_modular_training.py
python test_integration.py
```

Recommended style checks:

```bash
python -m black saara/ examples/ --check --line-length 100
python -m isort saara/ examples/ --check-only --profile black --line-length 100
```

---

## 10) Optional Dependencies

| Capability | Dependencies |
|---|---|
| Fine-tuning | torch, transformers, peft, trl |
| Local LLM providers | vllm, ollama |
| PDF and OCR pipeline | pymupdf, pdfplumber |
| Parquet and dataframe workflows | pyarrow, pandas |

Install only what your workflow needs to keep environments lightweight.

---

## 11) Best Practices

1. Start with Quickstart APIs for prototypes, then migrate to core modules for production control.
2. Keep configuration explicit via dataclasses to make runs reproducible.
3. Use token storage for repeated experiments to reduce preprocessing time.
4. Separate ingestion, training, and evaluation outputs by run directory.
5. Wrap external-provider calls with fallbacks for reliability.
6. Run integration tests before packaging or deployment.

---

## Appendix: Fast API Lookup

| Goal | API |
|---|---|
| One-line generation | quick_generate |
| Local model access | ollama_local |
| Build unified client | create_llm |
| Fast tokenization | quick_tokenize |
| Stage-based training | quick_train |
| Quick RAG setup | quick_rag |
| Auto-load files | load_from_file |
| Auto-save files | save_to_file |

