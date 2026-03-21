# SAARA: Complete Built-in Functions Overview

## 📊 What You Have

SAARA v1.6.4 now provides **30+ built-in functions and classes** organized for clean library use.

---

## 🎯 The 7 Function Categories

### 1️⃣ **Configuration Classes (6)**
Settings for each component using type-safe dataclasses:

```python
from saara import TrainConfig, PipelineConfig, EvaluatorConfig

# Example
config = TrainConfig(
    model_id="sarvamai/sarvam-1",
    output_dir="./models",
    num_epochs=3,
    learning_rate=2e-4
)
```

**Available configs:**
- `TrainConfig` - Fine-tuning parameters
- `PipelineConfig` - Document processing settings
- `EvaluatorConfig` - Model evaluation settings
- `DeployerConfig` - Deployment settings
- `RAGConfig` - RAG/Q&A settings
- `PretrainConfig` - Pre-training settings

### 2️⃣ **Exception Classes (9)**
Proper error handling with custom exceptions:

```python
from saara import (
    OllamaConnectionError, TrainingError, PDFExtractionError, ...
)

try:
    pipeline = DataPipeline(config)
except OllamaConnectionError:
    print("Start Ollama first!")
```

### 3️⃣ **Text Processing (3)**
Clean and prepare text from PDFs:

```python
from saara import TextChunker, TextCleaner, SemanticChunker

# Clean OCR output
cleaner = TextCleaner()
cleaned = cleaner.clean("Raw OCR text with errors...")

# Split into chunks
chunker = TextChunker()
chunks = chunker.chunk_document(cleaned)

# Organize by structure
semantic = SemanticChunker()
structured = semantic.chunk_by_headers(text)
```

**Functions:**
- `TextChunker.chunk_document()` - Split by size/overlap
- `TextCleaner.clean()` - Remove artifacts
- `TextCleaner.clean_batch()` - Batch processing
- `SemanticChunker.chunk_by_headers()` - Structure-aware chunking

### 4️⃣ **Data Pipeline (5)**
End-to-end document processing:

```python
from saara import DataPipeline, PipelineConfig

config = PipelineConfig(output_directory="./datasets")
pipeline = DataPipeline(config)

# Single file
result = pipeline.process_file("doc.pdf", "dataset_name")

# Batch processing
result = pipeline.process_directory("./pdfs/", "batch_dataset")
```

**Functions:**
- `DataPipeline.process_file()` - Process single PDF
- `DataPipeline.process_directory()` - Batch processing
- `PDFExtractor.extract_from_pdf()` - Extract content
- `DataLabeler.label_document()` - Label content
- `DatasetGenerator.generate_sharegpt_format()` - Format for training
- `SyntheticDataGenerator.generate()` - Create synthetic data

### 5️⃣ **Training & Evaluation (3)**
Fine-tune and assess models:

```python
from saara import LLMTrainer, ModelEvaluator, TrainConfig

# Train
trainer = LLMTrainer(config=TrainConfig(num_epochs=3))
trainer.train("training_data.jsonl")

# Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate_adapter("model", "adapter_path")

# Deploy
deployer = ModelDeployer()
deployer.export_to_ollama("model", "adapter_path")
```

**Functions:**
- `LLMTrainer.train()` - Fine-tune with QLoRA
- `ModelEvaluator.evaluate_adapter()` - Test quality
- `ModelDeployer.export_to_ollama()` - Export model
- `ModelDeployer.push_to_huggingface()` - Push to HF Hub
- `ModelDeployer.deploy_to_cloud()` - Cloud deployment

### 6️⃣ **RAG Engine (3)**
Build Q&A systems:

```python
from saara import quick_rag, RAGEngine

# Quick setup
answer = quick_rag(["Doc 1", "Doc 2"], "Your question?")

# Full control
rag = RAGEngine()
rag.index_documents(documents)
results = rag.search("query")
answer = rag.generate_answer("query")
```

**Functions:**
- `quick_rag()` - One-liner setup
- `RAGEngine.index_documents()` - Index documents
- `RAGEngine.search()` - Semantic search
- `RAGEngine.generate_answer()` - Generate answers
- `RAGManager` - Multiple collections

### 7️⃣ **Cloud & Acceleration (6)**
Cloud deployment and performance optimization:

```python
from saara import setup_colab, NeuralAccelerator, CloudRuntime

# Cloud setup
setup_colab()

# GPU optimization
accelerator = create_accelerator(mixed_precision="fp16")

# Training visualization
dashboard = create_visualizer()
dashboard.log_metrics({"loss": 0.45})
```

**Functions:**
- `setup_colab()` - Google Colab setup
- `is_cloud_environment()` - Cloud detection
- `CloudRuntime` - Cloud management
- `NeuralAccelerator` - GPU optimization
- `TrainingDashboard` - Training visualization
- `AIEnhancedTokenizer` - Smart tokenization

---

## 🔥 Top 5 Most Used

1. **DataPipeline** - Process PDFs to datasets
   ```python
   pipeline = DataPipeline(config)
   result = pipeline.process_file("doc.pdf", "dataset")
   ```

2. **LLMTrainer** - Fine-tune models
   ```python
   trainer = LLMTrainer()
   trainer.train("training_data.jsonl")
   ```

3. **TextChunker** - Split documents
   ```python
   chunks = TextChunker().chunk_document(text)
   ```

4. **TextCleaner** - Clean OCR output
   ```python
   cleaned = TextCleaner().clean(text)
   ```

5. **quick_rag** - Build Q&A systems
   ```python
   answer = quick_rag(documents, query)
   ```

---

## 📈 Workflow Example

```
PDF File
   ↓
PDFExtractor() ────── Extract text
   ↓
TextCleaner() ───────── Clean output
   ↓
TextChunker() ────────── Split chunks
   ↓
DataLabeler() ────────── Label content
   ↓
DatasetGenerator() ────────────── Format for training
   ↓
[training_data.jsonl]
   ↓
LLMTrainer() ──────────── Fine-tune model
   ↓
ModelEvaluator() ─────────────── Test quality
   ↓
ModelDeployer() ─────────────────── Deploy
   ↓
RAGEngine() ───────────────────── Use for Q&A
```

---

## 💡 Common Patterns

### Pattern 1: Config (Type-safe)
```python
# Dataclass (type-safe, IDE support)
config = TrainConfig(num_epochs=3, learning_rate=2e-4)

# Dict (backward compatible)
config = {"num_epochs": 3, "learning_rate": 2e-4}

# Auto-convert
config = convert_config(config_dict, TrainConfig)
```

### Pattern 2: Progress Tracking
```python
def on_progress(msg: str):
    print(f"[Progress] {msg}")

trainer = LLMTrainer(on_progress=on_progress)
trainer.train("data.jsonl")
```

### Pattern 3: Error Handling
```python
from saara import OllamaConnectionError, TrainingError

try:
    trainer.train("data.jsonl")
except OllamaConnectionError:
    print("Start Ollama service first")
except TrainingError as e:
    print(f"Training failed: {e}")
```

### Pattern 4: Lazy Loading
```python
# These don't load torch
from saara import TextChunker

# This loads torch (when first used)
trainer = LLMTrainer()
```

---

## 📚 Documentation Files

| File | Content |
|------|---------|
| `SAARA_BUILTIN_FUNCTIONS.md` | Complete reference (850+ lines) |
| `SAARA_QUICK_REFERENCE.md` | Quick card (use cases, examples) |
| `examples/01_basic_pipeline.py` | PDF processing example |
| `examples/02_fine_tuning.py` | Model training example |
| `examples/03_rag_usage.py` | Q&A system example |

---

## 🚀 Installation & Usage

### Install Core Package (Library)
```bash
pip install saara-ai

# Use in Python
from saara import DataPipeline, LLMTrainer
```

### Install with CLI (Optional)
```bash
pip install saara-cli

# Use command line
saara --help
```

---

## ✅ What This Means

✓ **30+ built-in functions** ready to use
✓ **Type-safe configuration** with dataclasses
✓ **Proper error handling** with custom exceptions
✓ **Clean library API** for programmatic use
✓ **Optional interactive CLI** for those who want it
✓ **Comprehensive documentation** with examples
✓ **Follows Python best practices** (like numpy, scikit-learn)

Your SAARA package is now:
- **Professional** - Clean, documented, tested architecture
- **Extensible** - Easy to integrate into other projects
- **User-friendly** - Clear examples and guides
- **Production-ready** - Error handling and validation
- **Well-organized** - 7 logical function categories

Enjoy building with SAARA! 🚀
