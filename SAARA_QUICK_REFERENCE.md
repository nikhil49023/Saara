# SAARA Built-in Functions - Quick Reference Card

## 🎯 By Use Case

### 📄 **PDF Processing Pipeline**
```
PDF Input
  ↓
PDFExtractor().extract_from_pdf(file)
  ↓
TextCleaner().clean(text)
  ↓
TextChunker().chunk_document(text)
  ↓
DataLabeler().label_document(chunk)
  ↓
DatasetGenerator().generate_sharegpt_format(labeled_doc)
  ↓
[Ready for Training]
```

### 🧠 **Model Fine-tuning**
```
TrainConfig(model_id, output_dir, num_epochs)
  ↓
LLMTrainer(config=config)
  ↓
trainer.train(training_data.jsonl)
  ↓
[Fine-tuned Model + Adapter Saved]
```

### 📊 **Model Evaluation**
```
EvaluatorConfig(teacher_provider, teacher_model)
  ↓
ModelEvaluator(config=config)
  ↓
evaluator.evaluate_adapter(base_model, adapter_path)
  ↓
[Quality Scores + Improvement Suggestions]
```

### 🚀 **Model Deployment**
```
ModelDeployer()
  ├→ export_to_ollama(base_model, adapter_path)
  ├→ push_to_huggingface(base_model, adapter_path)
  └→ deploy_to_cloud(base_model, adapter_path, provider)
```

### 🎓 **RAG / Q&A System**
```
Option 1 - One-liner:
quick_rag(documents=docs, query="Your question?")

Option 2 - Full control:
RAGEngine() → index_documents(docs) → search(query)
```

---

## 📚 All Built-in Functions (30+)

| Category | Functions | Purpose |
|----------|-----------|---------|
| **Config** | TrainConfig, PipelineConfig, EvaluatorConfig, DeployerConfig, RAGConfig, PretrainConfig | Settings for each component |
| **Exceptions** | SaaraException, ModelNotFoundError, OllamaConnectionError, TrainingError, etc. | Error handling (11 types) |
| **Text Processing** | TextChunker, TextCleaner, SemanticChunker | Clean, chunk, prepare text |
| **Data Pipeline** | DataPipeline, PDFExtractor, DatasetGenerator, DataLabeler, SyntheticDataGenerator | Document → Dataset |
| **Training** | LLMTrainer | Fine-tune models with QLoRA |
| **Evaluation** | ModelEvaluator | Assess model quality |
| **Deployment** | ModelDeployer | Deploy locally/cloud |
| **RAG** | RAGEngine, RAGManager, quick_rag | Q&A systems |
| **Cloud** | CloudRuntime, setup_colab, is_cloud_environment | Cloud deployment |
| **Acceleration** | NeuralAccelerator, create_accelerator | GPU optimization |
| **Visualization** | TrainingDashboard, ModelAnalyzer, create_visualizer | Training monitoring |
| **Tokenization** | AIEnhancedTokenizer, create_ai_tokenizer | Smart tokenization |

---

## 💡 Common Patterns

### Pattern 1: Configuration
```python
# Dict format (backward compatible)
config1 = {"num_epochs": 3, "output_dir": "./models"}

# Dataclass format (type-safe)
config2 = TrainConfig(num_epochs=3, output_dir="./models")

# Auto conversion
config3 = convert_config(config1, TrainConfig)
```

### Pattern 2: Progress Callbacks
```python
def progress_callback(msg: str):
    print(f"[Progress] {msg}")

trainer = LLMTrainer(on_progress=progress_callback)
trainer.train("data.jsonl")
```

### Pattern 3: Error Handling
```python
try:
    pipeline = DataPipeline(config)
except OllamaConnectionError:
    print("Start Ollama first!")
except PDFExtractionError as e:
    print(f"PDF error: {e}")
```

### Pattern 4: Lazy Loading
```python
# These DON'T load heavy dependencies
from saara import TextChunker, SyntheticDataGenerator

# These DO load torch when first accessed
trainer = LLMTrainer()  # torch, transformers loaded
evaluator = ModelEvaluator()  # torch, transformers loaded
```

---

## 🔥 Top 10 Most Used Functions

1. **DataPipeline** - Process PDFs to datasets
2. **LLMTrainer** - Fine-tune models
3. **TextChunker** - Split documents
4. **TextCleaner** - Clean OCR output
5. **ModelEvaluator** - Test model quality
6. **PDFExtractor** - Extract PDF content
7. **ModelDeployer** - Deploy models
8. **quick_rag** - Setup Q&A systems
9. **DatasetGenerator** - Format training data
10. **DataLabeler** - Label document content

---

## 📈 Performance Notes

| Function | Speed | Memory | Best For |
|----------|-------|--------|----------|
| TextChunker | ⚡⚡⚡ | 🟢 Low | Quick text splitting |
| TextCleaner | ⚡⚡ | 🟢 Low | OCR cleanup |
| PDFExtractor | ⚡⚡ | 🟡 Medium | PDF parsing |
| DataPipeline | ⚡ | 🟡 Medium | Full processing |
| LLMTrainer | 🐢 | 🔴 High | Model training |
| ModelEvaluator | 🐢 | 🔴 High | Quality testing |
| RAGEngine | ⚡⚡ | 🟡 Medium | Q&A systems |

---

## 🎯 When to Use Each Function

| Task | Use | Code |
|------|-----|------|
| Extract text from PDF | PDFExtractor | `PDFExtractor().extract_from_pdf(file)` |
| Clean messy text | TextCleaner | `TextCleaner().clean(text)` |
| Split long document | TextChunker | `TextChunker().chunk_document(text)` |
| Organize by topics | SemanticChunker | `SemanticChunker().chunk_by_headers(text)` |
| Label content | DataLabeler | `DataLabeler().label_document(text)` |
| Full pipeline | DataPipeline | `DataPipeline().process_file(pdf, dataset)` |
| Create training data | DatasetGenerator | `DatasetGenerator().generate_sharegpt_format(docs)` |
| Generate synthetic data | SyntheticDataGenerator | `SyntheticDataGenerator().generate(domain, num)` |
| Fine-tune model | LLMTrainer | `LLMTrainer().train(data.jsonl)` |
| Test model | ModelEvaluator | `ModelEvaluator().evaluate_adapter(model, adapter)` |
| Deploy model | ModelDeployer | `ModelDeployer().export_to_ollama(model, adapter)` |
| Q&A system | quick_rag | `quick_rag(documents, query)` |
| Advanced Q&A | RAGEngine | `RAGEngine().index_documents(docs).search(query)` |

---

## 🚀 Minimal Examples

### Minimal 1: Process PDF
```python
from saara import DataPipeline, PipelineConfig
pipeline = DataPipeline(PipelineConfig())
result = pipeline.process_file("doc.pdf", "dataset")
```

### Minimal 2: Fine-tune Model
```python
from saara import LLMTrainer
trainer = LLMTrainer()
trainer.train("training.jsonl")
```

### Minimal 3: Q&A
```python
from saara import quick_rag
answer = quick_rag(["Doc 1", "Doc 2"], "Question?")
```

### Minimal 4: Chunk Text
```python
from saara import TextChunker
chunks = TextChunker().chunk_document("Your text...")
```

### Minimal 5: Evaluate Model
```python
from saara import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.evaluate_adapter("base_model", "adapter_path")
```

---

## 📖 Documentation Links

- Full guide: `SAARA_BUILTIN_FUNCTIONS.md`
- Examples: `examples/`
  - `01_basic_pipeline.py`
  - `02_fine_tuning.py`
  - `03_rag_usage.py`
- CLI docs: `saara_cli/README.md`
