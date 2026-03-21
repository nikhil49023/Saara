# SAARA Built-in Functions & Classes Guide

## 📚 Overview

SAARA provides 30+ built-in functions organized into 7 main categories:

1. **Configuration Classes** - Settings for each component
2. **Exception Classes** - Error handling
3. **Text Processing** - Clean, chunk, and prepare text
4. **Data Pipeline** - End-to-end document processing
5. **Training & Evaluation** - Fine-tune and assess models
6. **RAG Engine** - Retrieval-augmented generation
7. **Cloud & Acceleration** - Cloud deployment and GPU optimization

---

## 1️⃣ CONFIGURATION CLASSES

These dataclasses define settings for each component. All accept both dict and dataclass formats.

### **TrainConfig**
Configure model fine-tuning parameters.

```python
from saara import TrainConfig

config = TrainConfig(
    model_id="sarvamai/sarvam-1",        # Base model
    output_dir="./models/fine_tuned",     # Where to save
    num_epochs=3,                         # Training epochs
    learning_rate=2e-4,                   # Learning rate
    per_device_train_batch_size=1,        # Batch size
    gradient_accumulation_steps=8,        # Accumulation steps
    max_seq_length=2048,                  # Max token length
)

# Or use dict (backward compatible)
config_dict = {
    "model_id": "sarvamai/sarvam-1",
    "output_dir": "./models",
    "num_epochs": 3,
}
```

### **PipelineConfig**
Configure document processing pipeline.

```python
from saara import PipelineConfig

config = PipelineConfig(
    output_directory="./datasets",        # Output folder
    model="granite",                      # Teacher model
    use_ocr=True,                         # Enable OCR
    ocr_model="qwen",                     # "qwen" or "moondream"
    chunk_size=1500,                      # Chunk size in chars
    chunk_overlap=200,                    # Overlap between chunks
    generate_synthetic=False,             # Generate synthetic data
)
```

### **EvaluatorConfig**
Configure model evaluation parameters.

```python
from saara import EvaluatorConfig

config = EvaluatorConfig(
    teacher_provider="ollama",            # "ollama", "openai", "gemini"
    teacher_model="granite",              # Judge model
    num_samples=10,                       # Samples to evaluate
    temperature=0.7,                      # Sampling temperature
    allow_improvement_data=True,          # Generate fixes
)
```

### **DeployerConfig, RAGConfig, PretrainConfig**
Similar pattern - dataclass for configuration.

---

## 2️⃣ EXCEPTION CLASSES

Custom exceptions for proper error handling.

```python
from saara import (
    SaaraException,                       # Base exception
    ModelNotFoundError,                   # Model not found
    OllamaConnectionError,                # Ollama not running
    TrainingError,                        # Training failed
    EvaluationError,                      # Evaluation failed
    PDFExtractionError,                   # PDF parsing failed
    DatasetGenerationError,               # Dataset creation failed
)

# Usage
try:
    pipeline = DataPipeline(config)
except OllamaConnectionError as e:
    print(f"Ollama not running: {e}")
except PDFExtractionError as e:
    print(f"PDF parsing failed: {e}")
```

---

## 3️⃣ TEXT PROCESSING FUNCTIONS

### **TextChunker**
Split documents into smaller, manageable chunks.

```python
from saara import TextChunker

chunker = TextChunker(config={
    "chunk_size": 1500,          # Target chunk size
    "chunk_overlap": 200,        # Overlap for context
    "min_chunk_size": 100,       # Minimum size
})

# Split text into chunks
chunks = chunker.chunk_document(
    text="Your long document text here...",
    sections=None  # Optional: section info
)

# Each chunk has:
# - chunk_id: Unique identifier
# - text: The chunk content
# - start_pos: Start position in original
# - end_pos: End position in original
# - page_numbers: Which pages (if from PDF)
# - word_count: Number of words
```

### **TextCleaner**
Remove OCR artifacts, normalize formatting.

```python
from saara import TextCleaner

cleaner = TextCleaner(config={})

# Clean OCR output
result = cleaner.clean(
    text="Raw OCR text with [errors]",
    use_llm=False  # Use LLM for deep cleaning
)

# Returns CleanedText with:
# - original: Original text
# - cleaned: Cleaned text
# - removed_phrases: What was removed
# - confidence: Cleaning confidence score

# Batch cleaning
texts = ["text1", "text2", "text3"]
cleaned_texts = cleaner.clean_batch(texts)
```

### **SemanticChunker**
Chunk text by semantic meaning (headers, paragraphs).

```python
from saara import SemanticChunker

chunker = SemanticChunker()

# Chunk by headers and semantic boundaries
chunks = chunker.chunk_by_headers(
    text="# Section 1\n\nContent...\n## Subsection\n\nMore content..."
)

# Returns chunks organized by semantic structure
```

---

## 4️⃣ DATA PIPELINE FUNCTIONS

### **DataPipeline**
End-to-end document → training dataset conversion.

```python
from saara import DataPipeline, PipelineConfig, PipelineResult

config = PipelineConfig(
    output_directory="./datasets",
    use_ocr=True,
    ocr_model="qwen",
)

pipeline = DataPipeline(config)

# Process single PDF
result = pipeline.process_file(
    file_path="document.pdf",
    dataset_name="my_dataset"
)

# Result is PipelineResult with:
if result.success:
    print(f"✓ Processed {result.documents_processed} docs")
    print(f"  Total chunks: {result.total_chunks}")
    print(f"  Total samples: {result.total_samples}")
    print(f"  Duration: {result.duration_seconds}s")
    print(f"  Output files: {result.output_files}")
else:
    print(f"✗ Errors: {result.errors}")

# Process directory of PDFs (batch mode)
result = pipeline.process_directory(
    directory="./pdfs",
    dataset_name="batch_dataset"
)
```

### **PDFExtractor**
Extract text, images, and metadata from PDFs.

```python
from saara import PDFExtractor

extractor = PDFExtractor()

# Extract from PDF
doc = extractor.extract_from_pdf("document.pdf")

# Returns Document with:
# - text: Full document text
# - images: Extracted images
# - metadata: PDF metadata
# - pages: Per-page information
```

### **DatasetGenerator**
Format data for training (Alpaca, ShareGPT).

```python
from saara import DatasetGenerator

generator = DatasetGenerator()

# Generate ShareGPT format
dataset = generator.generate_sharegpt_format(
    labeled_documents=documents,
    output_path="training_data.jsonl"
)

# Generate Alpaca format
dataset = generator.generate_alpaca_format(
    q_a_pairs=qa_pairs,
    output_path="alpaca_data.jsonl"
)
```

### **DataLabeler**
Label and categorize document content using Granite AI.

```python
from saara import DataLabeler

labeler = DataLabeler()

# Label a document
labeled = labeler.label_document(
    text="Your document content...",
    categories=["technical", "marketing", "general"]
)

# Returns LabeledDocument with:
# - original_text: Original content
# - labels: Assigned labels
# - summary: Auto-generated summary
# - key_points: Extracted key points
```

### **SyntheticDataGenerator**
Generate synthetic training data.

```python
from saara import SyntheticDataGenerator, DataType, QualityJudge

generator = SyntheticDataGenerator()

# Generate synthetic Q&A pairs
pairs = generator.generate(
    domain="Ayurveda",
    data_type=DataType.FACTUAL_QA,
    num_samples=100,
)

# Quality judgment
judge = QualityJudge()
quality_score = judge.judge(pair)
```

---

## 5️⃣ TRAINING & EVALUATION FUNCTIONS

### **LLMTrainer**
Fine-tune models using QLoRA (4-bit quantization).

```python
from saara import LLMTrainer, TrainConfig

config = TrainConfig(
    model_id="sarvamai/sarvam-1",
    output_dir="./models",
    num_epochs=3,
)

trainer = LLMTrainer(
    model_id=config.model_id,
    config=config,
    on_progress=lambda msg: print(f"[TRAIN] {msg}")  # Progress callback
)

# Single file training
trainer.train("training_data.jsonl")

# Batch training (multiple files)
trainer.train([
    "dataset1.jsonl",
    "dataset2.jsonl",
    "dataset3.jsonl"
])

# Resume from checkpoint
trainer.train(
    "training_data.jsonl",
    resume_from_checkpoint="./models/checkpoint-1000"
)

# After training:
# - Model saved to: ./models/model/
# - Adapter saved to: ./models/model/final_adapter
```

### **ModelEvaluator**
Evaluate fine-tuned model quality.

```python
from saara import ModelEvaluator, EvaluatorConfig

config = EvaluatorConfig(
    teacher_provider="ollama",
    teacher_model="granite",
    num_samples=10,
)

evaluator = ModelEvaluator(config)

# Evaluate adapter performance
results = evaluator.evaluate_adapter(
    base_model_id="sarvamai/sarvam-1",
    adapter_path="./models/final_adapter",
    test_prompts=["What is X?", "Explain Y"],
)

# Results include:
# - Scores for each response
# - Average quality score
# - Improvement suggestions
# - Generated training data for fixes
```

### **ModelDeployer**
Deploy trained models locally or to cloud.

```python
from saara import ModelDeployer

deployer = ModelDeployer()

# Deploy to local Ollama
deployer.export_to_ollama(
    base_model_id="sarvamai/sarvam-1",
    adapter_path="./models/final_adapter"
)

# Deploy to HuggingFace Hub
deployer.push_to_huggingface(
    base_model_id="sarvamai/sarvam-1",
    adapter_path="./models/final_adapter"
)

# Deploy to cloud (Google Cloud Run)
deployer.deploy_to_cloud(
    base_model_id="sarvamai/sarvam-1",
    adapter_path="./models/final_adapter",
    cloud_provider="gcp"
)
```

---

## 6️⃣ RAG ENGINE FUNCTIONS

### **RAGEngine**
Build retrieval-augmented generation systems.

```python
from saara import RAGEngine, RAGConfig

config = RAGConfig(
    vector_store="chromadb",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    top_k=5,
)

rag = RAGEngine(config)

# Index documents
documents = ["Doc 1 text...", "Doc 2 text...", "Doc 3 text..."]
rag.index_documents(documents)

# Query the RAG engine
query = "What is the main topic?"
results = rag.search(query)

# Results:
# - Relevant documents
# - Similarity scores
# - Ranked by relevance

# Generate answer with context
answer = rag.generate_answer(query)
```

### **RAGManager**
Manage multiple RAG indexes.

```python
from saara import RAGManager

manager = RAGManager()

# Create multiple collections
manager.create_collection("medical_docs")
manager.create_collection("legal_docs")

# Add to specific collection
manager.add_to_collection("medical_docs", documents)

# Search specific collection
results = manager.search("medical_docs", query)
```

### **quick_rag**
Simple one-liner RAG setup.

```python
from saara import quick_rag

# One-line RAG setup
answer = quick_rag(
    documents=["Doc 1", "Doc 2"],
    query="Your question here",
    top_k=3
)
```

---

## 7️⃣ ACCELERATION & VISUALIZATION

### **NeuralAccelerator**
GPU optimization and mixed precision training.

```python
from saara import NeuralAccelerator, create_accelerator

# Auto-create accelerator
accelerator = create_accelerator(
    mixed_precision="fp16",  # or "bf16", "no"
    gradient_accumulation_steps=8
)

# Manual creation
accelerator = NeuralAccelerator(
    use_mixed_precision=True,
    device_type="cuda"  # or "cpu"
)
```

### **TrainingDashboard**
Visualize training progress.

```python
from saara import TrainingDashboard, create_visualizer

# Create dashboard
dashboard = create_visualizer()

# Log metrics
dashboard.log_metrics({
    "loss": 0.45,
    "accuracy": 0.92,
    "epoch": 1
})

# Display live graphs
dashboard.show()
```

### **Cloud Runtime**
Deploy on cloud platforms (Colab, Kaggle).

```python
from saara import setup_colab, is_cloud_environment

# Auto-setup for Google Colab
setup_colab()

# Check if running in cloud
if is_cloud_environment():
    print("Running in cloud environment")
    # Use cloud-specific models (Gemini, etc.)
```

---

## 📋 QUICK START EXAMPLES

### Example 1: Process PDF → Generate Dataset
```python
from saara import DataPipeline, PipelineConfig

config = PipelineConfig(output_directory="./datasets")
pipeline = DataPipeline(config)
result = pipeline.process_file("book.pdf", "my_dataset")
```

### Example 2: Fine-tune Model
```python
from saara import LLMTrainer, TrainConfig

trainer = LLMTrainer(config=TrainConfig(num_epochs=3))
trainer.train("training_data.jsonl")
```

### Example 3: Q&A with RAG
```python
from saara import quick_rag

answer = quick_rag(
    documents=["Document 1...", "Document 2..."],
    query="What is mentioned?"
)
```

### Example 4: Evaluate Model
```python
from saara import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_adapter(
    base_model_id="sarvamai/sarvam-1",
    adapter_path="./models/final_adapter"
)
```

---

## 🔑 Key Design Patterns

### **Configuration Pattern**
All components use config dataclasses:
```python
from saara import TrainConfig, convert_config

# Use dataclass
config1 = TrainConfig(num_epochs=3)

# Use dict (backward compatible)
config2 = {"num_epochs": 3}

# Convert between them
config3 = convert_config(config2, TrainConfig)
```

### **Progress Callbacks**
Most long-running functions accept progress callbacks:
```python
def on_progress(msg: str):
    print(f"Progress: {msg}")

trainer = LLMTrainer(on_progress=on_progress)
trainer.train("data.jsonl")
```

### **Lazy Loading**
Heavy dependencies load only when needed:
```python
# No torch imported yet
from saara import TextChunker

chunker = TextChunker()  # Still no torch

# torch loads when first used
trainer = LLMTrainer()  # Now torch is imported
```

---

## 📊 Function Flow Diagram

```
PDF → PDFExtractor → TextCleaner → TextChunker
                                      ↓
                                 DataPipeline
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              DataLabeler    SyntheticDataGenerator  DatasetGenerator
                    ↓                 ↓                 ↓
                    └─────────────────┼─────────────────┘
                                      ↓
                           [training_data.jsonl]
                                      ↓
                                 LLMTrainer
                                      ↓
                           [fine_tuned_model]
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              ModelEvaluator   ModelDeployer         RAGEngine
```

This covers all 30+ built-in functions in SAARA! Each one is designed to work independently or as part of the larger pipeline.
