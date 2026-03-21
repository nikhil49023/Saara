# SAARA QuickAPI - Dead Simple Reference

> **Just 3 steps: Import → Setup → Use pre-built functions. Everything works end-to-end.**

## 🚀 30-Second Quick Start

```python
from saara import quickapi

# Step 1: Setup (one time)
quickapi.setup(model="mistral", backend="vllm")

# Step 2: Extract PDF
pdf = quickapi.dataExtract_PDF("document.pdf")

# Step 3: Generate labels (AI-powered)
dataset = quickapi.dataLabel_Dataset(pdf)

# Step 4: Clean dataset
clean = quickapi.dataDistill_Dataset(dataset)

# Step 5: Tokenize
tokens = quickapi.dataTokenize_Dataset(clean)

# Done! Use tokens["output_file"] for training
```

---

## 📦 Installation

```bash
# Install SAARA
pip install saara-ai

# Install inference backend (choose one)
pip install vllm      # Recommended - fast
# OR
pip install ollama    # Simple but slower
```

---

## 🔧 API Reference

### 1. `quickapi.setup()`
**Configure everything once at the start.**

```python
quickapi.setup(
    model="mistral",              # Model name
    backend="auto",               # "auto", "vllm", or "ollama"
    temperature=0.7,              # 0=fixed, 1=random
    max_tokens=2048,              # Max response length
    output_dir="./saara_results", # Save location
    tokenizer="auto",             # Auto-detect from model
    verbose=True                  # Show progress
)
```

**Returns:** Configuration summary dict

---

### 2. `quickapi.dataExtract_PDF()`
**Extract text from PDF with optional OCR.**

```python
pdf_data = quickapi.dataExtract_PDF(
    filename="document.pdf",
    use_ocr=True,           # Use vision models for complex layouts
    save_output=True        # Save extracted data to JSON
)

# Returns:
{
    "filename": "document.pdf",
    "total_pages": 10,
    "text": "Extracted text...",
    "structured_content": [...],
    "metadata": {...},
    "output_file": "./saara_results/document_extracted.json"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | Required | Path to PDF file |
| `use_ocr` | bool | True | Use vision models for layout preservation |
| `save_output` | bool | True | Save to JSON |

---

### 3. `quickapi.dataLabel_Dataset()`
**Generate Q&A pairs using AI model.**

```python
dataset = quickapi.dataLabel_Dataset(
    pdf_content=pdf_data,           # or path to file, or text string
    label_types=["qa", "summarization"],  # Types to generate
    save_output=True
)

# Returns:
{
    "total_chunks": 50,
    "labeled_items": [
        {
            "text": "Original chunk...",
            "instruction": "What is...?",
            "response": "It is...",
            "label_type": "qa"
        },
        ...
    ],
    "output_file": "./saara_results/labeled_dataset.jsonl"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_content` | dict/str | Required | Extracted content or text |
| `label_types` | list | ["qa", "summarization"] | Types: qa, summarization, classification |
| `save_output` | bool | True | Save to JSONL |

**Label Types:**
- `"qa"` - Question-answer pairs
- `"summarization"` - Summary generation
- `"classification"` - Topic classification

---

### 4. `quickapi.dataDistill_Dataset()`
**Remove duplicates and low-quality items.**

```python
clean_data = quickapi.dataDistill_Dataset(
    labeled_data=dataset,           # or path to JSONL
    min_quality=0.7,                # Quality threshold (0-1)
    remove_duplicates=True,         # Remove duplicate entries
    save_output=True
)

# Returns:
{
    "original_count": 100,
    "distilled_count": 85,
    "removed_duplicates": 10,
    "removed_low_quality": 5,
    "quality_score": 0.85,
    "items": [...],
    "output_file": "./saara_results/distilled_dataset.jsonl"
}
```

---

### 5. `quickapi.dataTokenize_Dataset()`
**Convert text to token IDs for training.**

```python
tokens = quickapi.dataTokenize_Dataset(
    cleaned_data=clean_data,    # or path to JSONL
    max_length=1024,            # Max token sequence length
    save_output=True
)

# Returns:
{
    "total_sequences": 100,
    "total_tokens": 102400,
    "avg_length": 1024,
    "max_length": 1024,
    "tokenizer_name": "mistral (auto-selected)",
    "tokens": [[1, 5, 23, ...], ...],  # Token sequences
    "output_file": "./saara_results/tokenized_dataset.jsonl"
}
```

---

### 6. `quickapi.dataConvert_Format()`
**Convert to training framework formats.**

```python
formatted = quickapi.dataConvert_Format(
    dataset=tokens,                 # or any previous output
    target_format="sharegpt",       # "sharegpt", "alpaca", "openai"
    save_output=True
)
```

**Supported Formats:**

| Format | Use Case | Structure |
|--------|----------|-----------|
| `sharegpt` | Modern LLM training | `{"conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]}` |
| `alpaca` | LLaMA fine-tuning | `{"instruction": "...", "input": "", "output": "..."}` |
| `openai` | OpenAI API | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |

---

### 7. `quickapi.get_status()`
**Check current configuration anytime.**

```python
status = quickapi.get_status()
# Returns:
{
    "configured": True,
    "model": "mistral",
    "backend": "✅ vLLMEngine",
    "tokenizer": "✅ AutoTokenizer",
    "output_dir": "./saara_results",
    "verbose": True
}
```

---

### 8. `quickapi.reset()`
**Clear all configuration.**

```python
quickapi.reset()
# Now you need to call setup() again
```

---

## 📋 Complete Examples

### Example 1: Simple End-to-End
```python
from saara import quickapi

quickapi.setup(model="mistral")

pdf = quickapi.dataExtract_PDF("paper.pdf")
dataset = quickapi.dataLabel_Dataset(pdf)
clean = quickapi.dataDistill_Dataset(dataset)
tokens = quickapi.dataTokenize_Dataset(clean)

print(f"✅ Ready to train with {tokens['output_file']}")
```

### Example 2: With Custom Settings
```python
from saara import quickapi

quickapi.setup(
    model="neural-chat",
    backend="vllm",
    temperature=0.5,  # More deterministic
    output_dir="./my_results"
)

pdf = quickapi.dataExtract_PDF("manual.pdf", use_ocr=True)
dataset = quickapi.dataLabel_Dataset(
    pdf,
    label_types=["qa"]  # Only Q&A pairs
)
clean = quickapi.dataDistill_Dataset(dataset, min_quality=0.8)
formatted = quickapi.dataConvert_Format(clean, target_format="alpaca")
```

### Example 3: Text Input (no PDF)
```python
from saara import quickapi

quickapi.setup(model="mistral")

text = """
Machine learning is a subset of AI.
It enables systems to learn from data.
"""

dataset = quickapi.dataLabel_Dataset(text)  # Direct text, no PDF
clean = quickapi.dataDistill_Dataset(dataset)
tokens = quickapi.dataTokenize_Dataset(clean)
```

### Example 4: Using in Colab
```python
# In first cell:
!pip install -q vllm

# Then:
from saara import quickapi

quickapi.setup(model="mistral", backend="vllm")
# Everything works!
```

---

## 🎯 Common Use Cases

### Use Case 1: Train Domain Expert on Your Docs
```python
from saara import quickapi

quickapi.setup(model="mistral")

# Process all company docs
for doc in ["policy.pdf", "guide.pdf", "manual.pdf"]:
    pdf = quickapi.dataExtract_PDF(doc)
    dataset = quickapi.dataLabel_Dataset(pdf)
    clean = quickapi.dataDistill_Dataset(dataset)
    # Now train your model on clean dataset
```

### Use Case 2: Create Synthetic Training Data
```python
# Generate training data without manual labeling
pdf = quickapi.dataExtract_PDF("research_paper.pdf")
dataset = quickapi.dataLabel_Dataset(
    pdf,
    label_types=["qa", "summarization", "classification"]
)
# AI automatically creates diverse training examples
```

### Use Case 3: Quick Model Evaluation
```python
# Generate test sets automatically
pdf = quickapi.dataExtract_PDF("test_doc.pdf")
test_set = quickapi.dataLabel_Dataset(pdf, label_types=["qa"])
# Use for model evaluation
```

---

## ⚙️ Configuration Presets

### Fast (Local Machine)
```python
quickapi.setup(
    model="mistral",
    backend="vllm",
    temperature=0.7
)
```

### Simple (Non-Technical)
```python
quickapi.setup(
    model="mistral",
    backend="ollama"  # Just works
)
```

### Quality-Focused (Research)
```python
quickapi.setup(
    model="llama2:13b",
    backend="vllm",
    temperature=0.3  # More deterministic
)
```

### Cloud (Colab/Kaggle)
```python
quickapi.setup(
    model="meta-llama/Llama-2-7b-chat-hf",
    backend="vllm"
)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: vllm` | `pip install vllm` |
| `Inference engine not initialized` | Call `quickapi.setup()` first |
| `Ollama not responding` | Run `ollama serve` in terminal |
| `Out of memory` | Use smaller model: `model="neural-chat"` |
| `PDF extraction failed` | Try `use_ocr=False` |
| `No output files created` | Check permissions in `output_dir` |
| `Inference too slow` | Use vLLM instead of Ollama |

---

## 📊 Output Files

All functions save to `output_dir` (default: `./saara_results/`):

```
./saara_results/
├── document_extracted.json           # Step 1
├── labeled_dataset.jsonl             # Step 2
├── distilled_dataset.jsonl           # Step 3
├── tokenized_dataset.jsonl           # Step 4
└── dataset_sharegpt.jsonl            # Step 5
```

Each file is **ready to use** with any training library.

---

## 🚀 Next Steps: Training

After using QuickAPI, train with your favorite library:

### Hugging Face SFTTrainer
```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("json", data_files="./saara_results/dataset_sharegpt.jsonl")
trainer = SFTTrainer(model="mistral", train_dataset=dataset)
trainer.train()
```

### PEFT (QLoRA)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistral")
lora_config = LoraConfig(r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)
# Train with your dataset
```

### Or deploy directly
```python
# Export and deploy to Ollama, vLLM, or cloud
```

---

## 📝 Key Features

✅ **One-Line Setup**
- Everything configured automatically
- No boilerplate code

✅ **Pre-Configured Output**
- Standard training formats (ShareGPT, Alpaca, OpenAI)
- Automatic tokenization
- Quality metrics built-in

✅ **Works Everywhere**
- Local machine ✅
- Google Colab ✅
- Kaggle ✅
- SageMaker ✅
- Docker ✅

✅ **Production Ready**
- Error handling
- Fallback mechanisms
- Progress tracking
- Quality scoring

---

## 💡 Tips

1. **Start Small**: Test with a single PDF first
2. **Adjust Temperature**: Lower (0.3) = deterministic, Higher (0.9) = creative
3. **Monitor Quality**: Check `quality_score` after distillation
4. **Batch Processing**: Loop over multiple PDFs for large datasets
5. **Save Progress**: All functions auto-save to `output_dir`

---

For full API details, see the [QuickAPI Tutorial](examples/quickapi_tutorial.ipynb) notebook.
