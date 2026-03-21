# SAARA QuickAPI - START HERE

## The Simplest Way to Use SAARA

**You don't need to understand the architecture. Just import, setup, and use 6 functions.**

---

## 🎯 The 3 Steps

### 1. Install
```bash
pip install saara-ai vllm
```

### 2. Configure (Once)
```python
from saara import quickapi

quickapi.setup(model="mistral", backend="vllm")
```

### 3. Use (All Features in 6 Functions)

```python
# Extract PDF
pdf = quickapi.dataExtract_PDF("document.pdf")

# Generate labels (AI-powered)
dataset = quickapi.dataLabel_Dataset(pdf)

# Clean dataset
clean = quickapi.dataDistill_Dataset(dataset)

# Tokenize
tokens = quickapi.dataTokenize_Dataset(clean)

# Convert format (optional)
formatted = quickapi.dataConvert_Format(tokens, target_format="sharegpt")

# Use formatted["output_file"] to train your model
```

**That's it. 6 functions. Everything else is magic.**

---

## 📚 What Each Function Does

| Function | Takes | Returns | Example |
|----------|-------|---------|---------|
| Setup | Model, backend | Config | `quickapi.setup(model="mistral")` |
| Extract | PDF file | Text + metadata | `quickapi.dataExtract_PDF("doc.pdf")` |
| Label | Extracted text | Q&A pairs | `quickapi.dataLabel_Dataset(pdf)` |
| Distill | Labeled data | Cleaned data | `quickapi.dataDistill_Dataset(dataset)` |
| Tokenize | Cleaned data | Token IDs | `quickapi.dataTokenize_Dataset(clean)` |
| Convert | Any data | Training format | `quickapi.dataConvert_Format(tokens)` |

---

## 🚀 5-Minute Example

```python
# 1. Import & Setup (30 seconds)
from saara import quickapi
quickapi.setup(model="mistral", backend="vllm")

# 2. Extract PDF (1 minute)
pdf = quickapi.dataExtract_PDF("research_paper.pdf")

# 3. Generate Labels (2 minutes)
dataset = quickapi.dataLabel_Dataset(pdf)

# 4. Clean Dataset (1 minute)
clean = quickapi.dataDistill_Dataset(dataset)

# 5. Tokenize (1 minute)
tokens = quickapi.dataTokenize_Dataset(clean)

# 6. Use for training (now!)
print(f"✅ Dataset ready: {tokens['output_file']}")
# → Use with HF Trainer, PEFT, or Ollama
```

---

## 💻 Where to Run This

### Local Machine
```bash
# Install once
pip install saara-ai vllm

# Run your script
python your_script.py
```

### Google Colab (Best for Testing)
```python
# Cell 1: Install
!pip install -q saara-ai vllm

# Cell 2: Your script (same as above)
from saara import quickapi
quickapi.setup(model="mistral")
# ... rest of your code
```

### Kaggle Notebooks
```python
# Same as Colab - just run the code
```

---

## 🎓 Complete Working Example

Copy-paste this into your Python script or Jupyter notebook:

```python
# ============================================================
# Complete SAARA QuickAPI Example
# Copy-paste this and change "document.pdf" to your file
# ============================================================

from saara import quickapi
import json

# Setup (one time)
print("🚀 Setting up SAARA...")
quickapi.setup(
    model="mistral",
    backend="vllm",
    output_dir="./saara_results",
    verbose=True
)

# PDF Processing Pipeline
print("\n📄 Step 1: Extracting PDF...")
pdf = quickapi.dataExtract_PDF("document.pdf", use_ocr=True)
print(f"  ✓ Extracted {pdf['total_pages']} pages")

print("\n🤖 Step 2: Generating labels...")
dataset = quickapi.dataLabel_Dataset(
    pdf,
    label_types=["qa", "summarization"],
    save_output=True
)
print(f"  ✓ Generated {len(dataset['labeled_items'])} training examples")

print("\n🧪 Step 3: Distilling dataset...")
clean = quickapi.dataDistill_Dataset(
    dataset,
    min_quality=0.7,
    remove_duplicates=True,
    save_output=True
)
print(f"  ✓ Kept {clean['distilled_count']} high-quality items")
print(f"  ✓ Quality score: {clean['quality_score']:.1%}")

print("\n🔢 Step 4: Tokenizing...")
tokens = quickapi.dataTokenize_Dataset(
    clean,
    max_length=1024,
    save_output=True
)
print(f"  ✓ Created {tokens['total_sequences']} sequences")
print(f"  ✓ Total tokens: {tokens['total_tokens']:,}")

print("\n📋 Step 5: Converting format...")
formatted = quickapi.dataConvert_Format(
    tokens,
    target_format="sharegpt",
    save_output=True
)
print(f"  ✓ Converted to ShareGPT format")

print("\n" + "="*60)
print("✅ PIPELINE COMPLETE!")
print("="*60)
print(f"\n📊 Results saved to: {quickapi.get_status()['output_dir']}\n")
print(f"Files created:")
print(f"  1. {pdf['output_file']}")
print(f"  2. {dataset['output_file']}")
print(f"  3. {clean['output_file']}")
print(f"  4. {tokens['output_file']}")
print(f"  5. {formatted['output_file']}")
print(f"\n🎯 Use {formatted['output_file']} to train your model!\n")
```

Save this as `quickapi_example.py` and run:
```bash
python quickapi_example.py
```

---

## ⚙️ What Backend Should I Use?

### vLLM (Recommended)
- **Fast**: 100-300 tokens/sec
- **Flexible**: Use any HuggingFace model
- **Cloud-Ready**: Works in Colab/Kaggle
- **Install**: `pip install vllm`

### Ollama (Simple)
- **Easy**: Graphical installer
- **Local-Only**: Not for cloud notebooks
- **Slower**: 30-50 tokens/sec
- **Install**: Download from ollama.ai

**Choice**: Use vLLM. It's faster and works everywhere.

---

## 🔧 Customization (Optional)

### Change Model
```python
quickapi.setup(model="llama2:13b")  # Larger, better quality
quickapi.setup(model="neural-chat")  # Faster
```

### Adjust Temperature
```python
quickapi.setup(temperature=0.3)  # More deterministic
quickapi.setup(temperature=0.9)  # More creative
```

### Custom Output Directory
```python
quickapi.setup(output_dir="./my_data")
```

### Different Label Types
```python
dataset = quickapi.dataLabel_Dataset(
    pdf,
    label_types=["qa"]  # Only Q&A pairs
)
```

---

## 🚨 Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: vllm` | `pip install vllm` |
| `Inference engine not initialized` | Call `quickapi.setup()` first |
| `Ollama not responding` | Run `ollama serve` in a terminal |
| `Out of memory` | Use smaller model or reduce `max_tokens` |
| `Slow` | Use vLLM instead of Ollama |

---

## 📊 What Gets Saved?

Everything is saved to `output_dir` (default: `./saara_results/`):

```
./saara_results/
├── document_extracted.json       # Raw extracted text
├── labeled_dataset.jsonl         # Raw generated Q&A
├── distilled_dataset.jsonl       # Cleaned data
├── tokenized_dataset.jsonl       # Token IDs
└── dataset_sharegpt.jsonl        # Training format
```

Each file is **production-ready** and can be used immediately.

---

## 🎯 Next Steps

After generating the dataset, train your model:

### Option 1: Hugging Face (Easiest)
```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("json", data_files="./saara_results/dataset_sharegpt.jsonl")
trainer = SFTTrainer(model="mistral", train_dataset=dataset)
trainer.train()
```

### Option 2: PEFT QLoRA (Memory-Efficient)
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16)
model = get_peft_model(model, config)
# Train on your dataset
```

### Option 3: Ollama (Local)
```bash
# Export model to GGUF and run locally
```

---

## 📖 For More Details

- **Full API**: [QUICKAPI_REFERENCE.md](QUICKAPI_REFERENCE.md)
- **Interactive Tutorial**: [examples/quickapi_tutorial.ipynb](examples/quickapi_tutorial.ipynb)
- **Architecture**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## ✅ Quick Checklist

- [ ] Installed `saara-ai` and `vllm`
- [ ] Ran `quickapi.setup()`
- [ ] Extracted a PDF
- [ ] Generated labels
- [ ] Distilled dataset
- [ ] Tokenized data
- [ ] Ready to train!

---

**You're all set. Go build something amazing! 🚀**

For support, check the troubleshooting section above or the [Full Reference](QUICKAPI_REFERENCE.md).
