# 🚀 SAARA QuickAPI - Get Started in 3 Minutes

> **The simplest way to turn PDFs into AI-ready training datasets**

---

## ⚡ 30-Second Setup

```python
from saara import quickapi

# One-time configuration (2 minutes)
quickapi.setup(model="mistral")

# Now use it! (3 lines per operation)
pdf_data = quickapi.dataExtract_PDF("document.pdf")
dataset = quickapi.dataLabel_Dataset(pdf_data)
final = quickapi.dataDistill_Dataset(dataset)
```

That's it. Your dataset is ready.

---

## 📋 What Each Function Does

### 1️⃣ **`setup(model)`** — Initialize Once
Creates configuration and loads your LLM model.

```python
quickapi.setup(
    model="mistral",                    # LLM to use
    inference_backend="auto"            # auto | vllm | ollama
)
```

✅ **When**: Run this once at the start of your script  
✅ **Time**: ~30 seconds (network dependent)

---

### 2️⃣ **`dataExtract_PDF(file_path)`** — Extract Text & Tables

Converts PDF into clean, searchable text with OCR support.

```python
pdf_output = quickapi.dataExtract_PDF("research_paper.pdf")

# Returns:
# {
#   "pages": [...],
#   "tables": [...],
#   "images": [...],
#   "metadata": {...}
# }
```

✅ **Handles**: Scanned PDFs, multi-page documents, mixed content  
✅ **OCR**: Automatic for image-only pages  
✅ **Output**: Structured JSON

---

### 3️⃣ **`dataLabel_Dataset(extracted_data)`** — Generate Q&A Pairs

Uses your LLM to create synthetic question-answer pairs from extracted content.

```python
labeled_data = quickapi.dataLabel_Dataset(pdf_output)

# Returns:
# {
#   "samples": [
#     {"question": "...", "answer": "..."},
#     {"question": "...", "answer": "..."}
#   ],
#   "quality_score": 0.92
# }
```

✅ **Generates**: 1 Q&A per ~100 tokens  
✅ **Quality**: Built-in filtering for relevance  
✅ **Speed**: ~30 sec per page (offline)

---

### 4️⃣ **`dataDistill_Dataset(labeled_data)`** — Filter & Clean

Removes low-quality samples, removes duplicates, validates structure.

```python
clean_data = quickapi.dataDistill_Dataset(labeled_data)

# Returns:
# {
#   "samples": [...],         # Only high-quality samples
#   "filtered_count": 3,      # Removed duplicates
#   "quality_metrics": {
#     "avg_relevance": 0.91,
#     "diversity": 0.87
#   }
# }
```

✅ **Filters**: By relevance, diversity, length  
✅ **Deduplicates**: Removes ~5-10% typical duplicates  
✅ **Speed**: Instant (local processing)

---

### 5️⃣ **`dataTokenize_Dataset(cleaned_data)`** — Prepare for Training

Converts text to token format ready for model training.

```python
tokens = quickapi.dataTokenize_Dataset(clean_data)

# Returns:
# {
#   "tokens": [...],
#   "token_count": 45320,
#   "format": "default"
# }
```

✅ **Output**: Training-ready format  
✅ **Handles**: BPE tokenization, normalization  
✅ **Speed**: Instant

---

### 6️⃣ **`dataConvert_Format(data, target_format)`** — Export Format

Converts dataset to standard training formats.

```python
alpaca_format = quickapi.dataConvert_Format(
    clean_data,
    target_format="alpaca"  # alpaca | chatml | sharegpt
)

# Returns dataset in desired format
```

✅ **Formats**: 
  - **Alpaca** — Fine-tuning (instruction-response)
  - **ChatML** — Chat-style conversations  
  - **ShareGPT** — Multi-turn dialogs

---

## 🔄 Complete Workflow Example

```python
from saara import quickapi

# Step 1: Setup (one time)
quickapi.setup(model="mistral")

# Step 2: Process multiple PDFs
pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

all_datasets = []
for pdf_file in pdfs:
    # Extract → Label → Distill
    extracted = quickapi.dataExtract_PDF(pdf_file)
    labeled = quickapi.dataLabel_Dataset(extracted)
    distilled = quickapi.dataDistill_Dataset(labeled)
    
    all_datasets.append(distilled)

# Step 3: Convert to final format
final_dataset = quickapi.dataConvert_Format(
    all_datasets,
    target_format="alpaca"
)

print(f"✅ Ready for training: {final_dataset['output_file']}")
```

---

## ⚙️ Configuration Tips

### Faster Processing (Less Accurate)
```python
quickapi.setup(
    model="phi2",                    # Smaller, faster model
    inference_backend="ollama"       # Lower latency
)
```

### Better Quality (Slower)
```python
quickapi.setup(
    model="mistral-7b",              # Larger, more accurate
    inference_backend="vllm",        # GPU acceleration
)
```

### Local-Only (No Auth)
```python
quickapi.setup(
    model="granite",                 # Use with Ollama
    inference_backend="ollama"
)
# Must run: ollama serve
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `ollama pull mistral` first |
| `Slow processing` | Use `inference_backend="vllm"` with GPU |
| `PDF extraction fails` | Ensure PDF is not encrypted |
| `Memory error` | Process fewer pages at once |

---

## 📖 Want More?

- **Full API Reference**: See [QUICKAPI_REFERENCE.md](QUICKAPI_REFERENCE.md)
- **Examples & Notebooks**: Check `examples/quickapi_tutorial.ipynb`
- **Performance Comparison**: See `examples/local_inference_comparison.ipynb`

---

## ✨ Key Features at a Glance

| Feature | Benefit |
|---------|---------|
| 🎯 **3-line functions** | No boilerplate, no complexity |
| 🔒 **All pre-configured** | Works out-of-box with sensible defaults |
| 📊 **Quality metrics** | See exactly what your data looks like |
| 💾 **Multiple formats** | Export to Alpaca, ChatML, ShareGPT |
| 🚀 **End-to-end** | PDF → tokens in one script |
| 📱 **Offline capable** | Works without internet (with Ollama) |
| ⚡ **Production-ready** | Error handling, logging, validation |

---

**Ready? Start with `quickapi.setup()` now!** 🎉
