# 🚀 SAARA QuickAPI - Get Started in 3 Minutes

> **The simplest way to turn PDFs into AI-ready training datasets**

---

## ⚡ 30-Second Setup

```python
from saara import quickapi

# One-time configuration (picks up Ollama automatically)
quickapi.setup("ollama")

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
    "ollama",                           # backend: ollama | vllm | auto
    model="mistral"                     # model name
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

### 5️⃣ **`dataGenerate_Distillation(seed_prompts)`** — Synthetic Data / Model Distillation

Run seed prompts through the teacher model and collect responses as training data for a smaller student model.

```python
# From a list of prompts (Ollama)
result = quickapi.dataGenerate_Distillation(
    seed_prompts=[
        "Explain the difference between TCP and UDP",
        "What is gradient descent and why is it used?",
        "Describe how attention mechanisms work in transformers",
    ],
    format="alpaca",
    responses_per_prompt=1
)
print(f"Generated {result['total_samples']} samples → {result['output_file']}")

# From a file (vLLM — faster for large batches)
quickapi.setup("vllm", model="mistral")
result = quickapi.dataGenerate_Distillation(
    seed_prompts="my_prompts.txt",   # one prompt per line
    format="chatml",
    responses_per_prompt=3,          # 3 diverse responses per prompt
    diversity_mode=True,             # vary temperature for diversity
    system_prompt="You are a precise technical expert."
)

# DPO dataset (chosen / rejected pairs)
result = quickapi.dataGenerate_Distillation(
    seed_prompts=["Compare SQL and NoSQL databases"],
    format="dpo",
    responses_per_prompt=2           # automatically splits chosen vs rejected
)
```

Or using the short alias:
```python
result = quickapi.synthesize(my_prompts, format="alpaca")
```

✅ **Teacher model**: Whatever backend you configured in `setup()`  
✅ **Formats**: alpaca, chatml, sharegpt, completion, dpo  
✅ **DPO support**: Pass `responses_per_prompt=2, format="dpo"`  
✅ **File input**: Pass path to `.txt` (one prompt/line) or `.jsonl`

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
quickapi.setup("ollama", model="mistral")

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

## ⚙️ Backend Options: Ollama vs vLLM

### 🦙 Ollama — Recommended for most users
```python
# Start Ollama first: ollama serve
# Pull a model:       ollama pull mistral

quickapi.setup(
    "ollama",
    model="mistral"          # or: llama3, granite3.1-dense:8b, qwen2.5, etc.
)
```
✅ CPU-friendly, installs in minutes, no GPU required  
✅ Best for: laptops, small servers, local dev

### ⚡ vLLM — For GPU servers (faster throughput)
```python
# Requires: pip install vllm  +  CUDA GPU

quickapi.setup(
    "vllm",
    model="mistralai/Mistral-7B-Instruct-v0.2",  # HuggingFace model ID
    temperature=0.7,
    max_tokens=2048
)
```
✅ 5-10× faster than Ollama on GPU  
✅ Best for: large-scale dataset generation, distillation jobs  
⚠️ Requires CUDA GPU + `pip install vllm`

### 🔄 Auto-detection
```python
quickapi.setup("auto")  # tries Ollama first, falls back to vLLM
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
| 💾 **Multiple formats** | Export to Alpaca, ChatML, ShareGPT, DPO |
| 🧬 **Distillation** | Generate synthetic datasets from any seed prompts |
| 🦙 **Ollama support** | Local, CPU-friendly, no GPU required |
| ⚡ **vLLM support** | GPU-accelerated, high-throughput generation |
| 🚀 **End-to-end** | PDF → dataset in one script |
| 📱 **Offline capable** | Works without internet (with Ollama) |
| ⚡ **Production-ready** | Error handling, logging, validation |

---

**Ready? Start with `quickapi.setup()` now!** 🎉
