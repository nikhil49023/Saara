# Running SAARA on Kaggle & Cloud Platforms

Complete guide for using SAARA on Kaggle, Google Colab, and other cloud notebook environments.

## TL;DR: Can I Run Ollama on Kaggle?

**Short Answer**: ❌ **Native Ollama won't work** on Kaggle (or most cloud notebooks)

**Better Answer**: ✅ **Use SAARA's Cloud Runtime** with API-based models (Gemini, OpenAI, etc.)

---

## Why Ollama Doesn't Work on Kaggle

### Technical Limitations

1. **No Persistent Service**
   - Ollama requires running as a background service (`ollama serve`)
   - Kaggle notebooks restart frequently, killing background processes
   - No systemd or service management

2. **Port Restrictions**
   - Ollama runs on `localhost:11434`
   - Kaggle doesn't allow custom port binding
   - Network isolation between sessions

3. **Resource Constraints**
   - Ollama models are large (4-70GB+)
   - Kaggle has limited disk space (20-40GB)
   - Download times are prohibitive

4. **Permission Issues**
   - Can't install system-level services
   - Limited sudo access
   - Docker not available

### What About Other Cloud Platforms?

| Platform | Ollama Support | Why? |
|----------|---------------|------|
| **Kaggle** | ❌ No | Port restrictions, no service persistence |
| **Google Colab** | ❌ No | Same issues as Kaggle |
| **AWS SageMaker** | ⚠️ Limited | Possible in persistent instances, complex setup |
| **Azure ML** | ⚠️ Limited | Same as SageMaker |
| **Paperspace** | ✅ Yes | Full VM access with persistent storage |
| **RunPod** | ✅ Yes | GPU pods with custom services |
| **Lambda Labs** | ✅ Yes | Full control instances |
| **Local** | ✅ Yes | Native support |

---

## Solution: SAARA Cloud Runtime

SAARA has **built-in cloud support** that automatically detects Kaggle and switches to API-based models.

### Architecture

```
Local Environment:
┌─────────────┐
│   SAARA     │ ──→ Ollama (localhost:11434) ──→ Local LLM
└─────────────┘

Cloud Environment (Kaggle/Colab):
┌─────────────┐
│   SAARA     │ ──→ Cloud API (Gemini/OpenAI) ──→ Cloud LLM
└─────────────┘     (Automatic fallback)
```

### Supported Cloud APIs

1. **Google Gemini** (Recommended for Kaggle)
   - Fast and free tier available
   - Gemini 2.0 Flash support
   - Built-in vision capabilities

2. **OpenAI**
   - GPT-4, GPT-3.5 Turbo
   - Requires API key

3. **Anthropic Claude**
   - Claude 3 models
   - High quality reasoning

---

## Quick Start: SAARA on Kaggle

### Method 1: Automatic Cloud Detection (Easiest)

```python
# Install SAARA
!pip install saara-ai google-generativeai

# Import
from saara import DataPipeline, CloudRuntime

# Setup cloud runtime (auto-detects Kaggle)
runtime = CloudRuntime()
runtime.setup(
    api_key="YOUR_GEMINI_API_KEY",  # Get from https://aistudio.google.com/
    provider="gemini"
)

# Use pipeline normally - it automatically uses Gemini instead of Ollama!
pipeline = DataPipeline()
result = pipeline.process_document("research_paper.pdf")
```

### Method 2: Manual Configuration

```python
# Install
!pip install saara-ai google-generativeai

# Configure to use Gemini API
from saara import DataPipeline

config = {
    "cloud": {
        "provider": "gemini",
        "api_key": "YOUR_GEMINI_API_KEY",
        "model": "gemini-2.0-flash"
    },
    "output": {
        "directory": "/kaggle/working/datasets"
    }
}

pipeline = DataPipeline(config)
result = pipeline.process_document("/kaggle/input/documents/paper.pdf")
```

### Method 3: Using Environment Variables

```python
# Set up API key securely in Kaggle Secrets
import os
os.environ['GEMINI_API_KEY'] = 'your-key-here'

# Or add to Kaggle Secrets (recommended):
# Settings → Add-ons → Secrets → Add Secret

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
gemini_key = user_secrets.get_secret("GEMINI_API_KEY")

# Use in SAARA
from saara import CloudRuntime
runtime = CloudRuntime()
runtime.setup(api_key=gemini_key, provider="gemini")
```

---

## Complete Kaggle Example

Here's a full working notebook for Kaggle:

```python
# Cell 1: Install dependencies
!pip install -q saara-ai google-generativeai

# Cell 2: Setup
from saara import CloudRuntime, DataPipeline, LLMTrainer
from saara import quick_tokenize, quick_train
from kaggle_secrets import UserSecretsClient

# Get API key from Kaggle Secrets
secrets = UserSecretsClient()
gemini_key = secrets.get_secret("GEMINI_API_KEY")

# Setup cloud runtime
runtime = CloudRuntime()
runtime.setup(api_key=gemini_key, provider="gemini")

print(f"✅ Running on: {runtime.environment.value}")
print(f"✅ Cloud client ready: {runtime.cloud_client.is_available()}")

# Cell 3: Process PDF documents
pipeline = DataPipeline({
    "cloud": {
        "provider": "gemini",
        "api_key": gemini_key
    },
    "output": {
        "directory": "/kaggle/working/datasets"
    }
})

# Process document from Kaggle dataset
result = pipeline.process_document(
    "/kaggle/input/research-papers/paper.pdf",
    output_name="research_data"
)

print(f"✅ Generated {result.total_samples} training samples")
print(f"✅ Output: {result.output_files}")

# Cell 4: Tokenize for training (Memory-efficient!)
tokens = quick_tokenize(
    "/kaggle/working/datasets/research_data.jsonl",
    "/kaggle/working/tokens/",
    tokenizer="google/gemma-2-2b",  # Use Gemma for Kaggle
    max_length=512
)

print(f"✅ Tokenized data: {tokens}")

# Cell 5: Train model with Kaggle GPU
result = quick_train(
    tokens,
    model_id="google/gemma-2-2b",  # Smaller model for Kaggle
    output_dir="/kaggle/working/model",
    num_epochs=2,
    batch_size=1,  # Small batch for GPU limits
    max_length=512
)

print(f"✅ Model trained: {result['model_path']}")

# Cell 6: Save outputs
# Kaggle notebooks save /kaggle/working/ as output
!ls -lh /kaggle/working/
```

---

## Cloud-Specific Optimizations

### Kaggle Optimizations

```python
from saara import TrainingPipelineConfig, TrainingPipeline

config = TrainingPipelineConfig(
    # Use Kaggle dataset path
    pdf_input="/kaggle/input/your-dataset/",

    # Use Google Gemma (optimized for Kaggle)
    model_id="google/gemma-2-2b",  # 2B model fits in Kaggle GPU
    tokenizer_id="google/gemma-2-2b",

    # Kaggle limits
    batch_size=1,  # Low for 16GB GPU
    max_length=512,  # Shorter sequences
    num_epochs=2,  # Faster training

    # Save to working directory (persisted as output)
    output_dir="/kaggle/working/trained_model",

    # Skip stages if needed (save time)
    skip_stages=[],  # Or skip [1] if using existing data
)

pipeline = TrainingPipeline(config)
result = pipeline.run()
```

### Google Colab Optimizations

```python
# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

from saara import TrainingPipeline, TrainingPipelineConfig

config = TrainingPipelineConfig(
    pdf_input="/content/drive/MyDrive/documents/",
    output_dir="/content/drive/MyDrive/saara_output/",

    # Use free Colab GPU (T4)
    model_id="google/gemma-2-2b",
    batch_size=2,  # T4 has more VRAM
    num_epochs=3,
)

pipeline = TrainingPipeline(config)
result = pipeline.run()
```

---

## Getting API Keys

### Google Gemini (Recommended for Kaggle)

1. Go to: https://aistudio.google.com/
2. Click "Get API Key"
3. Create new API key
4. Copy the key
5. Add to Kaggle Secrets:
   - Settings → Secrets → Add Secret
   - Name: `GEMINI_API_KEY`
   - Value: Your API key

**Free Tier**: 60 requests/minute, perfect for SAARA!

### OpenAI (Alternative)

1. Go to: https://platform.openai.com/api-keys
2. Create new secret key
3. Add to Kaggle Secrets as `OPENAI_API_KEY`

**Pricing**: Pay-as-you-go, ~$0.002 per 1K tokens

---

## Model Recommendations for Cloud

### Kaggle GPU Limits (16GB VRAM)

| Model | Parameters | VRAM | Speed | Quality |
|-------|-----------|------|-------|---------|
| **google/gemma-2-2b** | 2B | 4GB | ⚡⚡⚡ | ⭐⭐⭐ |
| google/gemma-2-7b | 7B | 14GB | ⚡⚡ | ⭐⭐⭐⭐ |
| meta-llama/Llama-3.2-3B | 3B | 6GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |

### Google Colab Free (T4 - 15GB VRAM)

Similar to Kaggle, use 2-7B models.

### Colab Pro (V100 - 27GB VRAM)

Can handle up to 13B models, or 20-30B with quantization.

---

## Troubleshooting Cloud Issues

### Issue 1: Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```

**Solution**:
```python
config = TrainingPipelineConfig(
    batch_size=1,        # Reduce batch size
    max_length=256,      # Shorter sequences
    model_id="google/gemma-2-2b"  # Smaller model
)
```

### Issue 2: API Rate Limits
```
Error: 429 Too Many Requests
```

**Solution**:
```python
# Add delays between requests
import time

# Process in smaller batches
for doc in documents[:10]:  # Process 10 at a time
    result = pipeline.process_document(doc)
    time.sleep(1)  # Wait between requests
```

### Issue 3: Session Timeouts
```
Session expired after 9 hours
```

**Solution**:
```python
# Save checkpoints frequently
config = TrainingPipelineConfig(
    output_dir="/kaggle/working/checkpoints",
    resume_from="/kaggle/working/checkpoints/latest"  # Resume if timeout
)

# Or use smaller training runs
config.num_epochs = 2  # Finish before timeout
```

### Issue 4: Kaggle Dataset Access
```
FileNotFoundError: /kaggle/input/...
```

**Solution**:
```python
# Add dataset in Kaggle notebook settings
# Input → Add Data → Search for dataset

# Verify path
import os
print(os.listdir("/kaggle/input/"))

# Use correct path
pdf_path = "/kaggle/input/your-dataset-name/document.pdf"
```

---

## Performance Comparison

### Local with Ollama
```
Training 1,000 samples:
├─ Document processing: 5 min (Ollama granite4)
├─ Pre-tokenization: 2 min
├─ Training: 15 min (RTX 4090)
└─ Total: 22 min
    Cost: $0 (local)
```

### Kaggle with Gemini API
```
Training 1,000 samples:
├─ Document processing: 3 min (Gemini 2.0 Flash API)
├─ Pre-tokenization: 2 min
├─ Training: 25 min (Kaggle P100 GPU)
└─ Total: 30 min
    Cost: ~$0.05 (Gemini free tier)
```

### Trade-offs

| Aspect | Local (Ollama) | Cloud (Kaggle + Gemini) |
|--------|---------------|------------------------|
| **Setup** | Complex (install Ollama, models) | Simple (pip install) |
| **Cost** | Free (but need hardware) | ~Free (free tiers) |
| **Speed** | Fast (local GPU) | Moderate (shared GPU) |
| **Persistence** | Permanent | 9hr timeout, save outputs |
| **Model Choice** | Any Ollama model | Limited by VRAM |
| **Internet** | Not required | Required |

---

## Best Practices for Cloud

1. **Use Pre-tokenization**
   ```python
   # Tokenize once, train multiple times
   tokens = quick_tokenize("data.jsonl", "/kaggle/working/tokens/")

   # Train with different configs (fast!)
   quick_train(tokens, lr=2e-4, epochs=2)
   quick_train(tokens, lr=5e-4, epochs=3)
   ```

2. **Save Intermediate Results**
   ```python
   # After each stage
   pipeline.save_checkpoint("/kaggle/working/checkpoint.pkl")

   # Resume if needed
   pipeline = TrainingPipeline.from_checkpoint("/kaggle/working/checkpoint.pkl")
   ```

3. **Use Kaggle Datasets**
   ```python
   # Add your PDFs as a dataset in Kaggle
   # Access via /kaggle/input/your-dataset-name/

   # Reusable across notebooks!
   ```

4. **Monitor Resource Usage**
   ```python
   import torch

   # Check GPU usage
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
       print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
       print(f"Used: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
   ```

---

## Alternative: Local Installation

If you **really** need Ollama features:

### Option 1: Use a Local Machine
```bash
# Install Ollama locally
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull granite4
ollama pull gemma2

# Run SAARA normally
pip install saara-ai
saara run
```

### Option 2: RunPod/Paperspace (Cloud with Ollama)

**RunPod** (~$0.20/hour for GPU):
```bash
# In RunPod container terminal
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull granite4

pip install saara-ai
# Use SAARA normally
```

**Paperspace** (~$0.50/hour):
```bash
# Same as RunPod
# Full VM access
```

---

## Summary

**Can Ollama run on Kaggle?** ❌ No

**Should you use Ollama on Kaggle?** ❌ No - use SAARA's Cloud Runtime instead

**Best approach for Kaggle:**
1. Install SAARA: `pip install saara-ai google-generativeai`
2. Setup Cloud Runtime with Gemini API key
3. Use SAARA normally - it handles cloud automatically
4. Enjoy 3-10x faster training with pre-tokenization

**When to use local Ollama:**
- Privacy-sensitive data
- No internet access
- Frequent/heavy usage (cost savings)
- Need specific Ollama-only models

**When to use cloud APIs:**
- Quick experiments
- One-off projects
- Learning/testing
- No local GPU available

---

## Additional Resources

- **Gemini API**: https://ai.google.dev/
- **Kaggle Secrets**: https://www.kaggle.com/docs/api#secrets
- **SAARA Docs**: README.md, MODULAR_TRAINING.md
- **Example Notebook**: [Link to Kaggle notebook]

## Questions?

Open an issue: https://github.com/nikhil49023/Saara/issues

---

© 2025-2026 Kilani Sai Nikhil
Part of SAARA AI SDK
