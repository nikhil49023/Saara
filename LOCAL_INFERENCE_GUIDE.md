# Local-First Inference: vLLM + Ollama Setup Guide

## Quick Summary
- **vLLM**: Fast, flexible, works locally & cloud ✅ **RECOMMENDED**
- **Ollama**: Simple, requires daemon, local only
- **Strategy**: Auto-select vLLM → fallback to Ollama

---

## 🚀 Setup by Environment

### Local Machine (Linux/Mac/Windows)

**Option A: vLLM (Recommended - 5-10x faster)**
```bash
# Install vLLM
pip install vllm

# Download model once
python -c "from vllm import LLM; LLM('mistral')"

# Use in code
from saara.local_inference import LocalInferenceEngine, InferenceConfig
config = InferenceConfig(model="mistral")
engine = LocalInferenceEngine(config)
response = engine.generate("Hello")
```

**Option B: Ollama (Fallback)**
```bash
# Install Ollama from ollama.ai (native installer)
# Then in terminal:
ollama serve

# In another terminal:
ollama pull mistral

# Use in code
config = InferenceConfig(backend="ollama", model="mistral")
engine = LocalInferenceEngine(config)
```

### Google Colab
```python
# Cell 1: Install dependencies
!pip install -q vllm torch

# Cell 2: Use vLLM
from saara.local_inference import LocalInferenceEngine, InferenceConfig
config = InferenceConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_tokens=1024
)
engine = LocalInferenceEngine(config)
response = engine.generate("Explain quantum computing")
print(response)
```

### Kaggle Notebooks
```python
# Cell 1: Install
!pip install -q vllm

# Cell 2: Use (exactly like Colab)
from saara.local_inference import LocalInferenceEngine, InferenceConfig
config = InferenceConfig(model="mistral:latest")
engine = LocalInferenceEngine(config)
```

### SageMaker / Lambda Labs
```python
# Install vLLM in SageMaker notebook
!pip install -q vllm

# vLLM handles GPU detection automatically
from saara.local_inference import LocalInferenceEngine
engine = LocalInferenceEngine()
```

---

## 📊 Model Selection Guide

### For vLLM (Local/Cloud)
| Model | Size | Speed | Quality | RAM | Notes |
|-------|------|-------|---------|-----|-------|
| `mistral:latest` | 7B | ⚡⚡⚡ | ⭐⭐⭐ | 16GB | Best all-rounder |
| `neural-chat` | 7B | ⚡⚡⚡ | ⭐⭐⭐ | 16GB | Fast instruction tuning |
| `meta-llama/Llama-2-13b` | 13B | ⚡⚡ | ⭐⭐⭐⭐ | 30GB | More capable |
| `gpt2` | 0.1B | ⚡⚡⚡⚡ | ⭐ | 1GB | Testing only |

### For Ollama (Local Only)
```bash
ollama pull mistral       # Fast, good quality
ollama pull neural-chat   # Instruction tuner
ollama pull llama2        # Larger, slower
```

---

## 💻 Code Examples

### Simple Generation
```python
from saara.local_inference import LocalInferenceEngine, InferenceConfig

# Auto-select best backend
engine = LocalInferenceEngine(InferenceConfig(model="mistral"))
response = engine.generate("What is AI?")
print(response)
```

### Streaming (Real-time output)
```python
for chunk in engine.generate_stream("Explain ML"):
    print(chunk, end="", flush=True)
```

### With System Prompt
```python
response = engine.generate(
    prompt="Calculate 2+2",
    system_prompt="You are a math tutor. Always explain your reasoning."
)
```

### Force Specific Backend
```python
# Force vLLM (fails if not available)
config = InferenceConfig(backend="vllm", model="mistral")
engine = LocalInferenceEngine(config)

# Force Ollama with fallback disabled
config = InferenceConfig(
    backend="ollama",
    model="mistral",
    enable_auto_fallback=False
)
engine = LocalInferenceEngine(config)
```

### Health Check
```python
engine = LocalInferenceEngine()
print(f"Using: {engine.backend_name}")
print(f"Health: {engine.health_check()}")
```

---

## ⚙️ Configuration Options

```python
from saara.local_inference import InferenceConfig

config = InferenceConfig(
    backend=None,                    # Auto-select, or "vllm"/"ollama"
    model="mistral",                 # Model name
    temperature=0.7,                 # 0=deterministic, 1=creative
    max_tokens=2048,                 # Max output length
    timeout=300,                     # Request timeout seconds
    base_url="http://localhost:11434",  # For Ollama only
    enable_auto_fallback=True        # Try next backend if current fails
)
```

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'vllm'`
```bash
pip install vllm
# For GPU support:
pip install vllm[cuda]
```

### `No inference backend available`
```bash
# Make sure at least one is installed:
pip install vllm     # OR
pip install ollama   # Then: ollama serve
```

### vLLM Slow on First Run
First run downloads and optimizes model (normal, ~1-5 min). Subsequent runs are fast.

### Ollama Not Responding
```bash
# Start daemon in terminal:
ollama serve

# Or check if running:
netstat -an | grep 11434
```

### Out of Memory (VRAM)
```python
# Use quantized model
config = InferenceConfig(model="TheBloke/Mistral-7B-GGUF")
# Or smaller model
config = InferenceConfig(model="gpt2")
```

---

## 📈 Performance Comparison

```
Local Machine (RTX 3090):
  vLLM Mistral:    150 tok/s  ⚡⚡⚡ (Fast)
  Ollama Mistral:   40 tok/s  ⚡   (Slow)
  
Google Colab (T4 GPU):
  vLLM:            80 tok/s   ⚡⚡⚡
  
Kaggle (P100 GPU):
  vLLM:           120 tok/s   ⚡⚡⚡
```

---

## 🎯 Recommendations

**I want speed** → Use vLLM (locally or Colab)
**I want simplicity** → Use Ollama (local only)
**I want both** → Use LocalInferenceEngine (auto-switches)
**I want cloud** → Use vLLM in Colab/Kaggle
**I want production** → Use vLLM with HF model repo

---

## Integration with SAARA

Replace Gemini/Ollama with local vLLM:

```python
from saara import DataPipeline, PipelineConfig
from saara.local_inference import LocalInferenceEngine, InferenceConfig

# Configure local inference
inference_config = InferenceConfig(model="mistral")
local_engine = LocalInferenceEngine(inference_config)

# Use in pipeline
config = PipelineConfig(
    output_directory="./datasets",
    use_ollama=True,  # Set to use local inference
)
```

This gives you:
- ✅ Fast local processing
- ✅ No API keys needed
- ✅ Private (data stays local)
- ✅ Works offline
- ✅ Cloud-compatible
