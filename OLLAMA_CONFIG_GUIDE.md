# Ollama Configuration Guide

Complete guide to configuring and using Ollama models in SAARA.

## Overview

SAARA uses **Ollama** for running local LLMs that power:
- 📄 **Document labeling** - Creating instruction-response pairs from PDFs
- 🔤 **Text processing** - Classification, entity extraction, summarization
- 🌐 **Translation** - Indian language translation via Saara AI
- 🤖 **Semantic analysis** - Topic extraction, quality assessment

## Configuration Methods

### 1. YAML Configuration File (Recommended)

Create a `config.yaml` file:

```yaml
ollama:
  base_url: http://localhost:11434
  model: granite4                  # Default model
  timeout: 300                     # Request timeout (seconds)
  max_retries: 3                   # Retry attempts

hardware:
  ram_gb: 15.8
  tier: light                      # light, medium, heavy
  vram_gb: 4.0

pdf:
  ocr_engine: moondream            # moondream or qwen

output:
  directory: datasets
```

**Available model options:**
- `granite4` - IBM Granite 4.0 (recommended for labeling)
- `qwen2.5:3b` - Qwen 2.5 3B (lightweight)
- `llama3.1` - Meta Llama 3.1
- `mistral` - Mistral 7B
- `mixtral` - Mixtral 8x7B
- Any model from [Ollama Library](https://ollama.com/library)

### 2. Python Dictionary Configuration

```python
from saara import DataPipeline

config = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "granite4",
        "timeout": 300,
        "max_retries": 3
    },
    "pdf": {
        "ocr_engine": "moondream"
    },
    "output": {
        "directory": "datasets"
    }
}

pipeline = DataPipeline(config)
```

### 3. Direct OllamaClient Configuration

```python
from saara.ollama_client import OllamaClient

# Minimal config
client = OllamaClient({
    "model": "granite4"
})

# Full config
client = OllamaClient({
    "base_url": "http://localhost:11434",
    "model": "llama3.1",
    "timeout": 300,
    "max_retries": 3
})
```

## Configuration Parameters

### Ollama Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | `http://localhost:11434` | Ollama API endpoint |
| `model` | str | `granite4` | Model to use for generation |
| `timeout` | int | `300` | Request timeout in seconds |
| `max_retries` | int | `3` | Retry attempts on failure |

### Model Selection Guide

```python
# For data labeling (Q&A generation, instructions)
config = {"ollama": {"model": "granite4"}}  # Best accuracy

# For lightweight tasks (classification, extraction)
config = {"ollama": {"model": "qwen2.5:3b"}}  # Fastest

# For complex reasoning
config = {"ollama": {"model": "mixtral"}}  # Most capable

# For Indian languages
config = {"ollama": {"model": "granite4"}}  # Best multilingual support
```

## Usage Patterns

### Pattern 1: Pipeline with Ollama

```python
from saara import DataPipeline

# Load config from YAML
pipeline = DataPipeline("config.yaml")

# Or use dict config
pipeline = DataPipeline({
    "ollama": {"model": "granite4"},
    "output": {"directory": "datasets"}
})

# Process document
result = pipeline.process_document("research_paper.pdf")
```

### Pattern 2: Direct Ollama Client

```python
from saara.ollama_client import OllamaClient

client = OllamaClient({"model": "granite4"})

# Generate text
response = client.generate(
    prompt="Explain quantum computing",
    system_prompt="You are a helpful physics teacher",
    temperature=0.7,
    max_tokens=2048
)

print(response.content)
```

### Pattern 3: JSON-Structured Responses

```python
from saara.ollama_client import OllamaClient

client = OllamaClient({"model": "granite4"})

# Generate structured JSON
result = client.generate_json(
    prompt="What are the main topics in this text: ...",
    system_prompt="Extract topics from text",
    schema={
        "type": "object",
        "properties": {
            "main_topic": {"type": "string"},
            "subtopics": {"type": "array"},
            "keywords": {"type": "array"}
        }
    }
)

print(result)
# {'main_topic': '...', 'subtopics': [...], 'keywords': [...]}
```

### Pattern 4: Streaming Responses

```python
from saara.ollama_client import OllamaClient

client = OllamaClient({"model": "granite4"})

# Stream response token by token
for chunk in client.stream_generate(
    prompt="Write a long essay about AI",
    system_prompt="You are an AI researcher"
):
    print(chunk, end='', flush=True)
```

## Built-in Prompt Templates

SAARA includes pre-configured prompt templates for common tasks:

```python
from saara.ollama_client import OllamaClient, PromptTemplates

client = OllamaClient({"model": "granite4"})

# 1. Document Classification
result = client.generate_json(
    prompt=PromptTemplates.CLASSIFY_DOCUMENT.format(text="..."),
)

# 2. Topic Extraction
result = client.generate_json(
    prompt=PromptTemplates.EXTRACT_TOPICS.format(text="..."),
)

# 3. Q&A Generation (for training data)
result = client.generate_json(
    prompt=PromptTemplates.GENERATE_QA.format(text="..."),
)

# 4. Summarization
result = client.generate_json(
    prompt=PromptTemplates.SUMMARIZE.format(text="..."),
)

# 5. Entity Extraction
result = client.generate_json(
    prompt=PromptTemplates.EXTRACT_ENTITIES.format(text="..."),
)

# 6. Instruction Creation
result = client.generate_json(
    prompt=PromptTemplates.CREATE_INSTRUCTION.format(text="..."),
)

# 7. Quality Assessment
result = client.generate_json(
    prompt=PromptTemplates.ASSESS_QUALITY.format(text="..."),
)
```

## Model Management

### Installing Models

```bash
# Install Granite 4.0 (recommended)
ollama pull granite4

# Install Qwen 2.5 3B (lightweight)
ollama pull qwen2.5:3b

# Install Llama 3.1
ollama pull llama3.1

# Install Mixtral
ollama pull mixtral
```

### Checking Model Availability

```python
from saara.ollama_client import OllamaClient

client = OllamaClient({"model": "granite4"})

# Check if Ollama is running and model is available
if client.check_health():
    print("✅ Ollama is ready")

    # Get model info
    info = client.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['parameter_size']}")
    print(f"Family: {info['family']}")
    print(f"Quantization: {info['quantization']}")
else:
    print("❌ Ollama not available")
    print("Run: ollama pull granite4")
```

### Pipeline Health Check

```python
from saara import DataPipeline

pipeline = DataPipeline("config.yaml")

# Check if all services (Ollama, OCR) are healthy
if pipeline.check_health():
    print("✅ Pipeline ready to process documents")
else:
    print("❌ Fix issues above before proceeding")
```

## Remote Ollama Server

To use Ollama running on a different machine:

```python
config = {
    "ollama": {
        "base_url": "http://192.168.1.100:11434",  # Remote IP
        "model": "granite4",
        "timeout": 600  # Longer timeout for network latency
    }
}

pipeline = DataPipeline(config)
```

## Performance Tuning

### Temperature Settings

```python
# Deterministic/factual responses (data labeling, extraction)
response = client.generate(prompt="...", temperature=0.3)

# Balanced creativity (Q&A generation)
response = client.generate(prompt="...", temperature=0.7)

# Creative/diverse responses (brainstorming)
response = client.generate(prompt="...", temperature=1.0)
```

### Max Tokens

```python
# Short responses (classification)
response = client.generate(prompt="...", max_tokens=256)

# Medium responses (Q&A)
response = client.generate(prompt="...", max_tokens=1024)

# Long responses (essays, summaries)
response = client.generate(prompt="...", max_tokens=4096)
```

### Retry Strategy

```python
# For critical operations, increase retries
client = OllamaClient({
    "model": "granite4",
    "max_retries": 5,  # More retries
    "timeout": 600     # Longer timeout
})
```

## Integration with SAARA Modules

### 1. DataLabeler
```python
from saara import DataLabeler

labeler = DataLabeler({
    "ollama": {"model": "granite4"}
})

labeled_doc = labeler.label_document(extracted_doc)
```

### 2. DataPipeline
```python
from saara import DataPipeline

pipeline = DataPipeline({
    "ollama": {"model": "qwen2.5:3b"}
})

result = pipeline.process_document("document.pdf")
```

### 3. SyntheticDataGenerator
```python
from saara import SyntheticDataGenerator

generator = SyntheticDataGenerator({
    "ollama": {"model": "mixtral"}
})

data = generator.generate_instruction_data(count=100)
```

### 4. Translator
```python
from saara.translator import Translator
from saara.ollama_client import OllamaClient

client = OllamaClient({"model": "granite4"})
translator = Translator({}, client)

# Translate to Hindi
hindi_text = translator.translate(
    text="Machine learning is fascinating",
    target_lang="hi"
)
```

## Common Issues & Solutions

### Issue 1: Connection Refused
```
Error: Failed to connect to Ollama at http://localhost:11434
```

**Solution:**
```bash
# Start Ollama service
ollama serve

# Or check if it's already running
curl http://localhost:11434
```

### Issue 2: Model Not Found
```
Warning: Model granite4 not found. Available: [...]
```

**Solution:**
```bash
# Pull the model
ollama pull granite4

# List all installed models
ollama list
```

### Issue 3: Timeout Errors
```
Attempt 1 failed: Request timeout
```

**Solution:**
```python
# Increase timeout
config = {
    "ollama": {
        "model": "granite4",
        "timeout": 600,  # 10 minutes
        "max_retries": 5
    }
}
```

### Issue 4: Invalid JSON Responses
```
Failed to parse JSON: Expecting value
```

**Solution:**
The `generate_json()` method includes automatic JSON extraction and error handling. If issues persist:

```python
# Use lower temperature for more consistent JSON
response = client.generate_json(
    prompt="...",
    system_prompt="...",
    temperature=0.1  # Very deterministic
)
```

## Advanced Configuration

### Custom Generation Options

```python
from saara.ollama_client import OllamaClient

client = OllamaClient({"model": "granite4"})

# Access low-level client for custom options
response = client.client.chat(
    model="granite4",
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    options={
        "temperature": 0.7,
        "num_predict": 2048,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 4096,  # Context window
        "num_thread": 8    # CPU threads
    }
)
```

### Multi-Model Strategy

```python
# Use different models for different tasks
from saara.ollama_client import OllamaClient

# Fast model for classification
classifier = OllamaClient({"model": "qwen2.5:3b"})

# Smart model for Q&A generation
qa_generator = OllamaClient({"model": "granite4"})

# Large model for complex reasoning
reasoner = OllamaClient({"model": "mixtral"})
```

## Best Practices

1. **Start Ollama before running SAARA**
   ```bash
   ollama serve
   ```

2. **Use health checks**
   ```python
   pipeline = DataPipeline("config.yaml")
   assert pipeline.check_health(), "Pipeline not ready"
   ```

3. **Choose appropriate models**
   - Labeling/Q&A: `granite4`
   - Speed-critical: `qwen2.5:3b`
   - Complex reasoning: `mixtral`

4. **Configure timeouts for large documents**
   ```python
   config = {"ollama": {"timeout": 600}}  # 10 min for large PDFs
   ```

5. **Use structured JSON for reliable parsing**
   ```python
   result = client.generate_json(prompt, schema={...})
   ```

## Reference

### Full Configuration Example

```yaml
# Complete config.yaml with all Ollama options
ollama:
  base_url: http://localhost:11434
  model: granite4
  timeout: 300
  max_retries: 3

hardware:
  ram_gb: 16
  tier: medium
  vram_gb: 8

pdf:
  ocr_engine: moondream
  dpi: 200
  max_pages: 1000

text:
  chunk_size: 512
  overlap: 50
  strategy: semantic

output:
  directory: datasets
  formats:
    - jsonl
    - csv

labeling:
  use_ollama: true
  include_metadata: true
  quality_threshold: 0.7

logging:
  level: INFO
  file: logs/saara.log
```

## Credits

Ollama Integration Module
© 2025-2026 Kilani Sai Nikhil
Part of SAARA AI SDK
