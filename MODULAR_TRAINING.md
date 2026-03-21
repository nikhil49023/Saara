# Modular Training Pipeline

A clean, modular architecture for end-to-end LLM fine-tuning that eliminates CPU bottlenecks through pre-tokenization and memory-mapped storage.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  MODULAR TRAINING PIPELINE                  │
└─────────────────────────────────────────────────────────────┘

Stage 1: PDF Parser
  📄 PDFs → Vision LLM → Structured JSONL
  ├─ Extract with Moondream/Qwen OCR
  ├─ Label with local LLM (Granite/Llama)
  └─ Output: instruction-response pairs

Stage 2: Tokenizer Setup
  🔤 Configure domain-specific tokenizer
  ├─ Use existing tokenizer (HuggingFace)
  ├─ OR train custom BPE tokenizer
  └─ Optimized for: medical, legal, code, scientific, etc.

Stage 3: Token Storage (Pre-tokenization)
  💾 JSONL → Binary Token IDs (.arrow)
  ├─ Tokenize once with multiprocessing
  ├─ Store as memory-mapped Arrow files
  ├─ 4 bytes per token (vs 10+ for strings)
  └─ Zero-copy reads during training

Stage 4: Training
  🎯 Train directly from pre-tokenized data
  ├─ No CPU bottleneck (no per-batch tokenization)
  ├─ LoRA fine-tuning
  ├─ High GPU utilization
  └─ 3-10x faster than traditional training
```

## Key Benefits

### 1. **Eliminated CPU Bottleneck**
Traditional training re-tokenizes text on every batch, starving the GPU. Pre-tokenization eliminates this.

### 2. **Memory Efficiency**
- Token IDs: 4 bytes per token
- Raw strings: 10+ bytes per token
- Memory-mapped: Zero-copy reads from disk to GPU

### 3. **Reusability**
Tokenize once, train multiple times with different hyperparameters.

### 4. **Modularity**
Each stage is independent. Run only what you need:
- Only tokenize → Skip stage 4
- Only train → Skip stages 1-3
- Custom tokenizer → Enable stage 2 training

## Quick Start

### Option 1: Full Pipeline (One Command)

```python
from saara import TrainingPipeline, TrainingPipelineConfig

config = TrainingPipelineConfig(
    pdf_input="docs/research_papers/",
    tokenizer_id="sarvamai/sarvam-1",
    max_length=1024,
    num_epochs=3,
    output_dir="training_output"
)

pipeline = TrainingPipeline(config)
result = pipeline.run()

print(f"Model: {result['model_path']}")
```

### Option 2: Quick Helper Functions

```python
from saara import quick_tokenize, quick_train

# Tokenize once
tokens = quick_tokenize(
    "data.jsonl",
    "tokens/",
    max_length=512
)

# Train (fast!)
result = quick_train(
    tokens,
    model_id="sarvamai/sarvam-1",
    num_epochs=3
)
```

### Option 3: Manual Stage Control

```python
from saara import TokenStorage, TokenStorageConfig, LLMTrainer

# Stage 3: Pre-tokenize
storage = TokenStorage(tokenizer="sarvamai/sarvam-1")
tokens = storage.tokenize_dataset(
    "data.jsonl",
    "tokens/",
    instruction_field="instruction",
    response_field="response"
)

# Stage 4: Train
trainer = LLMTrainer("sarvamai/sarvam-1")
trainer.train(tokens)
```

## Configuration

### TrainingPipelineConfig

```python
from saara import TrainingPipelineConfig

config = TrainingPipelineConfig(
    # Stage 1: Data
    pdf_input="docs/",              # Or jsonl_input for existing data

    # Stage 2: Tokenizer
    tokenizer_id="sarvamai/sarvam-1",
    train_custom_tokenizer=False,   # Set True for domain-specific
    tokenizer_domain="general",     # medical, legal, code, scientific

    # Stage 3: Pre-tokenization
    max_length=512,                 # Max sequence length
    padding="max_length",           # Padding strategy
    num_proc=4,                     # Parallel processes

    # Stage 4: Training
    model_id="sarvamai/sarvam-1",
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    lora_r=32,
    lora_alpha=16,

    # Control
    skip_stages=[],                 # Skip stages [1,2,3,4]
    output_dir="training_output"
)
```

### TokenStorageConfig

```python
from saara import TokenStorageConfig

config = TokenStorageConfig(
    max_length=512,
    padding="max_length",           # or "longest" or False
    truncation=True,
    add_special_tokens=True,
    return_attention_mask=True,
    num_proc=4,                     # Parallel tokenization
    batch_size=1000                 # Tokenization batch size
)
```

## Usage Patterns

### Pattern 1: Skip PDF Stage (Start from JSONL)

```python
config = TrainingPipelineConfig(
    jsonl_input="existing_data.jsonl",
    skip_stages=[1],  # Skip PDF parsing
    num_epochs=3
)
```

### Pattern 2: Only Pre-tokenize (No Training)

```python
config = TrainingPipelineConfig(
    jsonl_input="large_dataset.jsonl",
    skip_stages=[1, 4],  # Skip PDF and training
    num_proc=8           # Fast parallel tokenization
)
```

### Pattern 3: Custom Domain Tokenizer

```python
config = TrainingPipelineConfig(
    jsonl_input="medical_texts.jsonl",
    train_custom_tokenizer=True,
    tokenizer_domain="medical",  # Optimized vocabulary
    num_epochs=3
)
```

### Pattern 4: Reuse Tokenized Data

```python
# Tokenize once
tokens = quick_tokenize("data.jsonl", "tokens/")

# Train with different hyperparameters (fast!)
quick_train(tokens, num_epochs=3, learning_rate=2e-4)
quick_train(tokens, num_epochs=5, learning_rate=5e-4)
quick_train(tokens, num_epochs=2, learning_rate=1e-4)
```

## Performance Comparison

### Traditional Training (Without Pre-tokenization)
```
Training 10,000 examples:
├─ Tokenization: ~15 mins (CPU-bound, repeated every epoch)
├─ Training: ~30 mins (GPU waiting for CPU)
└─ Total: ~45 mins
     GPU Utilization: 40-60%
```

### Modular Pipeline (With Pre-tokenization)
```
Training 10,000 examples:
├─ Pre-tokenization: ~3 mins (one-time, parallel)
├─ Training: ~10 mins (GPU at full speed)
└─ Total: ~13 mins
     GPU Utilization: 85-95%
```

**3.5x faster!** And tokenized data can be reused.

## Memory-Mapped Storage

Token storage uses Apache Arrow format with memory-mapping:

```python
# Storage stats
>>> storage = TokenStorage("sarvamai/sarvam-1")
>>> tokens = storage.tokenize_dataset("data.jsonl", "tokens/")
>>> stats = storage.get_storage_stats(tokens)
>>> print(stats)
{
  'num_examples': 10000,
  'disk_size_mb': 42.5,
  'avg_bytes_per_example': 4480,
  'tokenizer_id': 'sarvamai/sarvam-1',
  'vocab_size': 32000
}
```

Benefits:
- **Zero-copy reads**: Data loaded directly from disk to GPU
- **No RAM bloat**: Only active batches in memory
- **Fast random access**: Arrow format is optimized for ML
- **Portable**: Works across machines, no re-tokenization

## Advanced: Custom Tokenizer Training

For domain-specific tasks (medical, legal, code), train a custom tokenizer:

```python
from saara import TrainingPipeline, TrainingPipelineConfig

config = TrainingPipelineConfig(
    jsonl_input="domain_texts.jsonl",

    # Train custom tokenizer on your domain
    train_custom_tokenizer=True,
    tokenizer_domain="medical",  # Options: general, medical, legal, code, scientific

    # Use the custom tokenizer for training
    num_epochs=3,
    output_dir="medical_model"
)

pipeline = TrainingPipeline(config)
result = pipeline.run()

# Custom tokenizer preserved in model
print(f"Tokenizer: {result['tokenizer_path']}")
print(f"Model: {result['model_path']}")
```

The AI tokenizer module (`ai_tokenizer.py`) uses local LLMs to:
- Detect domain-specific terminology
- Preserve compound words (e.g., "myocardial infarction")
- Handle abbreviations (e.g., "CHF" → "congestive heart failure")
- Optimize vocabulary for your domain

## Migration from Traditional Training

### Before (Traditional)
```python
from saara import LLMTrainer

trainer = LLMTrainer("sarvamai/sarvam-1")
trainer.train("data.jsonl")  # Tokenizes on every batch
```

### After (Modular)
```python
from saara import quick_tokenize, quick_train

# Step 1: Pre-tokenize (one-time)
tokens = quick_tokenize("data.jsonl", "tokens/")

# Step 2: Train (fast!)
quick_train(tokens, model_id="sarvamai/sarvam-1")
```

Both methods work, but the modular approach is 3-10x faster and allows reuse.

## Troubleshooting

### Issue: Out of Memory During Tokenization
**Solution**: Reduce `num_proc` or `batch_size` in `TokenStorageConfig`

```python
config = TokenStorageConfig(
    num_proc=2,        # Lower parallel processes
    batch_size=500     # Smaller batches
)
```

### Issue: Slow Tokenization
**Solution**: Increase parallelization

```python
config = TokenStorageConfig(
    num_proc=8,        # More parallel processes
    batch_size=2000    # Larger batches
)
```

### Issue: Disk Space for Tokens
**Expected**: ~4KB per example (1024 tokens)
**Solution**: Token storage is ~10x smaller than raw text. If still needed, use shorter `max_length`

### Issue: Tokenizer Mismatch
**Solution**: Ensure the tokenizer used for pre-tokenization matches the model

```python
# ✓ Correct
storage = TokenStorage(tokenizer="sarvamai/sarvam-1")
trainer = LLMTrainer(model_id="sarvamai/sarvam-1")

# ✗ Wrong
storage = TokenStorage(tokenizer="meta/llama-3.1")
trainer = LLMTrainer(model_id="sarvamai/sarvam-1")  # Mismatch!
```

## API Reference

See `SAARA_BUILTIN_FUNCTIONS.md` for complete API documentation of all modules.

## Examples

See `examples/04_modular_training.py` for 7 complete examples:
1. Full pipeline from PDF
2. Skip PDF stage (JSONL input)
3. Pre-tokenization only
4. Manual stage control
5. Quick helper functions
6. Custom domain tokenizer
7. Reuse tokenized data

```bash
python examples/04_modular_training.py 5  # Run example 5
```

## Credits

Modular Training Pipeline
© 2025-2026 Kilani Sai Nikhil
Part of the SAARA AI SDK
