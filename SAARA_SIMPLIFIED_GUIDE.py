"""
SAARA: Simple, Quick, and Efficient for Any Device

The new SAARA is designed to be:
✅ Simple - Just a few lines of code
✅ Quick - Fast inference with any LLM provider
✅ Efficient - Works on laptops, phones, cloud, anything
✅ Flexible - Multiple tokenizers, multiple providers
✅ Manual - You control file loading/saving

No automatic caching, no magic, no bloat. Pure Python.

Released under the MIT License.
"""

# =============================================================================
# INSTALLATION
# =============================================================================

INSTALLATION = """
pip install saara-ai

That's it! No CLI dependencies, pure library.

Optional: For specific features:
  - LLM providers: pip install anthropic google-generativeai openai
  - File formats: pip install pyarrow pandas
  - Everything: pip install saara-ai[all]
"""


# =============================================================================
# 5-MINUTE QUICK START
# =============================================================================

QUICKSTART_5MIN = """
from saara.quickstart import ollama_local, QuickTokenizer, QuickDataset
from saara.file_utils import save_jsonl

# 1. Use local LLM (Ollama - free, no API key)
llm = ollama_local("granite3.1-dense:8b")
response = llm.generate("Explain AI in one sentence")

# 2. Train a tokenizer
tokenizer = QuickTokenizer("bpe", vocab_size=32000)
tokenizer.train(["hello world", "how are you"])
tokens = tokenizer.encode("hello world")

# 3. Load and save data
from saara.file_utils import load_jsonl, save_jsonl
data = load_jsonl("data.jsonl")
save_jsonl(data, "output.jsonl")

That's it! 🎉
"""


# =============================================================================
# DEVICE-SPECIFIC SETUP
# =============================================================================

DEVICE_SETUP = """
┌─────────────────────────────────────────────────────────────────┐
│                  CHOOSE YOUR DEVICE & SETUP                     │
└─────────────────────────────────────────────────────────────────┘

🖥️  LAPTOP/DESKTOP (No API keys needed!)
────────────────────────────────────────
Option 1: Local Ollama (Recommended for beginners)
  1. Install Ollama: https://ollama.ai
  2. Run: ollama run granite3.1-dense:8b
  3. Use in Python:
     from saara.quickstart import ollama_local
     llm = ollama_local("granite3.1-dense:8b")
     response = llm.generate("Hello!")

☁️  GOOGLE COLAB / JUPYTER
──────────────────────────
  1. Get Gemini API key: https://aistudio.google.com/app/apikeys
  2. In your notebook:
     import os
     os.environ["GEMINI_API_KEY"] = "your-key-here"

     from saara.quickstart import gemini_api
     llm = gemini_api(os.environ["GEMINI_API_KEY"])
     llm.generate("Hello!")

🔗 OTHER APIS
─────────────
OpenAI ChatGPT:
  from saara.quickstart import openai_api
  llm = openai_api("sk-...", model="gpt-4")

Claude (Anthropic):
  from saara.quickstart import claude_api
  llm = claude_api("sk-ant-...", model="claude-opus-4-6")

NVIDIA Nemotron:
  from saara.quickstart import nemotron_api
  llm = nemotron_api("nvapi-...")

📱 MOBILE / EDGE DEVICES
─────────────────────────
For on-device inference, use quantized models:
  ollm = ollama_local("llama2-uncensored:7b-q4")

Or use byte tokenizer for any language/encoding:
  from saara.quickstart import QuickTokenizer
  tok = QuickTokenizer("byte")
  tokens = tok.encode("anything")
"""


# =============================================================================
# LLM PROVIDERS
# =============================================================================

LLM_PROVIDERS_GUIDE = """
┌─────────────────────────────────────────────────────────────────┐
│              SUPPORTED LLM PROVIDERS                             │
└─────────────────────────────────────────────────────────────────┘

All providers work with the same simple interface:

from saara.quickstart import QuickLLM

# Create LLM
llm = QuickLLM(
    provider="ollama",  # or "gemini", "openai", "anthropic", "nemotron"
    model="granite3.1-dense:8b",
    temperature=0.7,
    max_tokens=2048
)

# Generate text
response = llm.generate("Your prompt here")

# Check if available
if llm.is_available():
    print("Ready to use!")


PROVIDERS:

1️⃣  OLLAMA (Local, Free, No API key)
   Provider: "ollama"
   Models: llama2, granite, mistral, neural-chat, etc.
   Cost: FREE
   Speed: Fast (depends on your GPU)
   Setup: ollama run granite3.1-dense:8b

2️⃣  GEMINI (Google)
   Provider: "gemini"
   Models: gemini-2.0-flash, gemini-1.5-pro, etc.
   Cost: Free tier available
   Speed: Fast
   Setup: export GEMINI_API_KEY="..."

3️⃣  OPENAI (ChatGPT)
   Provider: "openai"
   Models: gpt-4, gpt-3.5-turbo, etc.
   Cost: Paid
   Speed: Very fast
   Setup: export OPENAI_API_KEY="sk-..."

4️⃣  ANTHROPIC (Claude)
   Provider: "anthropic"
   Models: claude-opus-4-6, claude-sonnet, etc.
   Cost: Paid
   Speed: Very fast
   Setup: export ANTHROPIC_API_KEY="sk-ant-..."

5️⃣  NEMOTRON (NVIDIA)
   Provider: "nemotron"
   Cost: Free tier available
   Speed: Fast
   Setup: export NEMOTRON_API_KEY="nvapi-..."

6️⃣  GROQ (Fastest Inference)
   Provider: "groq"
   Cost: Free tier
   Speed: Extremely fast
   Setup: export GROQ_API_KEY="..."


EXAMPLE: Using Multiple Providers

# Free local option
local_llm = ollama_local("mistral:7b")

# Cloud backup
cloud_llm = gemini_api("your-key")

# Automatic fallback
try:
    response = local_llm.generate("prompt")
except:
    response = cloud_llm.generate("prompt")
"""


# =============================================================================
# TOKENIZERS
# =============================================================================

TOKENIZERS_GUIDE = """
┌─────────────────────────────────────────────────────────────────┐
│              FLEXIBLE TOKENIZER SYSTEM                           │
└─────────────────────────────────────────────────────────────────┘

Three built-in tokenizers:

1️⃣  BPE (Byte Pair Encoding) - DEFAULT
   ├─ Fast, efficient
   ├─ Good compression ratio
   ├─ Works for most languages
   └─ Example:
      from saara.quickstart import QuickTokenizer
      tok = QuickTokenizer("bpe", vocab_size=32000)
      tok.train(texts)
      tokens = tok.encode("hello world")

2️⃣  WORDPIECE
   ├─ Used by BERT
   ├─ Great for English
   ├─ Lower compression than BPE
   └─ Example:
      tok = QuickTokenizer("wordpiece", vocab_size=30522)
      tok.train(texts)

3️⃣  BYTE-LEVEL
   ├─ Universal - works with ANY text/encoding
   ├─ Supports emojis, all languages
   ├─ More tokens but flexible
   └─ Example:
      tok = QuickTokenizer("byte")
      tok.train([])  # No training needed
      tokens = tok.encode("Hello 你好 مرحبا 🚀")


SAVING & LOADING:

# Save
tok.save("my_tokenizer_dir")

# Load later
loaded_tok = QuickTokenizer.load("my_tokenizer_dir", tokenizer_type="bpe")


BRING YOUR OWN TOKENIZER:

If you have a custom tokenizer:

from saara.tokenizers import TokenizerRegistry

class MyTokenizer(BaseTokenizer):
    def train(self, texts, vocab_size): ...
    def encode(self, text): ...
    def decode(self, tokens): ...
    def save(self, directory): ...
    @classmethod
    def load(cls, directory): ...

TokenizerRegistry.register("my_tokenizer", MyTokenizer)
tok = QuickTokenizer("my_tokenizer")
"""


# =============================================================================
# FILE HANDLING
# =============================================================================

FILE_HANDLING_GUIDE = """
┌─────────────────────────────────────────────────────────────────┐
│              SIMPLE FILE HANDLING (MANUAL)                       │
└─────────────────────────────────────────────────────────────────┘

You have full control - load and save data how you want!

LOADING DATA:

from saara.file_utils import load_from_file

# Auto-detect format
data = load_from_file("data.jsonl")        # JSONL
data = load_from_file("data.json")         # JSON
data = load_from_file("data.csv")          # CSV
data = load_from_file("data.txt")          # Text
data = load_from_file("data.parquet")      # Parquet

# Or use specific functions
from saara.file_utils import (
    load_jsonl, load_json, load_csv,
    load_text, load_texts
)

records = load_jsonl("data.jsonl")
texts = load_texts("data.txt")  # One line per text


SAVING DATA:

from saara.file_utils import save_to_file

save_to_file(data, "output.jsonl")         # JSONL
save_to_file(data, "output.json")          # JSON
save_to_file(data, "output.csv")           # CSV
save_to_file(text, "output.txt")           # Text

# Or use specific functions
from saara.file_utils import (
    save_jsonl, save_json, save_csv,
    save_text, save_texts
)

save_jsonl(records, "output.jsonl")


PROCESSING DATA:

from saara.file_utils import extract_texts, split_dataset

# Extract specific field
texts = extract_texts(records, "text")

# Split for training
train, val, test = split_dataset(records, train_ratio=0.8, val_ratio=0.1)


SUPPORTED FORMATS:

✓ Text (.txt)
✓ JSON (.json)
✓ JSONL (.jsonl) - One JSON per line, great for large datasets
✓ CSV (.csv, .tsv)
✓ Parquet (.parquet)
✓ Custom formats (bring your own loader)
"""


# =============================================================================
# COMPLETE WORKFLOWS
# =============================================================================

WORKFLOWS = """
┌─────────────────────────────────────────────────────────────────┐
│              COMMON WORKFLOWS                                    │
└─────────────────────────────────────────────────────────────────┘

WORKFLOW 1: Tokenize texts and save
──────────────────────────────────

from saara.file_utils import load_texts, save_jsonl
from saara.quickstart import QuickTokenizer

# 1. Load texts
texts = load_texts("corpus.txt")

# 2. Train tokenizer
tok = QuickTokenizer("bpe", vocab_size=32000)
tok.train(texts)
tok.save("my_tokenizer")

# 3. Tokenize and save
output = [
    {"text": t, "tokens": tok.encode(t)}
    for t in texts
]
save_jsonl(output, "tokenized.jsonl")


WORKFLOW 2: Generate synthetic data with LLM
──────────────────────────────────────────────

from saara.quickstart import ollama_local
from saara.file_utils import save_jsonl

llm = ollama_local()

prompts = [
    "Generate a customer support Q&A",
    "Generate a product review",
    "Generate a tech article title"
]

output = []
for prompt in prompts:
    response = llm.generate(prompt)
    output.append({"prompt": prompt, "response": response})

save_jsonl(output, "generated.jsonl")


WORKFLOW 3: Fine-tune a model
──────────────────────────────

from saara.file_utils import load_jsonl, save_jsonl
from saara.quickstart import QuickFineTune

# 1. Load training data
data = load_jsonl("training_data.jsonl")

# 2. Fine-tune
trainer = QuickFineTune(
    model_id="TinyLlama/TinyLlama-1.1B",
    output_dir="./my_model",
    learning_rate=2e-4
)
trainer.train(data, num_epochs=3)

# 3. Model saved to ./my_model


WORKFLOW 4: Complete pipeline
──────────────────────────────

from saara.quickstart import QuickDataset, QuickTokenizer, ollama_local
from saara.file_utils import save_jsonl

# 1. Load data
dataset = QuickDataset.from_file("raw_data.jsonl")

# 2. Extract and process
texts = dataset.get_texts("text")

# 3. Train tokenizer
tok = QuickTokenizer("bpe")
tok.train(texts)

# 4. Use LLM
llm = ollama_local()

# 5. Augment and save
output = []
for record in dataset.records:
    result = {
        **record,
        "tokens": tok.encode(record["text"])[:50],
        "summary": llm.generate(f"Summarize: {record['text']}")[:100]
    }
    output.append(result)

save_jsonl(output, "augmented.jsonl")
"""


# =============================================================================
# PERFORMANCE TIPS
# =============================================================================

PERFORMANCE_TIPS = """
┌─────────────────────────────────────────────────────────────────┐
│              PERFORMANCE & EFFICIENCY TIPS                       │
└─────────────────────────────────────────────────────────────────┘

FOR FASTER TOKENIZATION:
├─ Use BPE for good speed/compression balance
├─ Use byte tokenizer for simplicity
└─ Limit training corpus size for quick training

FOR FASTER LLM INFERENCE:
├─ Use Ollama with quantized models (4-bit, 8-bit)
├─ Use smaller models: llama2:7b instead of llama2:70b
├─ Cache results to avoid re-generating
└─ Use batch generation for multiple prompts

FOR REDUCED MEMORY:
├─ Process data in chunks instead of loading all at once
├─ Use smaller tokenizer vocab sizes (8000-16000)
├─ Use byte tokenizer instead of BPE for flexibility
└─ Don't keep all tokens in memory

FOR MOBILE/EDGE:
├─ Use quantized models (4-bit recommended)
├─ Use byte tokenizer (simpler, faster)
├─ Process one example at a time
└─ Save results to file instead of memory


BENCHMARK (Approximate):

BPE Tokenizer:
  Train: 1000 texts → 2-5 seconds
  Encode: 1000 texts → 100-200ms

LLM Generation (Ollama, GPU):
  Granite 8B: 50-100 tokens/sec
  Mistral 7B: 100-150 tokens/sec

LLM Generation (Ollama, CPU):
  Tiny Llama: 10-20 tokens/sec

File I/O:
  Load 100K JSONLs: 100-500ms
  Save 100K JSONLs: 200-800ms
"""


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

TROUBLESHOOTING = """
┌─────────────────────────────────────────────────────────────────┐
│              TROUBLESHOOTING                                     │
└─────────────────────────────────────────────────────────────────┘

Q: "Ollama connection refused"
A: Start Ollama first:
   ollama run granite3.1-dense:8b

Q: "API key not found"
A: Set environment variable:
   export GEMINI_API_KEY="your-key"
   export OPENAI_API_KEY="sk-..."

   Or pass directly:
   llm = gemini_api("your-key")

Q: "ImportError for transformers/torch"
A: These are only needed for full training
   pip install torch transformers

Q: "Tokenizer needs more vocab"
A: Increase vocab_size:
   tok = QuickTokenizer("bpe", vocab_size=50000)

Q: "Generation is too slow"
A: Use smaller model or GPU:
   llm = ollama_local("llama2:7b-q4")  # Quantized

   Or use cloud API (faster servers):
   llm = gemini_api("key")

Q: "Out of memory"
A: Process in smaller chunks:
   for batch in batches(records, 100):
       process(batch)

Q: "File format not supported"
A: Convert to JSONL first, then use:
   save_jsonl(data, "converted.jsonl")
"""


# =============================================================================
# MAIN GUIDE
# =============================================================================

if __name__ == "__main__":
    sections = [
        ("Installation", INSTALLATION),
        ("5-Minute Quickstart", QUICKSTART_5MIN),
        ("Device Setup", DEVICE_SETUP),
        ("LLM Providers", LLM_PROVIDERS_GUIDE),
        ("Tokenizers", TOKENIZERS_GUIDE),
        ("File Handling", FILE_HANDLING_GUIDE),
        ("Workflows", WORKFLOWS),
        ("Performance Tips", PERFORMANCE_TIPS),
        ("Troubleshooting", TROUBLESHOOTING),
    ]

    for title, content in sections:
        print(f"\n{'='*60}")
        print(f"{title}".center(60))
        print('='*60)
        print(content)

    print("\n" + "="*60)
    print("Ready to use SAARA! 🚀".center(60))
    print("="*60)
