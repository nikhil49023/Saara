"""
SAARA Quickstart Examples - The Simplest Way to Get Started

Choose your device type and run the examples!
"""

# =============================================================================
# EXAMPLE 1: Laptop/Desktop (Local Ollama - No API keys needed!)
# =============================================================================

def example_local_laptop():
    """
    Simple example: Run on local laptop with Ollama

    Prerequisites:
        1. Install Ollama: https://ollama.ai
        2. Run: ollama run granite3.1-dense:8b
        3. Then run this script

    This is the simplest way - NO API KEYS NEEDED!
    """
    from saara.quickstart import QuickLLM, QuickTokenizer, QuickDataset
    from saara.file_utils import save_jsonl

    # Step 1: Create local LLM (no API key needed!)
    print("📱 Connecting to local Ollama...")
    llm = QuickLLM("ollama", model="granite3.1-dense:8b")

    if not llm.is_available():
        print("❌ Ollama not running. Start it with: ollama run granite3.1-dense:8b")
        return

    # Step 2: Generate text
    print("✨ Generating text...")
    result = llm.generate(
        "Explain machine learning in simple terms for a 10-year-old"
    )
    print(f"Result:\n{result}\n")

    # Step 3: Create tokenizer
    print("🔤 Training tokenizer...")
    texts = [
        "Machine learning is when computers learn from examples",
        "Deep learning uses neural networks to find patterns",
        "Transformers are a type of deep learning model",
    ]

    tokenizer = QuickTokenizer("bpe", vocab_size=1000)
    tokenizer.train(texts)

    # Step 4: Tokenize and see results
    sample_text = "machine learning"
    tokens = tokenizer.encode(sample_text)
    print(f"Text: '{sample_text}' → Tokens: {tokens[:5]}\n")

    # Step 5: Save results
    output = [
        {
            "text": text,
            "tokens": tokenizer.encode(text)[:20],
            "llm_explanation": llm.generate(f"Explain: {text}")[:100]
        }
        for text in texts[:2]
    ]
    save_jsonl(output, "results_local.jsonl")
    print("✅ Results saved to results_local.jsonl")


# =============================================================================
# EXAMPLE 2: Cloud (Jupyter/Colab with Gemini API)
# =============================================================================

def example_cloud_gemini():
    """
    Example for Google Colab/Jupyter with Gemini

    Prerequisites:
        1. Get Gemini API key: https://aistudio.google.com/app/apikeys
        2. Set environment variable: export GEMINI_API_KEY="your-key"
        3. Run this script

    Copy-paste this in a Colab cell!
    """
    import os
    from saara.quickstart import gemini_api, QuickTokenizer, QuickDataset

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Set GEMINI_API_KEY environment variable")
        return

    # Create LLM
    print("🌐 Connecting to Gemini API...")
    llm = gemini_api(api_key, model="gemini-2.0-flash")

    # Generate text
    prompt = """You are a data scientist. Generate 3 realistic dataset descriptions
    in JSON format. Each should have: name, rows, columns, domain"""

    result = llm.generate(prompt)
    print(f"Generated:\n{result}\n")

    # Train tokenizer
    texts = [
        "Customer transaction data with 10000 rows",
        "Medical records from hospital systems",
        "Social media engagement metrics"
    ]

    tokenizer = QuickTokenizer("wordpiece", vocab_size=5000)
    tokenizer.train(texts)
    tokenizer.save("gemini_tokenizer")
    print("✅ Tokenizer saved!")


# =============================================================================
# EXAMPLE 3: OpenAI ChatGPT
# =============================================================================

def example_openai():
    """
    Example using OpenAI ChatGPT

    Prerequisites:
        1. Get OpenAI API key: https://platform.openai.com/api-keys
        2. Set: export OPENAI_API_KEY="sk-..."
        3. Run this script
    """
    import os
    from saara.quickstart import openai_api, QuickDataset

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable")
        return

    print("🤖 Using OpenAI ChatGPT...")
    llm = openai_api(api_key, model="gpt-4")

    # Generate training data
    result = llm.generate("Generate 3 Q&A pairs about Python")
    print(f"Generated:\n{result}")


# =============================================================================
# EXAMPLE 4: Anthropic Claude
# =============================================================================

def example_claude():
    """
    Example using Anthropic Claude

    Prerequisites:
        1. Get Claude API key: https://console.anthropic.com/
        2. Set: export ANTHROPIC_API_KEY="sk-ant-..."
        3. Run this script
    """
    import os
    from saara.quickstart import claude_api

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Set ANTHROPIC_API_KEY environment variable")
        return

    print("🧠 Using Claude...")
    llm = claude_api(api_key, model="claude-opus-4-6")

    result = llm.generate("Explain why Python is popular for AI/ML")
    print(f"Claude says:\n{result}")


# =============================================================================
# EXAMPLE 5: NVIDIA Nemotron
# =============================================================================

def example_nemotron():
    """
    Example using NVIDIA Nemotron

    Prerequisites:
        1. Get Nemotron API key: https://build.nvidia.com/
        2. Set: export NEMOTRON_API_KEY="nvapi-..."
        3. Run this script
    """
    import os
    from saara.quickstart import nemotron_api

    api_key = os.getenv("NEMOTRON_API_KEY")
    if not api_key:
        print("❌ Set NEMOTRON_API_KEY environment variable")
        return

    print("🚀 Using NVIDIA Nemotron...")
    llm = nemotron_api(api_key)

    result = llm.generate("What are the latest trends in AI?")
    print(f"Nemotron response:\n{result}")


# =============================================================================
# EXAMPLE 6: Complete Pipeline - Load Data → Process → LLM
# =============================================================================

def example_complete_pipeline():
    """
    Complete pipeline example

    This shows the end-to-end workflow:
    1. Load data from file
    2. Process it
    3. Augment with LLM
    4. Save results
    """
    from saara.quickstart import QuickDataset, QuickTokenizer, ollama_local
    from saara.file_utils import save_jsonl

    # Step 1: Create sample data
    sample_data = [
        {"id": 1, "text": "Machine learning is fascinating", "category": "ai"},
        {"id": 2, "text": "Deep learning uses neural networks", "category": "ai"},
        {"id": 3, "text": "Natural language processing helps understand text", "category": "nlp"},
    ]
    save_jsonl(sample_data, "input_data.jsonl")

    # Step 2: Load with QuickDataset
    print("📂 Loading dataset...")
    dataset = QuickDataset.from_file("input_data.jsonl")
    print(f"Loaded {len(dataset)} records")

    # Step 3: Extract texts
    texts = dataset.get_texts("text")
    print(f"Texts: {texts}\n")

    # Step 4: Train tokenizer
    print("🔤 Training tokenizer...")
    tokenizer = QuickTokenizer("bpe", vocab_size=2000)
    tokenizer.train(texts)

    # Step 5: Use LLM to augment
    print("🤖 Augmenting with LLM...")
    llm = ollama_local("granite3.1-dense:8b")

    output = []
    for record in dataset.records[:2]:
        text = record["text"]
        tokens = tokenizer.encode(text)
        summary = llm.generate(f"Summarize in one sentence: {text}")

        output.append({
            **record,
            "tokens": tokens[:30],
            "summary": summary[:100]
        })

    # Step 6: Save results
    save_jsonl(output, "output_augmented.jsonl")
    print("✅ Results saved to output_augmented.jsonl")


# =============================================================================
# EXAMPLE 7: Fine-tuning (simple)
# =============================================================================

def example_finetuning():
    """
    Simple fine-tuning example

    Prerequisites:
        - 4GB GPU (GPU optional, CPU works slow)
        - Training data in format: [{"prompt": "...", "response": "..."}]
    """
    from saara.quickstart import QuickFineTune
    from saara.file_utils import save_jsonl

    # Create sample training data
    training_data = [
        {
            "prompt": "What is AI?",
            "response": "AI is artificial intelligence - computer systems that perform tasks"
        },
        {
            "prompt": "What is ML?",
            "response": "ML is machine learning - systems that learn from data"
        },
    ]

    # Save training data
    save_jsonl(training_data, "train_data.jsonl")

    # Fine-tune
    print("🎓 Starting fine-tuning...")
    trainer = QuickFineTune(
        model_id="TinyLlama/TinyLlama-1.1B",
        output_dir="./my_finetuned_model",
        learning_rate=2e-4,
        batch_size=1
    )

    trainer.train(training_data, num_epochs=1, max_seq_length=256)
    print("✅ Fine-tuning complete! Model saved to ./my_finetuned_model")


# =============================================================================
# EXAMPLE 8: Multi-Tokenizer Comparison
# =============================================================================

def example_tokenizer_comparison():
    """
    Compare different tokenizers
    """
    from saara.quickstart import QuickTokenizer

    text = "Hello world! How are you? 你好 مرحبا 🚀"

    print(f"Original text: {text}\n")

    # BPE
    print("BPE Tokenizer:")
    tok_bpe = QuickTokenizer("bpe", vocab_size=5000)
    tok_bpe.train([text] * 10)
    tokens_bpe = tok_bpe.encode(text)
    print(f"  Tokens: {tokens_bpe[:20]}")
    print(f"  Count: {len(tokens_bpe)}\n")

    # WordPiece
    print("WordPiece Tokenizer:")
    tok_wp = QuickTokenizer("wordpiece", vocab_size=3000)
    tok_wp.train([text] * 10)
    tokens_wp = tok_wp.encode(text)
    print(f"  Tokens: {tokens_wp[:20]}")
    print(f"  Count: {len(tokens_wp)}\n")

    # Byte-level
    print("Byte-level Tokenizer:")
    tok_byte = QuickTokenizer("byte")
    tok_byte.train([text])
    tokens_byte = tok_byte.encode(text)
    print(f"  Tokens: {tokens_byte[:20]}")
    print(f"  Count: {len(tokens_byte)}\n")

    print("Summary:")
    print(f"  BPE: {len(tokens_bpe)} tokens")
    print(f"  WordPiece: {len(tokens_wp)} tokens")
    print(f"  Byte: {len(tokens_byte)} tokens")


# =============================================================================
# EXAMPLE 9: Data Processing Workflow
# =============================================================================

def example_data_processing():
    """
    Complete data processing workflow
    """
    from saara.file_utils import (
        load_jsonl, save_jsonl, extract_texts,
        extract_training_pairs, split_dataset
    )

    # Create sample data
    data = [
        {"prompt": "What is AI?", "response": "AI is artificial intelligence"},
        {"prompt": "What is ML?", "response": "ML is machine learning"},
        {"prompt": "What is DL?", "response": "DL is deep learning"},
    ]
    save_jsonl(data, "raw_data.jsonl")

    # Load data
    print("📂 Loading data...")
    records = load_jsonl("raw_data.jsonl")
    print(f"Loaded {len(records)} records\n")

    # Extract training pairs
    print("🔗 Extracting training pairs...")
    pairs = extract_training_pairs(records)
    print(f"Got {len(pairs)} pairs\n")

    # Split dataset
    print("✂️  Splitting dataset...")
    train, val, test = split_dataset(records, train_ratio=0.6, val_ratio=0.2)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}\n")

    # Save splits
    save_jsonl(train, "train.jsonl")
    save_jsonl(val, "val.jsonl")
    save_jsonl(test, "test.jsonl")
    print("✅ Splits saved!")


# =============================================================================
# MAIN - Run examples
# =============================================================================

if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Local Laptop (Ollama)", example_local_laptop),
        "2": ("Cloud - Gemini API", example_cloud_gemini),
        "3": ("OpenAI ChatGPT", example_openai),
        "4": ("Claude (Anthropic)", example_claude),
        "5": ("Nemotron (NVIDIA)", example_nemotron),
        "6": ("Complete Pipeline", example_complete_pipeline),
        "7": ("Fine-tuning", example_finetuning),
        "8": ("Tokenizer Comparison", example_tokenizer_comparison),
        "9": ("Data Processing", example_data_processing),
    }

    print("\n" + "=" * 60)
    print("SAARA Quickstart Examples".center(60))
    print("=" * 60)
    print()

    for key, (name, func) in examples.items():
        print(f"  {key}. {name}")

    print()
    choice = input("Pick an example (1-9) or 'all': ").strip()

    if choice == "all":
        for key, (name, func) in examples.items():
            print(f"\n⭐ Running: {name}\n")
            try:
                func()
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    elif choice in examples:
        name, func = examples[choice]
        print(f"\n⭐ Running: {name}\n")
        try:
            func()
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print("❌ Invalid choice")
        sys.exit(1)

    print("\n✅ Done!")
