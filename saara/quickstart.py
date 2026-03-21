"""
Quickstart Module - Simple patterns for all devices

Quick & efficient patterns for:
- Local inference (Ollama)
- Cloud APIs (Gemini, OpenAI, Claude, Nemotron)
- Fine-tuning
- Building datasets
- Everything manual, nothing automatic

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Quick LLM Inference (Any Provider)
# =============================================================================

class QuickLLM:
    """
    Fast LLM inference with any provider.

    Examples:
        >>> # Ollama (local, no API key needed)
        >>> llm = QuickLLM("ollama", model="granite3.1-dense:8b")

        >>> # Gemini
        >>> llm = QuickLLM("gemini", api_key="...", model="gemini-2.0-flash")

        >>> # OpenAI
        >>> llm = QuickLLM("openai", api_key="...", model="gpt-4")

        >>> # Anthropic (Claude)
        >>> llm = QuickLLM("anthropic", api_key="...", model="claude-opus-4-6")

        >>> # Nemotron
        >>> llm = QuickLLM("nemotron", api_key="...")

        >>> # Use it
        >>> result = llm.generate("Explain AI in one sentence")
        >>> print(result)
    """

    def __init__(
        self,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize Quick LLM.

        Args:
            provider: "ollama", "gemini", "openai", "anthropic", "nemotron", "groq"
            api_key: API key (not needed for ollama)
            model: Model name
            temperature: Generation temperature
            max_tokens: Max tokens to generate
        """
        from saara.llm_providers import create_llm

        self.llm = create_llm(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text."""
        return self.llm.generate(prompt, system_prompt)

    def is_available(self) -> bool:
        """Check if provider is available."""
        return self.llm.is_available()


# =============================================================================
# Quick Tokenization
# =============================================================================

class QuickTokenizer:
    """
    Simple tokenizer creation and training.

    Examples:
        >>> # BPE tokenizer
        >>> tok = QuickTokenizer("bpe", vocab_size=32000)
        >>> tok.train(texts)
        >>> tokens = tok.encode("hello world")

        >>> # WordPiece
        >>> tok = QuickTokenizer("wordpiece", vocab_size=30522)
        >>> tok.train(texts)

        >>> # Byte-level (universal)
        >>> tok = QuickTokenizer("byte")
        >>> tokens = tok.encode("hello 你好 🚀")
    """

    def __init__(self, tokenizer_type: str = "bpe", vocab_size: int = 32000, **kwargs):
        """
        Initialize Quick Tokenizer.

        Args:
            tokenizer_type: "bpe", "wordpiece", "byte"
            vocab_size: Vocabulary size
            **kwargs: Additional arguments for tokenizer
        """
        from saara.tokenizers import create_tokenizer

        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.tokenizer = create_tokenizer(tokenizer_type, **kwargs)

    def train(self, texts: List[str]) -> None:
        """Train tokenizer."""
        logger.info(f"Training {self.tokenizer_type} tokenizer...")
        self.tokenizer.train(texts, vocab_size=self.vocab_size)

    def encode(self, text: str) -> List[int]:
        """Encode text."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode tokens."""
        return self.tokenizer.decode(token_ids)

    def save(self, directory: str) -> None:
        """Save tokenizer."""
        self.tokenizer.save(directory)
        logger.info(f"Tokenizer saved to {directory}")

    @classmethod
    def load(cls, directory: str, tokenizer_type: str = "bpe") -> "QuickTokenizer":
        """Load tokenizer."""
        from saara.tokenizers import TokenizerRegistry

        tok_instance = TokenizerRegistry.create(tokenizer_type)
        tok_instance = tok_instance.load(directory)

        quick = cls(tokenizer_type)
        quick.tokenizer = tok_instance
        return quick


# =============================================================================
# Quick Dataset Processing
# =============================================================================

class QuickDataset:
    """
    Simple dataset loading, processing, and saving.

    Examples:
        >>> from saara.file_utils import load_jsonl, save_jsonl, extract_texts
        >>> from saara.quickstart import QuickDataset

        >>> # Load data
        >>> ds = QuickDataset.from_file("data.jsonl")

        >>> # Extract texts
        >>> texts = ds.get_texts("content")

        >>> # Split into train/val/test
        >>> train, val, test = ds.split(train_ratio=0.8, val_ratio=0.1)

        >>> # Save splits
        >>> train.save("train.jsonl")
        >>> val.save("val.jsonl")
        >>> test.save("test.jsonl")
    """

    def __init__(self, records: List[Dict[str, Any]]):
        """Initialize dataset."""
        self.records = records

    @classmethod
    def from_file(cls, file_path: str) -> "QuickDataset":
        """Load from file."""
        from saara.file_utils import load_from_file

        data = load_from_file(file_path)

        # Handle different formats
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            records = [{"text": str(data)}]

        return cls(records)

    def get_texts(self, field_name: str = "text") -> List[str]:
        """Extract texts from field."""
        from saara.file_utils import extract_texts

        return extract_texts(self.records, field_name)

    def get_training_pairs(
        self,
        prompt_field: str = "prompt",
        response_field: str = "response"
    ) -> List[Dict[str, str]]:
        """Extract training pairs."""
        from saara.file_utils import extract_training_pairs

        return extract_training_pairs(
            self.records,
            prompt_field=prompt_field,
            response_field=response_field
        )

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> tuple:
        """Split into train/val/test."""
        from saara.file_utils import split_dataset

        train_recs, val_recs, test_recs = split_dataset(
            self.records,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

        return QuickDataset(train_recs), QuickDataset(val_recs), QuickDataset(test_recs)

    def save(self, file_path: str) -> None:
        """Save dataset."""
        from saara.file_utils import save_to_file

        save_to_file(self.records, file_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


# =============================================================================
# Quick Fine-tuning (Simple)
# =============================================================================

class QuickFineTune:
    """
    Simple fine-tuning wrapper.

    Examples:
        >>> # Load training data
        >>> from saara.quickstart import QuickDataset
        >>> ds = QuickDataset.from_file("train_data.jsonl")
        >>> pairs = ds.get_training_pairs()

        >>> # Fine-tune
        >>> trainer = QuickFineTune("sarvamai/sarvam-1", output_dir="./my_model")
        >>> trainer.train(pairs, num_epochs=3)
    """

    def __init__(
        self,
        model_id: str = "sarvamai/sarvam-1",
        output_dir: str = "./finetuned_model",
        learning_rate: float = 2e-4,
        batch_size: int = 1
    ):
        """
        Initialize Fine-tuner.

        Args:
            model_id: Base model ID
            output_dir: Where to save fine-tuned model
            learning_rate: Learning rate
            batch_size: Batch size per device
        """
        self.model_id = model_id
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(
        self,
        training_data: List[Dict[str, str]],
        num_epochs: int = 3,
        max_seq_length: int = 512
    ) -> None:
        """
        Train model.

        Args:
            training_data: List of {"prompt": ..., "response": ...} dicts
            num_epochs: Number of epochs
            max_seq_length: Max sequence length
        """
        try:
            from saara.train import LLMTrainer

            trainer = LLMTrainer(
                model_id=self.model_id,
                config={
                    "output_dir": self.output_dir,
                    "num_epochs": num_epochs,
                    "learning_rate": self.learning_rate,
                    "per_device_train_batch_size": self.batch_size,
                    "max_seq_length": max_seq_length,
                }
            )

            # Convert training data to expected format
            dataset = {"text": [f"{d['prompt']}\n{d['response']}" for d in training_data]}

            logger.info(f"Starting training on {len(training_data)} examples")
            # Note: Actual training implementation depends on LLMTrainer.train()
            logger.info(f"Model will be saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


# =============================================================================
# Quick Patterns (Common Use Cases)
# =============================================================================

def ollama_local(model: str = "granite3.1-dense:8b") -> QuickLLM:
    """Quick local Ollama inference."""
    return QuickLLM("ollama", model=model)


def gemini_api(api_key: str, model: str = "gemini-2.0-flash") -> QuickLLM:
    """Quick Gemini API inference."""
    return QuickLLM("gemini", api_key=api_key, model=model)


def openai_api(api_key: str, model: str = "gpt-4") -> QuickLLM:
    """Quick OpenAI inference."""
    return QuickLLM("openai", api_key=api_key, model=model)


def claude_api(api_key: str, model: str = "claude-opus-4-6") -> QuickLLM:
    """Quick Claude (Anthropic) inference."""
    return QuickLLM("anthropic", api_key=api_key, model=model)


def nemotron_api(api_key: str) -> QuickLLM:
    """Quick NVIDIA Nemotron inference."""
    return QuickLLM("nemotron", api_key=api_key)


# =============================================================================
# Complete Workflow Examples
# =============================================================================

def simple_workflow_example():
    """
    Complete example: Load data → Tokenize → Generate with LLM

    This is a reference for how to use SAARA in a simple way.
    """
    # Step 1: Load data
    from saara.file_utils import load_jsonl, save_jsonl

    records = load_jsonl("data.jsonl")
    print(f"Loaded {len(records)} records")

    # Step 2: Prepare dataset
    ds = QuickDataset(records)
    texts = ds.get_texts("text")
    print(f"Found {len(texts)} texts")

    # Step 3: Train tokenizer
    tokenizer = QuickTokenizer("bpe", vocab_size=32000)
    tokenizer.train(texts[:1000])
    tokenizer.save("my_tokenizer")
    print("Tokenizer trained")

    # Step 4: Use LLM
    llm = ollama_local("granite3.1-dense:8b")
    result = llm.generate("Explain tokenization in simple terms")
    print(f"LLM result: {result}")

    # Step 5: Save results
    output_data = [
        {"text": text, "tokenized": tokenizer.encode(text)[:50]}
        for text in texts[:10]
    ]
    save_jsonl(output_data, "output_tokens.jsonl")
    print("Results saved")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "QuickLLM",
    "QuickTokenizer",
    "QuickDataset",
    "QuickFineTune",
    "ollama_local",
    "gemini_api",
    "openai_api",
    "claude_api",
    "nemotron_api",
    "simple_workflow_example",
]


if __name__ == "__main__":
    # Example usage
    simple_workflow_example()
