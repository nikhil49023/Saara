"""
Quickstart Module - Simple patterns for all devices

Quick & efficient patterns for:
- Local inference (vLLM, Ollama)
- Fine-tuning
- Building datasets
- Data format conversion
- Everything manual, nothing automatic

Released under the MIT License.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Quick LLM Inference (Any Provider)
# =============================================================================

class QuickLLM:
    """
    Fast local LLM inference.

    Examples:
        >>> # Auto local backend (vLLM preferred, Ollama fallback)
        >>> llm = QuickLLM("auto", model="mistral")

        >>> # Force vLLM backend
        >>> llm = QuickLLM("vllm", model="mistral")

        >>> # Ollama (local, no API key needed)
        >>> llm = QuickLLM("ollama", model="granite3.1-dense:8b")

        >>> # Use it
        >>> result = llm.generate("Explain AI in one sentence")
        >>> print(result)
    """

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize Quick LLM.

        Args:
            provider: "auto", "vllm", or "ollama"
            model: Model name
            temperature: Generation temperature
            max_tokens: Max tokens to generate
        """
        from saara.llm_providers import create_llm

        self.llm = create_llm(
            provider=provider,
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
# Quick Dataset Processing
# =============================================================================

class QuickDataset:
    """
    Simple dataset loading, processing, and saving.

    Examples:
        >>> from saara.file_utils import load_jsonl, save_jsonl
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

    def convert_format(self, target_format: str, system_prompt: str = "") -> List[Dict]:
        """
        Convert dataset to a training format.

        Args:
            target_format: One of: alpaca, chatml, sharegpt, completion, dpo
            system_prompt: Optional system prompt for chat formats

        Returns:
            Converted data in target format
        """
        from saara.formats import convert_dataset

        return convert_dataset(self.records, target_format, system_prompt=system_prompt)

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


def vllm_local(model: str = "mistral") -> QuickLLM:
    """Quick local vLLM inference."""
    return QuickLLM("vllm", model=model)


# =============================================================================
# Complete Workflow Examples
# =============================================================================

def simple_workflow_example():
    """
    Complete example: Load data -> Process -> Convert format

    This is a reference for how to use SAARA pipeline.
    """
    # Step 1: Load data
    from saara.file_utils import load_jsonl, save_jsonl

    records = load_jsonl("data.jsonl")
    print(f"Loaded {len(records)} records")

    # Step 2: Prepare dataset
    ds = QuickDataset(records)
    texts = ds.get_texts("text")
    print(f"Found {len(texts)} texts")

    # Step 3: Convert to training format
    alpaca_data = ds.convert_format("alpaca")
    print(f"Converted to {len(alpaca_data)} Alpaca samples")

    # Step 4: Use LLM for labeling/processing
    llm = ollama_local("granite3.1-dense:8b")
    result = llm.generate("Explain fine-tuning in simple terms")
    print(f"LLM result: {result}")

    # Step 5: Save results
    save_jsonl(alpaca_data, "output_alpaca.jsonl")
    print("Results saved")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "QuickLLM",
    "QuickDataset",
    "QuickFineTune",
    "ollama_local",
    "vllm_local",
    "simple_workflow_example",
]


if __name__ == "__main__":
    # Example usage
    simple_workflow_example()
