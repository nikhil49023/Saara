"""
Token Storage Module
Handles pre-tokenization and memory-mapped binary storage for efficient training.

Eliminates CPU bottlenecks by pre-tokenizing datasets and storing as .arrow files
that can be memory-mapped directly into GPU memory during training.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
import json

from datasets import Dataset, load_dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenStorageConfig:
    """Configuration for token storage."""
    max_length: int = 512
    padding: str = "max_length"  # "max_length" or "longest" or False
    truncation: bool = True
    add_special_tokens: bool = True
    return_attention_mask: bool = True

    # Storage settings
    num_proc: int = 4  # Parallel processing
    batch_size: int = 1000  # Tokenization batch size
    cache_file_name: Optional[str] = None  # Custom cache location


class TokenStorage:
    """
    Pre-tokenizes datasets and stores them as memory-mapped .arrow files.

    This eliminates the CPU bottleneck of tokenizing during training by doing
    it once upfront. Training then reads pre-tokenized token IDs directly.

    Benefits:
    - 3-10x faster training (no per-batch tokenization)
    - Lower memory usage (4 bytes per token vs 10+ for strings)
    - GPU utilization stays high (no CPU stalls)
    - Supports custom tokenizers

    Example:
        >>> storage = TokenStorage(tokenizer="sarvamai/sarvam-1")
        >>> storage.tokenize_dataset("data.jsonl", "tokens/", text_field="text")
        >>> # Later in training:
        >>> dataset = storage.load_tokenized("tokens/")
    """

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        config: Optional[TokenStorageConfig] = None,
        on_progress: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize TokenStorage.

        Args:
            tokenizer: Tokenizer ID or instance
            config: TokenStorageConfig or None for defaults
            on_progress: Optional callback for progress updates
        """
        self.config = config or TokenStorageConfig()
        self.on_progress = on_progress

        # Load tokenizer
        if isinstance(tokenizer, str):
            logger.info(f"Loading tokenizer: {tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer_id = tokenizer
        else:
            self.tokenizer = tokenizer
            self.tokenizer_id = "custom"

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._progress("TokenStorage initialized")

    def _progress(self, msg: str):
        """Send progress update."""
        logger.info(msg)
        if self.on_progress:
            self.on_progress(msg)

    def tokenize_dataset(
        self,
        data_path: Union[str, Dataset],
        output_dir: str,
        text_field: str = "text",
        instruction_field: Optional[str] = None,
        response_field: Optional[str] = None,
        prompt_template: Optional[str] = None,
        split: str = "train",
        **load_kwargs
    ) -> str:
        """
        Tokenize a dataset and save as memory-mapped .arrow file.

        Args:
            data_path: Path to JSONL file, HF dataset ID, or Dataset object
            output_dir: Directory to save tokenized data
            text_field: Field containing text (for completion tasks)
            instruction_field: Field containing instructions (for instruction tuning)
            response_field: Field containing responses (for instruction tuning)
            prompt_template: Template for combining fields (e.g., "### Instruction: {instruction}\n### Response: {response}")
            split: Dataset split to use
            **load_kwargs: Additional kwargs for load_dataset

        Returns:
            Path to tokenized dataset directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load dataset
        self._progress(f"Loading dataset from {data_path}")
        if isinstance(data_path, Dataset):
            dataset = data_path
        elif str(data_path).endswith(('.json', '.jsonl')):
            dataset = load_dataset("json", data_files=str(data_path), split=split, **load_kwargs)
        else:
            dataset = load_dataset(data_path, split=split, **load_kwargs)

        self._progress(f"Loaded {len(dataset)} examples")

        # Step 2: Define tokenization function
        def tokenize_function(examples):
            """Tokenize a batch of examples."""
            # Construct text based on fields
            if instruction_field and response_field:
                # Instruction-response pairs
                if prompt_template:
                    texts = [
                        prompt_template.format(
                            instruction=inst,
                            response=resp
                        )
                        for inst, resp in zip(
                            examples[instruction_field],
                            examples[response_field]
                        )
                    ]
                else:
                    # Default template
                    texts = [
                        f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                        for inst, resp in zip(
                            examples[instruction_field],
                            examples[response_field]
                        )
                    ]
            else:
                # Simple text field
                texts = examples[text_field]

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                add_special_tokens=self.config.add_special_tokens,
                return_attention_mask=self.config.return_attention_mask,
            )

            # Add labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Step 3: Apply tokenization with multiprocessing
        self._progress(f"Tokenizing with {self.config.num_proc} processes...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
            remove_columns=dataset.column_names,  # Remove original text columns
            desc="Tokenizing"
        )

        # Step 4: Save as Arrow format (memory-mappable)
        save_path = output_path / "tokenized"
        self._progress(f"Saving tokenized dataset to {save_path}")
        tokenized_dataset.save_to_disk(str(save_path))

        # Step 5: Save metadata
        metadata = {
            "tokenizer_id": self.tokenizer_id,
            "vocab_size": self.tokenizer.vocab_size,
            "num_examples": len(tokenized_dataset),
            "max_length": self.config.max_length,
            "text_field": text_field,
            "instruction_field": instruction_field,
            "response_field": response_field,
            "created_from": str(data_path),
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self._progress(f"✓ Tokenized dataset saved to {save_path}")
        self._progress(f"  - {len(tokenized_dataset)} examples")
        self._progress(f"  - Token IDs stored as .arrow (memory-mapped)")
        self._progress(f"  - Metadata saved to {metadata_path}")

        return str(save_path)

    def load_tokenized(self, tokenized_dir: str) -> Dataset:
        """
        Load a pre-tokenized dataset from disk.

        Uses memory-mapping for zero-copy reads directly from disk.

        Args:
            tokenized_dir: Directory containing tokenized dataset

        Returns:
            Memory-mapped Dataset
        """
        path = Path(tokenized_dir)

        # Check for metadata
        metadata_path = path.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Loading tokenized dataset: {metadata['num_examples']} examples")

        # Load as memory-mapped
        dataset = load_dataset(
            str(path),
            split="train",
            keep_in_memory=False  # Use memory-mapping
        )

        self._progress(f"✓ Loaded {len(dataset)} pre-tokenized examples (memory-mapped)")
        return dataset

    def get_storage_stats(self, tokenized_dir: str) -> Dict[str, Any]:
        """
        Get statistics about a tokenized dataset.

        Args:
            tokenized_dir: Directory containing tokenized dataset

        Returns:
            Dictionary with statistics
        """
        path = Path(tokenized_dir)

        # Load metadata
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.exists():
            return {"error": "No metadata found"}

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Calculate disk usage
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        stats = {
            **metadata,
            "disk_size_mb": total_size / (1024 * 1024),
            "avg_bytes_per_example": total_size / metadata['num_examples'] if metadata['num_examples'] > 0 else 0,
        }

        return stats


def quick_tokenize(
    data_path: str,
    output_dir: str,
    tokenizer: str = "sarvamai/sarvam-1",
    max_length: int = 512,
    text_field: str = "text",
    **kwargs
) -> str:
    """
    Quick helper to tokenize a dataset in one line.

    Example:
        >>> quick_tokenize("data.jsonl", "tokens/", max_length=1024)
        'tokens/tokenized'

    Args:
        data_path: Path to data file
        output_dir: Output directory
        tokenizer: Tokenizer ID
        max_length: Max sequence length
        text_field: Text field name
        **kwargs: Additional arguments for tokenize_dataset

    Returns:
        Path to tokenized dataset
    """
    config = TokenStorageConfig(max_length=max_length)
    storage = TokenStorage(tokenizer, config)
    return storage.tokenize_dataset(data_path, output_dir, text_field=text_field, **kwargs)
