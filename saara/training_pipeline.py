"""
Modular Training Pipeline
Orchestrates end-to-end training: PDF → Labeled Data → Tokenization → Training.

This module provides a clean, modular architecture where each stage is independent
and data is passed efficiently between stages using memory-mapped storage.

Architecture:
    Stage 1: PDF Parser      → Labeled JSONL (instruction-response pairs)
    Stage 2: Tokenizer Setup → Configure domain-specific tokenizer
    Stage 3: Token Storage   → Pre-tokenize & save as .arrow (memory-mapped)
    Stage 4: Training        → Train directly from pre-tokenized data

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingPipelineConfig:
    """Configuration for the complete training pipeline."""

    # Stage 1: Data Preparation
    pdf_input: Optional[str] = None  # Path to PDF or directory
    jsonl_input: Optional[str] = None  # Or use existing JSONL
    output_dir: str = "training_output"

    # Stage 2: Tokenizer
    tokenizer_id: str = "sarvamai/sarvam-1"
    custom_tokenizer_path: Optional[str] = None
    train_custom_tokenizer: bool = False
    tokenizer_domain: str = "general"  # general, medical, legal, code, scientific

    # Stage 3: Token Storage
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    num_proc: int = 4

    # Stage 4: Training
    model_id: str = "sarvamai/sarvam-1"
    num_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # General
    skip_stages: list = None  # List of stage numbers to skip
    resume_from: Optional[str] = None  # Resume from checkpoint

    def __post_init__(self):
        if self.skip_stages is None:
            self.skip_stages = []


class TrainingPipeline:
    """
    Modular training pipeline with independent, reusable stages.

    Example - Full pipeline from PDFs:
        >>> pipeline = TrainingPipeline(config={
        ...     "pdf_input": "docs/",
        ...     "tokenizer_id": "sarvamai/sarvam-1",
        ...     "max_length": 1024,
        ...     "num_epochs": 3
        ... })
        >>> result = pipeline.run()
        >>> print(f"Model saved to: {result['model_path']}")

    Example - Skip PDF stage (start from JSONL):
        >>> pipeline = TrainingPipeline(config={
        ...     "jsonl_input": "data/labeled.jsonl",
        ...     "skip_stages": [1]  # Skip PDF parsing
        ... })
        >>> result = pipeline.run()

    Example - Only tokenize (no training):
        >>> pipeline = TrainingPipeline(config={
        ...     "jsonl_input": "data.jsonl",
        ...     "skip_stages": [4]  # Skip training
        ... })
        >>> result = pipeline.run()
        >>> print(f"Tokens saved to: {result['tokenized_path']}")
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], TrainingPipelineConfig],
        on_progress: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the training pipeline.

        Args:
            config: Configuration dict or TrainingPipelineConfig
            on_progress: Optional callback for progress updates
        """
        if isinstance(config, dict):
            self.config = TrainingPipelineConfig(**config)
        else:
            self.config = config

        self.on_progress = on_progress
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage outputs
        self.labeled_data_path = None
        self.tokenized_data_path = None
        self.model_path = None

        self._progress("TrainingPipeline initialized")

    def _progress(self, msg: str):
        """Send progress update."""
        logger.info(msg)
        if self.on_progress:
            self.on_progress(msg)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Returns:
            Dictionary with paths and statistics
        """
        start_time = datetime.now()

        self._progress("=" * 60)
        self._progress("🚀 Starting Modular Training Pipeline")
        self._progress("=" * 60)

        result = {
            "success": True,
            "stages_completed": [],
            "errors": []
        }

        try:
            # Stage 1: PDF → Labeled Data
            if 1 not in self.config.skip_stages:
                self._progress("\n📄 Stage 1: PDF Parser → Labeled Data")
                self.labeled_data_path = self._run_stage1()
                result["stages_completed"].append(1)
                result["labeled_data_path"] = self.labeled_data_path
            else:
                self._progress("\n⏭️  Stage 1: Skipped (using existing data)")
                self.labeled_data_path = self.config.jsonl_input

            # Stage 2: Configure Tokenizer
            if 2 not in self.config.skip_stages:
                self._progress("\n🔤 Stage 2: Tokenizer Setup")
                tokenizer_path = self._run_stage2()
                result["stages_completed"].append(2)
                result["tokenizer_path"] = tokenizer_path
            else:
                self._progress("\n⏭️  Stage 2: Skipped (using default tokenizer)")

            # Stage 3: Pre-tokenize & Store
            if 3 not in self.config.skip_stages:
                self._progress("\n💾 Stage 3: Pre-tokenization → Token Storage")
                self.tokenized_data_path = self._run_stage3()
                result["stages_completed"].append(3)
                result["tokenized_data_path"] = self.tokenized_data_path
            else:
                self._progress("\n⏭️  Stage 3: Skipped")

            # Stage 4: Training
            if 4 not in self.config.skip_stages:
                self._progress("\n🎯 Stage 4: Training from Pre-tokenized Data")
                self.model_path = self._run_stage4()
                result["stages_completed"].append(4)
                result["model_path"] = self.model_path
            else:
                self._progress("\n⏭️  Stage 4: Skipped")

        except Exception as e:
            logger.exception("Pipeline failed")
            result["success"] = False
            result["errors"].append(str(e))

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        result["duration_seconds"] = duration

        # Save pipeline metadata
        self._save_metadata(result)

        self._progress("=" * 60)
        if result["success"]:
            self._progress(f"✅ Pipeline Complete ({duration:.1f}s)")
            self._progress(f"   Stages: {result['stages_completed']}")
        else:
            self._progress(f"❌ Pipeline Failed: {result['errors']}")
        self._progress("=" * 60)

        return result

    def _run_stage1(self) -> str:
        """Stage 1: PDF → Labeled JSONL."""
        from saara.pipeline import DataPipeline

        if not self.config.pdf_input:
            self._progress("⚠️  No PDF input specified, checking for existing JSONL")
            if self.config.jsonl_input:
                return self.config.jsonl_input
            raise ValueError("No PDF input or JSONL input provided")

        # Run document pipeline
        pipeline = DataPipeline()
        result = pipeline.process_document(
            self.config.pdf_input,
            output_name="labeled_data"
        )

        # Find generated JSONL
        output_files = result.output_files
        if 'instruction_tuning' in output_files:
            jsonl_path = output_files['instruction_tuning'].get('jsonl', [None])[0]
            if jsonl_path:
                self._progress(f"✓ Generated labeled data: {jsonl_path}")
                return jsonl_path

        raise ValueError("Failed to generate labeled data from PDF")

    def _run_stage2(self) -> Optional[str]:
        """Stage 2: Configure/Train Tokenizer."""
        if self.config.custom_tokenizer_path:
            self._progress(f"✓ Using custom tokenizer: {self.config.custom_tokenizer_path}")
            return self.config.custom_tokenizer_path

        if self.config.train_custom_tokenizer:
            from saara.ai_tokenizer import AITokenizer, AITokenizerConfig

            self._progress(f"Training domain-specific tokenizer ({self.config.tokenizer_domain})")

            config = AITokenizerConfig(
                domain=self.config.tokenizer_domain,
                vocab_size=32000
            )

            tokenizer = AITokenizer(config)

            # Train on labeled data
            if self.labeled_data_path:
                tokenizer_path = self.output_dir / "tokenizer"
                tokenizer.train_from_file(self.labeled_data_path, str(tokenizer_path))
                self._progress(f"✓ Trained tokenizer saved to: {tokenizer_path}")
                return str(tokenizer_path)

        self._progress(f"✓ Using base tokenizer: {self.config.tokenizer_id}")
        return self.config.tokenizer_id

    def _run_stage3(self) -> str:
        """Stage 3: Pre-tokenize and store as .arrow."""
        from saara.token_storage import TokenStorage, TokenStorageConfig

        if not self.labeled_data_path:
            raise ValueError("No labeled data available for tokenization")

        # Configure token storage
        storage_config = TokenStorageConfig(
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            num_proc=self.config.num_proc
        )

        # Determine tokenizer to use
        tokenizer_id = (
            self.config.custom_tokenizer_path or
            self.config.tokenizer_id
        )

        storage = TokenStorage(
            tokenizer=tokenizer_id,
            config=storage_config,
            on_progress=self.on_progress
        )

        # Tokenize and save
        token_dir = self.output_dir / "tokens"
        tokenized_path = storage.tokenize_dataset(
            self.labeled_data_path,
            str(token_dir),
            instruction_field="instruction",
            response_field="response"
        )

        # Print storage stats
        stats = storage.get_storage_stats(tokenized_path)
        self._progress(f"✓ Token storage stats:")
        self._progress(f"   - Examples: {stats['num_examples']}")
        self._progress(f"   - Disk size: {stats['disk_size_mb']:.2f} MB")
        self._progress(f"   - Avg bytes/example: {stats['avg_bytes_per_example']:.0f}")

        return tokenized_path

    def _run_stage4(self) -> str:
        """Stage 4: Train from pre-tokenized data."""
        from saara.train import LLMTrainer
        from saara.config import TrainConfig

        if not self.tokenized_data_path:
            raise ValueError("No tokenized data available for training")

        # Create training config
        train_config = TrainConfig(
            output_dir=str(self.output_dir / "model"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            max_seq_length=self.config.max_length,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )

        # Initialize trainer
        trainer = LLMTrainer(
            model_id=self.config.model_id,
            config=train_config.to_dict(),
            on_progress=self.on_progress
        )

        # Train from pre-tokenized data
        self._progress("🔥 Starting training from pre-tokenized data...")
        trainer.train(
            data_path=self.tokenized_data_path,
            resume_from_checkpoint=self.config.resume_from
        )

        model_path = train_config.output_dir
        self._progress(f"✓ Model saved to: {model_path}")
        return model_path

    def _save_metadata(self, result: Dict[str, Any]):
        """Save pipeline metadata."""
        metadata = {
            "config": asdict(self.config),
            "result": result,
            "created_at": datetime.now().isoformat()
        }

        metadata_path = self.output_dir / "pipeline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Pipeline metadata saved to: {metadata_path}")


def quick_train(
    data_path: str,
    model_id: str = "sarvamai/sarvam-1",
    output_dir: str = "quick_train",
    max_length: int = 512,
    num_epochs: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick helper to train a model in one line.

    Example:
        >>> result = quick_train(
        ...     "data.jsonl",
        ...     model_id="sarvamai/sarvam-1",
        ...     num_epochs=3,
        ...     max_length=1024
        ... )
        >>> print(result['model_path'])

    Args:
        data_path: Path to JSONL data
        model_id: Model to fine-tune
        output_dir: Output directory
        max_length: Max sequence length
        num_epochs: Number of epochs
        **kwargs: Additional config options

    Returns:
        Pipeline result dictionary
    """
    config = TrainingPipelineConfig(
        jsonl_input=data_path,
        model_id=model_id,
        output_dir=output_dir,
        max_length=max_length,
        num_epochs=num_epochs,
        skip_stages=[1],  # Skip PDF parsing
        **kwargs
    )

    pipeline = TrainingPipeline(config)
    return pipeline.run()
