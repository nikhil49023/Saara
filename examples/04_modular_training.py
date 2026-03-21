"""
04_modular_training.py

Demonstrates the modular training pipeline with:
  - Stage 1: PDF → Labeled Data (optional)
  - Stage 2: Tokenizer Setup (configurable)
  - Stage 3: Pre-tokenization → Memory-mapped storage
  - Stage 4: Efficient training from pre-tokenized data

This eliminates CPU bottlenecks by pre-tokenizing once and training
directly from memory-mapped token IDs.

Benefits:
  - 3-10x faster training (no per-batch tokenization)
  - Lower memory usage
  - Supports custom tokenizers
  - Each stage is independent and reusable
"""

from saara import TrainingPipeline, TrainingPipelineConfig
from saara import TokenStorage, TokenStorageConfig
from saara import quick_tokenize, quick_train


def example_1_full_pipeline():
    """Example 1: Full pipeline from PDF to trained model."""
    print("=" * 60)
    print("Example 1: Full Pipeline (PDF → Trained Model)")
    print("=" * 60)

    config = TrainingPipelineConfig(
        # Stage 1: PDF Input
        pdf_input="docs/research_paper.pdf",

        # Stage 2: Tokenizer
        tokenizer_id="sarvamai/sarvam-1",
        # Or use custom: train_custom_tokenizer=True, tokenizer_domain="scientific"

        # Stage 3: Pre-tokenization
        max_length=1024,
        num_proc=4,  # Parallel tokenization

        # Stage 4: Training
        model_id="sarvamai/sarvam-1",
        num_epochs=3,
        batch_size=2,
        learning_rate=2e-4,

        # Output
        output_dir="training_output/full_pipeline"
    )

    pipeline = TrainingPipeline(config)
    result = pipeline.run()

    if result["success"]:
        print(f"\n✅ Training complete!")
        print(f"   Model: {result['model_path']}")
        print(f"   Tokens: {result['tokenized_data_path']}")
        print(f"   Duration: {result['duration_seconds']:.1f}s")


def example_2_skip_pdf_stage():
    """Example 2: Start from existing JSONL (skip PDF parsing)."""
    print("\n" + "=" * 60)
    print("Example 2: Skip PDF Stage (Start from JSONL)")
    print("=" * 60)

    config = TrainingPipelineConfig(
        # Skip Stage 1, use existing data
        jsonl_input="data/labeled_data.jsonl",
        skip_stages=[1],  # Skip PDF parsing

        # Continue with stages 2-4
        tokenizer_id="sarvamai/sarvam-1",
        max_length=512,
        num_epochs=3,
        output_dir="training_output/from_jsonl"
    )

    pipeline = TrainingPipeline(config)
    result = pipeline.run()

    print(f"\n✅ Completed stages: {result['stages_completed']}")


def example_3_only_tokenize():
    """Example 3: Only pre-tokenize (skip training)."""
    print("\n" + "=" * 60)
    print("Example 3: Pre-tokenization Only (No Training)")
    print("=" * 60)

    config = TrainingPipelineConfig(
        jsonl_input="data/large_dataset.jsonl",
        skip_stages=[1, 4],  # Skip PDF and training

        tokenizer_id="sarvamai/sarvam-1",
        max_length=2048,
        num_proc=8,  # Fast parallel tokenization
        output_dir="tokens/large_dataset"
    )

    pipeline = TrainingPipeline(config)
    result = pipeline.run()

    print(f"\n✅ Tokenized dataset saved to: {result['tokenized_data_path']}")
    print("   This can be reused for multiple training runs without re-tokenization!")


def example_4_manual_stages():
    """Example 4: Manual control of each stage."""
    print("\n" + "=" * 60)
    print("Example 4: Manual Stage Control")
    print("=" * 60)

    # Stage 1: Already have labeled.jsonl from DataPipeline
    data_path = "data/instruction_data.jsonl"

    # Stage 2 & 3: Pre-tokenize with custom config
    print("\n📦 Stage 2+3: Pre-tokenizing...")
    storage_config = TokenStorageConfig(
        max_length=512,
        padding="max_length",
        num_proc=4
    )

    storage = TokenStorage(
        tokenizer="sarvamai/sarvam-1",
        config=storage_config
    )

    tokenized_path = storage.tokenize_dataset(
        data_path=data_path,
        output_dir="tokens/manual",
        instruction_field="instruction",
        response_field="response"
    )

    # View storage stats
    stats = storage.get_storage_stats(tokenized_path)
    print(f"\n📊 Token Storage Stats:")
    print(f"   Examples: {stats['num_examples']}")
    print(f"   Disk size: {stats['disk_size_mb']:.2f} MB")
    print(f"   Tokenizer: {stats['tokenizer_id']}")

    # Stage 4: Train from pre-tokenized data
    print(f"\n🎯 Stage 4: Training from {tokenized_path}")
    from saara import LLMTrainer, TrainConfig

    train_config = TrainConfig(
        output_dir="models/manual_training",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_seq_length=512
    )

    trainer = LLMTrainer(
        model_id="sarvamai/sarvam-1",
        config=train_config.to_dict()
    )

    trainer.train(data_path=tokenized_path)
    print("\n✅ Training complete!")


def example_5_quick_helpers():
    """Example 5: Using quick helper functions."""
    print("\n" + "=" * 60)
    print("Example 5: Quick Helper Functions")
    print("=" * 60)

    # Quick tokenization
    print("\n📦 Quick tokenization...")
    tokenized_path = quick_tokenize(
        data_path="data/small_dataset.jsonl",
        output_dir="tokens/quick",
        tokenizer="sarvamai/sarvam-1",
        max_length=512
    )
    print(f"✓ Tokenized: {tokenized_path}")

    # Quick training
    print("\n🎯 Quick training...")
    result = quick_train(
        data_path=tokenized_path,
        model_id="sarvamai/sarvam-1",
        output_dir="models/quick_train",
        num_epochs=2,
        max_length=512
    )
    print(f"✓ Model: {result['model_path']}")


def example_6_custom_tokenizer():
    """Example 6: Train domain-specific tokenizer."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Domain Tokenizer")
    print("=" * 60)

    config = TrainingPipelineConfig(
        jsonl_input="data/medical_texts.jsonl",
        skip_stages=[1],

        # Train custom medical tokenizer
        train_custom_tokenizer=True,
        tokenizer_domain="medical",  # Optimized for medical terms

        max_length=512,
        num_epochs=3,
        output_dir="training_output/medical_model"
    )

    pipeline = TrainingPipeline(config)
    result = pipeline.run()

    print(f"\n✅ Custom tokenizer trained and used!")
    print(f"   Tokenizer: {result.get('tokenizer_path')}")
    print(f"   Model: {result['model_path']}")


def example_7_reuse_tokenized_data():
    """Example 7: Reuse pre-tokenized data for multiple training runs."""
    print("\n" + "=" * 60)
    print("Example 7: Reuse Tokenized Data")
    print("=" * 60)

    # Tokenize once (this is slow)
    print("\n📦 Tokenizing once (slow, but one-time)...")
    tokenized_path = quick_tokenize(
        data_path="data/big_dataset.jsonl",
        output_dir="tokens/shared",
        max_length=512
    )

    # Now train with different hyperparameters (fast!)
    from saara import LLMTrainer, TrainConfig

    print("\n🎯 Training Run #1 (low learning rate)...")
    config1 = TrainConfig(
        output_dir="models/run1_lr2e-4",
        num_train_epochs=3,
        learning_rate=2e-4
    )
    trainer1 = LLMTrainer("sarvamai/sarvam-1", config=config1.to_dict())
    trainer1.train(tokenized_path)

    print("\n🎯 Training Run #2 (high learning rate)...")
    config2 = TrainConfig(
        output_dir="models/run2_lr5e-4",
        num_train_epochs=3,
        learning_rate=5e-4
    )
    trainer2 = LLMTrainer("sarvamai/sarvam-1", config=config2.to_dict())
    trainer2.train(tokenized_path)

    print("\n✅ Both runs complete - no re-tokenization needed!")
    print("   This saved hours of CPU time!")


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Full Pipeline", example_1_full_pipeline),
        "2": ("Skip PDF Stage", example_2_skip_pdf_stage),
        "3": ("Pre-tokenization Only", example_3_only_tokenize),
        "4": ("Manual Stage Control", example_4_manual_stages),
        "5": ("Quick Helpers", example_5_quick_helpers),
        "6": ("Custom Tokenizer", example_6_custom_tokenizer),
        "7": ("Reuse Tokenized Data", example_7_reuse_tokenized_data),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            _, func = examples[choice]
            func()
        else:
            print(f"Unknown example: {choice}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("🚀 Modular Training Pipeline Examples")
        print("=" * 60)
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print(f"\nUsage: python {sys.argv[0]} <example_number>")
        print(f"Example: python {sys.argv[0]} 5")
        print("\nOr run all examples:")
        print(f"  python {sys.argv[0]} all")

        if len(sys.argv) > 1 and sys.argv[1] == "all":
            for key, (name, func) in examples.items():
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Example {key} failed: {e}")
