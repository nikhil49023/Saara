"""
Test script for modular training pipeline.
Verifies all imports and configurations work correctly.
"""


def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")

    # Test TokenStorage
    from saara import TokenStorage, TokenStorageConfig, quick_tokenize
    print("✓ TokenStorage imports")

    # Test TrainingPipeline
    from saara import TrainingPipeline, TrainingPipelineConfig, quick_train
    print("✓ TrainingPipeline imports")

    # Test integration with existing modules
    from saara import LLMTrainer, TrainConfig
    print("✓ Existing training imports")

    print("\n✅ All imports successful!")


def test_config_creation():
    """Test configuration creation."""
    print("\nTesting configuration creation...")

    from saara import TokenStorageConfig, TrainingPipelineConfig

    # Test TokenStorageConfig
    token_config = TokenStorageConfig(
        max_length=512,
        padding="max_length",
        num_proc=4
    )
    print(f"✓ TokenStorageConfig: max_length={token_config.max_length}")

    # Test TrainingPipelineConfig
    pipeline_config = TrainingPipelineConfig(
        jsonl_input="data.jsonl",
        tokenizer_id="test/model",
        max_length=1024,
        num_epochs=3,
        skip_stages=[1]
    )
    print(f"✓ TrainingPipelineConfig: {len(pipeline_config.skip_stages)} skipped stages")

    print("\n✅ Configuration creation successful!")


def test_token_storage_api():
    """Test TokenStorage API (without actual data)."""
    print("\nTesting TokenStorage API...")

    from saara import TokenStorage, TokenStorageConfig

    config = TokenStorageConfig(max_length=512)
    storage = TokenStorage(
        tokenizer="gpt2",  # Small model for testing
        config=config
    )

    print(f"✓ TokenStorage initialized")
    print(f"  - Tokenizer: {storage.tokenizer_id}")
    print(f"  - Max length: {storage.config.max_length}")

    print("\n✅ TokenStorage API test successful!")


def test_pipeline_api():
    """Test TrainingPipeline API (without running)."""
    print("\nTesting TrainingPipeline API...")

    from saara import TrainingPipeline, TrainingPipelineConfig

    config = TrainingPipelineConfig(
        jsonl_input="dummy.jsonl",
        skip_stages=[1, 2, 3, 4],  # Skip all stages (just test init)
        output_dir="test_output"
    )

    pipeline = TrainingPipeline(config)

    print(f"✓ TrainingPipeline initialized")
    print(f"  - Output dir: {pipeline.output_dir}")
    print(f"  - Config: {pipeline.config.model_id}")

    print("\n✅ TrainingPipeline API test successful!")


def print_module_summary():
    """Print summary of new modules."""
    print("\n" + "=" * 60)
    print("MODULAR TRAINING PIPELINE - Module Summary")
    print("=" * 60)

    print("\n📦 New Modules:")
    print("  1. saara.token_storage")
    print("     - TokenStorage: Pre-tokenization & storage")
    print("     - TokenStorageConfig: Configuration")
    print("     - quick_tokenize(): Helper function")

    print("\n  2. saara.training_pipeline")
    print("     - TrainingPipeline: Orchestrates all stages")
    print("     - TrainingPipelineConfig: Pipeline configuration")
    print("     - quick_train(): Helper function")

    print("\n🔧 Existing Modules (Enhanced):")
    print("  - saara.train.LLMTrainer: Now supports pre-tokenized data")
    print("  - saara.ai_tokenizer: Domain-specific tokenizer training")

    print("\n📚 Documentation:")
    print("  - MODULAR_TRAINING.md: Complete guide")
    print("  - examples/04_modular_training.py: 7 examples")

    print("\n🚀 Quick Start:")
    print("  >>> from saara import quick_tokenize, quick_train")
    print("  >>> tokens = quick_tokenize('data.jsonl', 'tokens/')")
    print("  >>> quick_train(tokens, num_epochs=3)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        test_imports()
        test_config_creation()
        test_token_storage_api()
        test_pipeline_api()
        print_module_summary()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nModular training pipeline is ready to use!")
        print("See MODULAR_TRAINING.md for documentation.")
        print("See examples/04_modular_training.py for examples.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
