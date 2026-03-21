"""
Example 2: Fine-tune a Model

Shows how to use SAARA as a Python library to fine-tune a model
on your prepared dataset using QLoRA.
"""

from saara import LLMTrainer, TrainConfig


def main():
    """Fine-tuning example."""

    # Create training configuration
    config = TrainConfig(
        model_id="sarvamai/sarvam-1",  # Base model to fine-tune
        output_dir="./models/fine_tuned",
        num_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        gradient_accumulation_steps=8,
        max_seq_length=2048,
    )

    # Initialize trainer
    trainer = LLMTrainer(
        model_id=config.model_id,
        config=config,
        on_progress=log_progress  # Optional callback for progress updates
    )

    # Path to your prepared dataset
    dataset_path = "./datasets/my_dataset/training.jsonl"

    print(f"Starting fine-tuning on {config.model_id}...")
    print(f"Config: epochs={config.num_epochs}, lr={config.learning_rate}")

    # Run training
    try:
        trainer.train(dataset_path)
        print("✓ Training completed successfully!")
        print(f"Model saved to: {trainer.output_dir}/model")
        print(f"Adapter saved to: {trainer.output_dir}/model/final_adapter")
    except Exception as e:
        print(f"✗ Training failed: {e}")


def log_progress(message: str):
    """Callback for progress updates during training."""
    print(f"[TRAINING] {message}")


if __name__ == "__main__":
    main()
