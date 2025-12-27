"""
Fine-tuning Module for Sarvam-1
Trains the Sarvam-1 model on the distilled dataset using LoRA.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

class LLMTrainer:
    """
    Fine-tunes a base model using QLoRA.
    """
    
    def __init__(self, model_id: str = "sarvamai/sarvam-1", config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_id = model_id
        self.output_dir = Path(f"models/{model_id.split('/')[-1]}-finetuned")
        
        # Training hyperparameters (optimized for speed + quality)
        self.train_params = {
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 8,  # Larger batch for speed
            "gradient_accumulation_steps": 2,  # Reduced for faster updates
            "num_train_epochs": 1,             # 1 epoch for testing
            "max_seq_length": 1024,            # Reduced for faster processing
            "logging_steps": 5,                # More frequent logging
            "save_steps": 100,                 # Less frequent saves
            "optim": "adamw_torch_fused",      # Faster optimizer
            "warmup_steps": 10,                # Quick warmup
        }

    def train(self, data_path: str, resume_from_checkpoint: Optional[str] = None):
        """
        Start fine-tuning process.
        
        Args:
            data_path: Path to the JSONL training data
            resume_from_checkpoint: Path to a checkpoint to resume from (optional)
        """
        from rich.table import Table
        from rich.panel import Panel
        
        # Display training configuration in a nice table
        config_table = Table(title="🚀 Fine-tuning Configuration", show_header=True, header_style="bold cyan")
        config_table.add_column("Parameter", style="green")
        config_table.add_column("Value", style="yellow")
        
        config_table.add_row("Base Model", self.model_id)
        config_table.add_row("Training Data", str(data_path))
        config_table.add_row("Output Directory", str(self.output_dir))
        config_table.add_row("Batch Size", str(self.train_params["per_device_train_batch_size"]))
        config_table.add_row("Learning Rate", str(self.train_params["learning_rate"]))
        config_table.add_row("Epochs", str(self.train_params["num_train_epochs"]))
        config_table.add_row("Max Seq Length", str(self.train_params["max_seq_length"]))
        if resume_from_checkpoint:
            config_table.add_row("Resume From", resume_from_checkpoint)
        
        console.print(config_table)
        console.print()
        
        # 1. Load Dataset
        try:
            dataset = load_dataset("json", data_files=data_path, split="train")
            console.print(f"[green]✅ Loaded {len(dataset)} training examples[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load dataset: {e}[/red]")
            return
        
        # 1.5 Data Preparation with Granite 4 (Optional optimization)
        dataset = self._prepare_dataset(dataset)

        console.print(f"\n[bold yellow]🔄 Pulling/Loading Model & Tokenizer: {self.model_id}...[/bold yellow]")

        # 2. Load Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = tokenizer.eos_token
            console.print("✅ Tokenizer Loaded")
        except Exception as e:
            console.print(f"[red]Failed to load tokenizer: {e}[/red]")
            return
        
        # 3. Load Base Model (Quantized)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True 
        )
        
        model.config.use_cache = False # Silence warnings
        model.config.pretraining_tp = 1
        model = prepare_model_for_kbit_training(model)
        
        # 4. LoRA Config
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=32,
            bias="none",
            task_type="CAUSAL_LM", 
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 5. Training Config (using SFTConfig for new TRL API)
        from trl import SFTConfig
        
        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.train_params["num_train_epochs"],
            per_device_train_batch_size=self.train_params["per_device_train_batch_size"],
            gradient_accumulation_steps=self.train_params["gradient_accumulation_steps"],
            optim=self.train_params["optim"],
            save_steps=self.train_params["save_steps"],
            logging_steps=self.train_params["logging_steps"],
            learning_rate=self.train_params["learning_rate"],
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_steps=self.train_params["warmup_steps"],
            group_by_length=True,
            lr_scheduler_type="linear",  # Faster than cosine for short runs
            report_to="tensorboard",
            max_length=self.train_params["max_seq_length"],
            gradient_checkpointing=True,  # Reduce memory usage
        )
        
        # 6. Initialize Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_args,
            formatting_func=self._format_prompts
        )
        
        # 7. Train
        console.print("\n[bold green]▶️ Starting training loop...[/bold green]\n")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 8. Save
        adapter_path = self.output_dir / "final_adapter"
        console.print("\n[bold cyan]💾 Saving adapter model...[/bold cyan]")
        trainer.model.save_pretrained(adapter_path)
        
        # Display success summary
        from rich.panel import Panel
        success_msg = f"""
[bold green]✅ Training Complete![/bold green]

[yellow]Adapter Model Saved To:[/yellow]
  {adapter_path}

[yellow]To use this model:[/yellow]
  from peft import PeftModel
  model = PeftModel.from_pretrained(base_model, "{adapter_path}")
"""
        console.print(Panel(success_msg, title="🎉 Success", border_style="green"))



    def _format_prompts(self, example):
        """
        Dynamically format prompts based on the dataset structure.
        Supports: ShareGPT, Alpaca, and Raw Text.
        New TRL API - receives single example, returns single string.
        """
        
        # 1. ShareGPT Format (Multi-turn conversation)
        if 'conversations' in example:
            conversation_list = example['conversations']
            text = ""
            for msg in conversation_list:
                role = msg['from']
                content = msg['value']
                if role == 'human':
                    text += f"User: {content}\n"
                elif role == 'gpt':
                    text += f"Assistant: {content}\n"
                elif role == 'system':
                    text += f"System: {content}\n"
            return text
                
        # 2. Alpaca Format (Instruction following)
        elif 'instruction' in example and 'output' in example:
            instr = example['instruction']
            inp = example.get('input', '')
            out = example['output']
            
            if inp:
                return f"Instruction: {instr}\nInput: {inp}\nResponse: {out}"
            else:
                return f"Instruction: {instr}\nResponse: {out}"
                
        # 3. Raw Text Format (Pre-training / Completion)
        elif 'text' in example:
            return example['text']
            
        else:
            # Fallback for unknown format
            logger.warning("Unknown dataset format. Expected 'conversations', 'instruction'/'output', or 'text'.")
            first_col = list(example.keys())[0]
            return str(example[first_col])

    def _prepare_dataset(self, dataset):
        """
        Prepare and optimize dataset for training.
        Uses Granite 4 via Ollama to validate and fix data issues.
        """
        console.print("\n[bold yellow]🔧 Preparing Dataset with Granite 4...[/bold yellow]")
        
        original_count = len(dataset)
        
        # Step 1: Filter out empty/invalid samples
        def is_valid(example):
            if 'conversations' in example:
                convs = example['conversations']
                return isinstance(convs, list) and len(convs) >= 2
            elif 'instruction' in example and 'output' in example:
                return bool(example['instruction']) and bool(example['output'])
            elif 'text' in example:
                return bool(example['text']) and len(example['text']) > 50
            return False
        
        dataset = dataset.filter(is_valid)
        filtered_count = len(dataset)
        
        if filtered_count < original_count:
            console.print(f"  [dim]Removed {original_count - filtered_count} invalid samples[/dim]")
        
        # Step 2: Truncate very long samples for faster training
        def truncate_sample(example):
            max_chars = 4000  # ~1000 tokens
            if 'conversations' in example:
                for msg in example['conversations']:
                    if len(msg.get('value', '')) > max_chars:
                        msg['value'] = msg['value'][:max_chars] + "..."
            elif 'text' in example:
                if len(example['text']) > max_chars:
                    example['text'] = example['text'][:max_chars] + "..."
            return example
        
        dataset = dataset.map(truncate_sample)
        
        # Step 3: Shuffle for better training
        dataset = dataset.shuffle(seed=42)
        
        console.print(f"  [green]✅ Dataset ready: {len(dataset)} samples[/green]")
        
        # Optional: Use Granite 4 for quality scoring (can be slow for large datasets)
        # Skipping by default for speed, but can be enabled
        # dataset = self._granite_quality_filter(dataset)
        
        return dataset
    
    def _granite_quality_filter(self, dataset, sample_size=50):
        """
        Use Granite 4 to score sample quality (optional, for advanced cleaning).
        Only processes a sample to avoid slowdown.
        """
        try:
            from src.ollama_client import OllamaClient
            client = OllamaClient(self.config.get('ollama', {}))
            
            if not client.check_health():
                console.print("[yellow]Granite 4 not available, skipping quality filter[/yellow]")
                return dataset
            
            console.print(f"  [dim]Running quality check on {sample_size} samples...[/dim]")
            
            # Sample a subset for quality check
            indices_to_check = list(range(min(sample_size, len(dataset))))
            low_quality_indices = []
            
            for idx in indices_to_check:
                example = dataset[idx]
                # Format for quality check
                if 'conversations' in example:
                    sample_text = str(example['conversations'])[:500]
                elif 'text' in example:
                    sample_text = example['text'][:500]
                else:
                    continue
                
                prompt = f"""Rate this training sample quality (1-10). Reply with just a number.
Sample: {sample_text}
Rating:"""
                
                response = client.generate(prompt, max_tokens=5)
                try:
                    score = int(response.strip().split()[0])
                    if score < 4:
                        low_quality_indices.append(idx)
                except:
                    pass
            
            if low_quality_indices:
                console.print(f"  [dim]Flagged {len(low_quality_indices)} low-quality samples[/dim]")
            
        except Exception as e:
            console.print(f"[dim]Quality filter skipped: {e}[/dim]")
        
        return dataset

if __name__ == "__main__":
    # Test
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "datasets/distilled_train.jsonl"
        
    trainer = LLMTrainer()
    trainer.train(path)
