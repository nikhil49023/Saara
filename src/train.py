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

class SarvamTrainer:
    """
    Fine-tunes Sarvam-1 using QLoRA.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_id = "sarvamai/sarvam-1" 
        self.output_dir = Path("models/sarvam-finetuned")
        
        # Training hyperparameters
        self.train_params = {
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4, 
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 1, # Start with 1 epoch for testing
            "max_seq_length": 2048,
            "logging_steps": 10,
            "save_steps": 50,
            "optim": "paged_adamw_32bit"
        }

    def train(self, data_path: str):
        """
        Start fine-tuning process.
        
        Args:
            data_path: Path to the JSONL training data
        """
        console.print(f"[bold cyan]🚀 Starting Sarvam-1 Fine-tuning[/bold cyan]")
        console.print(f"Model: {self.model_id}")
        console.print(f"Data: {data_path}")
        
        # 1. Load Dataset
        try:
            dataset = load_dataset("json", data_files=data_path, split="train")
            console.print(f"Loaded {len(dataset)} examples")
        except Exception as e:
            console.print(f"[red]Failed to load dataset: {e}[/red]")
            return

        # 2. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
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
        
        # 5. Training Arguments
        training_args = TrainingArguments(
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
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="tensorboard"
        )
        
        # 6. Initialize Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text", # SFTTrainer expects 'text' if not formatting func
            max_seq_length=self.train_params["max_seq_length"],
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=self._format_prompts
        )
        
        # 7. Train
        console.print("[bold green]Starting training loop...[/bold green]")
        trainer.train()
        
        # 8. Save
        console.print("[bold cyan]Saving adapter model...[/bold cyan]")
        trainer.model.save_pretrained(self.output_dir / "final_adapter")
        console.print(f"[green]Training complete! Model saved to {self.output_dir}[/green]")



    # Better formatting function that handles list of dicts
    # SFTTrainer passes the batch (dict of lists) to this function
    
    def _format_prompts(self, examples):
        output_texts = []
        for conversation_list in examples['conversations']:
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
            output_texts.append(text)
        return output_texts

if __name__ == "__main__":
    # Test
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "datasets/distilled_train.jsonl"
        
    trainer = SarvamTrainer()
    trainer.train(path)
