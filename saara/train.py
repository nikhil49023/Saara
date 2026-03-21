"""
Fine-tuning Module for Sarvam-1
Trains the Sarvam-1 model on the distilled dataset using LoRA.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable

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

logger = logging.getLogger(__name__)

class LLMTrainer:
    """
    Fine-tunes a base model using QLoRA.
    """

    def __init__(
        self,
        model_id: str = "sarvamai/sarvam-1",
        adapter_path: Optional[str] = None,
        config: Dict[str, Any] = None,
        on_progress: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize LLMTrainer.

        Args:
            model_id: Base model ID (default: sarvamai/sarvam-1)
            adapter_path: Path to existing adapter for continuation training
            config: Configuration dictionary or TrainConfig dataclass
            on_progress: Optional callback for progress updates
        """
        from saara.config import convert_config, TrainConfig

        # Convert config if needed
        if config is not None and isinstance(config, dict):
            self.config = convert_config(config, TrainConfig).to_dict()
        else:
            self.config = config or {}

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.on_progress = on_progress

        # Setup logging callback
        self._progress = self._make_progress_fn()

        # Use provided output directory or default
        if self.config.get("output_dir"):
            self.output_dir = Path(self.config["output_dir"])
        elif adapter_path:
             # Create a unique name for the continuation
             base_name = model_id.split('/')[-1]
             parent_name = Path(adapter_path).parent.name
             self.output_dir = Path(f"models/{parent_name}-refined")
        else:
             self.output_dir = Path(f"models/{model_id.split('/')[-1]}-finetuned")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for separation
        self.model_output_dir = self.output_dir / "model"
        self.dataset_output_dir = self.output_dir / "dataset"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters (optimized for CPU/GPU)
        self.train_params = {
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 1,  # Small batch for memory safety
            "gradient_accumulation_steps": 8,  # Accumulate for effective batch of 8
            "num_train_epochs": 1,             # 1 epoch for testing
            "max_seq_length": 512,             # Reduced for memory safety
            "logging_steps": 5,                # More frequent logging
            "save_steps": 100,                 # Less frequent saves
            "optim": "adamw_torch",            # Standard optimizer
            "warmup_steps": 10,                # Quick warmup
            "dataloader_pin_memory": False,    # Disable for CPU
            "fp16": False,                     # Default false, enabled if GPU available
            "bf16": False,
        }

    def _make_progress_fn(self) -> Callable[[str], None]:
        """Create a progress function that logs and calls callback."""
        def progress(msg: str) -> None:
            logger.info(msg)
            if self.on_progress:
                self.on_progress(msg)
        return progress

    def train(self, data_path: str, resume_from_checkpoint: Optional[str] = None, dataset_config_name: Optional[str] = None):
        """
        Start fine-tuning process.

        Args:
            data_path: Path to the JSONL training data OR Hugging Face dataset ID
            resume_from_checkpoint: Path to a checkpoint to resume from (optional)
            dataset_config_name: Optional configuration name for Hugging Face datasets (e.g., 'english', 'v1.0')
        """
        from peft import PeftModel

        # Log training configuration
        logger.debug("=" * 60)
        logger.debug("Fine-tuning Configuration")
        logger.debug("=" * 60)
        logger.debug(f"Base Model: {self.model_id}")
        if self.adapter_path:
            logger.debug(f"Starting Adapter: {self.adapter_path}")

        if isinstance(data_path, list):
            logger.debug(f"Training Data: {len(data_path)} files (batch mode)")
        else:
            logger.debug(f"Training Data: {data_path}")
            if dataset_config_name:
                logger.debug(f"Dataset Config: {dataset_config_name}")

        logger.debug(f"Output Directory: {self.output_dir}")
        logger.debug(f"Batch Size: {self.train_params['per_device_train_batch_size']}")
        logger.debug(f"Learning Rate: {self.train_params['learning_rate']}")
        logger.debug(f"Epochs: {self.train_params['num_train_epochs']}")
        logger.debug(f"Max Seq Length: {self.train_params['max_seq_length']}")
        if resume_from_checkpoint:
            logger.debug(f"Resume From: {resume_from_checkpoint}")
        logger.debug("=" * 60)
        
        # 1. Load Dataset (supports single file, list of files, or Hub ID)
        try:
            from datasets import concatenate_datasets
            
            if isinstance(data_path, list):
                # Batch loading - load each file and concatenate
                logger.info(f"Loading {len(data_path)} files...")
                datasets_list = []
                
                for i, file_path in enumerate(data_path):
                    try:
                        ds = load_dataset("json", data_files=file_path, split="train")
                        datasets_list.append(ds)
                        logger.info(f"  [dim]+ {Path(file_path).name}: {len(ds)} samples[/dim]")
                    except Exception as e:
                        logger.info(f"  [red]Skipped {Path(file_path).name}: {e}[/red]")
                
                if not datasets_list:
                    logger.error(f"Failed to load any datasets!")
                    return
                
                # Concatenate all datasets
                dataset = concatenate_datasets(datasets_list)
                logger.info(f"SUCCESS: Loaded {len(dataset)} total training examples from {len(datasets_list)} files")
            else:
                # Single file or Hub loading
                if str(data_path).endswith('.json') or str(data_path).endswith('.jsonl'):
                    dataset = load_dataset("json", data_files=data_path, split="train")
                    logger.info(f"SUCCESS: Loaded {len(dataset)} training examples from file")
                else:
                    # Assume Hugging Face Hub ID
                    try:
                        logger.info(f"⏳ Downloading dataset from Hub: {data_path} ({dataset_config_name or 'default'})...")
                        dataset = load_dataset(data_path, dataset_config_name, split="train")
                        logger.info(f"SUCCESS: Loaded {len(dataset)} training examples from Hub")
                    except ValueError as e:
                        # Handle missing config name error
                        error_str = str(e)
                        if "Config name is missing" in error_str or "pick one among" in error_str:
                            logger.warning(f"This dataset requires a specific configuration name.")
                            logger.debug(f"Error: {error_str}")
                            
                            new_config = input("Please enter a configuration name (e.g., \'stage1\') [default: \1]: ").strip() or "stage1"
                            
                            logger.info(f"⏳ Retrying with config: {new_config}...")
                            dataset = load_dataset(data_path, new_config, split="train")
                            logger.info(f"SUCCESS: Loaded {len(dataset)} training examples from Hub")
                        else:
                            raise e
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
        
        # 1.5 Data Preparation (Optional optimization)
        dataset = self._prepare_dataset(dataset)
        
        # Save prepared dataset to output folder
        logger.info(f"\n==== 💾 Saving prepared dataset to {self.dataset_output_dir}... ====")
        try:
            dataset.save_to_disk(str(self.dataset_output_dir))
            logger.info(f"SUCCESS: Dataset saved successfully")
        except Exception as e:
            logger.warning(f"Could not save dataset copy: {e}")

        logger.info(f"\n==== 🔄 Pulling/Loading Model & Tokenizer: {self.model_id}... ====")

        # 2. Load Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"SUCCESS: Tokenizer Loaded")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
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
        
        peft_config = None
        
        if self.adapter_path:
            # 4a. Load existing adapter
            logger.info(f"[bold cyan]🔄 Loading existing adapter: {self.adapter_path}...[/bold cyan]")
            model = PeftModel.from_pretrained(model, self.adapter_path, is_trainable=True)
            logger.info("SUCCESS: Adapter loaded and set to trainable")
        else:
            # 4b. Create new LoRA Config
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
            output_dir=str(self.model_output_dir),
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
        logger.info("\n[bold green]▶️ Starting training loop...[/bold green]\n")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 8. Save
        adapter_path = self.model_output_dir / "final_adapter"
        logger.info("\n==== 💾 Saving adapter model... ====")
        trainer.model.save_pretrained(adapter_path)
        
        # Display success summary
        success_msg = (
            "[bold green]✅ Training Complete![/bold green]\n\n"
            "[yellow]Adapter Model Saved To:[/yellow]\n"
            f"  {adapter_path}\n\n"
            "[yellow]To use this model:[/yellow]\n"
            "  from peft import PeftModel\n"
            f"  model = PeftModel.from_pretrained(base_model, \"{adapter_path}\")"
        )
        logger.info("=" * 70)
        logger.info("Process complete")
        logger.info("=" * 70)
        logger.info(success_msg)


    def _format_prompts(self, example):
        """
        Dynamically format prompts based on the dataset structure.
        Supports: ShareGPT, Alpaca, Q&A, and Raw Text.
        New TRL API - receives single example, returns single string.
        """
        
        # 1. ShareGPT Format (Multi-turn conversation)
        if 'conversations' in example:
            conversation_list = example['conversations']
            text = ""
            for msg in conversation_list:
                role = msg.get('from', msg.get('role', ''))
                content = msg.get('value', msg.get('content', ''))
                # Handle both ShareGPT (human/gpt) and standard (user/assistant) formats
                if role in ['human', 'user']:
                    text += f"User: {content}\n"
                elif role in ['gpt', 'assistant']:
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
        
        # 3. Q&A Format (New Support)
        elif 'question' in example and 'answer' in example:
            return f"Question: {example['question']}\nAnswer: {example['answer']}"
                
        # 4. Raw Text Format (Pre-training / Completion)
        elif 'text' in example:
            return example['text']
            
        else:
            # Fallback for unknown format
            # Try to return first available column to avoid crashing with None
            try:
                first_col = list(example.keys())[0]
                return str(example[first_col])
            except Exception:
                return "" # Safest fallback (will likely be filtered out by tokenizer or short length)

    def _prepare_dataset(self, dataset):
        """
        Prepare and optimize dataset for training.
        Uses Granite 4 via Ollama to validate and fix data issues.
        """
        logger.info("\n==== 🔧 Preparing Dataset... ====")
        
        original_count = len(dataset)
        
        # Step 0: Normalize role names (ShareGPT human/gpt -> user/assistant)
        def normalize_roles(example):
            if 'conversations' in example and example['conversations'] is not None:
                convs = example['conversations']
                if isinstance(convs, list):
                    for msg in convs:
                        if isinstance(msg, dict):
                            if msg.get('from') == 'human':
                                msg['from'] = 'user'
                            elif msg.get('from') == 'gpt':
                                msg['from'] = 'assistant'
                            # Also handle 'role' key
                            if msg.get('role') == 'human':
                                msg['role'] = 'user'
                            elif msg.get('role') == 'gpt':
                                msg['role'] = 'assistant'
            return example
        
        dataset = dataset.map(normalize_roles)
        
        # Step 1: Filter out empty/invalid samples
        def is_valid(example):
            try:
                if 'conversations' in example:
                    convs = example['conversations']
                    if convs is None or not isinstance(convs, list) or len(convs) < 2:
                        return False
                    # Check that at least 2 messages have valid content
                    valid_msgs = 0
                    for msg in convs:
                        if msg is None or not isinstance(msg, dict):
                            continue
                        content = msg.get('value') or msg.get('content')
                        if content and len(str(content).strip()) > 0:
                            valid_msgs += 1
                    return valid_msgs >= 2
                elif 'instruction' in example and 'output' in example:
                    return bool(example['instruction']) and bool(example['output'])
                elif 'question' in example and 'answer' in example:
                    return bool(example['question']) and bool(example['answer'])
                elif 'text' in example:
                    text = example['text']
                    return text is not None and bool(text) and len(str(text)) > 50
                return False
            except Exception:
                return False
        
        dataset = dataset.filter(is_valid)
        filtered_count = len(dataset)
        
        if filtered_count < original_count:
            logger.info(f"  [dim]Removed {original_count - filtered_count} invalid samples[/dim]")
        
        # Step 2: Truncate very long samples for faster training
        def truncate_sample(example):
            max_chars = 4000  # ~1000 tokens
            if 'conversations' in example and example['conversations'] is not None:
                convs = example['conversations']
                if isinstance(convs, list):
                    for msg in convs:
                        if msg is None or not isinstance(msg, dict):
                            continue
                        # Handle both 'value' (ShareGPT) and 'content' (standard) keys
                        for key in ['value', 'content']:
                            if key in msg and msg[key] is not None and len(str(msg[key])) > max_chars:
                                msg[key] = str(msg[key])[:max_chars] + "..."
            elif 'text' in example and example['text'] is not None:
                if len(str(example['text'])) > max_chars:
                    example['text'] = str(example['text'])[:max_chars] + "..."
            return example
        
        dataset = dataset.map(truncate_sample)
        
        # Step 3: Shuffle for better training
        dataset = dataset.shuffle(seed=42)
        
        logger.info(f"  [green]✅ Dataset ready: {len(dataset)} samples")
        
        return dataset


# ============================================================================
# Autonomous Fine-tuning with Teacher AI
# ============================================================================

class AutonomousFineTuner:
    """
    Autonomous fine-tuning pipeline with teacher AI (Sarvam/Gemini).
    
    The teacher generates high-quality Q&A pairs for fine-tuning,
    which is much more practical than pre-training from scratch.
    """
    
    def __init__(self,
                 base_model: str = "google/gemma-2-2b",
                 teacher_config: Dict[str, Any] = None,
                 output_dir: str = "datasets"):
        
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Teacher configuration
        self.teacher_config = teacher_config or {
            "provider": "google",
            "model": "gemini-2.0-flash-exp"
        }
        
        self.teacher = None
        
    def _init_teacher(self):
        """Initialize teacher model."""
        from saara.evaluator import TeacherClient
        self.teacher = TeacherClient(self.teacher_config)
        
    def generate_qa_pairs(self, topic: str, num_pairs: int = 10) -> list:
        """Generate Q&A pairs on a topic."""
        if not self.teacher:
            self._init_teacher()
        
        prompt = f"""Generate {num_pairs} high-quality question-answer pairs about: {topic}

Requirements:
- Questions should be diverse (what, why, how, when, compare, explain)
- Answers should be detailed (50-200 words each)
- Include specific facts, examples, and explanations
- Cover different difficulty levels (beginner to advanced)

Format EXACTLY as:
Q: [question]
A: [detailed answer]

Q: [question]
A: [detailed answer]

Generate Q&A pairs about {topic}:"""

        response = self.teacher.generate(prompt)
        
        # Parse Q&A pairs
        pairs = []
        lines = response.split('\n')
        current_q = None
        current_a = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                if current_q and current_a:
                    pairs.append({
                        "question": current_q,
                        "answer": ' '.join(current_a)
                    })
                current_q = line[2:].strip()
                current_a = []
            elif line.startswith('A:'):
                current_a = [line[2:].strip()]
            elif current_a is not None and line:
                current_a.append(line)
        
        # Don't forget the last pair
        if current_q and current_a:
            pairs.append({
                "question": current_q,
                "answer": ' '.join(current_a)
            })
        
        return pairs
    
    def generate_curriculum(self, domain: str, num_topics: int = 20) -> list:
        """Generate curriculum topics for the domain."""
        if not self.teacher:
            self._init_teacher()
            
        prompt = f"""Generate {num_topics} specific topics for teaching an AI about: {domain}

Order from basic to advanced. Output ONLY a numbered list.

Topics for {domain}:"""
        
        response = self.teacher.generate(prompt)
        
        topics = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                topic = line.lstrip('0123456789.-) ').strip()
                if topic and len(topic) > 5:
                    topics.append(topic)
        
        # Fallback if parsing fails
        if not topics:
            topics = [
                f"Introduction to {domain}",
                f"Basic concepts of {domain}",
                f"History of {domain}",
                f"Key principles in {domain}",
                f"Practical applications of {domain}",
            ]
        
        return topics[:num_topics]
    
    def run_autonomous_generation(self,
                                  domain: str,
                                  target_pairs: int = 500,
                                  quality_threshold: int = 7) -> str:
        """
        Autonomously generate fine-tuning dataset.
        
        Returns path to generated JSONL file.
        """
        import json
        
        logger.info("=" * 70)
        logger.info("Autonomous Fine-tuning Data Generation")
        logger.info("=" * 70)
        
        if not self.teacher:
            self._init_teacher()
        
        # Step 1: Generate curriculum
        logger.info("\n==== Step 1: Generating Curriculum ====")
        topics = self.generate_curriculum(domain, num_topics=25)
        logger.info(f"[green]✓ Generated {len(topics)} topics[/green]")
        
        for i, t in enumerate(topics[:5], 1):
            logger.info(f"  [dim]{i}. {t}[/dim]")
        if len(topics) > 5:
            logger.info(f"  [dim]... and {len(topics) - 5} more[/dim]")
        
        # Step 2: Generate Q&A pairs
        logger.info("\n==== Step 2: Generating Q&A Pairs ====")
        
        all_pairs = []
        pairs_per_topic = max(5, target_pairs // len(topics))
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task(f"Generating Q&A pairs...", total=len(topics))
            
            for topic in topics:
                try:
                    pairs = self.generate_qa_pairs(f"{domain}: {topic}", num_pairs=pairs_per_topic)
                    all_pairs.extend(pairs)
                    logger.info(f"  [dim]Generated {len(pairs)} pairs for: {topic[:40]}...[/dim]")
                except Exception as e:
                    logger.info(f"  [yellow]⚠ Error for {topic}: {e}[/yellow]")
                
                progress.advance(task)
                
                if len(all_pairs) >= target_pairs * 1.2:
                    break
        
        logger.info(f"[green]✓ Generated {len(all_pairs)} total Q&A pairs[/green]")
        
        # Step 3: Convert to ShareGPT format
        logger.info("\n==== Step 3: Formatting for Fine-tuning ====")
        
        formatted_data = []
        for pair in all_pairs:
            if pair['question'] and pair['answer'] and len(pair['answer']) > 20:
                formatted_data.append({
                    "conversations": [
                        {"role": "user", "content": pair['question']},
                        {"role": "assistant", "content": pair['answer']}
                    ]
                })
        
        # Step 4: Save dataset
        output_file = self.output_dir / f"{domain.replace(' ', '_').lower()}_finetune.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in formatted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
                logger.info("=" * 70)
                logger.info("Process complete")
                logger.info("=" * 70)
                logger.info(
                        "✅ Fine-tuning dataset ready | "
                        f"Q&A Pairs: {len(formatted_data)} | "
                        f"Output File: {output_file} | "
                        "Format: ShareGPT"
                )
        
        return str(output_file)
    
    def run_full_pipeline(self,
                          domain: str,
                          target_pairs: int = 500,
                          train_model: bool = True) -> Optional[str]:
        """Run full autonomous fine-tuning pipeline."""
        
        # Generate dataset
        dataset_path = self.run_autonomous_generation(domain, target_pairs)
        
        if not train_model:
            return dataset_path
        
        # Fine-tune model
        logger.info("\n==== Step 4: Fine-tuning Model ====")
        
        trainer = LLMTrainer(model_id=self.base_model)
        trainer.train(dataset_path)
        
        return str(trainer.output_dir)


if __name__ == "__main__":
    # Test
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "datasets/distilled_train.jsonl"
        
    trainer = LLMTrainer()
    trainer.train(path)

