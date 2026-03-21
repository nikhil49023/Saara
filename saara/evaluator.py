"""
Model Evaluator - Tests fine-tuned models using Granite 4 as a judge.
Implements a self-improvement loop for continuous model enhancement.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)


class TeacherClient:
    """
    Abstraction layer for different teacher models (Ollama, OpenAI, Google, Sarvam, etc.)
    """
    def __init__(self, config: Dict[str, Any]):
        self.provider = config.get("provider", "ollama")
        self.model_name = config.get("model", "granite4")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url") 
        
    def generate(self, prompt: str) -> str:
        """Generate response from the configured teacher."""
        try:
            if self.provider == "ollama":
                import ollama
                response = ollama.generate(model=self.model_name, prompt=prompt)
                return response['response'].strip()
                
            elif self.provider == "openai" or self.provider == "deepseek":
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                return response.text.strip()
            
            elif self.provider == "sarvam":
                # Sarvam AI - Sanskrit and Indian language specialist
                from sarvamai import SarvamAI
                client = SarvamAI(api_subscription_key=self.api_key)
                response = client.chat.completions(
                    messages=[{"role": "user", "content": prompt}]
                )
                # Extract text from response
                if hasattr(response, 'choices') and response.choices:
                    return response.choices[0].message.content.strip()
                elif isinstance(response, dict):
                    return response.get('choices', [{}])[0].get('message', {}).get('content', str(response))
                return str(response)
                
            elif self.provider == "huggingface":
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=self.api_key)
                # For open weights, we use text-generation
                response = client.text_generation(prompt, model=self.model_name, max_new_tokens=512)
                return response.strip()
                
            else:
                return f"[Error: Unknown provider {self.provider}]"
                
        except Exception as e:
            return f"[Error: {self.provider} generation failed: {e}]"


class ModelEvaluator:
    """
    Evaluates fine-tuned models using a teacher/judge model.
    Generates improvement data for iterative training.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Default to Gemini for high-quality evaluations
        # User will be prompted for API key when needed
        self.default_teacher_config = {
            "provider": "google",
            "model": "gemini-2.0-flash",
            "api_key": self.config.get('gemini', {}).get('api_key')
        }
        self.results_dir = Path("evaluations")
        self.results_dir.mkdir(exist_ok=True)
        
    def evaluate_adapter(
        self, 
        base_model_id: str,
        adapter_path: str,
        test_prompts: List[str] = None,
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate a fine-tuned adapter model.
        
        Args:
            base_model_id: HuggingFace ID of the base model
            adapter_path: Path to the LoRA adapter
            test_prompts: Custom test prompts (optional)
            num_samples: Number of test samples to evaluate
            
        Returns:
            Evaluation results with scores and improvement suggestions
        """
        console.print(Panel.fit(
            f"[bold cyan]🧪 Model Evaluation[/bold cyan]\n\n"
            f"Base Model: {base_model_id}\n"
            f"Adapter: {adapter_path}",
            title="Testing Fine-tuned Model",
            border_style="cyan"
        ))
        
        # Default test prompts if none provided
        if not test_prompts:
            test_prompts = self._get_default_test_prompts()
        
        # Sample prompts
        test_prompts = test_prompts[:num_samples]
        
        # Load the fine-tuned model
        console.print("\n[yellow]Loading fine-tuned model...[/yellow]")
        model, tokenizer = self._load_finetuned_model(base_model_id, adapter_path)
        
        if model is None:
            console.print("[red]Failed to load model[/red]")
            return {"success": False, "error": "Failed to load model"}
        
        # Generate responses
        console.print("\n[yellow]Generating responses...[/yellow]")
        responses = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Testing model...", total=len(test_prompts))
            
            for prompt in test_prompts:
                response = self._generate_response(model, tokenizer, prompt)
                responses.append({
                    "prompt": prompt,
                    "response": response
                })
                progress.advance(task)
        
        # Evaluate with Granite 4
        console.print("\n[yellow]Evaluating with Granite 4...[/yellow]")
        evaluations = self._evaluate_with_granite(responses)
        
        # Generate improvement data
        improvement_data = self._generate_improvement_data(evaluations)
        
        # Calculate overall score
        scores = [e.get("score", 5) for e in evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Display results
        self._display_results(evaluations, avg_score)
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "base_model": base_model_id,
            "adapter_path": adapter_path,
            "average_score": avg_score,
            "evaluations": evaluations,
            "improvement_data": improvement_data
        }
        
        results_file = self.results_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]Results saved to: {results_file}[/green]")
        
        # Save improvement data for next training round
        if improvement_data:
            improvement_file = self.results_dir / "corrections_for_training.jsonl"
            with open(improvement_file, 'a', encoding='utf-8') as f:
                for item in improvement_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            console.print(f"[green]Improvement data saved to: {improvement_file}[/green]")
        
        return results
    
    def _load_finetuned_model(self, base_model_id: str, adapter_path: str):
        """Load the base model with the fine-tuned adapter."""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model with 8-bit quantization for lower memory
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True  # Lower memory usage
            )
            
            # Load adapter
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            
            console.print("[green]✅ Model loaded successfully[/green]")
            return model, tokenizer
            
        except Exception as e:
            error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
            console.print(f"[red]Error loading model: {error_msg}[/red]")
            return None, None
    
    def _generate_response(self, model, tokenizer, prompt: str, max_length: int = 256) -> str:
        """Generate a response from the fine-tuned model."""
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"[Error: {e}]"
    
    def _evaluate_with_granite(self, responses: List[Dict]) -> List[Dict]:
        """Use Granite 4 to evaluate the model's responses."""
        try:
            import ollama
        except ImportError:
            console.print("[yellow]Ollama not available, using basic evaluation[/yellow]")
            return [{"prompt": r["prompt"], "response": r["response"], "score": 5, "feedback": "N/A"} for r in responses]
        
        evaluations = []
        
        for r in responses:
            eval_prompt = f"""You are an AI quality evaluator. Rate this AI response on a scale of 1-10.
Provide a brief explanation and suggest improvements.

User Question: {r['prompt']}

AI Response: {r['response']}

Respond in this exact JSON format:
{{"score": <1-10>, "feedback": "<brief explanation>", "improved_response": "<your better answer if score < 7>"}}
"""
            try:
                result = ollama.generate(model=self.default_teacher_config['model'], prompt=eval_prompt)
                response_text = result.get('response', '{}')
                
                # Try to parse JSON from response
                try:
                    # Find JSON in response
                    import re
                    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                    if json_match:
                        eval_data = json.loads(json_match.group())
                    else:
                        eval_data = {"score": 5, "feedback": response_text[:200]}
                except:
                    eval_data = {"score": 5, "feedback": response_text[:200]}
                
                evaluations.append({
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "score": eval_data.get("score", 5),
                    "feedback": eval_data.get("feedback", ""),
                    "improved_response": eval_data.get("improved_response", "")
                })
                
            except Exception as e:
                evaluations.append({
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "score": 5,
                    "feedback": f"Evaluation error: {e}",
                    "improved_response": ""
                })
        
        return evaluations
    
    def _generate_improvement_data(self, evaluations: List[Dict]) -> List[Dict]:
        """
        Generate training data from low-scoring responses.
        Uses Granite 4's improved responses as ground truth.
        """
        improvement_data = []
        
        for eval_item in evaluations:
            if eval_item.get("score", 10) < 7 and eval_item.get("improved_response"):
                # Create ShareGPT format correction data
                improvement_data.append({
                    "conversations": [
                        {"from": "human", "value": eval_item["prompt"]},
                        {"from": "gpt", "value": eval_item["improved_response"]}
                    ],
                    "source": "granite4_correction",
                    "original_score": eval_item["score"]
                })
        
        return improvement_data
    
    def _display_results(self, evaluations: List[Dict], avg_score: float):
        """Display evaluation results in a nice table."""
        results_table = Table(title="📊 Evaluation Results", show_header=True, header_style="bold magenta")
        results_table.add_column("#", style="cyan", width=4)
        results_table.add_column("Prompt", style="dim", max_width=30)
        results_table.add_column("Score", style="yellow", width=8)
        results_table.add_column("Feedback", max_width=40)
        
        for i, item in enumerate(evaluations, 1):
            score = item.get("score", 0)
            score_color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
            results_table.add_row(
                str(i),
                item["prompt"][:30] + "...",
                f"[{score_color}]{score}/10[/{score_color}]",
                item.get("feedback", "")[:40]
            )
        
        console.print(results_table)
        
        # Summary panel
        score_color = "green" if avg_score >= 7 else "yellow" if avg_score >= 5 else "red"
        summary = f"""
[bold]Average Score:[/bold] [{score_color}]{avg_score:.1f}/10[/{score_color}]

[bold]Recommendations:[/bold]
"""
        if avg_score < 5:
            summary += "• Model needs significant improvement\n• Consider more training data\n• Review low-scoring examples"
        elif avg_score < 7:
            summary += "• Model is decent but can improve\n• Use correction data for next training round"
        else:
            summary += "• Model is performing well!\n• Consider testing on harder examples"
        
        console.print(Panel(summary, title="📈 Summary", border_style="cyan"))
    
    def _get_default_test_prompts(self) -> List[str]:
        """Default test prompts for Ayurveda domain."""
        return [
            "What is Ayurveda and its main principles?",
            "Explain the three doshas in Ayurveda.",
            "What are the benefits of Ashwagandha?",
            "How does Ayurveda treat digestive disorders?",
            "What is Panchakarma therapy?",
            "Explain the concept of Prakriti in Ayurveda.",
            "What herbs are used for immunity in Ayurveda?",
            "How does Ayurveda view the mind-body connection?",
            "What is the role of diet in Ayurvedic treatment?",
            "Describe Ayurvedic remedies for common cold.",
        ]

    
    def run_autonomous_learning(self, base_model_id: str, adapter_path: str, topic: str, 
                                num_iterations: int = 10, teacher_config: Dict = None):
        """
        Run an autonomous learning session where the model 'chats' with a Teacher Model.
        """
        teacher_config = teacher_config or self.default_teacher_config
        teacher_name = f"{teacher_config['provider']}/{teacher_config['model']}"
        
        console.print(Panel.fit(
            f"[bold cyan]🧠 Autonomous Learning Session[/bold cyan]\n\n"
            f"Topic: {topic}\n"
            f"Iterations: {num_iterations}\n"
            f"Teacher: {teacher_name}",
            title="Self-Improvement Mode",
            border_style="green"
        ))
        
        # Initialize Teacher
        teacher = TeacherClient(teacher_config)
        
        # Load Student Model
        console.print("[yellow]Loading Student Model (Fine-tuned)...[/yellow]")
        model, tokenizer = self._load_finetuned_model(base_model_id, adapter_path)
        if not model:
            return

        new_training_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Learning...", total=num_iterations)
            
            for i in range(num_iterations):
                # 1. Teacher generates a question
                progress.update(task, description=f"Generating question {i+1}/{num_iterations}...")
                teacher_prompt = f"Ask a challenging question about '{topic}' that an expert should know. Output ONLY the question."
                teacher_q = teacher.generate(teacher_prompt)
                
                # Validation
                if "[Error" in teacher_q:
                    console.print(f"[red]{teacher_q}[/red]")
                    break
                
                # 2. Student answers
                progress.update(task, description=f"Student answering Q{i+1}...")
                student_ans = self._generate_response(model, tokenizer, teacher_q)
                
                # 3. Teacher evaluates and provides ideal answer
                progress.update(task, description=f"Teacher correcting Q{i+1}...")
                critique_prompt = f"""
                You are an expert teacher. 
                Question: {teacher_q}
                Student Answer: {student_ans}
                
                Analyze the student's answer. If it is perfect, output "PERFECT".
                If it has errors or missing info, provide the IDEAL, CORRECT answer.
                Output ONLY the ideal answer.
                """
                teacher_feedback = teacher.generate(critique_prompt)
                
                if "PERFECT" not in teacher_feedback and len(teacher_feedback) > 10 and "[Error" not in teacher_feedback:
                    # Create training sample from Teacher's correction
                    new_training_data.append({
                        "conversations": [
                            {"from": "user", "value": teacher_q},
                            {"from": "assistant", "value": teacher_feedback}
                        ],
                        "source": "autonomous_learning",
                        "topic": topic
                    })
                    console.print(f"[green]✓ Learned new concept: {teacher_q[:30]}...[/green]")
                else:
                    console.print(f"[dim]✓ Student answered correctly: {teacher_q[:30]}...[/dim]")
                
                progress.advance(task)
        
        # Save learned data
        if new_training_data:
            save_path = self.results_dir / f"learned_data_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(save_path, 'w', encoding='utf-8') as f:
                for item in new_training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            console.print(Panel(f"""
[bold green]Learning Session Complete![/bold green]

[yellow]New Training Samples:[/yellow] {len(new_training_data)}
[yellow]Saved To:[/yellow] {save_path}

You can now retrain your model with this data to improve its knowledge on '{topic}'.
""", title="Session Summary", border_style="green"))
            
            return str(save_path)
        else:
            console.print("[yellow]No new data generated (Student answered everything correctly!).[/yellow]")
            return None
    
    def _load_pretrained_model(self, model_path: str):
        """Load a pre-trained model directly (no adapter)."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()
            
            console.print("[green]✅ Pre-trained model loaded successfully[/green]")
            return model, tokenizer
            
        except Exception as e:
            error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
            console.print(f"[red]Error loading model: {error_msg}[/red]")
            return None, None
    
    def evaluate_pretrained(self, model_path: str, test_prompts: List[str] = None, num_samples: int = 10) -> Dict[str, Any]:
        """
        Evaluate a pre-trained model (without adapter).
        """
        console.print(Panel.fit(
            f"[bold cyan]🧪 Pre-trained Model Evaluation[/bold cyan]\n\n"
            f"Model: {model_path}",
            title="Testing Pre-trained Model",
            border_style="cyan"
        ))
        
        # Default test prompts if none provided
        if not test_prompts:
            test_prompts = self._get_default_test_prompts()
        
        # Sample prompts
        test_prompts = test_prompts[:num_samples]
        
        # Load the pre-trained model
        console.print("\n[yellow]Loading pre-trained model...[/yellow]")
        model, tokenizer = self._load_pretrained_model(model_path)
        
        if model is None:
            console.print("[red]Failed to load model[/red]")
            return {"success": False, "error": "Failed to load model"}
        
        # Generate responses
        console.print("\n[yellow]Generating responses...[/yellow]")
        responses = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Testing model...", total=len(test_prompts))
            
            for prompt in test_prompts:
                response = self._generate_response(model, tokenizer, prompt)
                responses.append({
                    "prompt": prompt,
                    "response": response
                })
                progress.advance(task)
        
        # Evaluate with Granite 4
        console.print("\n[yellow]Evaluating with Granite 4...[/yellow]")
        evaluations = self._evaluate_with_granite(responses)
        
        # Calculate overall score
        scores = [e.get("score", 5) for e in evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Display results
        self._display_results(evaluations, avg_score)
        
        return {"success": True, "average_score": avg_score, "evaluations": evaluations}
    
    def run_autonomous_learning_pretrained(self, model_path: str, topic: str, 
                                           iterations: int = 10, teacher_config: Dict = None):
        """
        Run autonomous learning for a pre-trained model.
        """
        teacher_config = teacher_config or self.default_teacher_config
        teacher_name = f"{teacher_config['provider']}/{teacher_config['model']}"
        
        console.print(Panel.fit(
            f"[bold cyan]🧠 Autonomous Learning Session[/bold cyan]\n\n"
            f"Model: {model_path}\n"
            f"Topic: {topic}\n"
            f"Iterations: {iterations}\n"
            f"Teacher: {teacher_name}",
            title="Self-Improvement Mode (Pre-trained)",
            border_style="green"
        ))
        
        # Initialize Teacher
        teacher = TeacherClient(teacher_config)
        
        # Load Pre-trained Model
        console.print("[yellow]Loading Pre-trained Model...[/yellow]")
        model, tokenizer = self._load_pretrained_model(model_path)
        if not model:
            return None

        new_training_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Learning...", total=iterations)
            
            for i in range(iterations):
                # 1. Teacher generates a question
                progress.update(task, description=f"Generating question {i+1}/{iterations}...")
                teacher_prompt = f"Ask a challenging question about '{topic}' that an expert should know. Output ONLY the question."
                teacher_q = teacher.generate(teacher_prompt)
                
                # Validation
                if "[Error" in teacher_q:
                    console.print(f"[red]{teacher_q}[/red]")
                    break
                
                # 2. Student answers
                progress.update(task, description=f"Student answering Q{i+1}...")
                student_ans = self._generate_response(model, tokenizer, teacher_q)
                
                # 3. Teacher evaluates and provides ideal answer
                progress.update(task, description=f"Teacher correcting Q{i+1}...")
                critique_prompt = f"""
                You are an expert teacher. 
                Question: {teacher_q}
                Student Answer: {student_ans}
                
                Analyze the student's answer. If it is perfect, output "PERFECT".
                If it has errors or missing info, provide the IDEAL, CORRECT answer.
                Output ONLY the ideal answer.
                """
                teacher_feedback = teacher.generate(critique_prompt)
                
                if "PERFECT" not in teacher_feedback and len(teacher_feedback) > 10 and "[Error" not in teacher_feedback:
                    # Create training sample from Teacher's correction
                    new_training_data.append({
                        "conversations": [
                            {"from": "user", "value": teacher_q},
                            {"from": "assistant", "value": teacher_feedback}
                        ],
                        "source": "autonomous_learning",
                        "topic": topic
                    })
                    console.print(f"[green]✓ Learned new concept: {teacher_q[:30]}...[/green]")
                else:
                    console.print(f"[dim]✓ Student answered correctly: {teacher_q[:30]}...[/dim]")
                
                progress.advance(task)
        
        # Save learned data
        if new_training_data:
            save_path = self.results_dir / f"pretrained_learned_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(save_path, 'w', encoding='utf-8') as f:
                for item in new_training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            console.print(Panel(f"""
[bold green]Learning Session Complete![/bold green]

[yellow]New Training Samples:[/yellow] {len(new_training_data)}
[yellow]Saved To:[/yellow] {save_path}

You can now fine-tune your pre-trained model with this data to improve its knowledge on '{topic}'.
""", title="Session Summary", border_style="green"))
            
            return str(save_path)
        else:
            console.print("[yellow]No new data generated (Student answered everything correctly!).[/yellow]")
            return None


def run_evaluation(base_model: str, adapter_path: str, config: dict = None):
    """Run model evaluation from CLI."""
    evaluator = ModelEvaluator(config)
    return evaluator.evaluate_adapter(base_model, adapter_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        base = sys.argv[1]
        adapter = sys.argv[2]
    else:
        base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        adapter = "models/TinyLlama-1.1B-Chat-v1.0-finetuned/final_adapter"
    
    run_evaluation(base, adapter)
