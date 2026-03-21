"""
Model Deployment Module - Deploy fine-tuned models locally or to cloud.
Supports: Ollama (local), Google Cloud Run, HuggingFace Hub
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

console = Console()


class ModelDeployer:
    """
    Handles model deployment to various platforms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.exports_dir = Path("exports")
        self.exports_dir.mkdir(exist_ok=True)
    
    def deploy_menu(self, base_model_id: str, adapter_path: str):
        """Show deployment options menu."""
        console.print(Panel.fit(
            f"[bold cyan]🚀 Model Deployment[/bold cyan]\n\n"
            f"Base Model: {base_model_id}\n"
            f"Adapter: {adapter_path}",
            title="Deploy Your Model",
            border_style="green"
        ))
        
        deploy_table = Table(title="Deployment Options", show_header=True, header_style="bold magenta")
        deploy_table.add_column("Option", style="cyan", width=8)
        deploy_table.add_column("Platform", style="green")
        deploy_table.add_column("Description", style="dim")
        
        deploy_table.add_row("1", "Local Chat", "Test interactively in terminal")
        deploy_table.add_row("2", "Export to Ollama", "Convert to GGUF for Ollama")
        deploy_table.add_row("3", "Push to HuggingFace", "Upload to HuggingFace Hub")
        deploy_table.add_row("4", "Deploy to Cloud", "Google Cloud Run / Docker")
        deploy_table.add_row("5", "Export Merged Model", "Merge adapter with base model")
        deploy_table.add_row("6", "Back", "Return to main menu")
        
        console.print(deploy_table)
        console.print()
        
        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6"], default="1")
        
        if choice == "1":
            self.run_local_chat(base_model_id, adapter_path)
        elif choice == "2":
            self.export_to_ollama(base_model_id, adapter_path)
        elif choice == "3":
            self.push_to_huggingface(base_model_id, adapter_path)
        elif choice == "4":
            self.deploy_to_cloud(base_model_id, adapter_path)
        elif choice == "5":
            self.export_merged_model(base_model_id, adapter_path)
        else:
            return
    
    def run_local_chat(self, base_model_id: str, adapter_path: str):
        """Run interactive chat session with the fine-tuned or pre-trained model."""
        console.print("\n[bold cyan]Starting Local Chat...[/bold cyan]")
        console.print("[dim]Type 'quit' or 'exit' to stop.[/dim]\n")
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            console.print("[yellow]Loading model...[/yellow]")
            
            # Determine if this is a pre-trained model (no adapter) or fine-tuned (with adapter)
            is_pretrained = adapter_path is None or adapter_path == "None"
            
            if is_pretrained:
                # Pre-trained model - load directly
                console.print(f"[dim]Loading pre-trained model from {base_model_id}[/dim]")
                
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
            else:
                # Fine-tuned model with adapter
                from peft import PeftModel
                
                console.print(f"[dim]Loading base model: {base_model_id}[/dim]")
                console.print(f"[dim]Loading adapter: {adapter_path}[/dim]")
                
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_8bit=True  # Lower memory usage
                )
                model = PeftModel.from_pretrained(model, adapter_path)
            
            model.eval()
            
            # Move to CPU if no GPU
            if not torch.cuda.is_available():
                model = model.to("cpu")
                console.print("[yellow]Running on CPU (slower but works)[/yellow]")
            
            console.print("[green]Model loaded! Start chatting:[/green]\n")
            
            while True:
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                inputs = tokenizer(user_input, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(user_input):].strip()
                
                # Escape any brackets that might be in the response
                response_safe = response.replace("[", "\\[").replace("]", "\\]")
                console.print(f"[bold green]Assistant:[/bold green] {response_safe}\n")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Escape brackets in error message to prevent Rich markup errors
            error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
            console.print(f"[red]Error: {error_msg}[/red]")
    
    def export_to_ollama(self, base_model_id: str, adapter_path: str):
        """Export model to GGUF format for use with Ollama."""
        console.print("\n[bold cyan]Exporting to Ollama Format[/bold cyan]")
        
        model_name = base_model_id.split('/')[-1]
        merged_path = self.exports_dir / f"{model_name}-merged"
        
        # First merge the model
        if not merged_path.exists():
            console.print("[yellow]Merging adapter with base model first...[/yellow]")
            self.export_merged_model(base_model_id, adapter_path)
        
        # Create Modelfile
        modelfile_path = merged_path / "Modelfile"
        modelfile_content = f'''FROM .

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM "You are AyurGuru, an expert Ayurvedic AI assistant."
'''
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        console.print(Panel(f'''
[bold]To create an Ollama model:[/bold]

1. Install Ollama: https://ollama.com

2. Convert to GGUF (requires llama.cpp):
   python convert_hf_to_gguf.py {merged_path}

3. Create Ollama model:
   cd {merged_path}
   ollama create {model_name}-finetuned -f Modelfile

4. Run your model:
   ollama run {model_name}-finetuned
''', title="Ollama Export Instructions", border_style="cyan"))
    
    def push_to_huggingface(self, base_model_id: str, adapter_path: str):
        """Push the fine-tuned model to HuggingFace Hub."""
        console.print("\n[bold cyan]Push to HuggingFace Hub[/bold cyan]")
        
        try:
            from huggingface_hub import login, HfApi
            
            # Login
            if Confirm.ask("Do you need to login to HuggingFace?"):
                token = Prompt.ask("Enter your HuggingFace token", password=True)
                login(token=token)
            
            # Get repo name
            repo_name = Prompt.ask("Enter repository name (e.g. username/my-model)")
            
            console.print("[yellow]Uploading to HuggingFace Hub...[/yellow]")
            
            api = HfApi()
            api.upload_folder(
                folder_path=adapter_path,
                repo_id=repo_name,
                repo_type="model",
                create_pr=False
            )
            
            console.print(f"[green]Successfully pushed to: https://huggingface.co/{repo_name}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def deploy_to_cloud(self, base_model_id: str, adapter_path: str):
        """Deploy model to cloud platforms."""
        console.print("\n[bold cyan]Cloud Deployment Options[/bold cyan]")
        
        cloud_table = Table(show_header=True, header_style="bold magenta")
        cloud_table.add_column("Option", style="cyan", width=8)
        cloud_table.add_column("Platform", style="green")
        cloud_table.add_column("Requirements")
        
        cloud_table.add_row("1", "Google Cloud Run", "gcloud CLI, Docker")
        cloud_table.add_row("2", "Docker Container", "Docker installed")
        cloud_table.add_row("3", "FastAPI Server", "Creates API endpoint")
        
        console.print(cloud_table)
        
        choice = Prompt.ask("Select platform", choices=["1", "2", "3"], default="3")
        
        if choice == "1":
            self._deploy_to_gcloud(base_model_id, adapter_path)
        elif choice == "2":
            self._create_docker(base_model_id, adapter_path)
        else:
            self._create_fastapi_server(base_model_id, adapter_path)
    
    def _create_fastapi_server(self, base_model_id: str, adapter_path: str):
        """Create a FastAPI server for the model."""
        server_dir = self.exports_dir / "api_server"
        server_dir.mkdir(exist_ok=True)
        
        # Create main.py
        server_code = f'''"""
FastAPI Server for Fine-tuned Model
Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="Fine-tuned LLM API")

# Load model on startup
BASE_MODEL = "{base_model_id}"
ADAPTER_PATH = "{adapter_path}"

tokenizer = None
model = None

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 256
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    inputs = tokenizer(request.message, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.message):].strip()
    
    return ChatResponse(response=response)

@app.get("/health")
async def health():
    return {{"status": "healthy", "model": BASE_MODEL}}
'''
        
        with open(server_dir / "main.py", 'w') as f:
            f.write(server_code)
        
        # Create requirements.txt
        requirements = '''fastapi
uvicorn
torch
transformers
peft
accelerate
'''
        with open(server_dir / "requirements.txt", 'w') as f:
            f.write(requirements)
        
        console.print(Panel(f'''
[bold green]FastAPI Server Created![/bold green]

[yellow]Location:[/yellow] {server_dir}

[yellow]To run locally:[/yellow]
  cd {server_dir}
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

[yellow]API Endpoints:[/yellow]
  POST /chat - Send messages
  GET /health - Health check

[yellow]Test with:[/yellow]
  curl -X POST http://localhost:8000/chat \\
    -H "Content-Type: application/json" \\
    -d '{{"message": "What is Ayurveda?"}}'
''', title="API Server Ready", border_style="green"))
    
    def _create_docker(self, base_model_id: str, adapter_path: str):
        """Create Docker configuration."""
        docker_dir = self.exports_dir / "docker"
        docker_dir.mkdir(exist_ok=True)
        
        # First create the FastAPI server
        self._create_fastapi_server(base_model_id, adapter_path)
        
        dockerfile = f'''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(docker_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        console.print(Panel(f'''
[bold green]Docker Configuration Created![/bold green]

[yellow]Location:[/yellow] {docker_dir}

[yellow]To build and run:[/yellow]
  cd {docker_dir}
  docker build -t my-llm-api .
  docker run -p 8000:8000 my-llm-api

[yellow]To push to registry:[/yellow]
  docker tag my-llm-api your-registry/my-llm-api
  docker push your-registry/my-llm-api
''', title="Docker Ready", border_style="green"))
    
    def _deploy_to_gcloud(self, base_model_id: str, adapter_path: str):
        """Deploy to Google Cloud Run."""
        console.print("\n[bold]Google Cloud Run Deployment[/bold]")
        
        # Create Docker first
        self._create_docker(base_model_id, adapter_path)
        
        project_id = Prompt.ask("Enter your GCP Project ID")
        service_name = Prompt.ask("Enter service name", default="llm-api")
        region = Prompt.ask("Enter region", default="us-central1")
        
        console.print(Panel(f'''
[bold]Deploy to Google Cloud Run:[/bold]

1. Authenticate:
   gcloud auth login
   gcloud config set project {project_id}

2. Build and push:
   cd exports/docker
   gcloud builds submit --tag gcr.io/{project_id}/{service_name}

3. Deploy:
   gcloud run deploy {service_name} \\
     --image gcr.io/{project_id}/{service_name} \\
     --platform managed \\
     --region {region} \\
     --memory 8Gi \\
     --cpu 4 \\
     --allow-unauthenticated
''', title="GCloud Deployment Steps", border_style="cyan"))
    
    def export_merged_model(self, base_model_id: str, adapter_path: str):
        """Merge adapter with base model and export."""
        console.print("\n[bold cyan]Merging Adapter with Base Model...[/bold cyan]")
        
        model_name = base_model_id.split('/')[-1]
        output_path = self.exports_dir / f"{model_name}-merged"
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            console.print("[yellow]Loading base model...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            console.print("[yellow]Loading adapter...[/yellow]")
            model = PeftModel.from_pretrained(model, adapter_path)
            
            console.print("[yellow]Merging weights...[/yellow]")
            model = model.merge_and_unload()
            
            console.print("[yellow]Saving merged model...[/yellow]")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            console.print(Panel(f'''
[bold green]Model Merged Successfully![/bold green]

[yellow]Saved to:[/yellow] {output_path}

[yellow]This merged model can be:[/yellow]
  - Loaded directly with transformers
  - Converted to GGUF for Ollama
  - Deployed to cloud platforms
''', title="Export Complete", border_style="green"))
            
        except Exception as e:
            console.print(f"[red]Error merging model: {e}[/red]")


def run_deployment_wizard(base_model_id: str, adapter_path: str, config: dict = None):
    """Run the deployment wizard."""
    deployer = ModelDeployer(config)
    deployer.deploy_menu(base_model_id, adapter_path)
