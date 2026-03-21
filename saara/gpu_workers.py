"""
GPU Workers Module
Enables connecting cloud GPU workers (Kaggle, Colab, etc.) to SAARA CLI for remote training.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import os
import sys
import json
import time
import secrets
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)


# ============================================================================
# Worker & Job Models
# ============================================================================

class WorkerStatus(Enum):
    """Worker connection status."""
    PENDING = "pending"
    CONNECTED = "connected"
    BUSY = "busy"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class JobStatus(Enum):
    """Job execution status."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Types of jobs that workers can execute."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    PRETRAINING = "pretraining"
    INFERENCE = "inference"


@dataclass
class GPUInfo:
    """GPU hardware information."""
    name: str = "Unknown"
    memory_gb: float = 0.0
    cuda_version: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Worker:
    """Represents a connected GPU worker."""
    worker_id: str
    token: str
    worker_type: str  # kaggle, colab, runpod, etc.
    gpu_info: Optional[GPUInfo] = None
    status: WorkerStatus = WorkerStatus.PENDING
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    current_job_id: Optional[str] = None
    jobs_completed: int = 0
    
    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if worker is still alive based on heartbeat."""
        return (datetime.now() - self.last_heartbeat).total_seconds() < timeout_seconds
    
    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "gpu_info": self.gpu_info.to_dict() if self.gpu_info else None,
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "current_job_id": self.current_job_id,
            "jobs_completed": self.jobs_completed,
            "is_alive": self.is_alive()
        }


@dataclass
class Job:
    """Represents a job to be executed by a worker."""
    job_id: str
    job_type: JobType
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_worker_id: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "payload": self.payload,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assigned_worker_id": self.assigned_worker_id,
            "result": self.result,
            "error": self.error
        }


# ============================================================================
# Worker Token Manager
# ============================================================================

class TokenManager:
    """Manages worker authentication tokens."""
    
    TOKEN_PREFIX = "saara_worker_"
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path.home() / ".saara"
        self.tokens_file = self.config_dir / "worker_tokens.json"
        self._tokens: Dict[str, Dict] = {}
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from file."""
        if self.tokens_file.exists():
            try:
                with open(self.tokens_file, "r") as f:
                    self._tokens = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load tokens: {e}")
                self._tokens = {}
    
    def _save_tokens(self):
        """Save tokens to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.tokens_file, "w") as f:
            json.dump(self._tokens, f, indent=2)
    
    def generate_token(self, name: str = None, expires_hours: int = 24) -> str:
        """Generate a new worker token."""
        token = f"{self.TOKEN_PREFIX}{secrets.token_hex(8)}"
        
        self._tokens[token] = {
            "name": name or f"Worker Token {len(self._tokens) + 1}",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=expires_hours)).isoformat() if expires_hours > 0 else None,
            "used": False,
            "worker_id": None
        }
        
        self._save_tokens()
        return token
    
    def validate_token(self, token: str) -> bool:
        """Check if a token is valid."""
        if token not in self._tokens:
            return False
        
        token_info = self._tokens[token]
        
        # Check expiration
        if token_info.get("expires_at"):
            expires_at = datetime.fromisoformat(token_info["expires_at"])
            if datetime.now() > expires_at:
                return False
        
        return True
    
    def mark_used(self, token: str, worker_id: str):
        """Mark a token as used by a worker."""
        if token in self._tokens:
            self._tokens[token]["used"] = True
            self._tokens[token]["worker_id"] = worker_id
            self._save_tokens()
    
    def list_tokens(self) -> List[Dict]:
        """List all tokens."""
        result = []
        for token, info in self._tokens.items():
            result.append({
                "token": token,
                **info,
                "is_valid": self.validate_token(token)
            })
        return result
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self._tokens:
            del self._tokens[token]
            self._save_tokens()
            return True
        return False


# ============================================================================
# Worker Manager
# ============================================================================

class WorkerManager:
    """Manages connected GPU workers and job queue."""
    
    def __init__(self, token_manager: TokenManager = None):
        self.token_manager = token_manager or TokenManager()
        self._workers: Dict[str, Worker] = {}
        self._jobs: Dict[str, Job] = {}
        self._job_queue: List[str] = []  # Job IDs in queue order
        self._lock = threading.Lock()
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the worker manager background tasks."""
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def stop(self):
        """Stop the worker manager."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def _cleanup_loop(self):
        """Background loop to clean up dead workers."""
        while self._running:
            try:
                self._cleanup_dead_workers()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            time.sleep(10)
    
    def _cleanup_dead_workers(self):
        """Remove workers that haven't sent heartbeats."""
        with self._lock:
            dead_workers = [
                w_id for w_id, w in self._workers.items()
                if not w.is_alive(timeout_seconds=90)
            ]
            
            for w_id in dead_workers:
                worker = self._workers[w_id]
                
                # Re-queue any assigned jobs
                if worker.current_job_id and worker.current_job_id in self._jobs:
                    job = self._jobs[worker.current_job_id]
                    if job.status in [JobStatus.ASSIGNED, JobStatus.RUNNING]:
                        job.status = JobStatus.QUEUED
                        job.assigned_worker_id = None
                        if job.job_id not in self._job_queue:
                            self._job_queue.insert(0, job.job_id)  # Priority re-queue
                
                worker.status = WorkerStatus.DISCONNECTED
                logger.info(f"Worker {w_id} marked as disconnected (no heartbeat)")
    
    # Worker Management
    
    def register_worker(self, token: str, worker_type: str, gpu_info: Dict = None) -> Optional[Worker]:
        """Register a new worker with token authentication."""
        if not self.token_manager.validate_token(token):
            return None
        
        worker_id = f"worker_{secrets.token_hex(4)}"
        
        gpu = None
        if gpu_info:
            gpu = GPUInfo(
                name=gpu_info.get("name", "Unknown"),
                memory_gb=gpu_info.get("memory_gb", 0.0),
                cuda_version=gpu_info.get("cuda_version", "")
            )
        
        worker = Worker(
            worker_id=worker_id,
            token=token,
            worker_type=worker_type,
            gpu_info=gpu,
            status=WorkerStatus.CONNECTED
        )
        
        with self._lock:
            self._workers[worker_id] = worker
        
        self.token_manager.mark_used(token, worker_id)
        
        logger.info(f"Worker registered: {worker_id} ({worker_type})")
        return worker
    
    def heartbeat(self, worker_id: str) -> bool:
        """Record a heartbeat from a worker."""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = datetime.now()
                return True
        return False
    
    def get_worker(self, worker_id: str) -> Optional[Worker]:
        """Get worker by ID."""
        return self._workers.get(worker_id)
    
    def list_workers(self, include_disconnected: bool = False) -> List[Worker]:
        """List all workers."""
        with self._lock:
            workers = list(self._workers.values())
            if not include_disconnected:
                workers = [w for w in workers if w.status != WorkerStatus.DISCONNECTED]
            return workers
    
    def get_connected_workers(self) -> List[Worker]:
        """Get only connected and alive workers."""
        return [w for w in self.list_workers() if w.is_alive() and w.status == WorkerStatus.CONNECTED]
    
    # Job Management
    
    def create_job(self, job_type: JobType, payload: Dict) -> Job:
        """Create a new job and add to queue."""
        job_id = f"job_{secrets.token_hex(8)}"
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            payload=payload
        )
        
        with self._lock:
            self._jobs[job_id] = job
            self._job_queue.append(job_id)
        
        logger.info(f"Job created: {job_id} ({job_type.value})")
        return job
    
    def get_next_job(self, worker_id: str) -> Optional[Job]:
        """Get next available job for a worker."""
        with self._lock:
            if worker_id not in self._workers:
                return None
            
            worker = self._workers[worker_id]
            if worker.status != WorkerStatus.CONNECTED or worker.current_job_id:
                return None
            
            # Find next queued job
            for job_id in self._job_queue:
                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.QUEUED:
                    # Assign job to worker
                    job.status = JobStatus.ASSIGNED
                    job.assigned_worker_id = worker_id
                    job.started_at = datetime.now()
                    
                    worker.status = WorkerStatus.BUSY
                    worker.current_job_id = job_id
                    
                    self._job_queue.remove(job_id)
                    
                    logger.info(f"Job {job_id} assigned to worker {worker_id}")
                    return job
        
        return None
    
    def update_job_progress(self, job_id: str, progress: int, status: str = None):
        """Update job progress."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.progress = min(100, max(0, progress))
                if status:
                    try:
                        job.status = JobStatus(status)
                    except ValueError:
                        pass
    
    def complete_job(self, job_id: str, result: Dict = None, error: str = None):
        """Mark a job as completed or failed."""
        with self._lock:
            if job_id not in self._jobs:
                return
            
            job = self._jobs[job_id]
            job.completed_at = datetime.now()
            job.result = result
            job.error = error
            job.status = JobStatus.COMPLETED if not error else JobStatus.FAILED
            job.progress = 100 if not error else job.progress
            
            # Update worker
            if job.assigned_worker_id and job.assigned_worker_id in self._workers:
                worker = self._workers[job.assigned_worker_id]
                worker.current_job_id = None
                worker.status = WorkerStatus.CONNECTED
                worker.jobs_completed += 1
            
            logger.info(f"Job {job_id} {'completed' if not error else 'failed'}")
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(self, status: JobStatus = None) -> List[Job]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status in [JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING]:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    
                    if job_id in self._job_queue:
                        self._job_queue.remove(job_id)
                    
                    # Free up worker
                    if job.assigned_worker_id and job.assigned_worker_id in self._workers:
                        worker = self._workers[job.assigned_worker_id]
                        worker.current_job_id = None
                        worker.status = WorkerStatus.CONNECTED
                    
                    return True
        return False
    
    # Stats
    
    def get_stats(self) -> Dict:
        """Get worker and job statistics."""
        workers = self.list_workers(include_disconnected=True)
        jobs = self.list_jobs()
        
        return {
            "workers": {
                "total": len(workers),
                "connected": len([w for w in workers if w.status == WorkerStatus.CONNECTED]),
                "busy": len([w for w in workers if w.status == WorkerStatus.BUSY]),
                "disconnected": len([w for w in workers if w.status == WorkerStatus.DISCONNECTED])
            },
            "jobs": {
                "total": len(jobs),
                "queued": len([j for j in jobs if j.status == JobStatus.QUEUED]),
                "running": len([j for j in jobs if j.status in [JobStatus.ASSIGNED, JobStatus.RUNNING]]),
                "completed": len([j for j in jobs if j.status == JobStatus.COMPLETED]),
                "failed": len([j for j in jobs if j.status == JobStatus.FAILED])
            },
            "queue_length": len(self._job_queue)
        }


# ============================================================================
# FastAPI Worker Server
# ============================================================================

def create_worker_server(worker_manager: WorkerManager, host: str = "0.0.0.0", port: int = 8765):
    """Create FastAPI server for worker connections."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import Optional
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(
        title="SAARA GPU Worker Server",
        description="API for cloud GPU workers to connect and execute training jobs",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request/Response models
    class RegisterRequest(BaseModel):
        token: str
        worker_type: str
        gpu_info: Optional[Dict] = None
    
    class ProgressRequest(BaseModel):
        progress: int
        status: Optional[str] = None
    
    class CompleteRequest(BaseModel):
        result: Optional[Dict] = None
        error: Optional[str] = None
    
    class JobRequest(BaseModel):
        job_type: str
        payload: Dict[str, Any]

    # Endpoints
    @app.get("/")
    async def root():
        return {"status": "ok", "service": "SAARA GPU Worker Server"}
    
    @app.get("/status")
    async def status():
        return worker_manager.get_stats()
    
    @app.post("/workers/register")
    async def register_worker(request: RegisterRequest):
        worker = worker_manager.register_worker(
            token=request.token,
            worker_type=request.worker_type,
            gpu_info=request.gpu_info
        )
        
        if not worker:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        return {"worker_id": worker.worker_id, "status": "registered"}
    
    @app.post("/workers/{worker_id}/heartbeat")
    async def heartbeat(worker_id: str):
        if not worker_manager.heartbeat(worker_id):
            raise HTTPException(status_code=404, detail="Worker not found")
        return {"status": "ok"}
    
    @app.get("/workers/{worker_id}/jobs/next")
    async def get_next_job(worker_id: str):
        job = worker_manager.get_next_job(worker_id)
        if job:
            return {"job": job.to_dict()}
        return {"job": None}
    
    @app.post("/workers/jobs/{job_id}/progress")
    async def update_progress(job_id: str, request: ProgressRequest):
        worker_manager.update_job_progress(job_id, request.progress, request.status)
        return {"status": "ok"}
    
    @app.post("/workers/jobs/{job_id}/complete")
    async def complete_job(job_id: str, request: CompleteRequest):
        worker_manager.complete_job(job_id, result=request.result, error=request.error)
        return {"status": "ok"}
    
    @app.get("/workers")
    async def list_workers():
        workers = worker_manager.list_workers(include_disconnected=True)
        return {"workers": [w.to_dict() for w in workers]}
    
    @app.get("/jobs")
    async def list_jobs():
        jobs = worker_manager.list_jobs()
        return {"jobs": [j.to_dict() for j in jobs]}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = worker_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.to_dict()

    @app.post("/jobs/submit")
    async def submit_job(request: JobRequest):
        try:
            # Map string to enum
            j_type = JobType(request.job_type)
        except ValueError:
            j_type = JobType.TRAINING
            
        job = worker_manager.create_job(job_type=j_type, payload=request.payload)
        return {"job_id": job.job_id, "status": "queued"}
    
    return app


# ============================================================================
# Notebook Generator
# ============================================================================

def generate_colab_notebook(server_url: str, token: str) -> str:
    """Generate a Colab notebook for GPU worker."""
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"gpuType": "T4"}
        },
        "nbformat_minor": 0,
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "markdown",
                "source": f"""# 🚀 SAARA GPU Worker - Google Colab

This notebook connects to your SAARA application and provides free GPU compute.

**Pre-configured:**
- Server URL: `{server_url}`
- Worker Token: `{token[:20]}...` (hidden for security)

**Instructions:**
1. Enable GPU: Runtime > Change runtime type > GPU
2. Run all cells (Ctrl+F9)
3. Keep this notebook open while training
""",
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": f"""# Configuration
SAARA_URL = "{server_url}"
WORKER_TOKEN = "{token}"
""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """!pip install -q transformers datasets accelerate peft bitsandbytes trl""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """import requests
import time

class SaaraWorker:
    def __init__(self, base_url, token):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.worker_id = None
        
    def _get_gpu_info(self):
        if torch.cuda.is_available():
            return {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "cuda_version": torch.version.cuda or ""
            }
        return None
    
    def register(self):
        try:
            resp = requests.post(f"{self.base_url}/workers/register", json={
                "token": self.token,
                "worker_type": "colab",
                "gpu_info": self._get_gpu_info()
            }, timeout=30)
            if resp.status_code == 200:
                self.worker_id = resp.json()["worker_id"]
                print(f"✅ Registered: {self.worker_id}")
                return True
            print(f"❌ Failed: {resp.text}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def heartbeat(self):
        try:
            requests.post(f"{self.base_url}/workers/{self.worker_id}/heartbeat", timeout=10)
        except: pass
    
    def get_next_job(self):
        try:
            resp = requests.get(f"{self.base_url}/workers/{self.worker_id}/jobs/next", timeout=30)
            if resp.status_code == 200:
                return resp.json().get("job")
        except: pass
        return None
    
    def update_progress(self, job_id, progress, status="running"):
        try:
            requests.post(f"{self.base_url}/workers/jobs/{job_id}/progress", 
                          json={"progress": progress, "status": status}, timeout=10)
        except: pass
    
    def complete_job(self, job_id, result=None, error=None):
        try:
            requests.post(f"{self.base_url}/workers/jobs/{job_id}/complete",
                          json={"result": result, "error": error}, timeout=30)
        except Exception as e:
            print(f"Failed: {e}")

print("✅ Worker class loaded")
""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
import json

def run_training_job(job, worker):
    payload = job["payload"]
    job_id = job["job_id"]
    print(f"🚀 Starting training job: {job_id}")
    worker.update_progress(job_id, 5, "running")
    
    try:
        model_name = payload.get("model_name", "microsoft/DialoGPT-small")
        dataset_path = payload.get("dataset_path")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        worker.update_progress(job_id, 20)
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=payload.get("lora_r", 8), 
            lora_alpha=payload.get("lora_alpha", 16), 
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)
        worker.update_progress(job_id, 30)
        
        # Load dataset
        if dataset_path and dataset_path.startswith("http"):
            # Download dataset from URL
            resp = requests.get(dataset_path)
            data = [json.loads(line) for line in resp.text.strip().split("\\n")]
            dataset = Dataset.from_list(data)
        else:
            # Use sample dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]")
        
        def tokenize_function(examples):
            return tokenizer(
                examples.get("text", examples.get("content", [""])),
                truncation=True, 
                max_length=payload.get("max_length", 512), 
                padding="max_length"
            )
        
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        worker.update_progress(job_id, 50)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=payload.get("epochs", 1),
            per_device_train_batch_size=payload.get("batch_size", 4),
            learning_rate=payload.get("learning_rate", 2e-5),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized
        )
        
        trainer.train()
        worker.update_progress(job_id, 90)
        
        # Save model
        model.save_pretrained("./trained_model")
        tokenizer.save_pretrained("./trained_model")
        
        result = {
            "status": "completed",
            "model": model_name,
            "epochs": payload.get("epochs", 1),
            "samples": len(tokenized)
        }
        worker.complete_job(job_id, result=result)
        print(f"✅ Training completed: {result}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        worker.complete_job(job_id, error=str(e))

def run_evaluation_job(job, worker):
    payload = job["payload"]
    job_id = job["job_id"]
    print(f"🔍 Starting evaluation job: {job_id}")
    worker.update_progress(job_id, 10, "running")
    
    try:
        model_name = payload.get("model_name", "gpt2")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        worker.update_progress(job_id, 50)
        
        # Simple perplexity evaluation
        test_text = payload.get("test_text", "The quick brown fox jumps over the lazy dog.")
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            perplexity = torch.exp(outputs.loss).item()
        
        result = {
            "status": "completed",
            "perplexity": round(perplexity, 2),
            "model": model_name
        }
        worker.complete_job(job_id, result=result)
        print(f"✅ Evaluation completed: PPL={perplexity:.2f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        worker.complete_job(job_id, error=str(e))

print("✅ Job handlers loaded")
""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """# Start Worker
print("🚀 Starting SAARA GPU Worker...")

worker = SaaraWorker(SAARA_URL, WORKER_TOKEN)
if worker.register():
    print("\\n🔄 Waiting for jobs from SAARA...")
    print("(Keep this cell running to receive training jobs)\\n")
    
    last_heartbeat = time.time()
    dots = 0
    
    while True:
        try:
            # Send heartbeat every 30 seconds
            if time.time() - last_heartbeat > 30:
                worker.heartbeat()
                last_heartbeat = time.time()
            
            # Check for jobs
            job = worker.get_next_job()
            if job:
                job_type = job.get("job_type")
                print(f"\\n📥 Received job: {job['job_id']} (type: {job_type})")
                
                if job_type == "training":
                    run_training_job(job, worker)
                elif job_type == "evaluation":
                    run_evaluation_job(job, worker)
                else:
                    worker.complete_job(job["job_id"], error=f"Unknown job type: {job_type}")
                
                print("\\n🔄 Waiting for next job...")
                dots = 0
            else:
                # Print progress indicator
                dots = (dots + 1) % 60
                if dots == 0:
                    print(".", end="", flush=True)
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\\n\\n👋 Worker stopped by user")
            break
        except Exception as e:
            print(f"\\n⚠️ Error: {e}")
            time.sleep(10)
else:
    print("❌ Failed to register worker. Check your token and server URL.")
""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            }
        ]
    }
    
    return json.dumps(notebook, indent=2)


# ============================================================================
# Swarm Visualization
# ============================================================================

def display_swarm_status(worker_manager: WorkerManager):
    """
    Display a rich table of the GPU swarm status.
    Combines local and cloud workers in a unified view.
    Attempts to fetch data from local server if available.
    """
    if not RICH_AVAILABLE:
        print("Rich library not available for swarm visualization.")
        return

    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console
    from rich import box
    import requests
    
    console = Console()
    
    # Try to fetch from server first
    server_stats = None
    server_workers = []
    try:
        resp_status = requests.get("http://127.0.0.1:8765/status", timeout=0.2)
        resp_workers = requests.get("http://127.0.0.1:8765/workers", timeout=0.2)
        
        if resp_status.ok and resp_workers.ok:
            server_stats = resp_status.json()
            workers_data = resp_workers.json().get("workers", [])
            
            # Reconstruct worker objects from dicts for visualization compatibility
            for w_data in workers_data:
                 # Manually reconstruct minimal Worker object or mock for display
                 # Since Worker is a dataclass, we can instantiate it if we parse fields carefully
                 # But for display, a simple object wrapper is enough or we parse it map to the existing Worker class 
                 
                 # Create GPU info
                 gpu_dict = w_data.get("gpu_info")
                 gpu = None
                 if gpu_dict:
                     gpu = GPUInfo(**gpu_dict)
                     
                 # Convert string status back to Enum if needed, or just use string
                 # The display logic uses .value comparison, so we need to be careful.
                 # Let's map string to Enum
                 try:
                     status_enum = WorkerStatus(w_data["status"])
                 except:
                     status_enum = w_data["status"]

                 w = Worker(
                     worker_id=w_data["worker_id"],
                     token="remote", # dummy
                     worker_type=w_data["worker_type"],
                     gpu_info=gpu,
                     status=status_enum,
                     current_job_id=w_data.get("current_job_id"),
                     jobs_completed=w_data.get("jobs_completed", 0)
                 )
                 server_workers.append(w)
                 
    except Exception:
        pass # Server not running or unreachable
        
    if server_stats:
        stats = server_stats
        workers = server_workers
        # Append " (Server Mode)" to title or something?
    else:
        stats = worker_manager.get_stats()
        workers = worker_manager.list_workers(include_disconnected=True)
    
    # 1. Header with Stats
    total_gpus = stats['workers']['total']
    active_gpus = stats['workers']['connected'] + stats['workers']['busy']
    
    console.print(Panel(
        f"[bold cyan]GPU Swarm Status[/bold cyan]\n"
        f"Active Nodes: [green]{active_gpus}[/green] / {total_gpus} | "
        f"Queue: [yellow]{stats['queue_length']}[/yellow] jobs | "
        f"Completed: [green]{stats['jobs']['completed']}[/green]",
        title="🕸️ Decentralized Compute Grid",
        border_style="cyan"
    ))
    
    # 2. Worker Table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta", expand=True)
    table.add_column("ID / Name", style="white")
    table.add_column("Type", width=10)
    table.add_column("GPU", style="cyan")
    table.add_column("Memory", style="yellow")
    table.add_column("Status")
    table.add_column("Current Job", style="dim")
    
    # Add Local Machine (Simulated for visualization if not strictly a "worker")
    import torch
    if torch.cuda.is_available():
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        gpu_name = torch.cuda.get_device_name(0)
        table.add_row(
            "[bold]Local Host[/bold]",
            "Local",
            gpu_name,
            f"{vram} GB",
            "[green]● Active[/green]",
            "Controller"
        )
    else:
        table.add_row(
            "[bold]Local Host[/bold]",
            "Local",
            "CPU Only",
            "-",
            "[yellow]● Active[/yellow]",
            "Controller"
        )
        
    # Add Remote Workers
    for w in workers:
        status_val = w.status.value if isinstance(w.status, WorkerStatus) else str(w.status)
        
        status_style = {
            "connected": "[green]● Ready[/green]",
            "busy": "[blue]▶ Working[/blue]",
            "disconnected": "[red]○ Offline[/red]",
            "error": "[red]⚠ Error[/red]",
            "pending": "[yellow]○ Pending[/yellow]"
        }.get(status_val, status_val)
        
        gpu_name = w.gpu_info.name if w.gpu_info else "Unknown"
        memory = f"{w.gpu_info.memory_gb} GB" if w.gpu_info else "-"
        job_info = f"Job: {w.current_job_id}" if w.current_job_id else "-"
        
        table.add_row(
            w.worker_id[:12],
            w.worker_type.upper(),
            gpu_name,
            memory,
            status_style,
            job_info
        )

    if not workers and not torch.cuda.is_available():
         table.add_row("-", "-", "-", "-", "-", "-")

    console.print(table)
    console.print()



def generate_kaggle_notebook(server_url: str, token: str) -> str:
    """Generate a Kaggle notebook for GPU worker."""
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "dockerImageVersionId": 31260,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True
            }
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "markdown",
                "source": f"""# 🚀 SAARA GPU Worker - Kaggle

This notebook connects to your SAARA application and provides free GPU compute.

**Pre-configured:**
- Server URL: `{server_url}`
- Worker Token: Generated

**Instructions:**
1. Enable GPU: Settings > Accelerator > GPU
2. Enable Internet: Settings > Internet > On
3. Run all cells
4. Keep this notebook running while training
""",
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": f"""# Configuration
SAARA_URL = "{server_url}"
WORKER_TOKEN = "{token}"

print("✅ Configuration loaded")
print(f"Server: {{SAARA_URL}}")
""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """!pip install -q transformers datasets accelerate peft bitsandbytes trl""",
                "metadata": {"trusted": True},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """import requests
import time

class SaaraWorker:
    def __init__(self, base_url, token):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.worker_id = None
        
    def _get_gpu_info(self):
        if torch.cuda.is_available():
            return {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "cuda_version": torch.version.cuda or ""
            }
        return None
    
    def register(self):
        try:
            resp = requests.post(f"{self.base_url}/workers/register", json={
                "token": self.token,
                "worker_type": "kaggle",
                "gpu_info": self._get_gpu_info()
            }, timeout=30)
            if resp.status_code == 200:
                self.worker_id = resp.json()["worker_id"]
                print(f"✅ Registered: {self.worker_id}")
                return True
            print(f"❌ Failed: {resp.text}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def heartbeat(self):
        try:
            requests.post(f"{self.base_url}/workers/{self.worker_id}/heartbeat", timeout=10)
        except: pass
    
    def get_next_job(self):
        try:
            resp = requests.get(f"{self.base_url}/workers/{self.worker_id}/jobs/next", timeout=30)
            if resp.status_code == 200:
                return resp.json().get("job")
        except: pass
        return None
    
    def update_progress(self, job_id, progress, status="running"):
        try:
            requests.post(f"{self.base_url}/workers/jobs/{job_id}/progress", 
                          json={"progress": progress, "status": status}, timeout=10)
        except: pass
    
    def complete_job(self, job_id, result=None, error=None):
        try:
            requests.post(f"{self.base_url}/workers/jobs/{job_id}/complete",
                          json={"result": result, "error": error}, timeout=30)
        except Exception as e:
            print(f"Failed: {e}")

print("✅ Worker class loaded")
""",
                "metadata": {"trusted": True, "editable": False},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
import json

def run_training_job(job, worker):
    payload = job["payload"]
    job_id = job["job_id"]
    print(f"🚀 Starting training: {job_id}")
    worker.update_progress(job_id, 5, "running")
    
    try:
        model_name = payload.get("model_name", "microsoft/DialoGPT-small")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        worker.update_progress(job_id, 20)
        
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, lora_config)
        worker.update_progress(job_id, 30)
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]")
        
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
        
        tokenized = dataset.map(tokenize, batched=True)
        worker.update_progress(job_id, 50)
        
        args = TrainingArguments(
            output_dir="./results", num_train_epochs=1, per_device_train_batch_size=4,
            learning_rate=2e-5, logging_steps=10, save_strategy="no", report_to="none"
        )
        
        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()
        worker.update_progress(job_id, 90)
        
        result = {"status": "completed", "model": model_name}
        worker.complete_job(job_id, result=result)
        print(f"✅ Done: {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        worker.complete_job(job_id, error=str(e))

def run_evaluation_job(job, worker):
    payload = job["payload"]
    job_id = job["job_id"]
    print(f"🔍 Eval: {job_id}")
    worker.update_progress(job_id, 10, "running")
    
    try:
        model_name = payload.get("model_name", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        worker.update_progress(job_id, 50)
        
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            ppl = torch.exp(outputs.loss).item()
        
        result = {"status": "completed", "perplexity": ppl}
        worker.complete_job(job_id, result=result)
        print(f"✅ Done: PPL={ppl:.2f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        worker.complete_job(job_id, error=str(e))

print("✅ Handlers loaded")
""",
                "metadata": {"trusted": True, "editable": False},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "source": """# Start Worker
if not WORKER_TOKEN or not SAARA_URL:
    print("❌ Set SAARA_URL and WORKER_TOKEN!")
else:
    worker = SaaraWorker(SAARA_URL, WORKER_TOKEN)
    if worker.register():
        print("\\n🔄 Waiting for jobs...")
        last_hb = time.time()
        while True:
            try:
                if time.time() - last_hb > 30:
                    worker.heartbeat()
                    last_hb = time.time()
                
                job = worker.get_next_job()
                if job:
                    jt = job.get("job_type")
                    if jt == "training": run_training_job(job, worker)
                    elif jt == "evaluation": run_evaluation_job(job, worker)
                    else: worker.complete_job(job["job_id"], error=f"Unknown: {jt}")
                else:
                    print(".", end="", flush=True)
                    time.sleep(5)
            except KeyboardInterrupt:
                print("\\n👋 Stopped")
                break
            except Exception as e:
                print(f"\\n⚠️ {e}")
                time.sleep(10)
""",
                "metadata": {"trusted": True, "editable": False},
                "outputs": [],
                "execution_count": None
            }
        ]
    }
    
    return json.dumps(notebook, indent=2)


# ============================================================================
# Display Functions
# ============================================================================

def display_workers(workers: List[Worker]):
    """Display workers in a rich table."""
    if not RICH_AVAILABLE:
        for w in workers:
            print(f"  {w.worker_id}: {w.worker_type} ({w.status.value})")
        return
    
    table = Table(title="🖥️ Connected GPU Workers", show_header=True, header_style="bold cyan")
    table.add_column("Worker ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("GPU", style="yellow")
    table.add_column("Memory", style="magenta")
    table.add_column("Status", style="blue")
    table.add_column("Jobs Done", style="green")
    
    for w in workers:
        gpu_name = w.gpu_info.name if w.gpu_info else "N/A"
        gpu_mem = f"{w.gpu_info.memory_gb:.1f} GB" if w.gpu_info else "N/A"
        
        status_style = {
            WorkerStatus.CONNECTED: "[green]● Connected[/green]",
            WorkerStatus.BUSY: "[yellow]● Busy[/yellow]",
            WorkerStatus.DISCONNECTED: "[red]● Disconnected[/red]",
            WorkerStatus.PENDING: "[dim]○ Pending[/dim]",
            WorkerStatus.ERROR: "[red]✗ Error[/red]"
        }.get(w.status, w.status.value)
        
        table.add_row(
            w.worker_id,
            w.worker_type.capitalize(),
            gpu_name[:25] + "..." if len(gpu_name) > 25 else gpu_name,
            gpu_mem,
            status_style,
            str(w.jobs_completed)
        )
    
    console.print(table)


def display_jobs(jobs: List[Job]):
    """Display jobs in a rich table."""
    if not RICH_AVAILABLE:
        for j in jobs:
            print(f"  {j.job_id}: {j.job_type.value} ({j.status.value}) - {j.progress}%")
        return
    
    table = Table(title="📋 Job Queue", show_header=True, header_style="bold cyan")
    table.add_column("Job ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Status", style="blue")
    table.add_column("Progress", style="magenta")
    table.add_column("Worker", style="yellow")
    
    for j in jobs:
        status_style = {
            JobStatus.QUEUED: "[dim]○ Queued[/dim]",
            JobStatus.ASSIGNED: "[yellow]● Assigned[/yellow]",
            JobStatus.RUNNING: "[cyan]● Running[/cyan]",
            JobStatus.COMPLETED: "[green]✓ Completed[/green]",
            JobStatus.FAILED: "[red]✗ Failed[/red]",
            JobStatus.CANCELLED: "[dim]✗ Cancelled[/dim]"
        }.get(j.status, j.status.value)
        
        progress_bar = f"[{'█' * (j.progress // 10)}{'░' * (10 - j.progress // 10)}] {j.progress}%"
        
        table.add_row(
            j.job_id[:16] + "...",
            j.job_type.value.capitalize(),
            status_style,
            progress_bar if j.status in [JobStatus.RUNNING, JobStatus.ASSIGNED] else f"{j.progress}%",
            j.assigned_worker_id[:12] + "..." if j.assigned_worker_id else "-"
        )
    
    console.print(table)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'WorkerStatus',
    'JobStatus', 
    'JobType',
    
    # Models
    'GPUInfo',
    'Worker',
    'Job',
    
    # Managers
    'TokenManager',
    'WorkerManager',
    
    # Server
    'create_worker_server',
    
    # Generators
    'generate_colab_notebook',
    'generate_kaggle_notebook',
    
    # Display
    'display_workers',
    'display_jobs',
]
