"""
FastAPI Web Interface
Provides REST endpoints for uploading documents and monitoring pipeline progress.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import shutil
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid

from .pipeline import load_pipeline, PipelineResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Data Pipeline API",
    description="API for processing documents into training datasets using Granite 4.0",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
try:
    pipeline = load_pipeline("config.yaml")
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None

# Job store (in-memory for simplicity)
jobs: Dict[str, Dict] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    filename: str
    dataset_name: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    model_info: Dict


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and pipeline health."""
    if not pipeline:
        return {
            "status": "unhealthy", 
            "ollama_connected": False, 
            "model_info": {"error": "Pipeline not initialized"}
        }
        
    is_healthy = pipeline.ollama_client.check_health()
    model_info = pipeline.ollama_client.get_model_info() if is_healthy else {}
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "ollama_connected": is_healthy,
        "model_info": model_info
    }


def process_document_task(job_id: str, file_path: str, dataset_name: str):
    """Background task to process a document."""
    try:
        jobs[job_id]["status"] = "running"
        
        # Process the file
        result = pipeline.process_file(file_path, dataset_name=dataset_name)
        
        # Update job status
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        if result.success:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "documents_processed": result.documents_processed,
                "total_chunks": result.total_chunks,
                "total_samples": result.total_samples,
                "output_files": result.output_files,
                "duration_seconds": result.duration_seconds
            }
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "; ".join(result.errors)
            
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


@app.post("/upload", response_model=JobStatus)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dataset_name: Optional[str] = None
):
    """Upload a PDF document for processing."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline service unavailable")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Set default dataset name
    if not dataset_name:
        dataset_name = Path(file.filename).stem
        
    # Create job entry
    job_info = {
        "job_id": job_id,
        "status": "pending",
        "filename": file.filename,
        "dataset_name": dataset_name,
        "created_at": datetime.now().isoformat(),
        "file_path": str(file_path)
    }
    jobs[job_id] = job_info
    
    # Start background processing
    background_tasks.add_task(process_document_task, job_id, str(file_path), dataset_name)
    
    return job_info


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return jobs[job_id]


@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all jobs."""
    return list(jobs.values())


@app.get("/datasets")
async def list_datasets():
    """List generated datasets."""
    output_dir = Path("datasets")
    if not output_dir.exists():
        return []
        
    datasets = []
    for file in output_dir.glob("*_stats.json"):
        try:
            with open(file, "r") as f:
                import json
                stats = json.load(f)
                datasets.append(stats)
        except Exception:
            pass
            
    return datasets
