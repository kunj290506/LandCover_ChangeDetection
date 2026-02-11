import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from landcover_common.logging import configure_logging
from landcover_common.settings import Settings
from .trainer import run_training


app = FastAPI(title="Land Cover Training", version="1.0.0")
executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent training jobs


class TrainRequest(BaseModel):
    dataset_uri: str
    config_uri: str = "s3://landcover/configs/train_config.yaml"  # Default config
    register_model: bool = True


class TrainResponse(BaseModel):
    run_id: str
    status: str
    message: str


# In-memory job tracking (use Redis in production)
training_jobs = {}


def training_worker(job_id: str, dataset_uri: str, config_uri: str, register_model: bool):
    """Background worker for training"""
    try:
        training_jobs[job_id] = {"status": "running", "message": "Training in progress"}
        
        logging.info(f"Starting training job {job_id}")
        run_id = run_training(dataset_uri, config_uri, register_model)
        
        training_jobs[job_id] = {
            "status": "completed", 
            "message": f"Training completed successfully. MLflow run: {run_id}",
            "run_id": run_id
        }
        logging.info(f"Training job {job_id} completed with run_id {run_id}")
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        training_jobs[job_id] = {"status": "failed", "message": error_msg}
        logging.error(f"Training job {job_id} failed: {e}")


@app.on_event("startup")
def startup():
    settings = Settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    logging.info("Training service started")


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    """Start a training job asynchronously"""
    import uuid
    
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {"status": "queued", "message": "Training job queued"}
    
    # Submit training job to thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, 
        training_worker, 
        job_id, 
        req.dataset_uri, 
        req.config_uri, 
        req.register_model
    )
    
    return TrainResponse(
        run_id=job_id,
        status="started",
        message="Training job submitted successfully"
    )


@app.get("/train/{job_id}")
def get_training_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]


@app.get("/train")
def list_training_jobs():
    """List all training jobs"""
    return {"jobs": training_jobs}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ready"}
