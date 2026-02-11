import uuid
import logging

import requests
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
from redis import Redis
from rq import Queue
import mlflow
from mlflow.tracking import MlflowClient

from landcover_common.auth import create_access_token, decode_token, require_role
from landcover_common.logging import configure_logging
from landcover_common.settings import Settings


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Land Cover Gateway", version="1.0.0")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


class TokenRequest(BaseModel):
    username: str
    password: str


class InferRequest(BaseModel):
    image_before_uri: str
    image_after_uri: str
    model_version: str = "production"


class BatchPair(BaseModel):
    image_before_uri: str
    image_after_uri: str


class BatchRequest(BaseModel):
    pairs: list[BatchPair]
    model_version: str = "production"


class TrainRequest(BaseModel):
    dataset_uri: str
    config_uri: str
    register_model: bool = True


class PromoteRequest(BaseModel):
    name: str
    version: str
    stage: str


@app.on_event("startup")
def startup():
    settings = Settings()
    configure_logging(settings.log_level)
    logging.getLogger(__name__).info("gateway.startup", extra={"env": settings.app_env})


@app.post("/v1/auth/token")
@limiter.limit("10/minute")
def token(req: TokenRequest, settings: Settings = Depends(Settings)):
    if req.username != settings.admin_user or req.password != settings.admin_password:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token_value = create_access_token(req.username, ["admin", "trainer", "inference"], settings)
    return {"access_token": token_value, "token_type": "bearer", "expires_in": settings.access_token_expire_seconds}


@app.post("/v1/infer")
@limiter.limit("30/minute")
def infer(req: InferRequest, payload=Depends(decode_token), settings: Settings = Depends(Settings)):
    try:
        response = requests.post(f"{settings.inference_url}/infer", json=req.dict(), timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="inference service unavailable") from exc
    return response.json()


@app.post("/v1/batch")
@limiter.limit("10/minute")
def batch(req: BatchRequest, payload=Depends(require_role("inference")), settings: Settings = Depends(Settings)):
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue("batch", connection=redis_conn)
    job = queue.enqueue("worker.app.tasks.run_batch_job", req.dict())
    return {"job_id": job.id, "status": "queued"}


@app.get("/v1/batch/{job_id}")
@limiter.limit("60/minute")
def batch_status(job_id: str, payload=Depends(decode_token), settings: Settings = Depends(Settings)):
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue("batch", connection=redis_conn)
    job = queue.fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job.id, "status": job.get_status(), "result": job.result}


@app.post("/v1/train")
@limiter.limit("5/minute")
def train(req: TrainRequest, payload=Depends(require_role("trainer")), settings: Settings = Depends(Settings)):
    try:
        response = requests.post(f"{settings.training_url}/train", json=req.dict(), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="training service unavailable") from exc
    return response.json()


@app.get("/v1/models")
@limiter.limit("30/minute")
def list_models(payload=Depends(decode_token), settings: Settings = Depends(Settings)):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{settings.model_name}'")
    models = [
        {"name": v.name, "version": v.version, "stage": v.current_stage}
        for v in versions
    ]
    return {"models": models}


@app.post("/v1/models/promote")
@limiter.limit("5/minute")
def promote(req: PromoteRequest, payload=Depends(require_role("admin")), settings: Settings = Depends(Settings)):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    client.transition_model_version_stage(req.name, req.version, req.stage)
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ready"}
