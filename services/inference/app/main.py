import tempfile
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from landcover_common.logging import configure_logging
from landcover_common.settings import Settings
from .model import load_model, preprocess_pair, postprocess_mask, _download_from_s3, _upload_to_s3


app = FastAPI(title="Land Cover Inference", version="1.0.0")


class InferRequest(BaseModel):
    image_before_uri: str
    image_after_uri: str
    model_version: str = "production"


class InferResponse(BaseModel):
    job_id: str
    output_mask_uri: str
    metrics: dict


@app.on_event("startup")
def startup():
    settings = Settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    app.state.model = load_model(settings)


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    settings: Settings = app.state.settings
    model = app.state.model

    try:
        before_path = _download_from_s3(req.image_before_uri, settings)
        after_path = _download_from_s3(req.image_after_uri, settings)
        img_before, img_after = preprocess_pair(before_path, after_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        img_before = img_before.to(device)
        img_after = img_after.to(device)

        with torch.no_grad():
            logits = model(img_before, img_after)
            probs = torch.sigmoid(logits)

        mask_image = postprocess_mask(probs)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        mask_image.save(tmp.name)

        output_uri = _upload_to_s3(tmp.name, req.image_before_uri.replace("before", "output"), settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="inference failed") from exc

    return InferResponse(job_id="sync", output_mask_uri=output_uri, metrics={})


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ready"}
