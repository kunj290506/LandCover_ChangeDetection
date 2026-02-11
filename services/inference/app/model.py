import os
import tempfile
from typing import Tuple

import boto3
import torch
from PIL import Image
from torchvision import transforms
import mlflow
import mlflow.pytorch

from landcover_common.settings import Settings
from models.snunet import SNUNet


def _download_from_s3(uri: str, settings: Settings) -> str:
    if not uri.startswith("s3://"):
        return uri
    _, path = uri.split("s3://", 1)
    bucket, key = path.split("/", 1)
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1])
    client.download_file(bucket, key, tmp.name)
    return tmp.name


def _upload_to_s3(local_path: str, uri: str, settings: Settings) -> str:
    if not uri.startswith("s3://"):
        return local_path
    _, path = uri.split("s3://", 1)
    bucket, key = path.split("/", 1)
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    client.upload_file(local_path, bucket, key)
    return uri


def load_model(settings: Settings, version: str | None = None) -> torch.nn.Module:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if version is None or version.lower() == "production":
        model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
    else:
        model_uri = f"models:/{settings.model_name}/{version}"
    try:
        return mlflow.pytorch.load_model(model_uri)
    except Exception:
        model = SNUNet(3, 1, use_attention=True)
        model.eval()
        return model


def preprocess_pair(img_before_path: str, img_after_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    image_before = Image.open(img_before_path).convert("RGB")
    image_after = Image.open(img_after_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image_before).unsqueeze(0), transform(image_after).unsqueeze(0)


def postprocess_mask(mask: torch.Tensor) -> Image.Image:
    mask = (mask > 0.5).float().squeeze(0).squeeze(0)
    mask = (mask * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(mask)
