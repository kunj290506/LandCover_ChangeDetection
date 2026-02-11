# API Contract (Gateway)

Base URL: `/v1`

## Auth

### POST `/auth/token`

Request:
```json
{
  "username": "string",
  "password": "string"
}
```

Response:
```json
{
  "access_token": "jwt",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## Inference

### POST `/infer`

Headers:
- `Authorization: Bearer <token>`

Request:
```json
{
  "image_before_uri": "s3://bucket/path/before.tif",
  "image_after_uri": "s3://bucket/path/after.tif",
  "model_version": "production"
}
```

Response:
```json
{
  "job_id": "uuid",
  "output_mask_uri": "s3://bucket/path/output.tif",
  "metrics": {
    "f1": 0.0,
    "iou": 0.0,
    "precision": 0.0,
    "recall": 0.0
  }
}
```

## Batch

### POST `/batch`

Request:
```json
{
  "pairs": [
    {
      "image_before_uri": "s3://bucket/a.tif",
      "image_after_uri": "s3://bucket/b.tif"
    }
  ],
  "model_version": "production"
}
```

Response:
```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### GET `/batch/{job_id}`

Response:
```json
{
  "job_id": "uuid",
  "status": "running|succeeded|failed",
  "outputs": ["s3://bucket/output1.tif"]
}
```

## Training

### POST `/train`

Request:
```json
{
  "dataset_uri": "s3://bucket/datasets/levir",
  "config_uri": "s3://bucket/configs/train.yaml",
  "register_model": true
}
```

Response:
```json
{
  "run_id": "mlflow_run_id",
  "status": "started"
}
```

## Models

### GET `/models`

Response:
```json
{
  "models": [
    {"name": "snunet_cbam", "version": "12", "stage": "Production"}
  ]
}
```

### POST `/models/promote`

Request:
```json
{
  "name": "snunet_cbam",
  "version": "12",
  "stage": "Production"
}
```

Response:
```json
{
  "status": "ok"
}
```

## Health

- GET `/healthz`
- GET `/readyz`
