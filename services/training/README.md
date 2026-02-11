# Training Service

Production-grade training service for Land Cover Change Detection using SNUNet-CBAM.

## Features

- **Complete Training Pipeline**: Full implementation with data loading, augmentation, training, and validation
- **MLflow Integration**: Experiment tracking, metrics logging, and model registry
- **Comprehensive Metrics**: F1, IoU, Precision, Recall with automated evaluation
- **Early Stopping**: Smart training termination based on validation performance
- **GPU Optimization**: Automatic GPU detection and utilization
- **Configuration Management**: YAML-based flexible configuration
- **Asynchronous Training**: Non-blocking API with job status tracking
- **Model Versioning**: Automatic model registration and stage management

## API Endpoints

### Start Training Job
```bash
POST /train
{
  "dataset_uri": "s3://landcover/datasets/levir-cd",
  "config_uri": "s3://landcover/configs/train_config.yaml", 
  "register_model": true
}
```

### Check Training Status
```bash
GET /train/{job_id}
```

### List All Training Jobs
```bash
GET /train
```

## Configuration

Training parameters are configured via YAML files. See `config/train_config.yaml` for defaults.

Key parameters:
- `epochs`: Training epochs
- `batch_size`: Batch size  
- `lr`: Learning rate
- `base_channel`: Model complexity
- `use_attention`: Enable CBAM attention
- `weight_bce`/`weight_dice`: Loss function weights

## Model Architecture

- **SNUNet-CBAM**: Siamese Nested U-Net with Convolutional Block Attention Module
- **Hybrid Loss**: BCE + Dice loss combination
- **Data Augmentation**: Rotation, flipping, color jittering, Gaussian blur
- **Metrics**: IoU, F1, Precision, Recall calculated per batch and epoch

## Usage

The training service automatically:
1. Downloads datasets and configs from S3
2. Sets up data loaders with augmentation
3. Trains SNUNet-CBAM model
4. Logs metrics to MLflow
5. Registers best models to MLflow registry
6. Manages model staging (None → Staging → Production)

## Integration

Works seamlessly with:
- Gateway service for API orchestration
- MLflow for experiment tracking
- MinIO/S3 for dataset storage
- Kubernetes for orchestration

## Monitoring

All training metrics are logged to MLflow and can be visualized in the MLflow UI:
- Training/validation loss curves
- F1, IoU, Precision, Recall trends
- Learning rate schedules
- Model artifacts and versioning