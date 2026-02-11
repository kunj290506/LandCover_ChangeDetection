# Quick Start Guide

## Local Development

1. **Start the stack**:
   ```bash
   docker compose up
   ```

2. **Get JWT token**:
   ```bash
   curl -X POST http://localhost:8000/v1/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"admin123"}'
   ```

3. **Start training**:
   ```bash
   curl -X POST http://localhost:8000/v1/train \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_uri": "data/LEVIR-CD-patches",
       "config_uri": "config/train_config.yaml",
       "register_model": true
     }'
   ```

4. **Monitor progress**:
   - MLflow UI: http://localhost:5000
   - Grafana: http://localhost:3000

## Production Deployment

1. **Deploy to Kubernetes**:
   ```bash
   kubectl apply -k infra/k8s/overlays/prod
   ```

2. **Port-forward services** (if needed):
   ```bash
   kubectl port-forward svc/gateway 8000:8000 -n landcover
   kubectl port-forward svc/mlflow 5000:5000 -n landcover
   ```

3. **Scale services**:
   ```bash
   kubectl scale deployment gateway --replicas=3 -n landcover
   ```

## Training Configuration

Edit `config/train_config.yaml` or provide your own via S3:

```yaml
epochs: 100
batch_size: 16
lr: 3e-4
base_channel: 64
use_attention: true
```

## Model Performance

The training pipeline will automatically:
- Calculate IoU, F1, Precision, Recall
- Log metrics to MLflow
- Register models with F1 > 0.5
- Promote best models to Staging

Expected performance on LEVIR-CD:
- **F1 Score**: 0.65-0.75
- **IoU**: 0.48-0.60 
- **Training Time**: ~2-4 hours on GPU

## Troubleshooting

**OOM errors**: Reduce batch_size in config
**Slow training**: Ensure GPU nodes available in K8s
**Model not registering**: Check F1 > min_f1_for_registration threshold