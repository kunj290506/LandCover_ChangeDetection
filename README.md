# Land Cover Change Detection

Deep learning system for bi-temporal satellite image analysis using Siamese Nested U-Net with CBAM attention mechanisms.

## Overview

This system analyzes pairs of satellite images from different time periods to detect and map land cover changes. It leverages a Siamese architecture with nested skip connections and CBAM attention modules for accurate change detection in remote sensing data.

## Project Structure

```
LandCover_ChangeDetection/
├── src/
│   ├── models/          # SNUNet, CBAM, and baseline architectures
│   ├── utils/           # Loss functions, metrics, utilities
│   ├── dataset.py       # Dataset loader with augmentation
│   ├── train.py         # Standard training script
│   ├── train_baseline.py
│   ├── train_kfold.py   # K-fold cross-validation
│   ├── tune_hyperparams.py
│   ├── inference.py
│   └── compare_models.py
├── scripts/
│   └── download_data.py
├── train_fixed.py       # Optimized training (recommended)
├── train_list.txt
├── val_list.txt
└── test_list.txt

Enterprise deployment assets are in `services/`, `infra/`, and `docs/`.

**Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for rapid deployment.

## Enterprise Services

- **Gateway API**: FastAPI service with JWT auth, RBAC, rate limiting
- **Inference Service**: GPU-enabled real-time change detection
- **Training Service**: Production ML training with MLflow tracking  
- **Batch Worker**: Asynchronous processing for large datasets
- **MLflow**: Experiment tracking and model registry
- **Monitoring**: Prometheus + Grafana observability stack
```

## Requirements

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn tqdm
```

Requires Python 3.8+ and PyTorch 1.10+.

## Dataset

Expected structure for LEVIR-CD or similar bi-temporal datasets:

```
data/LEVIR-CD/
├── train/
│   ├── A/          # Time 1 images
│   ├── B/          # Time 2 images
│   └── label/      # Change masks
├── val/
└── test/
```

Data list format (train_list.txt, val_list.txt, test_list.txt):
```
train/A/image_001.png train/B/image_001.png train/label/image_001.png
```

## Training

### Quick Start

```bash
python train_fixed.py --epochs 50 --batch_size 8
```

### Custom Configuration

```bash
python train_fixed.py \
    --data_root ./data/LEVIR-CD \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --base_channel 32
```

### Standard Training (High-performance systems)

```bash
python src/train.py \
    --data_root ./data/LEVIR-CD \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-4
```

### Hyperparameter Optimization

```bash
python src/tune_hyperparams.py --n_trials 100
```

### K-Fold Cross-Validation

```bash
python src/train_kfold.py --n_folds 5
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| --epochs | 50 | Number of training epochs |
| --batch_size | 8 | Batch size |
| --lr | 1e-4 | Learning rate |
| --base_channel | 32 | Model complexity (16/32/64) |
| --checkpoint_dir | ./checkpoints_optimized | Output directory |
| --use_attention | True | Enable CBAM attention |

## Inference

Single image pair:
```bash
python src/inference.py \
    --model_path checkpoints_optimized/best_model.pth \
    --image1 before.png \
    --image2 after.png \
    --output change_map.png
```

Batch processing:
```bash
python src/inference.py \
    --model_path checkpoints_optimized/best_model.pth \
    --input_dir ./image_pairs/ \
    --output_dir ./results/
```

## Model Architecture

**SNUNet-CBAM**: Siamese Nested U-Net with Convolutional Block Attention Module

Architecture components:
- Shared-weight Siamese encoder (5 levels, VGG-style blocks)
- Nested skip connections (UNet++ architecture)
- CBAM attention modules (channel + spatial attention)
- Hybrid loss function (BCE + Dice)

Model variants by base_channel parameter:
- base_channel=16: ~500K parameters (lightweight)
- base_channel=32: ~11.7M parameters (standard)
- base_channel=64: ~47M parameters (heavy)

## Performance

Results on LEVIR-CD dataset (256x256 patches):

| Model | F1 Score | IoU | Parameters |
|-------|----------|-----|------------|
| SNUNet-CBAM (base=32) | 0.65 | 0.48 | 11.7M |
| FC-EF Baseline | 0.42 | 0.26 | 1.5M |

Training configuration: AdamW optimizer, Cosine Annealing LR scheduler, 50-100 epochs with early stopping.

## Advanced Usage

**Resume training:**
```bash
python train_fixed.py --resume --checkpoint_dir ./checkpoints_optimized
```

**CPU-only mode:**
```bash
python train_fixed.py --cpu --batch_size 4
```

**Quick testing (3 epochs):**
```bash
python train_fixed.py --epochs 3
```

## Troubleshooting

**Out of Memory:**
- Reduce batch size: `--batch_size 4` or lower
- Use smaller model: `--base_channel 16`
- Disable attention: `--no_attention`

**Slow Training:**
- Ensure CUDA is available and being used
- Increase batch size if memory permits
- Use fewer data augmentations

**Poor Performance:**
- Train longer: `--epochs 150`
- Increase model capacity: `--base_channel 64`
- Run hyperparameter optimization
- Verify data quality and augmentation settings

## License

MIT License

## Citation

```
@misc{landcover_change_detection,
  title={Land Cover Change Detection using SNUNet-CBAM},
  author={Your Name},
  year={2026},
  url={https://github.com/username/LandCover_ChangeDetection}
```

## References

- LEVIR-CD Dataset: https://justchenhao.github.io/LEVIR/
- SNUNet: "SNUNet-CD: A Densely Connected Siamese Network for Change Detection"
- CBAM: "CBAM: Convolutional Block Attention Module"

## Contact

For questions or issues, please open an issue on GitHub.
