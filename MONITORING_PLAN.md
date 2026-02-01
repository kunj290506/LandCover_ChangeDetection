# Training Monitoring Plan

## Current Status (Epoch ~6-7)
- Best Val F1: 0.6662
- Training Loss: Decreasing (1.54 → 1.43)
- Status: Healthy learning trajectory

## Target Metrics
- **F1 Score:** >0.85 (current: 0.6662)
- **IoU:** >0.75 (current: ~0.48)
- **Gap to Target F1:** 0.1838 (need +28% improvement)

## Monitoring Strategy

### Automatic Checkpoints
Will check training progress at:
- **Epoch 10** (Est. ~1.5h from start)
- **Epoch 15** (Est. ~2.5h from start)  
- **Epoch 20** (Est. ~3.5h from start)
- **Epoch 30** (Est. ~5h from start)
- **Epoch 40** (Est. ~6.5h from start)
- **Epoch 50** (Final - Est. ~7.5h from start)

### Metrics to Track Per Epoch

**Primary Metrics:**
- Validation F1 Score (target: >0.85)
- Validation IoU (target: >0.75)
- Validation Kappa (target: >0.6)

**Loss Metrics:**
- Training Loss (target: 0.15-0.35)
- Validation Loss (target: 0.15-0.35)
- Loss Gap (train-val, alert if >0.1)

**Performance Gap:**
- Train F1 vs Val F1 (alert if gap >0.15)

## Intervention Triggers

### DO NOT INTERVENE if:
- ✓ Metrics improving steadily
- ✓ Loss gap <0.1
- ✓ F1 gap <0.15
- ✓ No plateau >5 epochs

### INTERVENE if:
- ✗ Val metrics plateau for >5 consecutive epochs
- ✗ Train/Val F1 gap >0.15 (overfitting)
- ✗ Val loss increases while train loss decreases
- ✗ Metrics degrade

### Intervention Actions (if needed)
1. **For Plateau:** 
   - Adjust learning rate schedule
   - Consider increasing model capacity

2. **For Overfitting (gap >0.15):**
   - Increase regularization (weight decay)
   - Add dropout layers
   - Strengthen data augmentation

3. **For Underfitting:**
   - Increase base_channels to 64
   - Train longer
   - Reduce regularization

## Success Criteria

**Excellent Performance:**
- Val F1 >0.85
- Val IoU >0.75
- Kappa >0.6
- Train/Val gap <0.10

**Good Performance:**
- Val F1 >0.75
- Val IoU >0.60
- Kappa >0.4
- Stable improvement trend

**Needs Intervention:**
- Val F1 <0.70 after 20 epochs
- Plateau >5 epochs
- Large train/val gap

## Current Recommendations

**Action:** Let training continue uninterrupted for next 20-30 epochs

**Reasoning:**
- Model showing healthy improvement
- Only 2-3% improvement so far (early stage)
- No signs of overfitting or underfitting
- GPU training stable
- Plenty of room for improvement

**Next Review:** Epoch 15-20 for trend analysis

---
Last Updated: 2026-02-01 22:28 IST
