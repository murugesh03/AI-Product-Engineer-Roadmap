# Phase 3: Deep Learning

## 3.1 Neural Network Development
**Why it matters:** production AI products use trained deep models.
**What it is:** architecture design in PyTorch/TensorFlow.
**Goal:** working model with training loop and eval metrics.

### Steps
1. Build model class and dataset loader.
2. Implement training and validation loops.
3. Track metrics and log in tensorboard.
4. Save best checkpoints and test inference.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `ml/deep/train.py`
- `ml/deep/config.yaml`

---

## 3.2 Experiment Tracking
**Why it matters:** iterate faster with history and hyperparameters.
**What it is:** use MLflow/W&B or simple logs.
**Goal:** track at least 10 experiments and pick best model.

### Steps
1. Integrate MLflow/W&B into training script.
2. Log hyperparameters and metric curves.
3. Tag best runs and add notes.
4. Develop repeatable pipeline stage.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `ml/experiments/README.md`
