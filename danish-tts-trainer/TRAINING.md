# Training Guide

## Quick Start

### 1. Start Full Training

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
./scripts/train_full.sh
```

### 2. Resume from Checkpoint

```bash
python src/danish_tts/train.py \
    --config configs/coral_danish.yaml \
    --resume checkpoints/checkpoint_00050000.pt \
    --experiment_name coral_danish_resumed
```

## Configuration Details

### Training Hyperparameters
- **Batch Size**: 32 (per GPU)
- **Gradient Accumulation**: 2 steps
- **Effective Batch Size**: 64
- **Workers**: 8 data loading workers
- **Max Steps**: 600,000 (~2-3 days on single GPU)
- **Learning Rate**: 2.0e-4 with warmup + cosine schedule
- **Mixed Precision**: bfloat16 (more stable than float16)

### Loss Weights
- **Reconstruction**: 45.0 (high weight for mel accuracy)
- **Adversarial**: 1.0
- **Style KL**: 1.0
- **Duration**: 1.0
- **Pitch**: 0.1

### Checkpointing
- **Save Interval**: Every 5,000 steps
- **Keep**: 10 most recent checkpoints
- **Validation**: Every 2,500 steps (20 samples)

### Discriminator
- **Multi-scale**: MPD + MSD + WavLM
- **WavLM Hidden**: 256
- **WavLM Layers**: 3
- **Update Frequency**: Every step
- **Learning Rate**: 2.0e-4

## Monitoring

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

### Check Logs
```bash
tail -f logs/coral_danish_*/training.log
```

## Output Structure

```
danish-tts-trainer/
├── checkpoints/
│   └── coral_danish_*/
│       ├── checkpoint_00005000.pt
│       ├── checkpoint_00010000.pt
│       └── ...
└── logs/
    └── coral_danish_*/
        ├── training.log
        ├── tensorboard/
        └── samples/
```

## Resume Training

The training script automatically saves:
- Model state
- Optimizer state
- Discriminator state
- Discriminator optimizer state
- Current step number
- Configuration

Resume with:
```bash
python src/danish_tts/train.py \
    --config configs/coral_danish.yaml \
    --resume checkpoints/coral_danish_*/checkpoint_NNNNNNNN.pt
```
