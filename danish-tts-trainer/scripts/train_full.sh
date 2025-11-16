#!/bin/bash

# Full CoRal Danish TTS Training Script

set -e  # Exit on error

# Configuration
CONFIG="configs/coral_danish.yaml"
EXPERIMENT_NAME="coral_danish_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXPERIMENT_NAME}"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "checkpoints/${EXPERIMENT_NAME}"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0  # Set GPU

# Set espeak-ng paths (Linux or macOS)
if [ -f "/usr/lib/x86_64-linux-gnu/libespeak-ng.so" ]; then
    # Linux
    export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so
    export PHONEMIZER_ESPEAK_PATH=/usr/bin/espeak-ng
else
    # macOS
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
    export PHONEMIZER_ESPEAK_PATH=/opt/homebrew/bin/espeak-ng
fi

# Print system info
echo "========================================="
echo "Starting Danish TTS Training"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Config: ${CONFIG}"
echo "Date: $(date)"
echo "========================================="

# Check if GPU is available (skip on macOS)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "Running on CPU (no nvidia-smi found)"
fi

echo "========================================="

# Start training
python src/danish_tts/train.py \
    --config "${CONFIG}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    2>&1 | tee "${LOG_DIR}/training.log"

echo "Training complete! Logs saved to ${LOG_DIR}"
