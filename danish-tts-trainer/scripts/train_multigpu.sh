#!/bin/bash

# Multi-GPU Training Script for Danish TTS
# Optimized for 4x RTX 5090 (127.4GB total VRAM)

set -e  # Exit on error

# Configuration
CONFIG="${1:-configs/coral_danish_multigpu.yaml}"  # Default to multi-GPU config
EXPERIMENT_NAME="coral_danish_4gpu_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXPERIMENT_NAME}"
NUM_GPUS="${2:-4}"  # Default to 4 GPUs, can override with second argument

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "checkpoints/${EXPERIMENT_NAME}"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# CUDA settings for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation

# NCCL settings for better multi-GPU communication
export NCCL_DEBUG=INFO  # Enable NCCL debugging
export NCCL_TREE_THRESHOLD=0  # Optimize for small messages

# Set espeak-ng paths (Linux)
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so
export PHONEMIZER_ESPEAK_PATH=/usr/bin/espeak-ng

# Print system info
echo "========================================="
echo "Starting Multi-GPU Danish TTS Training"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Config: ${CONFIG}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Date: $(date)"
echo "========================================="

# Check GPU availability
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "========================================="

# Show GPU topology for multi-GPU setup
nvidia-smi topo -m
echo "========================================="

# Start distributed training
echo "Launching distributed training on ${NUM_GPUS} GPUs..."
python src/danish_tts/train_distributed.py \
    --config "${CONFIG}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --num_gpus ${NUM_GPUS} \
    2>&1 | tee "${LOG_DIR}/training.log"

echo "Training complete! Logs saved to ${LOG_DIR}"