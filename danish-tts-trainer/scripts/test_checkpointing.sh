#!/bin/bash

# Test training with gradient checkpointing

set -e  # Exit on error

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0  # Set GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation

# Set espeak-ng paths (macOS)
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
export PHONEMIZER_ESPEAK_PATH=/opt/homebrew/bin/espeak-ng
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

echo "========================================="
echo "Testing gradient checkpointing"
echo "Date: $(date)"
echo "========================================="

# Run test
python test_training_with_checkpointing.py

echo "Test complete!"