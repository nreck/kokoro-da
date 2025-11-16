#!/usr/bin/env python3
"""Test multi-GPU setup and configuration."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time

def test_gpu_setup(rank, world_size):
    """Test function for each GPU."""
    # Initialize distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"Rank {rank}/{world_size} | GPU {device}: {gpu_name} | Memory: {gpu_memory:.1f} GB")

    # Test tensor operations
    tensor_size = (1024, 1024, 256)  # ~1GB tensor
    print(f"Rank {rank} | Creating {tensor_size} tensor...")

    x = torch.randn(tensor_size, device=device)
    print(f"Rank {rank} | Tensor created, memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    # Test all-reduce operation
    dist.all_reduce(x)
    print(f"Rank {rank} | All-reduce completed")

    # Clean up
    del x
    torch.cuda.empty_cache()

    dist.destroy_process_group()

def test_data_distribution():
    """Test data distribution across GPUs."""
    print("\n" + "="*60)
    print("Testing Data Distribution for Multi-GPU Training")
    print("="*60)

    # Simulate dataset
    dataset_size = 17919  # CoRal dataset size
    batch_size_per_gpu = 8
    num_gpus = 4
    gradient_accumulation = 2

    effective_batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation

    print(f"Dataset size: {dataset_size} samples")
    print(f"Batch size per GPU: {batch_size_per_gpu}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Gradient accumulation steps: {gradient_accumulation}")
    print(f"Effective batch size: {effective_batch_size}")

    # Calculate training metrics
    samples_per_gpu_per_epoch = dataset_size // num_gpus
    batches_per_gpu_per_epoch = samples_per_gpu_per_epoch // batch_size_per_gpu
    optimizer_steps_per_epoch = batches_per_gpu_per_epoch // gradient_accumulation

    print(f"\nPer Epoch:")
    print(f"  Samples per GPU: {samples_per_gpu_per_epoch}")
    print(f"  Batches per GPU: {batches_per_gpu_per_epoch}")
    print(f"  Optimizer steps: {optimizer_steps_per_epoch}")

    # Calculate time estimates
    time_per_step = 0.5  # Estimated seconds per optimizer step with 4 GPUs
    time_per_epoch = optimizer_steps_per_epoch * time_per_step
    total_steps = 600000
    epochs_needed = total_steps / optimizer_steps_per_epoch
    total_time_hours = (total_steps * time_per_step) / 3600

    print(f"\nTraining Estimates:")
    print(f"  Time per epoch: {time_per_epoch/60:.1f} minutes")
    print(f"  Epochs needed for {total_steps} steps: {epochs_needed:.1f}")
    print(f"  Total training time: {total_time_hours:.1f} hours")

    # Memory estimates
    model_size_gb = 0.5  # Approximate model size
    batch_memory_gb = 0.8  # Memory per sample
    total_memory_per_gpu = model_size_gb + (batch_size_per_gpu * batch_memory_gb)

    print(f"\nMemory Estimates per GPU:")
    print(f"  Model size: {model_size_gb:.1f} GB")
    print(f"  Batch memory: {batch_size_per_gpu * batch_memory_gb:.1f} GB")
    print(f"  Total expected: {total_memory_per_gpu:.1f} GB")
    print(f"  Available: 32.0 GB per GPU")
    print(f"  Headroom: {32.0 - total_memory_per_gpu:.1f} GB")

def test_multigpu():
    """Main test function."""
    print("="*60)
    print("Multi-GPU Setup Test")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA devices")

    if num_gpus == 0:
        print("ERROR: No GPUs found!")
        return

    # Show all GPUs
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # Test distributed training
    if num_gpus > 1:
        print(f"\nTesting distributed training with {num_gpus} GPUs...")
        try:
            mp.spawn(test_gpu_setup, args=(num_gpus,), nprocs=num_gpus, join=True)
            print("✓ Distributed training test PASSED")
        except Exception as e:
            print(f"✗ Distributed training test FAILED: {e}")
    else:
        print("\nOnly 1 GPU found, skipping distributed test")

    # Test data distribution
    test_data_distribution()

    print("\n" + "="*60)
    print("Multi-GPU setup test completed!")
    print("="*60)

    # Recommendations based on GPU count
    if num_gpus == 4:
        print("\n📌 Recommendations for 4x RTX 5090:")
        print("  • Use batch_size=8 per GPU (32 total)")
        print("  • Use gradient_accumulation=2 (effective batch=64)")
        print("  • Disable gradient checkpointing for speed")
        print("  • Enable mixed precision (bfloat16)")
        print("  • Use the coral_danish_multigpu.yaml config")
        print("\nTo start training:")
        print("  ./scripts/train_multigpu.sh")
    elif num_gpus == 1:
        print("\n📌 Recommendations for single GPU:")
        print("  • Use batch_size=1")
        print("  • Use gradient_accumulation=4")
        print("  • Enable gradient checkpointing")
        print("  • Use the coral_danish.yaml config")
        print("\nTo start training:")
        print("  ./scripts/train_full.sh")

if __name__ == "__main__":
    test_multigpu()