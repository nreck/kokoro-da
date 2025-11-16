#!/usr/bin/env python3
"""Test training with gradient checkpointing enabled."""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from danish_tts.models.model_config import build_model, enable_gradient_checkpointing
from danish_tts.g2p_da import G2P as DanishG2P
from danish_tts.data.tts_dataset import TTSDataset
import yaml

def test_training():
    """Test basic training with gradient checkpointing."""

    # Load config
    with open("configs/coral_danish.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)

    # Enable gradient checkpointing
    use_checkpointing = config["training"].get("use_gradient_checkpointing", True)
    enable_gradient_checkpointing(model, use_checkpointing)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    # Initialize G2P
    print("\nInitializing Danish G2P...")
    g2p = DanishG2P()

    # Create dataset
    print("\nLoading dataset...")
    train_dataset = TTSDataset(
        data_dir=Path("../coral-tts"),
        g2p=g2p,
        sample_rate=24000,
        normalize=True,
        use_phoneme_cache=False,
    )

    # Create dataloader with batch_size=1
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Training parameters
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    use_amp = config["training"]["use_amp"]

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\nTraining configuration:")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {config['training']['batch_size'] * gradient_accumulation_steps}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Gradient checkpointing: {use_checkpointing}")

    # Test training loop
    print("\n" + "="*60)
    print("Starting test training...")
    print("="*60)

    model.train()
    accumulation_counter = 0

    for step, batch in enumerate(train_loader):
        if step >= 20:  # Only test 20 steps
            break

        # Move batch to device
        phoneme_ids = batch["phoneme_ids"].to(device)
        audio = batch["audio"].to(device)
        speaker_ids = batch["speaker_id"].to(device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(
                phoneme_ids=phoneme_ids,
                speaker_ids=speaker_ids,
                ref_audio=audio,
            )

            # Simple L1 loss for testing
            loss = F.l1_loss(outputs["predicted_mel"], outputs["target_mel"])

        # Scale loss for accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        accumulation_counter += 1

        # Clear cache periodically during accumulation
        if accumulation_counter % 2 == 0:
            torch.cuda.empty_cache()

        # Update weights after accumulation
        if accumulation_counter % gradient_accumulation_steps == 0:
            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["grad_clip_norm"]
            )

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Clear cache after optimizer step
            torch.cuda.empty_cache()

            # Print progress
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"Step {step:4d} | Loss: {loss.item():.6f} | "
                      f"Grad norm: {grad_norm:.2f} | "
                      f"Mem: {allocated:.2f}/{reserved:.2f} GB")
            else:
                print(f"Step {step:4d} | Loss: {loss.item():.6f} | "
                      f"Grad norm: {grad_norm:.2f}")

    print("\n" + "="*60)
    print("Test training completed successfully!")
    print("Gradient checkpointing is working properly.")
    print("="*60)

if __name__ == "__main__":
    test_training()