#!/usr/bin/env python3
"""
Distributed training script for Danish StyleTTS2 with multi-GPU support.

Supports both single-GPU and multi-GPU training using PyTorch DistributedDataParallel.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import yaml
import torchaudio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from danish_tts.models.model_config import build_model, enable_gradient_checkpointing
from danish_tts.models.discriminators import MultiScaleDiscriminator, MultiPeriodDiscriminator, WavLMDiscriminator
from danish_tts.data.tts_dataset import TTSDataset
from danish_tts.g2p_da import G2P

logger = logging.getLogger(__name__)


def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def setup_logging(rank, experiment_name):
    """Setup logging for each process."""
    log_level = logging.INFO if rank == 0 else logging.WARNING

    logging.basicConfig(
        level=log_level,
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    return logging.getLogger(__name__)


def compute_losses(outputs, config, device):
    """Compute all training losses."""
    losses = {}
    loss_weights = config["loss"]

    # Reconstruction loss (L1)
    if outputs["predicted_mel"] is not None and outputs["target_mel"] is not None:
        losses["reconstruction"] = F.l1_loss(
            outputs["predicted_mel"],
            outputs["target_mel"]
        ) * loss_weights["reconstruction"]
    else:
        losses["reconstruction"] = torch.tensor(0.0, device=device, requires_grad=True)

    # KL divergence loss for style VAE
    if "style_mean" in outputs and "style_log_var" in outputs:
        kl_loss = -0.5 * torch.mean(
            1 + outputs["style_log_var"] - outputs["style_mean"].pow(2) - outputs["style_log_var"].exp()
        )
        losses["style_kl"] = kl_loss * loss_weights["style_kl"]
    else:
        losses["style_kl"] = torch.tensor(0.0, device=device, requires_grad=True)

    # Duration loss
    if "durations" in outputs:
        # Simple L1 loss on log durations for now
        target_durations = torch.ones_like(outputs["durations"]) * 0.1
        losses["duration"] = F.l1_loss(
            outputs["durations"],
            target_durations
        ) * loss_weights["duration"]
    else:
        losses["duration"] = torch.tensor(0.0, device=device, requires_grad=True)

    # Pitch loss
    if "pitch" in outputs and outputs["pitch"] is not None:
        target_pitch = torch.zeros_like(outputs["pitch"])
        losses["pitch"] = F.l1_loss(
            outputs["pitch"],
            target_pitch
        ) * loss_weights.get("pitch", 0.1)
    else:
        losses["pitch"] = torch.tensor(0.0, device=device, requires_grad=True)

    # Total generator loss (without adversarial)
    losses["total_g"] = (
        losses["reconstruction"] +
        losses["style_kl"] +
        losses["duration"] +
        losses["pitch"]
    )

    return losses


def train_epoch(rank, world_size, model, train_loader, optimizer, scaler,
                discriminator, disc_optimizer, config, epoch, global_step, writer):
    """Train for one epoch in distributed mode."""
    model.train()
    discriminator.train() if discriminator else None

    accumulation_steps = config["training"]["gradient_accumulation_steps"]
    use_amp = config["training"]["use_amp"]
    disc_warmup_steps = 100
    accumulation_counter = 0

    # Set sampler epoch for proper shuffling
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(train_loader):
        # Move batch to device
        phoneme_ids = batch["phoneme_ids"].to(rank)
        audio = batch["audio"].to(rank)
        speaker_ids = batch["speaker_id"].to(rank)

        # Generator forward pass
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(
                phoneme_ids=phoneme_ids,
                speaker_ids=speaker_ids,
                ref_audio=audio,
            )

            # Compute losses
            losses = compute_losses(outputs, config, rank)

            # Add discriminator loss if warmed up
            if discriminator and global_step > disc_warmup_steps:
                # Get discriminator predictions on generated audio
                fake_audio = outputs.get("predicted_audio", audio)

                # Discriminator on fake (for generator training)
                disc_fake = discriminator(fake_audio)
                disc_loss_g = -torch.mean(disc_fake)  # Maximize discriminator output
                losses["adversarial"] = disc_loss_g * config["loss"]["adversarial"]
                losses["total_g"] = losses["total_g"] + losses["adversarial"]
            else:
                losses["adversarial"] = torch.tensor(0.0, device=rank)

        # Scale loss for gradient accumulation
        loss = losses["total_g"] / accumulation_steps

        # Generator backward pass
        scaler.scale(loss).backward()
        accumulation_counter += 1

        # Clear cache periodically during accumulation
        if accumulation_counter % 2 == 0:
            torch.cuda.empty_cache()

        # Update generator weights after accumulation
        if accumulation_counter % accumulation_steps == 0:
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

            # Train discriminator if available and warmed up
            if discriminator and global_step > disc_warmup_steps:
                disc_optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    # Real audio
                    real_audio = audio.detach()
                    disc_real = discriminator(real_audio)
                    loss_d_real = F.relu(1.0 - disc_real).mean()

                    # Fake audio (detached from generator)
                    fake_audio = outputs.get("predicted_audio", audio).detach()
                    disc_fake = discriminator(fake_audio)
                    loss_d_fake = F.relu(1.0 + disc_fake).mean()

                    # Total discriminator loss
                    loss_d = (loss_d_real + loss_d_fake) / 2

                # Discriminator backward
                scaler.scale(loss_d).backward()

                # Discriminator optimizer step
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    config["training"]["grad_clip_norm"]
                )
                scaler.step(disc_optimizer)
                scaler.update()
            else:
                loss_d = torch.tensor(0.0, device=rank)

            # Clear cache after optimizer steps
            torch.cuda.empty_cache()

            # Logging (only from rank 0)
            if rank == 0:
                if global_step % config["logging"]["log_interval"] == 0:
                    # Log to tensorboard
                    writer.add_scalar("loss/reconstruction", losses["reconstruction"].item(), global_step)
                    writer.add_scalar("loss/style_kl", losses["style_kl"].item(), global_step)
                    writer.add_scalar("loss/duration", losses["duration"].item(), global_step)
                    writer.add_scalar("loss/pitch", losses["pitch"].item(), global_step)
                    writer.add_scalar("loss/adversarial", losses["adversarial"].item(), global_step)
                    writer.add_scalar("loss/total_g", losses["total_g"].item(), global_step)
                    writer.add_scalar("loss/discriminator", loss_d.item(), global_step)
                    writer.add_scalar("training/grad_norm", grad_norm.item(), global_step)
                    writer.add_scalar("training/learning_rate", optimizer.param_groups[0]["lr"], global_step)

                    # Memory stats
                    allocated = torch.cuda.memory_allocated(rank) / 1e9
                    reserved = torch.cuda.memory_reserved(rank) / 1e9
                    writer.add_scalar("memory/allocated_gb", allocated, global_step)
                    writer.add_scalar("memory/reserved_gb", reserved, global_step)

                    # Print progress
                    logger.info(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"Loss: {losses['total_g'].item():.4f} | "
                        f"Rec: {losses['reconstruction'].item():.4f} | "
                        f"KL: {losses['style_kl'].item():.4f} | "
                        f"Adv: {losses['adversarial'].item():.4f} | "
                        f"Disc: {loss_d.item():.4f} | "
                        f"Grad: {grad_norm:.2f} | "
                        f"Mem: {allocated:.2f}/{reserved:.2f} GB"
                    )

            global_step += 1

            # Checkpoint saving (only from rank 0)
            if rank == 0 and global_step % config["training"]["checkpoint_interval"] == 0:
                save_checkpoint(
                    model.module if hasattr(model, 'module') else model,
                    discriminator.module if hasattr(discriminator, 'module') else discriminator,
                    optimizer, disc_optimizer, global_step, epoch, config
                )

    return global_step


def save_checkpoint(model, discriminator, optimizer, disc_optimizer, global_step, epoch, config):
    """Save training checkpoint."""
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{global_step}.pt"

    checkpoint = {
        "model": model.state_dict(),
        "discriminator": discriminator.state_dict() if discriminator else None,
        "optimizer": optimizer.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer else None,
        "global_step": global_step,
        "epoch": epoch,
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Keep only last N checkpoints
    keep_n = config["training"].get("keep_n_checkpoints", 10)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if len(checkpoints) > keep_n:
        for old_ckpt in checkpoints[:-keep_n]:
            old_ckpt.unlink()
            logger.info(f"Removed old checkpoint {old_ckpt}")


def train_distributed(rank, world_size, config, experiment_name):
    """Main distributed training function for each process."""
    # Setup distributed environment
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Setup logging for this rank
    logger = setup_logging(rank, experiment_name)

    if rank == 0:
        logger.info(f"Starting distributed training with {world_size} GPUs")
        logger.info(f"Experiment: {experiment_name}")

    # Initialize model
    logger.info(f"Rank {rank}: Building model...")
    model = build_model(config).to(rank)

    # Enable gradient checkpointing if configured
    if config["training"].get("use_gradient_checkpointing", True):
        enable_gradient_checkpointing(model, True)
        if rank == 0:
            logger.info("Gradient checkpointing enabled")

    # Wrap model with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # Initialize discriminator
    if config["discriminator"]["use_wavlm"]:
        discriminator = WavLMDiscriminator(
            hidden=config["discriminator"]["wavlm_hidden"],
            nlayers=config["discriminator"]["wavlm_nlayers"],
            initial_channel=config["discriminator"]["wavlm_initial_channel"],
        ).to(rank)

        if world_size > 1:
            discriminator = DDP(discriminator, device_ids=[rank], output_device=rank)
    else:
        discriminator = None

    # Initialize G2P
    logger.info(f"Rank {rank}: Initializing Danish G2P...")
    g2p = G2P()

    # Create datasets
    logger.info(f"Rank {rank}: Loading datasets...")
    train_dataset = TTSDataset(
        data_dir=Path(config["data"]["coral_data_dir"]),
        g2p=g2p,
        sample_rate=config["data"]["sample_rate"],
        normalize=True,
        use_phoneme_cache=(rank != 0),  # Cache phonemes for non-primary ranks
    )

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None

    # Scale batch size for multi-GPU
    # Each GPU processes batch_size samples
    per_gpu_batch_size = config["training"]["batch_size"]

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Setup optimizers
    # Scale learning rate for effective batch size
    effective_batch_size = (
        per_gpu_batch_size *
        config["training"]["gradient_accumulation_steps"] *
        world_size
    )
    base_lr = config["training"]["learning_rate"]
    scaled_lr = base_lr * (effective_batch_size / 4)  # Linear scaling from base batch size of 4

    if rank == 0:
        logger.info(f"Per-GPU batch size: {per_gpu_batch_size}")
        logger.info(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Base learning rate: {base_lr}")
        logger.info(f"Scaled learning rate: {scaled_lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config["discriminator"]["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ) if discriminator else None

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["use_amp"])

    # Tensorboard writer (only rank 0)
    writer = None
    if rank == 0:
        log_dir = Path(config["logging"]["tensorboard_dir"]) / experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)

    # Training loop
    global_step = 0
    max_steps = config["training"]["max_steps"]

    epoch = 0
    while global_step < max_steps:
        epoch += 1

        if rank == 0:
            logger.info(f"Starting epoch {epoch}")

        global_step = train_epoch(
            rank, world_size, model, train_loader, optimizer, scaler,
            discriminator, disc_optimizer, config, epoch, global_step, writer
        )

        if global_step >= max_steps:
            break

    # Final checkpoint
    if rank == 0:
        save_checkpoint(
            model.module if hasattr(model, 'module') else model,
            discriminator.module if hasattr(discriminator, 'module') else discriminator,
            optimizer, disc_optimizer, global_step, epoch, config
        )
        logger.info("Training completed!")
        writer.close()

    # Cleanup
    if world_size > 1:
        cleanup_distributed()


def main():
    """Main entry point for distributed training."""
    parser = argparse.ArgumentParser(description="Train Danish StyleTTS2 with multi-GPU support")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update config with resume checkpoint if provided
    if args.resume_from:
        config["paths"]["resume_from"] = args.resume_from

    # Generate experiment name if not provided
    if args.experiment_name is None:
        experiment_name = f"coral_danish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        experiment_name = args.experiment_name

    # Determine number of GPUs
    if args.num_gpus is not None:
        world_size = min(args.num_gpus, torch.cuda.device_count())
    else:
        world_size = torch.cuda.device_count()

    if world_size == 0:
        raise RuntimeError("No GPUs available for training")

    print(f"Training with {world_size} GPU(s)")

    if world_size > 1:
        # Multi-GPU training
        mp.spawn(
            train_distributed,
            args=(world_size, config, experiment_name),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        train_distributed(0, 1, config, experiment_name)


if __name__ == "__main__":
    main()