"""Training script for Danish StyleTTS2."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict) -> nn.Module:
    """
    Initialize StyleTTS2 model.

    Args:
        config: Model configuration

    Returns:
        Model instance
    """
    # TODO: Integrate actual StyleTTS2 model
    # For now, placeholder
    raise NotImplementedError("Integrate StyleTTS2 model from yl4579/StyleTTS2")


def setup_dataloader(config: dict, g2p, split: str = "train") -> DataLoader:
    """
    Create data loader.

    Args:
        config: Data configuration
        g2p: Danish G2P instance
        split: 'train' or 'val'

    Returns:
        DataLoader instance
    """
    from danish_tts.data.tts_dataset import TTSDataset
    from danish_tts.data.collate import collate_fn

    data_dir = Path(config["data"]["coral_data_dir"])

    dataset = TTSDataset(
        data_dir=data_dir,
        g2p=g2p,
        sample_rate=config["data"]["sample_rate"],
    )

    # Split dataset into train/val
    # For now, use 95/5 split
    total_size = len(dataset)
    train_size = int(0.95 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    dataset_to_use = train_dataset if split == "train" else val_dataset

    dataloader = DataLoader(
        dataset_to_use,
        batch_size=config["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
    losses: dict,
    do_backward: bool = True,
    do_step: bool = True,
) -> dict:
    """Single training step.

    Args:
        model: StyleTTS2 model
        batch: Training batch from DataLoader
        optimizer: Optimizer
        config: Training configuration
        losses: Dictionary of loss functions
        do_backward: Whether to call backward()
        do_step: Whether to step the optimizer

    Returns:
        Dictionary with loss values
    """
    model.train()

    if do_step:
        optimizer.zero_grad()

    # Move batch to device
    device = next(model.parameters()).device
    phoneme_ids = batch["phoneme_ids"].to(device)
    audio = batch["audio"].to(device)
    speaker_ids = batch["speaker_ids"].to(device)
    phoneme_lengths = batch["phoneme_lengths"].to(device)
    audio_lengths = batch["audio_lengths"].to(device)

    # Forward pass
    outputs = model(
        phoneme_ids=phoneme_ids,
        speaker_ids=speaker_ids,
        ref_audio=audio,  # Use ground truth as reference
        phoneme_lengths=phoneme_lengths,
    )

    # Compute losses
    loss_weights = config["loss"]

    # Reconstruction loss (mel/STFT)
    loss_recon = losses["reconstruction"](
        outputs["predicted_mel"],
        outputs["target_mel"],
    )

    # Style KL divergence
    loss_kl = losses["kl_divergence"](
        outputs["style_mean"],
        outputs["style_log_var"],
    )

    # Duration loss (if model returns durations)
    loss_dur = 0
    if "durations" in outputs and "target_durations" in batch:
        loss_dur = losses["duration"](
            outputs["durations"],
            batch["target_durations"].to(device),
            phoneme_lengths,
        )

    # Total loss
    total_loss = (
        loss_weights["reconstruction"] * loss_recon +
        loss_weights["style_kl"] * loss_kl +
        loss_weights.get("duration", 1.0) * loss_dur
    )

    # Backward pass
    if do_backward:
        total_loss.backward()

    # Gradient clipping and optimizer step
    if do_step:
        if "grad_clip_norm" in config["training"]:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["grad_clip_norm"],
            )
        optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "reconstruction_loss": loss_recon.item(),
        "kl_loss": loss_kl.item(),
        "duration_loss": loss_dur.item() if isinstance(loss_dur, torch.Tensor) else 0,
    }


def discriminator_step(
    discriminator: nn.Module,
    real_audio: torch.Tensor,
    fake_audio: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
) -> dict:
    """Single discriminator training step.

    Args:
        discriminator: Multi-discriminator model
        real_audio: Real audio samples [batch, audio_samples]
        fake_audio: Generated audio samples [batch, audio_samples]
        optimizer: Discriminator optimizer
        loss_fn: AdversarialLoss instance

    Returns:
        Dictionary with discriminator losses
    """
    discriminator.train()
    optimizer.zero_grad()

    # Forward pass on real and fake audio
    real_logits = discriminator(real_audio)
    fake_logits = discriminator(fake_audio.detach())  # Detach to avoid generator gradients

    # Compute discriminator loss for each discriminator
    total_disc_loss = 0
    loss_real_total = 0
    loss_fake_total = 0

    for real_l_list, fake_l_list in zip(real_logits, fake_logits):
        # Each discriminator (MPD, MSD, WavLM) returns a list of logits
        # Handle both list and single tensor cases
        if isinstance(real_l_list, list):
            for real_l, fake_l in zip(real_l_list, fake_l_list):
                disc_loss = loss_fn.forward_discriminator(real_l, fake_l)
                total_disc_loss += disc_loss

                # Track individual components
                loss_real_total += torch.mean(F.relu(1.0 - real_l))
                loss_fake_total += torch.mean(F.relu(1.0 + fake_l))
        else:
            # Single tensor case
            disc_loss = loss_fn.forward_discriminator(real_l_list, fake_l_list)
            total_disc_loss += disc_loss

            loss_real_total += torch.mean(F.relu(1.0 - real_l_list))
            loss_fake_total += torch.mean(F.relu(1.0 + fake_l_list))

    # Backward pass
    total_disc_loss.backward()
    optimizer.step()

    return {
        "disc_loss": total_disc_loss.item(),
        "disc_loss_real": loss_real_total.item(),
        "disc_loss_fake": loss_fake_total.item(),
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    config: dict,
    step: int,
    writer: SummaryWriter,
) -> float:
    """
    Validation loop.

    Args:
        model: StyleTTS2 model
        val_loader: Validation data loader
        config: Configuration
        step: Current training step
        writer: TensorBoard writer

    Returns:
        Average validation loss
    """
    # TODO: Implement validation
    raise NotImplementedError("Implement validation")


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train Danish StyleTTS2")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="danish_tts",
        help="Experiment name for logging",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Training for {config['training']['max_steps']} steps")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize G2P
    print("Loading Danish G2P...")
    from danish_tts.g2p_da import G2P
    g2p = G2P()

    # Setup model
    print("Building model...")
    from danish_tts.models.model_config import build_model
    model = build_model(config)
    model = model.to(device)

    # Setup dataloaders
    print("Setting up dataloaders...")
    train_loader = setup_dataloader(config, g2p, split="train")
    val_loader = setup_dataloader(config, g2p, split="val")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # Setup losses
    from danish_tts.losses import ReconstructionLoss, KLDivergenceLoss, AdversarialLoss, DurationLoss
    losses = {
        "reconstruction": ReconstructionLoss(loss_type="l1"),
        "kl_divergence": KLDivergenceLoss(),
        "adversarial": AdversarialLoss(loss_type="hinge"),
        "duration": DurationLoss(),
    }

    # Setup discriminator
    print("Building discriminator...")
    from danish_tts.discriminators import MultiDiscriminator
    discriminator = MultiDiscriminator(
        use_wavlm=config.get("discriminator", {}).get("use_wavlm", True),
        wavlm_hidden=config.get("discriminator", {}).get("wavlm_hidden", 768),
        wavlm_nlayers=config.get("discriminator", {}).get("wavlm_nlayers", 13),
        wavlm_initial_channel=config.get("discriminator", {}).get("wavlm_initial_channel", 64),
    )
    discriminator = discriminator.to(device)

    # Discriminator optimizer
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.get("discriminator", {}).get("learning_rate", 2.0e-4),
    )

    # Setup TensorBoard
    writer = SummaryWriter(config["logging"]["tensorboard_dir"])

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume:
        print(f"\n{'='*50}")
        print(f"Resuming from checkpoint: {args.resume}")
        print(f"{'='*50}")

        if not args.resume.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        checkpoint = torch.load(args.resume, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model"])
        print("  Loaded model state")

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("  Loaded optimizer state")

        # Load discriminator if present
        if "discriminator" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator"])
            print("  Loaded discriminator state")

        # Load discriminator optimizer if present
        if "disc_optimizer" in checkpoint:
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
            print("  Loaded discriminator optimizer state")

        # Resume from step
        start_step = checkpoint.get("step", 0)
        print(f"\n  Resumed from step {start_step}")
        print(f"  Remaining steps: {max_steps - start_step}")
        print(f"{'='*50}\n")

    # Training loop
    step = start_step
    max_steps = config["training"]["max_steps"]
    gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
    accumulation_counter = 0

    print(f"\nStarting training from step {step}...")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {config['training']['batch_size'] * gradient_accumulation_steps}\n")

    while step < max_steps:
        for batch in tqdm(train_loader, desc=f"Step {step}/{max_steps}"):
            # Generator step (with gradient accumulation)
            is_accumulation_step = (accumulation_counter < gradient_accumulation_steps - 1)

            loss_dict = train_step(model, batch, optimizer, config, losses)

            # Only step optimizer after accumulating enough gradients
            if not is_accumulation_step:
                accumulation_counter = 0
            else:
                accumulation_counter += 1
                continue  # Skip discriminator and logging until accumulation is done

            # Discriminator step (every N steps)
            disc_update_freq = config["training"].get("disc_update_freq", 1)
            if step % disc_update_freq == 0:
                # Generate fake audio from model
                with torch.no_grad():
                    outputs = model(
                        phoneme_ids=batch["phoneme_ids"].to(device),
                        speaker_ids=batch["speaker_ids"].to(device),
                        ref_audio=batch["audio"].to(device),
                        phoneme_lengths=batch["phoneme_lengths"].to(device),
                    )
                    # Assume model now outputs audio (with iSTFTNet decoder)
                    # For now, use predicted_mel as placeholder since we don't have full audio decoder yet
                    # In production: fake_audio = outputs.get("predicted_audio", outputs.get("audio"))
                    fake_audio = outputs.get("predicted_audio", batch["audio"].to(device))

                real_audio = batch["audio"].to(device)

                disc_loss_dict = discriminator_step(
                    discriminator=discriminator,
                    real_audio=real_audio,
                    fake_audio=fake_audio,
                    optimizer=disc_optimizer,
                    loss_fn=losses["adversarial"],
                )

                loss_dict.update(disc_loss_dict)

            # Logging
            if step % config["logging"]["log_interval"] == 0:
                for key, value in loss_dict.items():
                    writer.add_scalar(f"train/{key}", value, step)

                print(f"Step {step}: {loss_dict}")

            # Validation
            if step % config["training"]["val_interval"] == 0:
                val_loss = validate(model, val_loader, config, step, writer)
                print(f"Validation loss: {val_loss:.4f}")

            # Checkpointing
            if step % config["training"]["checkpoint_interval"] == 0:
                checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = checkpoint_dir / f"checkpoint_{step:08d}.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "disc_optimizer": disc_optimizer.state_dict(),
                    "config": config,
                }, checkpoint_path)

                print(f"Saved checkpoint: {checkpoint_path}")

            step += 1
            if step >= max_steps:
                break

    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    main()
