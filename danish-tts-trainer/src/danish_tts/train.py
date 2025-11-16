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


def setup_dataloader(config: dict, g2p, split: str = "train", full_dataset=None) -> DataLoader:
    """
    Create data loader.

    Args:
        config: Data configuration
        g2p: Danish G2P instance
        split: 'train' or 'val'
        full_dataset: Pre-created dataset to use (avoids recreating)

    Returns:
        DataLoader instance
    """
    from danish_tts.data.tts_dataset import TTSDataset
    from danish_tts.data.collate import collate_fn

    if full_dataset is None:
        data_dir = Path(config["data"]["coral_data_dir"])

        # Create dataset without caching (will cache separately)
        dataset = TTSDataset(
            data_dir=data_dir,
            g2p=g2p,
            sample_rate=config["data"]["sample_rate"],
            use_phoneme_cache=False,
        )
    else:
        dataset = full_dataset

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

    # Use fewer workers for validation to avoid espeak-ng thread safety issues
    num_workers = config["training"]["num_workers"] if split == "train" else 0

    dataloader = DataLoader(
        dataset_to_use,
        batch_size=config["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=num_workers,
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
    amp_context = None,
    scaler = None,
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

    # Forward pass with mixed precision
    if amp_context is not None:
        with amp_context:
            outputs = model(
                phoneme_ids=phoneme_ids,
                speaker_ids=speaker_ids,
                ref_audio=audio,  # Use ground truth as reference
                phoneme_lengths=phoneme_lengths,
            )
    else:
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
    loss_dur = torch.tensor(0.0, device=device)
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

    # Scale loss by gradient accumulation steps for proper averaging
    gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
    total_loss = total_loss / gradient_accumulation_steps

    # Backward pass with gradient scaling for AMP
    if do_backward:
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

    # Gradient clipping and optimizer step
    if do_step:
        if scaler is not None:
            scaler.unscale_(optimizer)  # Unscale before clipping

        if "grad_clip_norm" in config["training"]:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["grad_clip_norm"],
            )

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    return {
        "total_loss": total_loss.item() * gradient_accumulation_steps,  # Unscale for logging
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
    model.eval()
    total_loss = 0
    num_batches = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in val_loader:
            phoneme_ids = batch["phoneme_ids"].to(device)
            audio = batch["audio"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            phoneme_lengths = batch["phoneme_lengths"].to(device)

            # Forward pass
            outputs = model(
                phoneme_ids=phoneme_ids,
                speaker_ids=speaker_ids,
                ref_audio=audio,
                phoneme_lengths=phoneme_lengths,
            )

            # Compute reconstruction loss only for validation
            loss = torch.nn.functional.l1_loss(
                outputs["predicted_mel"],
                outputs["target_mel"],
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # Log to tensorboard
    writer.add_scalar("val/loss", avg_loss, step)

    model.train()
    return avg_loss


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
    print(f"Validation using single-threaded DataLoader (espeak-ng thread safety)")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # Setup mixed precision training
    use_amp = config["training"].get("use_amp", False)
    amp_dtype = config["training"].get("amp_dtype", "float16")
    scaler = None
    amp_context = None

    if use_amp:
        if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            print("Using bfloat16 mixed precision training")
            amp_context = torch.amp.autocast('cuda', dtype=torch.bfloat16)
        else:
            print("Using float16 mixed precision training")
            amp_context = torch.amp.autocast('cuda', dtype=torch.float16)
            scaler = torch.cuda.amp.GradScaler()
    else:
        print("Mixed precision training disabled")
        amp_context = torch.amp.autocast('cuda', enabled=False)

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

            # Pass gradient accumulation flags and AMP context to train_step
            loss_dict = train_step(
                model, batch, optimizer, config, losses,
                do_backward=True,  # Always backward for accumulation
                do_step=not is_accumulation_step,  # Only step when done accumulating
                amp_context=amp_context,
                scaler=scaler
            )

            # Only step optimizer after accumulating enough gradients
            if not is_accumulation_step:
                accumulation_counter = 0
                optimizer.zero_grad()  # Explicitly clear gradients after step
                # Clear cache after optimizer step to free memory
                if step % 5 == 0:  # Every 5 steps (more aggressive)
                    torch.cuda.empty_cache()
            else:
                accumulation_counter += 1
                # Clear cache during accumulation to prevent buildup
                if accumulation_counter % 2 == 0:  # Clear every 2 accumulations
                    torch.cuda.empty_cache()
                continue  # Skip discriminator and logging until accumulation is done

            # Discriminator step (every N steps, skip early steps for warmup)
            disc_update_freq = config["training"].get("disc_update_freq", 1)
            warmup_steps = 100  # Allow model to stabilize first
            if step >= warmup_steps and step % disc_update_freq == 0:
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

                # Check for NaN in audio before discriminator step
                if torch.isnan(fake_audio).any() or torch.isnan(real_audio).any():
                    print(f"Warning: NaN detected in audio at step {step}")
                    disc_loss_dict = {
                        "disc_loss": 0.0,
                        "disc_loss_real": 0.0,
                        "disc_loss_fake": 0.0,
                    }
                else:
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

            # Validation (skip at step 0 to avoid initial crashes)
            if step > 0 and step % config["training"]["val_interval"] == 0:
                val_loss = validate(model, val_loader, config, step, writer)
                print(f"Validation loss: {val_loss:.4f}")

            # Checkpointing (skip step 0)
            if step > 0 and step % config["training"]["checkpoint_interval"] == 0:
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
