"""Tests for discriminator training step."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.train import discriminator_step


def test_discriminator_step():
    """Test discriminator training step."""
    from danish_tts.discriminators import MultiDiscriminator
    from danish_tts.losses import AdversarialLoss

    # Create discriminator and optimizer
    discriminator = MultiDiscriminator(use_wavlm=False)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # Create fake batch
    batch_size = 2
    real_audio = torch.randn(batch_size, 24000)
    fake_audio = torch.randn(batch_size, 24000)

    # Training step
    loss_fn = AdversarialLoss()
    loss_dict = discriminator_step(
        discriminator=discriminator,
        real_audio=real_audio,
        fake_audio=fake_audio,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    # Check outputs
    assert "disc_loss" in loss_dict
    assert "disc_loss_real" in loss_dict
    assert "disc_loss_fake" in loss_dict
    assert loss_dict["disc_loss"] > 0
