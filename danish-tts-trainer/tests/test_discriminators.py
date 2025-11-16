"""Tests for discriminator models."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.discriminators import MultiDiscriminator


def test_multi_discriminator_initialization():
    """Test multi-discriminator initializes correctly."""
    disc = MultiDiscriminator()

    # Should have MPD, MSD, and WavLM discriminators
    assert hasattr(disc, 'mpd')
    assert hasattr(disc, 'msd')
    assert hasattr(disc, 'wd')


def test_discriminator_forward():
    """Test discriminator forward pass."""
    disc = MultiDiscriminator()

    # Create fake and real audio
    batch_size = 2
    audio_length = 24000
    real_audio = torch.randn(batch_size, audio_length)
    fake_audio = torch.randn(batch_size, audio_length)

    # Forward pass
    with torch.no_grad():
        real_logits = disc(real_audio)
        fake_logits = disc(fake_audio)

    # Each should return list of logits from each discriminator
    assert isinstance(real_logits, list)
    assert isinstance(fake_logits, list)
    assert len(real_logits) == 3  # MPD + MSD + WavLM
    assert len(fake_logits) == 3


def test_discriminator_loss():
    """Test discriminator loss computation."""
    from danish_tts.losses import AdversarialLoss

    disc = MultiDiscriminator()
    loss_fn = AdversarialLoss()

    real_audio = torch.randn(2, 24000)
    fake_audio = torch.randn(2, 24000)

    with torch.no_grad():
        real_logits_groups = disc(real_audio)
        fake_logits_groups = disc(fake_audio)

    # Compute discriminator loss
    # Each group contains multiple sub-discriminator outputs
    disc_loss = 0
    for real_group, fake_group in zip(real_logits_groups, fake_logits_groups):
        # Iterate over sub-discriminators in each group
        for real_l, fake_l in zip(real_group, fake_group):
            disc_loss += loss_fn.forward_discriminator(real_l, fake_l)

    assert isinstance(disc_loss, torch.Tensor)
    assert disc_loss.ndim == 0  # Scalar
